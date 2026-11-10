//! `langtraind`: the local HTTP daemon.
//!
//! - OpenAI-compatible: `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
//!   `/v1/fine_tuning/jobs`
//! - Ollama-compatible: `/api/tags`, `/api/chat`, `/api/generate`, `/api/version`
//! - Studio web UI at `/` (from the `langtrain-web` crate)

pub mod billing;
pub mod ollama;
pub mod openai;
pub mod state;
pub mod tune;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

pub use state::{AppState, BackendChoice, ServerConfig};

pub fn router(state: Arc<AppState>) -> Router {
    let cloud = state.billing.is_some();

    let inference = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate));

    let jobs = Router::new()
        .route("/v1/fine_tuning/jobs", post(tune::create).get(tune::list))
        .route("/v1/fine_tuning/jobs/{id}", get(tune::get))
        .route("/v1/fine_tuning/jobs/{id}/events", get(tune::events))
        .route("/v1/fine_tuning/jobs/{id}/cancel", post(tune::cancel));

    let public = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/system", get(system))
        .route("/v1/models", get(openai::models))
        .route("/api/version", get(ollama::version))
        .route("/api/tags", get(ollama::tags))
        .merge(langtrain_web::router());

    let app = if cloud {
        use billing::routes as b;
        let key = axum::middleware::from_fn_with_state(state.clone(), b::require_key);
        let admin = axum::middleware::from_fn_with_state(state.clone(), b::require_admin);
        let account = Router::new()
            .route("/v1/account", get(b::account))
            .route("/v1/keys", get(b::list_keys).post(b::create_key))
            .route("/v1/keys/{id}", axum::routing::delete(b::revoke_key))
            .route("/v1/billing/checkout", post(b::checkout))
            .route("/v1/billing/ledger", get(b::ledger))
            .route("/v1/usage", get(b::usage));
        let operator = Router::new()
            .route("/v1/admin/accounts", get(b::admin_accounts))
            .route("/v1/admin/grant", post(b::admin_grant))
            // Cloud fine-tuning for customers is Phase 2; until then only the operator runs jobs.
            .merge(jobs);
        public
            .route("/v1/billing/config", get(b::config))
            .route("/v1/signup", post(b::signup))
            .route("/v1/billing/webhooks/razorpay", post(b::razorpay_webhook))
            .route("/v1/billing/webhooks/stripe", post(b::stripe_webhook))
            .merge(inference.merge(account).route_layer(key))
            .merge(operator.route_layer(admin))
    } else {
        public.merge(inference).merge(jobs)
    };

    // Browser tools and the Tauri webview call the API cross-origin.
    app.layer(CorsLayer::permissive()).with_state(state)
}

async fn system(State(state): State<Arc<AppState>>) -> Json<Value> {
    if state.billing.is_some() {
        // Public in cloud mode: no hardware details or file paths.
        return Json(json!({
            "version": env!("CARGO_PKG_VERSION"),
            "cloud": true,
            "models": state.list_models().into_iter().map(|m| json!({ "name": m.name, "adapters": m.adapters, "size_bytes": m.size_bytes })).collect::<Vec<_>>(),
        }));
    }
    let loaded = state.loaded_model().await.map(|(model, mut v)| {
        v["model"] = json!(model);
        v
    });
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "hardware": state.config.hardware,
        "profile": state.config.profile,
        "models_dir": state.config.models_dir(),
        "models": state.list_models(),
        "loaded": loaded,
        "cloud": false,
    }))
}

pub async fn serve(config: ServerConfig, addr: SocketAddr) -> anyhow::Result<()> {
    for dir in [config.models_dir(), config.jobs_dir()] {
        std::fs::create_dir_all(&dir)?;
    }
    let state = AppState::new(config)?;
    if state.billing.is_some() {
        tracing::info!("Langtrain Cloud mode: API keys and prepaid credits required");
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Langtrain Studio on http://{}", listener.local_addr()?);
    axum::serve(listener, router(state)).with_graceful_shutdown(shutdown()).await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use langtrain_engine::hardware;
    use tower::ServiceExt;

    fn app() -> Router {
        let hw = hardware::detect();
        let profile = hw.profile();
        let home = std::env::temp_dir().join(format!("langtrain-test-{}", uuid::Uuid::new_v4().simple()));
        router(AppState::with_billing(ServerConfig {
            home,
            enable_mock: true,
            langtune_dir: None,
            python: "python3".into(),
            hardware: hw,
            profile,
            backend: BackendChoice::Auto,
            cloud: false,
        }, None))
    }

    async fn call(app: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, String) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(if body.is_null() { Body::empty() } else { Body::from(body.to_string()) })
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn openai_chat_non_streaming() {
        let app = app();
        let (s, body) = call(&app, "POST", "/v1/chat/completions",
            json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}], "temperature": 0})).await;
        assert_eq!(s, StatusCode::OK, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Langtrain.");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn openai_chat_streaming_with_adapter_and_usage() {
        let app = app();
        let (s, body) = call(&app, "POST", "/v1/chat/completions", json!({
            "model": "mock@pirate", "stream": true, "temperature": 0,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        })).await;
        assert_eq!(s, StatusCode::OK);
        let mut text = String::new();
        let mut saw_usage = false;
        for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if line == "[DONE]" { continue; }
            let v: Value = serde_json::from_str(line).unwrap();
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str() { text.push_str(t); }
            saw_usage |= v.get("usage").is_some();
        }
        assert_eq!(text, "[pirate] Hello from Langtrain.");
        assert!(saw_usage);
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }

    #[tokio::test]
    async fn ollama_chat_streams_ndjson() {
        let app = app();
        let (s, body) = call(&app, "POST", "/api/chat",
            json!({"model": "mock:latest", "messages": [{"role": "user", "content": "hi"}], "options": {"temperature": 0}})).await;
        assert_eq!(s, StatusCode::OK);
        let lines: Vec<Value> = body.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let text: String = lines.iter().filter_map(|v| v["message"]["content"].as_str()).collect();
        assert_eq!(text, "Hello from Langtrain.");
        let last = lines.last().unwrap();
        assert_eq!(last["done"], true);
        assert!(last["eval_count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn ollama_generate_non_streaming_and_tags() {
        let app = app();
        let (s, body) = call(&app, "POST", "/api/generate",
            json!({"model": "mock", "prompt": "hi", "stream": false, "options": {"temperature": 0, "num_predict": 5}})).await;
        assert_eq!(s, StatusCode::OK);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["response"], "Hello");
        assert_eq!(v["done_reason"], "length");

        let (_, body) = call(&app, "GET", "/api/tags", Value::Null).await;
        assert!(body.contains("\"mock@pirate\""));
    }

    #[tokio::test]
    async fn errors_are_openai_shaped() {
        let app = app();
        let (s, body) = call(&app, "POST", "/v1/chat/completions",
            json!({"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]})).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(body.contains("\"error\""));
        let (s, _) = call(&app, "POST", "/v1/chat/completions",
            json!({"model": "mock@nope", "messages": [{"role": "user", "content": "hi"}]})).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ui_and_system_are_served() {
        let app = app();
        let (s, body) = call(&app, "GET", "/", Value::Null).await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("Langtrain Studio"));
        let (s, body) = call(&app, "GET", "/v1/system", Value::Null).await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("\"profile\""));
    }

    #[tokio::test]
    async fn fine_tune_validates_inputs() {
        let app = app();
        let (s, _) = call(&app, "POST", "/v1/fine_tuning/jobs", json!({"model": "mock", "training_file": "x.jsonl"})).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = call(&app, "POST", "/v1/fine_tuning/jobs", json!({"model": "missing", "training_file": "x.jsonl"})).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod cloud_tests {
    use super::*;
    use crate::billing::payments::{sign_razorpay, sign_stripe, Secrets};
    use crate::billing::store::{Currency, Store, MICRO};
    use crate::billing::{Billing, Caller, CloudConfig};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use langtrain_engine::hardware;
    use langtrain_engine::sampling::SamplingParams;
    use langtrain_engine::{ChatMessage, Event, Prompt};
    use tower::ServiceExt;

    fn cloud() -> (Router, Arc<AppState>) {
        let hw = hardware::detect();
        let profile = hw.profile();
        let secrets = Secrets {
            razorpay_key_id: Some("rzp_test_key".into()),
            razorpay_key_secret: Some("rzp_test_secret".into()),
            razorpay_webhook_secret: Some("rzp_whsec".into()),
            stripe_secret_key: Some("sk_test".into()),
            stripe_webhook_secret: Some("whsec_stripe".into()),
            admin_token: Some("admin-token".into()),
        };
        let billing = Arc::new(Billing::new(Store::in_memory().unwrap(), CloudConfig::default(), secrets));
        let home = std::env::temp_dir().join(format!("langtrain-cloud-test-{}", uuid::Uuid::new_v4().simple()));
        let state = AppState::with_billing(
            ServerConfig { home, enable_mock: true, langtune_dir: None, python: "python3".into(), hardware: hw, profile, backend: BackendChoice::Auto, cloud: true },
            Some(billing),
        );
        (router(state.clone()), state)
    }

    async fn call(app: &Router, method: &str, uri: &str, auth: Option<&str>, headers: &[(&str, &str)], body: &str) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
        if let Some(a) = auth {
            req = req.header("authorization", format!("Bearer {a}"));
        }
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let res = app.clone().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    const CHAT: &str = r#"{"model": "mock", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}"#;

    async fn signup(app: &Router, email: &str) -> String {
        let (s, v) = call(app, "POST", "/v1/signup", None, &[], &format!(r#"{{"email": "{email}", "currency": "INR"}}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        v["api_key"].as_str().unwrap().to_string()
    }

    async fn balance(app: &Router, key: &str) -> i64 {
        call(app, "GET", "/v1/account", Some(key), &[], "").await.1["balance"].as_i64().unwrap()
    }

    #[tokio::test]
    async fn pay_first_flow_with_razorpay() {
        let (app, state) = cloud();
        // No key, no generation.
        let (s, _) = call(&app, "POST", "/v1/chat/completions", None, &[], CHAT).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
        let (s, _) = call(&app, "POST", "/v1/chat/completions", Some("lt_live_bogus"), &[], CHAT).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);

        // A new account has no credit: the request is refused before it runs.
        let key = signup(&app, "buyer@example.com").await;
        let (s, v) = call(&app, "POST", "/v1/chat/completions", Some(&key), &[], CHAT).await;
        assert_eq!(s, StatusCode::PAYMENT_REQUIRED, "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains("insufficient credit"));

        // The customer pays for the ₹2,000 pack (5% bonus). Checkout normally
        // records the order; record it directly since tests have no network.
        let b = state.billing.as_ref().unwrap();
        let acct = b.store.account_by_email("buyer@example.com").unwrap().unwrap();
        b.store.record_payment("order_test1", "razorpay", &acct.id, "inr-2000", 200_000, Currency::INR).unwrap();
        let event = r#"{"event":"order.paid","payload":{"payment":{"entity":{"id":"pay_1","order_id":"order_test1","amount":200000,"currency":"INR","status":"captured"}}}}"#;

        // A forged signature is rejected and credits nothing.
        let (s, _) = call(&app, "POST", "/v1/billing/webhooks/razorpay", None, &[("x-razorpay-signature", "00ff")], event).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(balance(&app, &key).await, 0);

        let sig = sign_razorpay("rzp_whsec", event.as_bytes());
        for _ in 0..2 {
            // Providers retry webhooks; the second delivery must not double-credit.
            let (s, _) = call(&app, "POST", "/v1/billing/webhooks/razorpay", None, &[("x-razorpay-signature", &sig)], event).await;
            assert_eq!(s, StatusCode::OK);
        }
        assert_eq!(balance(&app, &key).await, 2100 * MICRO);

        // A webhook whose amount does not match the order is ignored.
        b.store.record_payment("order_test2", "razorpay", &acct.id, "inr-500", 50_000, Currency::INR).unwrap();
        let cheap = r#"{"event":"order.paid","payload":{"payment":{"entity":{"id":"pay_2","order_id":"order_test2","amount":100,"currency":"INR","status":"captured"}}}}"#;
        let sig2 = sign_razorpay("rzp_whsec", cheap.as_bytes());
        call(&app, "POST", "/v1/billing/webhooks/razorpay", None, &[("x-razorpay-signature", &sig2)], cheap).await;
        assert_eq!(balance(&app, &key).await, 2100 * MICRO);

        // Now the request runs and is charged its actual cost.
        let (s, v) = call(&app, "POST", "/v1/chat/completions", Some(&key), &[], CHAT).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Langtrain.");
        // Settlement happens on the metering task; give it a moment.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let after = balance(&app, &key).await;
        let usage = call(&app, "GET", "/v1/usage", Some(&key), &[], "").await.1;
        let row = &usage["data"][0];
        let cost = row["cost"].as_i64().unwrap();
        assert!(cost > 0 && after == 2100 * MICRO - cost, "cost {cost}, balance {after}");
        assert_eq!(row["completion_tokens"], 21);
        let acct_view = call(&app, "GET", "/v1/account", Some(&key), &[], "").await.1;
        assert_eq!(acct_view["held"], 0);

        let ledger = call(&app, "GET", "/v1/billing/ledger", Some(&key), &[], "").await.1;
        let kinds: Vec<&str> = ledger["data"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["charge", "bonus", "topup"]);
    }

    #[tokio::test]
    async fn stripe_webhook_credits_once() {
        let (app, state) = cloud();
        let (_, v) = call(&app, "POST", "/v1/signup", None, &[], r#"{"email": "us@example.com", "currency": "USD"}"#).await;
        let key = v["api_key"].as_str().unwrap().to_string();
        let b = state.billing.as_ref().unwrap();
        let acct = b.store.account_by_email("us@example.com").unwrap().unwrap();
        b.store.record_payment("cs_test_1", "stripe", &acct.id, "usd-10", 1000, Currency::USD).unwrap();
        let event = r#"{"type":"checkout.session.completed","data":{"object":{"id":"cs_test_1","payment_status":"paid","amount_total":1000,"currency":"usd"}}}"#;
        let header = sign_stripe("whsec_stripe", event.as_bytes(), crate::state::unix_now() as i64);
        for _ in 0..2 {
            let (s, _) = call(&app, "POST", "/v1/billing/webhooks/stripe", None, &[("stripe-signature", &header)], event).await;
            assert_eq!(s, StatusCode::OK);
        }
        assert_eq!(balance(&app, &key).await, 10 * MICRO);
        // An old signature is a replay and is refused.
        let stale = sign_stripe("whsec_stripe", event.as_bytes(), crate::state::unix_now() as i64 - 3600);
        let (s, _) = call(&app, "POST", "/v1/billing/webhooks/stripe", None, &[("stripe-signature", &stale)], event).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn disconnected_client_is_charged_for_what_ran() {
        let (app, state) = cloud();
        let key = signup(&app, "dropper@example.com").await;
        let b = state.billing.clone().unwrap();
        let acct = b.store.account_by_email("dropper@example.com").unwrap().unwrap();
        b.store.credit(&acct.id, "grant", 100 * MICRO, None, None, None).unwrap();
        let (caller_acct, key_id) = b.store.authenticate(&key).unwrap().unwrap();
        let caller = Caller { account: caller_acct, key_id };

        let msgs = vec![ChatMessage::new("user", "hi")];
        let mut rx = state.generate_for(Some(&caller), "mock", Prompt::Chat(msgs), SamplingParams::greedy()).await.unwrap();
        assert!(matches!(rx.recv().await, Some(Event::Text(_))));
        drop(rx); // the client goes away mid-answer
        for _ in 0..50 {
            if b.store.balance(&acct.id).unwrap().held == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let bal = b.store.balance(&acct.id).unwrap();
        assert_eq!(bal.held, 0, "hold must be settled after a disconnect");
        let usage = b.store.usage(&acct.id, 5).unwrap();
        assert_eq!(usage.len(), 1);
        assert!(usage[0].cost > 0 && bal.available == 100 * MICRO - usage[0].cost);
    }

    #[tokio::test]
    async fn keys_admin_and_limits() {
        let (app, state) = cloud();
        let key = signup(&app, "team@example.com").await;
        let (s, v) = call(&app, "POST", "/v1/keys", Some(&key), &[], r#"{"name": "ci"}"#).await;
        assert_eq!(s, StatusCode::OK);
        let second = v["api_key"].as_str().unwrap().to_string();
        let second_id = v["key"]["id"].as_str().unwrap().to_string();
        let keys = call(&app, "GET", "/v1/keys", Some(&key), &[], "").await.1;
        assert_eq!(keys["data"].as_array().unwrap().len(), 2);
        let (s, _) = call(&app, "DELETE", &format!("/v1/keys/{second_id}"), Some(&key), &[], "").await;
        assert_eq!(s, StatusCode::OK);
        let (s, _) = call(&app, "GET", "/v1/account", Some(&second), &[], "").await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);

        // Operator endpoints need the admin token, not a customer key.
        let grant = r#"{"email": "team@example.com", "amount": 50, "note": "refund"}"#;
        let (s, _) = call(&app, "POST", "/v1/admin/grant", Some(&key), &[], grant).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, v) = call(&app, "POST", "/v1/admin/grant", Some("admin-token"), &[], grant).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(balance(&app, &key).await, 50 * MICRO);
        let (s, _) = call(&app, "GET", "/v1/fine_tuning/jobs", Some(&key), &[], "").await;
        assert_eq!(s, StatusCode::FORBIDDEN);

        // Per-account concurrency limit.
        let b = state.billing.clone().unwrap();
        let (a, kid) = b.store.authenticate(&key).unwrap().unwrap();
        let caller = Caller { account: a, key_id: kid };
        let held: Vec<_> = (0..b.config.max_concurrent_per_account).map(|_| b.admit(&caller, MICRO).unwrap().unwrap()).collect();
        assert!(matches!(b.admit(&caller, MICRO).unwrap(), Err(crate::billing::Refusal::TooManyRequests { .. })));
        drop(held);
        assert!(b.admit(&caller, MICRO).unwrap().is_ok());
        assert_eq!(b.store.balance(&caller.account.id).unwrap().held, 0);

        // Public system info hides hardware details in cloud mode.
        let (_, sys) = call(&app, "GET", "/v1/system", None, &[], "").await;
        assert_eq!(sys["cloud"], true);
        assert!(sys.get("hardware").is_none());
    }
}
