//! `langtraind`: the local HTTP daemon.
//!
//! - OpenAI-compatible: `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
//!   `/v1/fine_tuning/jobs`
//! - Ollama-compatible: `/api/tags`, `/api/chat`, `/api/generate`, `/api/version`
//! - Studio web UI at `/` (from the `langtrain-web` crate)

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
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/system", get(system))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/fine_tuning/jobs", post(tune::create).get(tune::list))
        .route("/v1/fine_tuning/jobs/{id}", get(tune::get))
        .route("/v1/fine_tuning/jobs/{id}/events", get(tune::events))
        .route("/v1/fine_tuning/jobs/{id}/cancel", post(tune::cancel))
        .route("/api/version", get(ollama::version))
        .route("/api/tags", get(ollama::tags))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .merge(langtrain_web::router())
        // Browser tools and the Tauri webview call the API cross-origin.
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn system(State(state): State<Arc<AppState>>) -> Json<Value> {
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
    }))
}

pub async fn serve(config: ServerConfig, addr: SocketAddr) -> anyhow::Result<()> {
    for dir in [config.models_dir(), config.jobs_dir()] {
        std::fs::create_dir_all(&dir)?;
    }
    let state = AppState::new(config);
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
        router(AppState::new(ServerConfig {
            home,
            enable_mock: true,
            langtune_dir: None,
            python: "python3".into(),
            hardware: hw,
            profile,
            backend: BackendChoice::Auto,
        }))
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
