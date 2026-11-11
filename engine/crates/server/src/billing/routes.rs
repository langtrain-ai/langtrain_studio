//! HTTP routes for accounts, keys, credits and payment webhooks.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use super::payments::{self, Paid};
use super::store::{Currency, MICRO};
use super::{format_money, Billing, Caller};
use crate::state::{ApiError, AppState};

fn billing(state: &AppState) -> Result<&Arc<Billing>, ApiError> {
    state.billing.as_ref().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "billing is not enabled on this server".into()))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ").map(str::trim)
}

/// Require a valid API key; the caller is attached to the request.
pub async fn require_key(State(state): State<Arc<AppState>>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let b = billing(&state)?;
    let key = bearer(req.headers()).ok_or_else(|| {
        ApiError(StatusCode::UNAUTHORIZED, "missing API key: send `Authorization: Bearer lt_live_...`".into())
    })?;
    let (account, key_id) = b
        .store
        .authenticate(key)?
        .ok_or_else(|| ApiError(StatusCode::UNAUTHORIZED, "invalid or revoked API key".into()))?;
    req.extensions_mut().insert(Caller { account, key_id });
    Ok(next.run(req).await)
}

/// Require the operator's admin token (`LANGTRAIN_ADMIN_TOKEN`).
pub async fn require_admin(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Result<Response, ApiError> {
    let b = billing(&state)?;
    let expected = b.secrets.admin_token.as_deref().ok_or_else(|| {
        ApiError(StatusCode::FORBIDDEN, "admin endpoints are disabled: set LANGTRAIN_ADMIN_TOKEN".into())
    })?;
    match bearer(req.headers()) {
        // Compare hashes so the check takes the same time for any input.
        Some(t) if super::store::hash_key(t) == super::store::hash_key(expected) => Ok(next.run(req).await),
        _ => Err(ApiError(StatusCode::FORBIDDEN, "admin token required".into())),
    }
}

/// Public: prices, packs and which payment providers are live.
pub async fn config(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    Ok(Json(json!({
        "open_signup": b.config.open_signup,
        "packs": b.config.packs,
        "prices": b.config.default_prices,
        "model_prices": b.config.model_prices,
        "providers": {
            "razorpay": b.secrets.razorpay_ready(),
            "stripe": b.secrets.stripe_ready(),
        },
        "razorpay_key_id": b.secrets.razorpay_key_id,
        "default_max_tokens": b.config.default_max_tokens,
        "max_tokens_cap": b.config.max_tokens_cap,
    })))
}

#[derive(Deserialize)]
pub struct Signup {
    email: String,
    #[serde(default)]
    currency: Option<String>,
}

/// Create an account and its first API key. The key is shown once.
pub async fn signup(State(state): State<Arc<AppState>>, Json(req): Json<Signup>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    if !b.config.open_signup {
        return Err(ApiError(StatusCode::FORBIDDEN, "sign-up is closed; ask the operator for an account".into()));
    }
    let currency = Currency::parse(req.currency.as_deref().unwrap_or("INR")).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let account = b.store.create_account(&req.email, currency).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let trial = match currency {
        Currency::INR => b.config.signup_credit_inr,
        Currency::USD => b.config.signup_credit_usd,
    };
    if trial > 0 {
        b.store.credit(&account.id, "bonus", trial * MICRO, None, Some("trial credit"), Some(&format!("trial:{}", account.id)))?;
    }
    let (key, secret) = b.store.create_key(&account.id, "default")?;
    Ok(Json(json!({ "account": account, "key": key, "api_key": secret })))
}

pub async fn account(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    let bal = b.store.balance(&c.account.id)?;
    Ok(Json(json!({
        "account": c.account,
        "balance": bal.available,
        "held": bal.held,
        "balance_display": format_money(bal.available, c.account.currency),
        "prices": b.config.default_prices.get(c.account.currency),
    })))
}

pub async fn list_keys(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    Ok(Json(json!({ "data": b.store.keys(&c.account.id)?, "current": c.key_id })))
}

#[derive(Deserialize, Default)]
pub struct NewKey {
    #[serde(default)]
    name: String,
}

pub async fn create_key(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>, body: Option<Json<NewKey>>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    let name = body.map(|Json(k)| k.name).unwrap_or_default();
    if b.store.keys(&c.account.id)?.iter().filter(|k| !k.revoked).count() >= 20 {
        return Err(ApiError::bad_request("an account can have at most 20 active keys"));
    }
    let (key, secret) = b.store.create_key(&c.account.id, &name)?;
    Ok(Json(json!({ "key": key, "api_key": secret })))
}

pub async fn revoke_key(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    if id == c.key_id {
        return Err(ApiError::bad_request("this is the key you are using; revoke it from another key"));
    }
    if !b.store.revoke_key(&c.account.id, &id)? {
        return Err(ApiError(StatusCode::NOT_FOUND, "key not found".into()));
    }
    Ok(Json(json!({ "revoked": id })))
}

pub async fn ledger(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    Ok(Json(json!({ "data": b.store.ledger(&c.account.id, 100)?, "currency": c.account.currency })))
}

pub async fn usage(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    Ok(Json(json!({ "data": b.store.usage(&c.account.id, 100)?, "currency": c.account.currency })))
}

#[derive(Deserialize)]
pub struct Checkout {
    pack_id: String,
}

/// Start a payment for a credit pack. Credits are added by the webhook.
pub async fn checkout(State(state): State<Arc<AppState>>, Extension(c): Extension<Caller>, Json(req): Json<Checkout>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    let pack = b
        .config
        .packs
        .iter()
        .find(|p| p.id == req.pack_id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown pack '{}'", req.pack_id)))?;
    if pack.currency != c.account.currency {
        return Err(ApiError::bad_request(format!("this account pays in {}", c.account.currency.code())));
    }
    let product = format!("Langtrain credits {}{}", pack.currency.symbol(), pack.amount);
    // Razorpay for rupees (UPI and Indian cards), Stripe for dollars; either as fallback.
    let use_razorpay = match pack.currency {
        Currency::INR => b.secrets.razorpay_ready() || !b.secrets.stripe_ready(),
        Currency::USD => !b.secrets.stripe_ready() && b.secrets.razorpay_ready(),
    };
    if use_razorpay && b.secrets.razorpay_ready() {
        let receipt = format!("lt_{}", &uuid::Uuid::new_v4().simple().to_string()[..20]);
        let order = payments::create_razorpay_order(&b.http, &b.secrets, pack.amount_minor(), pack.currency, &receipt, &c.account.id).await?;
        b.store.record_payment(&order, "razorpay", &c.account.id, &pack.id, pack.amount_minor(), pack.currency)?;
        return Ok(Json(json!({
            "provider": "razorpay",
            "order_id": order,
            "key_id": b.secrets.razorpay_key_id,
            "amount": pack.amount_minor(),
            "currency": pack.currency.code(),
            "name": "Langtrain",
            "description": product,
            "email": c.account.email,
        })));
    }
    if b.secrets.stripe_ready() {
        let (id, url) = payments::create_stripe_session(
            &b.http,
            &b.secrets,
            pack.amount_minor(),
            pack.currency,
            &product,
            &c.account.id,
            &c.account.email,
            &format!("{}/", b.config.public_url.trim_end_matches('/')),
        )
        .await?;
        b.store.record_payment(&id, "stripe", &c.account.id, &pack.id, pack.amount_minor(), pack.currency)?;
        return Ok(Json(json!({ "provider": "stripe", "url": url })));
    }
    Err(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "payments are not set up on this server yet (the operator needs to configure Razorpay or Stripe)".into(),
    ))
}

/// Apply a verified payment: credit the pack (and bonus) exactly once.
fn apply_payment(b: &Billing, provider: &str, paid: &Paid) -> Result<(), ApiError> {
    let Some((recorded_provider, account_id, pack_id, amount_minor, currency, _status)) = b.store.payment(&paid.checkout_id)? else {
        tracing::warn!(provider, checkout = %paid.checkout_id, "webhook for a checkout this server did not create; ignored");
        return Ok(());
    };
    if recorded_provider != provider || amount_minor != paid.amount_minor || currency.code() != paid.currency {
        tracing::error!(provider, checkout = %paid.checkout_id, "webhook amount or currency does not match the checkout; not credited");
        return Ok(());
    }
    let Some(pack) = b.config.packs.iter().find(|p| p.id == pack_id) else {
        tracing::error!(pack = %pack_id, "paid pack no longer configured; credit it manually");
        return Ok(());
    };
    let key = format!("{provider}:{}", paid.checkout_id);
    let applied = b.store.credit(&account_id, "topup", pack.credit_micro(), Some(&paid.checkout_id), Some(&format!("{} pack", pack.id)), Some(&key))?;
    if pack.bonus_micro() > 0 {
        b.store.credit(&account_id, "bonus", pack.bonus_micro(), Some(&paid.checkout_id), Some(&format!("{}% bonus", pack.bonus_pct)), Some(&format!("{key}:bonus")))?;
    }
    if applied {
        b.store.mark_paid(&paid.checkout_id)?;
        tracing::info!(provider, account = %account_id, pack = %pack.id, "credits added");
    }
    Ok(())
}

pub async fn razorpay_webhook(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Result<StatusCode, ApiError> {
    let b = billing(&state)?;
    let secret = b.secrets.razorpay_webhook_secret.as_deref().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Razorpay is not configured".into()))?;
    let sig = headers.get("x-razorpay-signature").and_then(|v| v.to_str().ok()).unwrap_or_default();
    if !payments::verify_razorpay(secret, &body, sig) {
        return Err(ApiError(StatusCode::BAD_REQUEST, "invalid signature".into()));
    }
    let event: Value = serde_json::from_slice(&body).map_err(|_| ApiError::bad_request("invalid JSON"))?;
    if let Some(paid) = payments::razorpay_paid(&event) {
        apply_payment(b, "razorpay", &paid)?;
    }
    Ok(StatusCode::OK)
}

pub async fn stripe_webhook(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Result<StatusCode, ApiError> {
    let b = billing(&state)?;
    let secret = b.secrets.stripe_webhook_secret.as_deref().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "Stripe is not configured".into()))?;
    let sig = headers.get("stripe-signature").and_then(|v| v.to_str().ok()).unwrap_or_default();
    if !payments::verify_stripe(secret, &body, sig, crate::state::unix_now() as i64) {
        return Err(ApiError(StatusCode::BAD_REQUEST, "invalid signature".into()));
    }
    let event: Value = serde_json::from_slice(&body).map_err(|_| ApiError::bad_request("invalid JSON"))?;
    if let Some(paid) = payments::stripe_paid(&event) {
        apply_payment(b, "stripe", &paid)?;
    }
    Ok(StatusCode::OK)
}

pub async fn admin_accounts(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    let rows: Vec<Value> = b
        .store
        .accounts()?
        .into_iter()
        .map(|(a, bal)| json!({ "account": a, "balance": bal.available, "held": bal.held, "balance_display": format_money(bal.available, a.currency) }))
        .collect();
    Ok(Json(json!({ "data": rows })))
}

#[derive(Deserialize)]
pub struct Grant {
    email: String,
    /// Whole currency units.
    amount: f64,
    #[serde(default)]
    note: Option<String>,
}

/// Add credit by hand (refunds, goodwill, offline payments).
pub async fn admin_grant(State(state): State<Arc<AppState>>, Json(g): Json<Grant>) -> Result<Json<Value>, ApiError> {
    let b = billing(&state)?;
    let account = b.store.account_by_email(&g.email)?.ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "no account with that email".into()))?;
    let micro = (g.amount * MICRO as f64).round() as i64;
    if micro <= 0 {
        return Err(ApiError::bad_request("amount must be positive"));
    }
    b.store.credit(&account.id, "grant", micro, None, g.note.as_deref(), None)?;
    let bal = b.store.balance(&account.id)?;
    Ok(Json(json!({ "account": account.id, "balance_display": format_money(bal.available, account.currency) })))
}
