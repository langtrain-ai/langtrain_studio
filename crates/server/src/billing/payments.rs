//! Razorpay and Stripe: creating checkouts and verifying webhooks.
//!
//! Credits are added only from a verified webhook for a checkout we created,
//! never from the browser's redirect, so a customer cannot fake a payment.

use anyhow::{anyhow, bail, Result};
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{json, Value};
use sha2::Sha256;

use super::store::Currency;

type HmacSha256 = Hmac<Sha256>;

/// Webhooks older than this are rejected (replay protection, Stripe's default).
const STRIPE_TOLERANCE_SECS: i64 = 300;

#[derive(Debug, Clone, Default)]
pub struct Secrets {
    pub razorpay_key_id: Option<String>,
    pub razorpay_key_secret: Option<String>,
    pub razorpay_webhook_secret: Option<String>,
    pub stripe_secret_key: Option<String>,
    pub stripe_webhook_secret: Option<String>,
    pub admin_token: Option<String>,
}

impl Secrets {
    pub fn from_env() -> Self {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Self {
            razorpay_key_id: get("RAZORPAY_KEY_ID"),
            razorpay_key_secret: get("RAZORPAY_KEY_SECRET"),
            razorpay_webhook_secret: get("RAZORPAY_WEBHOOK_SECRET"),
            stripe_secret_key: get("STRIPE_SECRET_KEY"),
            stripe_webhook_secret: get("STRIPE_WEBHOOK_SECRET"),
            admin_token: get("LANGTRAIN_ADMIN_TOKEN"),
        }
    }

    pub fn razorpay_ready(&self) -> bool {
        self.razorpay_key_id.is_some() && self.razorpay_key_secret.is_some() && self.razorpay_webhook_secret.is_some()
    }

    pub fn stripe_ready(&self) -> bool {
        self.stripe_secret_key.is_some() && self.stripe_webhook_secret.is_some()
    }
}

fn hmac_hex(secret: &str, data: &[u8]) -> String {
    let mut m = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    m.update(data);
    hex::encode(m.finalize().into_bytes())
}

/// Constant-time check of a hex HMAC-SHA256 signature over `data`.
fn verify_hex(secret: &str, data: &[u8], sig_hex: &str) -> bool {
    let Ok(sig) = hex::decode(sig_hex.trim()) else { return false };
    let mut m = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    m.update(data);
    m.verify_slice(&sig).is_ok()
}

/// `X-Razorpay-Signature`: hex HMAC-SHA256 of the raw body with the webhook secret.
pub fn verify_razorpay(secret: &str, body: &[u8], signature: &str) -> bool {
    verify_hex(secret, body, signature)
}

/// `Stripe-Signature: t=<unix>,v1=<hex>[,v1=...]` over `"<t>.<body>"`.
pub fn verify_stripe(secret: &str, body: &[u8], header: &str, now: i64) -> bool {
    let mut t: Option<i64> = None;
    let mut sigs: Vec<&str> = Vec::new();
    for part in header.split(',') {
        match part.trim().split_once('=') {
            Some(("t", v)) => t = v.parse().ok(),
            Some(("v1", v)) => sigs.push(v),
            _ => {}
        }
    }
    let Some(t) = t else { return false };
    if (now - t).abs() > STRIPE_TOLERANCE_SECS {
        return false;
    }
    let mut signed = format!("{t}.").into_bytes();
    signed.extend_from_slice(body);
    sigs.iter().any(|s| verify_hex(secret, &signed, s))
}

/// A verified payment, as reported by a provider webhook.
#[derive(Debug, PartialEq, Eq)]
pub struct Paid {
    /// The order (Razorpay) or checkout session (Stripe) we created.
    pub checkout_id: String,
    pub amount_minor: i64,
    pub currency: String,
}

/// Extract a completed payment from a verified Razorpay event, if it is one.
pub fn razorpay_paid(event: &Value) -> Option<Paid> {
    match event.get("event")?.as_str()? {
        "order.paid" | "payment.captured" => {}
        _ => return None,
    }
    let payment = event.pointer("/payload/payment/entity")?;
    if payment.get("status").and_then(Value::as_str) != Some("captured") {
        return None;
    }
    Some(Paid {
        checkout_id: payment.get("order_id")?.as_str()?.to_string(),
        amount_minor: payment.get("amount")?.as_i64()?,
        currency: payment.get("currency")?.as_str()?.to_ascii_uppercase(),
    })
}

/// Extract a completed payment from a verified Stripe event, if it is one.
pub fn stripe_paid(event: &Value) -> Option<Paid> {
    match event.get("type")?.as_str()? {
        "checkout.session.completed" | "checkout.session.async_payment_succeeded" => {}
        _ => return None,
    }
    let s = event.pointer("/data/object")?;
    if s.get("payment_status").and_then(Value::as_str) != Some("paid") {
        return None;
    }
    Some(Paid {
        checkout_id: s.get("id")?.as_str()?.to_string(),
        amount_minor: s.get("amount_total")?.as_i64()?,
        currency: s.get("currency")?.as_str()?.to_ascii_uppercase(),
    })
}

/// Create a Razorpay order; the browser completes it with Razorpay Checkout.
pub async fn create_razorpay_order(
    http: &reqwest::Client,
    secrets: &Secrets,
    amount_minor: i64,
    currency: Currency,
    receipt: &str,
    account_id: &str,
) -> Result<String> {
    let (Some(id), Some(secret)) = (&secrets.razorpay_key_id, &secrets.razorpay_key_secret) else {
        bail!("Razorpay is not configured");
    };
    let res = http
        .post("https://api.razorpay.com/v1/orders")
        .basic_auth(id, Some(secret))
        .json(&json!({
            "amount": amount_minor,
            "currency": currency.code(),
            "receipt": receipt,
            "notes": { "account_id": account_id },
        }))
        .send()
        .await?;
    let status = res.status();
    let body: Value = res.json().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Razorpay order failed ({status}): {}", body.pointer("/error/description").and_then(Value::as_str).unwrap_or("unknown error"));
    }
    body.get("id").and_then(Value::as_str).map(str::to_owned).ok_or_else(|| anyhow!("Razorpay returned no order id"))
}

/// Create a Stripe Checkout session; returns (session id, hosted payment URL).
#[allow(clippy::too_many_arguments)]
pub async fn create_stripe_session(
    http: &reqwest::Client,
    secrets: &Secrets,
    amount_minor: i64,
    currency: Currency,
    product: &str,
    account_id: &str,
    email: &str,
    return_url: &str,
) -> Result<(String, String)> {
    let Some(key) = &secrets.stripe_secret_key else { bail!("Stripe is not configured") };
    let amount = amount_minor.to_string();
    let form = [
        ("mode", "payment"),
        ("line_items[0][quantity]", "1"),
        ("line_items[0][price_data][currency]", &currency.code().to_ascii_lowercase()),
        ("line_items[0][price_data][unit_amount]", &amount),
        ("line_items[0][price_data][product_data][name]", product),
        ("client_reference_id", account_id),
        ("customer_email", email),
        ("success_url", &format!("{return_url}?checkout=success")),
        ("cancel_url", &format!("{return_url}?checkout=cancelled")),
    ];
    let res = http.post("https://api.stripe.com/v1/checkout/sessions").bearer_auth(key).form(&form).send().await?;
    let status = res.status();
    let body: Value = res.json().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Stripe checkout failed ({status}): {}", body.pointer("/error/message").and_then(Value::as_str).unwrap_or("unknown error"));
    }
    let id = body.get("id").and_then(Value::as_str).ok_or_else(|| anyhow!("Stripe returned no session id"))?;
    let url = body.get("url").and_then(Value::as_str).ok_or_else(|| anyhow!("Stripe returned no checkout URL"))?;
    Ok((id.to_string(), url.to_string()))
}

/// Sign a body the way Razorpay does (for tests and local webhook replay).
pub fn sign_razorpay(secret: &str, body: &[u8]) -> String {
    hmac_hex(secret, body)
}

/// Build a `Stripe-Signature` header (for tests and local webhook replay).
pub fn sign_stripe(secret: &str, body: &[u8], t: i64) -> String {
    let mut signed = format!("{t}.").into_bytes();
    signed.extend_from_slice(body);
    format!("t={t},v1={}", hmac_hex(secret, &signed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn razorpay_signature() {
        let body = br#"{"event":"order.paid"}"#;
        let sig = sign_razorpay("whsec", body);
        assert!(verify_razorpay("whsec", body, &sig));
        assert!(!verify_razorpay("other", body, &sig));
        assert!(!verify_razorpay("whsec", b"tampered", &sig));
        assert!(!verify_razorpay("whsec", body, "not-hex"));
    }

    #[test]
    fn stripe_signature_and_replay_window() {
        let body = br#"{"type":"checkout.session.completed"}"#;
        let header = sign_stripe("whsec_1", body, 1_700_000_000);
        assert!(verify_stripe("whsec_1", body, &header, 1_700_000_100));
        assert!(!verify_stripe("whsec_1", body, &header, 1_700_000_000 + 301));
        assert!(!verify_stripe("whsec_2", body, &header, 1_700_000_100));
        assert!(!verify_stripe("whsec_1", b"{}", &header, 1_700_000_100));
        // Extra v1 entries (key rotation) are fine as long as one matches.
        let rotated = format!("{header},v1=deadbeef");
        assert!(verify_stripe("whsec_1", body, &rotated, 1_700_000_100));
    }

    #[test]
    fn parses_paid_events() {
        let rzp = json!({"event": "order.paid", "payload": {"payment": {"entity": {
            "id": "pay_1", "order_id": "order_9", "amount": 50000, "currency": "INR", "status": "captured"}}}});
        assert_eq!(razorpay_paid(&rzp), Some(Paid { checkout_id: "order_9".into(), amount_minor: 50000, currency: "INR".into() }));
        let failed = json!({"event": "payment.failed", "payload": {}});
        assert_eq!(razorpay_paid(&failed), None);

        let stripe = json!({"type": "checkout.session.completed", "data": {"object": {
            "id": "cs_1", "payment_status": "paid", "amount_total": 1000, "currency": "usd"}}});
        assert_eq!(stripe_paid(&stripe), Some(Paid { checkout_id: "cs_1".into(), amount_minor: 1000, currency: "USD".into() }));
        let unpaid = json!({"type": "checkout.session.completed", "data": {"object": {
            "id": "cs_2", "payment_status": "unpaid", "amount_total": 1000, "currency": "usd"}}});
        assert_eq!(stripe_paid(&unpaid), None);
    }
}
