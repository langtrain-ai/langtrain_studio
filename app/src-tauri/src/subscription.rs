use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const GRACE_PERIOD_DAYS: u64 = 7;
const CACHE_FILE: &str = "subscription.json";

/// Cached subscription state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSubscription {
    pub plan: String,
    pub is_active: bool,
    pub verified_at: u64,   // Unix timestamp
    pub expires_at: Option<u64>,
    pub features: Vec<String>,
}

/// Result of a Pro access check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProAccessResult {
    pub allowed: bool,
    pub plan: String,
    pub source: String,  // "live", "cached", "denied"
    pub message: String,
    pub days_until_expiry: Option<i64>,
}

fn cache_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".langtrain").join(CACHE_FILE)
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Save subscription state to disk.
fn save_cache(sub: &CachedSubscription) -> Result<(), String> {
    let path = cache_path();
    let dir = path.parent().unwrap();
    let _ = fs::create_dir_all(dir);
    let json = serde_json::to_string_pretty(sub).map_err(|e| e.to_string())?;
    fs::write(&path, json).map_err(|e| e.to_string())
}

/// Load cached subscription from disk.
fn load_cache() -> Option<CachedSubscription> {
    let path = cache_path();
    let data = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Check Pro access — tries live API first, falls back to cache.
#[tauri::command]
pub async fn check_pro_access(auth_token: Option<String>) -> ProAccessResult {
    // 1. Try live verification if we have a token
    if let Some(token) = &auth_token {
        if let Ok(live_result) = verify_live(token).await {
            // Cache the result
            let cached = CachedSubscription {
                plan: live_result.plan.clone(),
                is_active: live_result.is_active,
                verified_at: now_unix(),
                expires_at: live_result.expires_at,
                features: live_result.features.clone(),
            };
            let _ = save_cache(&cached);

            let is_pro = live_result.plan != "free" && live_result.is_active;
            return ProAccessResult {
                allowed: is_pro,
                plan: live_result.plan,
                source: "live".to_string(),
                message: if is_pro {
                    "Pro subscription verified".to_string()
                } else {
                    "Upgrade to Pro to use offline features. Visit app.langtrain.xyz/billing".to_string()
                },
                days_until_expiry: live_result.expires_at.map(|exp| {
                    let now = now_unix() as i64;
                    (exp as i64 - now) / 86400
                }),
            };
        }
    }

    // 2. Fallback to cache
    if let Some(cached) = load_cache() {
        let age_days = (now_unix() - cached.verified_at) / 86400;
        let within_grace = age_days <= GRACE_PERIOD_DAYS;
        let is_pro = cached.plan != "free" && cached.is_active && within_grace;

        return ProAccessResult {
            allowed: is_pro,
            plan: cached.plan,
            source: "cached".to_string(),
            message: if is_pro {
                format!(
                    "Offline mode — subscription cached {} day(s) ago. Grace period: {} days remaining.",
                    age_days,
                    GRACE_PERIOD_DAYS.saturating_sub(age_days)
                )
            } else if !within_grace {
                "Cached subscription expired. Connect to the internet to re-verify.".to_string()
            } else {
                "Free plan — upgrade to Pro for offline features.".to_string()
            },
            days_until_expiry: Some((GRACE_PERIOD_DAYS as i64) - (age_days as i64)),
        };
    }

    // 3. No cache, no token — denied
    ProAccessResult {
        allowed: false,
        plan: "unknown".to_string(),
        source: "denied".to_string(),
        message: "Please log in and verify your Pro subscription to use offline features.".to_string(),
        days_until_expiry: None,
    }
}

/// Live subscription response from the server.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LiveSubscription {
    is_active: bool,
    plan: String,
    expires_at: Option<u64>,
    features: Vec<String>,
}

/// Verify subscription against the live API.
async fn verify_live(token: &str) -> Result<LiveSubscription, String> {
    let client = reqwest::Client::new();
    let res = client
        .get("https://api.langtrain.xyz/v1/subscription/status")
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "Langtrain-Studio-Desktop/1.0")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !res.status().is_success() {
        return Err(format!("Server returned {}", res.status()));
    }

    res.json::<LiveSubscription>()
        .await
        .map_err(|e| e.to_string())
}
