//! Langtrain Cloud billing: prepaid credits, API keys and metering.
//!
//! Enabled with `langtrain serve --cloud`. Every generation is paid for
//! before it runs: an upper-bound cost is held from the account's balance
//! (402 if it cannot be covered), and when the request ends, even if the
//! client disconnects, the hold is released and the actual cost is charged.

pub mod payments;
pub mod routes;
pub mod store;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use langtrain_engine::{Event, FinishReason, Usage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use payments::Secrets;
use store::{Account, Currency, Hold, Store, MICRO};

/// Price per 1M tokens, in whole currency units (₹ or $).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Price {
    pub input_per_m: f64,
    /// Prompt tokens served from the prefix cache cost less: no compute.
    pub cached_input_per_m: f64,
    pub output_per_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyPrices {
    pub inr: Price,
    pub usd: Price,
}

impl CurrencyPrices {
    pub fn get(&self, c: Currency) -> Price {
        match c {
            Currency::INR => self.inr,
            Currency::USD => self.usd,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pack {
    pub id: String,
    pub currency: Currency,
    /// What the customer pays, in whole units (₹500).
    pub amount: i64,
    /// Extra credit on top, as a percentage of `amount`.
    pub bonus_pct: u32,
}

impl Pack {
    pub fn credit_micro(&self) -> i64 {
        self.amount * MICRO
    }
    pub fn bonus_micro(&self) -> i64 {
        self.amount * MICRO * self.bonus_pct as i64 / 100
    }
    pub fn amount_minor(&self) -> i64 {
        self.amount * 100 // paise / cents
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CloudConfig {
    /// Public base URL, used for Stripe's return links.
    pub public_url: String,
    pub default_prices: CurrencyPrices,
    /// Per-model overrides, keyed by base model name.
    pub model_prices: HashMap<String, CurrencyPrices>,
    pub packs: Vec<Pack>,
    /// Anyone can create an account with `POST /v1/signup`.
    pub open_signup: bool,
    /// Trial credit for new accounts, in whole units.
    pub signup_credit_inr: i64,
    pub signup_credit_usd: i64,
    /// Used when a request sets no `max_tokens`.
    pub default_max_tokens: usize,
    /// Hard limit per request, which also bounds the hold.
    pub max_tokens_cap: usize,
    /// Requests one account may run at the same time.
    pub max_concurrent_per_account: usize,
}

impl Default for CloudConfig {
    fn default() -> Self {
        let pack = |id: &str, currency, amount, bonus_pct| Pack { id: id.into(), currency, amount, bonus_pct };
        Self {
            public_url: "http://127.0.0.1:11435".into(),
            default_prices: CurrencyPrices {
                inr: Price { input_per_m: 12.0, cached_input_per_m: 6.0, output_per_m: 24.0 },
                usd: Price { input_per_m: 0.15, cached_input_per_m: 0.075, output_per_m: 0.30 },
            },
            model_prices: HashMap::new(),
            packs: vec![
                pack("inr-500", Currency::INR, 500, 0),
                pack("inr-2000", Currency::INR, 2000, 5),
                pack("inr-8000", Currency::INR, 8000, 10),
                pack("usd-10", Currency::USD, 10, 0),
                pack("usd-25", Currency::USD, 25, 5),
                pack("usd-100", Currency::USD, 100, 10),
            ],
            open_signup: true,
            signup_credit_inr: 0,
            signup_credit_usd: 0,
            default_max_tokens: 1024,
            max_tokens_cap: 4096,
            max_concurrent_per_account: 4,
        }
    }
}

/// An authenticated API caller.
#[derive(Debug, Clone)]
pub struct Caller {
    pub account: Account,
    pub key_id: String,
}

pub struct Billing {
    pub store: Store,
    pub config: CloudConfig,
    pub secrets: Secrets,
    pub http: reqwest::Client,
    inflight: Mutex<HashMap<String, usize>>,
}

/// Why a request was not started.
#[derive(Debug)]
pub enum Refusal {
    /// 402: the balance cannot cover the worst-case cost.
    InsufficientCredit { available: i64, needed: i64, currency: Currency },
    /// 429: too many requests running for this account.
    TooManyRequests { limit: usize },
}

pub fn format_money(micro: i64, c: Currency) -> String {
    format!("{}{:.2}", c.symbol(), micro as f64 / MICRO as f64)
}

impl Billing {
    /// Open `billing.db` and `cloud.json` under `home` (writing defaults on first run).
    pub fn open(home: &Path) -> Result<Self> {
        let cfg_path = home.join("cloud.json");
        let config: CloudConfig = if cfg_path.exists() {
            serde_json::from_slice(&std::fs::read(&cfg_path)?).with_context(|| format!("reading {}", cfg_path.display()))?
        } else {
            let c = CloudConfig::default();
            std::fs::create_dir_all(home)?;
            std::fs::write(&cfg_path, serde_json::to_vec_pretty(&c)?)?;
            c
        };
        Ok(Self::new(Store::open(&home.join("billing.db"))?, config, Secrets::from_env()))
    }

    pub fn new(store: Store, config: CloudConfig, secrets: Secrets) -> Self {
        Self { store, config, secrets, http: reqwest::Client::new(), inflight: Mutex::new(HashMap::new()) }
    }

    pub fn price(&self, base_model: &str, c: Currency) -> Price {
        self.config.model_prices.get(base_model).unwrap_or(&self.config.default_prices).get(c)
    }

    /// Actual cost of a finished request, rounded up to the micro-unit.
    pub fn cost(price: Price, u: &Usage) -> i64 {
        let uncached = u.prompt_tokens.saturating_sub(u.cached_tokens) as f64;
        let v = uncached * price.input_per_m + u.cached_tokens as f64 * price.cached_input_per_m + u.completion_tokens as f64 * price.output_per_m;
        v.ceil() as i64
    }

    /// Upper bound on a request's cost. Byte-level BPE never produces more
    /// tokens than bytes, and the margin covers chat-template tokens.
    pub fn estimate(price: Price, prompt_bytes: usize, n_messages: usize, max_tokens: usize) -> i64 {
        let prompt = (prompt_bytes + 16 * n_messages + 64) as f64;
        (prompt * price.input_per_m + max_tokens as f64 * price.output_per_m).ceil() as i64
    }

    /// The `max_tokens` a cloud request actually runs with.
    pub fn clamp_max_tokens(&self, requested: Option<usize>) -> usize {
        requested.unwrap_or(self.config.default_max_tokens).clamp(1, self.config.max_tokens_cap)
    }

    /// Reserve credit and a concurrency slot. On success the returned guard
    /// must be settled (see [`Billing::meter`]).
    pub fn admit(self: &Arc<Self>, caller: &Caller, estimate: i64) -> Result<std::result::Result<Admission, Refusal>> {
        {
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            let n = inflight.entry(caller.account.id.clone()).or_default();
            if *n >= self.config.max_concurrent_per_account {
                return Ok(Err(Refusal::TooManyRequests { limit: self.config.max_concurrent_per_account }));
            }
            *n += 1;
        }
        match self.store.place_hold(&caller.account.id, estimate)? {
            Ok(hold) => Ok(Ok(Admission { billing: self.clone(), hold, caller: caller.clone(), done: false })),
            Err(store::HoldError::Insufficient { available, needed }) => {
                self.release_slot(&caller.account.id);
                Ok(Err(Refusal::InsufficientCredit { available, needed, currency: caller.account.currency }))
            }
        }
    }

    fn release_slot(&self, account_id: &str) {
        let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = inflight.get_mut(account_id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inflight.remove(account_id);
            }
        }
    }

    /// Forward engine events to the client and settle the hold when the
    /// request ends. If the client goes away, generation is cancelled and the
    /// tokens produced so far are still charged.
    pub fn meter(
        admission: Admission,
        model: String,
        price: Price,
        mut engine: mpsc::UnboundedReceiver<Event>,
        cancel: Arc<AtomicBool>,
    ) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut admission = admission;
            let mut client_gone = false;
            while let Some(ev) = engine.recv().await {
                if let Event::Done { reason, usage } = &ev {
                    admission.settle(&model, price, usage, *reason);
                }
                let last = matches!(ev, Event::Done { .. } | Event::Error(_));
                if !client_gone && tx.send(ev).is_err() {
                    client_gone = true;
                    cancel.store(true, Ordering::Relaxed);
                }
                if last {
                    return;
                }
            }
            // Engine stopped without a final report: charge nothing.
        });
        rx
    }
}

/// A request that holds credit and a concurrency slot. Dropping it without
/// settling releases both and charges nothing.
pub struct Admission {
    billing: Arc<Billing>,
    hold: Hold,
    caller: Caller,
    done: bool,
}

impl Admission {
    fn settle(&mut self, model: &str, price: Price, usage: &Usage, reason: FinishReason) {
        if self.done {
            return;
        }
        self.done = true;
        let cost = Billing::cost(price, usage);
        let finish = reason.as_str();
        if let Err(e) = self.billing.store.settle(&self.hold, cost, Some((&self.caller.key_id, model, usage, finish))) {
            tracing::error!(account = %self.caller.account.id, "failed to settle hold {}: {e:#}", self.hold.id);
        }
        self.billing.release_slot(&self.caller.account.id);
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.billing.store.settle(&self.hold, 0, None);
            self.billing.release_slot(&self.caller.account.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_and_estimate() {
        let p = Price { input_per_m: 12.0, cached_input_per_m: 6.0, output_per_m: 24.0 };
        let u = Usage { prompt_tokens: 1000, cached_tokens: 400, completion_tokens: 500, ..Default::default() };
        // 600*12 + 400*6 + 500*24 micro-rupees
        assert_eq!(Billing::cost(p, &u), 7200 + 2400 + 12000);
        // The estimate always covers the real cost of a request that fits it.
        let est = Billing::estimate(p, 1000, 2, 500);
        assert!(est >= Billing::cost(p, &Usage { prompt_tokens: 1000, completion_tokens: 500, ..Default::default() }));
        let pack = Pack { id: "x".into(), currency: Currency::INR, amount: 2000, bonus_pct: 5 };
        assert_eq!((pack.amount_minor(), pack.bonus_micro()), (200_000, 100 * MICRO));
        assert_eq!(format_money(1_234_567, Currency::INR), "₹1.23");
    }
}
