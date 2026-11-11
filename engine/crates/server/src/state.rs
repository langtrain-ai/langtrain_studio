use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{anyhow, bail, Result};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use langtrain_engine::hardware::{Hardware, Profile};
use langtrain_engine::sampling::SamplingParams;
use langtrain_engine::{EngineConfig, EngineHandle, Event, GenerateRequest, Prompt};
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, Mutex};

use std::sync::atomic::AtomicBool;

use crate::billing::{format_money, Billing, Caller, Refusal};
use crate::tune::Jobs;

/// Name of the built-in test model, available with `--mock`.
pub const MOCK_MODEL: &str = "mock";

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Root for models, adapters and jobs (default `~/.langtrain`).
    pub home: PathBuf,
    pub enable_mock: bool,
    /// Directory of the `langtune` Python package, used for fine-tuning jobs.
    pub langtune_dir: Option<PathBuf>,
    /// Python interpreter for fine-tuning jobs.
    pub python: String,
    pub hardware: Hardware,
    pub profile: Profile,
    pub backend: BackendChoice,
    /// Langtrain Cloud mode: API keys, prepaid credits and metering.
    pub cloud: bool,
}

/// Which inference engine serves GGUF models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// The native engine when it supports the model, else llama.cpp.
    Auto,
    Native,
    Llama,
}

impl std::str::FromStr for BackendChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "native" => Ok(Self::Native),
            "llama" | "llama.cpp" => Ok(Self::Llama),
            o => Err(format!("unknown backend '{o}' (auto, native, llama)")),
        }
    }
}

impl ServerConfig {
    pub fn models_dir(&self) -> PathBuf {
        self.home.join("models")
    }
    pub fn adapters_dir(&self, base: &str) -> PathBuf {
        self.home.join("adapters").join(base)
    }
    pub fn jobs_dir(&self) -> PathBuf {
        self.home.join("jobs")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: Option<PathBuf>,
    pub size_bytes: u64,
    pub modified: Option<SystemTime>,
    pub adapters: Vec<String>,
}

/// `base@adapter` selects a LoRA adapter; Ollama-style `:latest` tags are ignored.
pub fn parse_model_name(name: &str) -> (String, Option<String>) {
    let name = name.strip_suffix(":latest").unwrap_or(name);
    match name.split_once('@') {
        Some((base, adapter)) if !adapter.is_empty() => (base.to_string(), Some(adapter.to_string())),
        _ => (name.trim_end_matches('@').to_string(), None),
    }
}

fn gguf_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")))
        .collect();
    files.sort();
    files
}

fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string()
}

struct Loaded {
    base: String,
    adapters: Vec<String>,
    handle: Arc<EngineHandle>,
}

pub struct AppState {
    pub config: ServerConfig,
    loaded: Mutex<Option<Loaded>>,
    pub jobs: Jobs,
    pub billing: Option<Arc<Billing>>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Result<Arc<Self>> {
        let billing = if config.cloud { Some(Arc::new(Billing::open(&config.home)?)) } else { None };
        Ok(Self::with_billing(config, billing))
    }

    pub fn with_billing(config: ServerConfig, billing: Option<Arc<Billing>>) -> Arc<Self> {
        let jobs = Jobs::new(config.jobs_dir());
        Arc::new(Self { config, loaded: Mutex::new(None), jobs, billing })
    }

    pub fn adapter_files(&self, base: &str) -> BTreeMap<String, PathBuf> {
        gguf_files(&self.config.adapters_dir(base)).into_iter().map(|p| (stem(&p), p)).collect()
    }

    pub fn list_models(&self) -> Vec<ModelEntry> {
        let mut out: Vec<ModelEntry> = gguf_files(&self.config.models_dir())
            .into_iter()
            .map(|p| {
                let meta = std::fs::metadata(&p).ok();
                let name = stem(&p);
                ModelEntry {
                    adapters: self.adapter_files(&name).into_keys().collect(),
                    size_bytes: meta.as_ref().map(|m| m.len()).unwrap_or(0),
                    modified: meta.and_then(|m| m.modified().ok()),
                    path: Some(p),
                    name,
                }
            })
            .collect();
        if self.config.enable_mock {
            out.push(ModelEntry {
                name: MOCK_MODEL.into(),
                path: None,
                size_bytes: 0,
                modified: None,
                adapters: vec!["pirate".into()],
            });
        }
        out
    }

    /// Get (loading if needed) the engine serving `base`. One model is resident
    /// at a time; switching models unloads the previous one first.
    pub async fn engine(&self, base: &str) -> Result<Arc<EngineHandle>> {
        let adapters = self.adapter_files(base);
        let adapter_names: Vec<String> = adapters.keys().cloned().collect();
        let mut loaded = self.loaded.lock().await;
        if let Some(l) = loaded.as_ref() {
            if l.base == base && (base == MOCK_MODEL || l.adapters == adapter_names) {
                return Ok(l.handle.clone());
            }
        }
        // Free the old model's memory before loading the next one.
        if let Some(old) = loaded.take() {
            tokio::task::spawn_blocking(move || drop(old.handle)).await?;
        }

        let handle = if base == MOCK_MODEL && self.config.enable_mock {
            let slots = self.config.profile.parallel_slots;
            EngineHandle::spawn(EngineConfig::default(), move || {
                Ok(Box::new(langtrain_engine::MockBackend::new(8192, slots, 512)) as Box<dyn langtrain_engine::Backend>)
            })?
        } else {
            let path = self.config.models_dir().join(format!("{base}.gguf"));
            if !path.exists() {
                bail!("model '{base}' not found; pull it with `langtrain pull` or put a .gguf in {}", self.config.models_dir().display());
            }
            let use_native = match self.config.backend {
                BackendChoice::Native => true,
                BackendChoice::Llama => false,
                BackendChoice::Auto => match langtrain_native::supports(&path) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::info!("native engine cannot run {base} ({e:#}); using llama.cpp");
                        !cfg!(feature = "llama")
                    }
                },
            };
            if use_native {
                self.spawn_native(path, adapters).await?
            } else {
                self.spawn_llama(path, adapters).await?
            }
        };
        tracing::info!(model = base, backend = %handle.info.backend, n_ctx = handle.info.n_ctx, "model loaded");
        let handle = Arc::new(handle);
        *loaded = Some(Loaded { base: base.to_string(), adapters: adapter_names, handle: handle.clone() });
        Ok(handle)
    }

    async fn spawn_native(&self, path: PathBuf, adapters: BTreeMap<String, PathBuf>) -> Result<EngineHandle> {
        use langtrain_native::{NativeBackend, NativeOptions};
        let p = &self.config.profile;
        let opts = NativeOptions {
            model_path: path,
            adapters,
            parallel_slots: p.parallel_slots,
            kv_tokens: p.parallel_slots * p.context_per_slot,
            n_batch: 512,
            threads: p.threads,
        };
        tokio::task::spawn_blocking(move || {
            EngineHandle::spawn(EngineConfig::default(), move || {
                Ok(Box::new(NativeBackend::load(&opts)?) as Box<dyn langtrain_engine::Backend>)
            })
        })
        .await?
    }

    #[cfg(feature = "llama")]
    async fn spawn_llama(&self, path: PathBuf, adapters: BTreeMap<String, PathBuf>) -> Result<EngineHandle> {
        use langtrain_engine::llama::{LlamaCppBackend, LlamaOptions};
        let p = &self.config.profile;
        let opts = LlamaOptions {
            model_path: path,
            adapters,
            parallel_slots: p.parallel_slots,
            context_per_slot: p.context_per_slot,
            n_batch: 512,
            threads: p.threads,
            threads_batch: p.threads_batch,
            gpu_layers: p.gpu_layers,
            kv_q8: p.kv_cache_type == "q8_0",
        };
        tokio::task::spawn_blocking(move || {
            EngineHandle::spawn(EngineConfig::default(), move || {
                Ok(Box::new(LlamaCppBackend::load(&opts)?) as Box<dyn langtrain_engine::Backend>)
            })
        })
        .await?
    }

    #[cfg(not(feature = "llama"))]
    async fn spawn_llama(&self, _path: PathBuf, _adapters: BTreeMap<String, PathBuf>) -> Result<EngineHandle> {
        bail!("this build has no llama.cpp backend; rebuild with --features llama")
    }

    /// Unload the model if it is `base`, so new adapters are picked up on next use.
    pub async fn invalidate(&self, base: &str) {
        let mut loaded = self.loaded.lock().await;
        if loaded.as_ref().is_some_and(|l| l.base == base) {
            if let Some(old) = loaded.take() {
                let _ = tokio::task::spawn_blocking(move || drop(old.handle)).await;
            }
        }
    }

    pub async fn loaded_model(&self) -> Option<(String, serde_json::Value)> {
        let loaded = self.loaded.lock().await;
        loaded.as_ref().map(|l| {
            (l.base.clone(), json!({ "info": l.handle.info, "metrics": l.handle.metrics.snapshot() }))
        })
    }

    /// Resolve a model name and start generating. Events arrive on the returned channel.
    pub async fn generate(&self, model: &str, prompt: Prompt, params: SamplingParams) -> Result<mpsc::UnboundedReceiver<Event>> {
        Ok(self.start(model, prompt, params).await?.0)
    }

    async fn start(&self, model: &str, prompt: Prompt, params: SamplingParams) -> Result<(mpsc::UnboundedReceiver<Event>, Arc<AtomicBool>)> {
        let (base, adapter) = parse_model_name(model);
        if base.is_empty() {
            return Err(anyhow!("model is required"));
        }
        let engine = self.engine(&base).await?;
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest::new(prompt, params, adapter, tx);
        let cancel = req.cancel.clone();
        engine.submit(req)?;
        Ok((rx, cancel))
    }

    /// Generate on behalf of an API caller. In cloud mode the request is paid
    /// for first: its worst-case cost is held from the caller's credit (402 if
    /// short), and the actual cost is charged when it ends.
    pub async fn generate_for(
        &self,
        caller: Option<&Caller>,
        model: &str,
        prompt: Prompt,
        mut params: SamplingParams,
    ) -> Result<mpsc::UnboundedReceiver<Event>, ApiError> {
        let Some(billing) = &self.billing else {
            return Ok(self.generate(model, prompt, params).await?);
        };
        let caller = caller.ok_or_else(|| ApiError(StatusCode::UNAUTHORIZED, "missing API key".into()))?;
        let max_tokens = billing.clamp_max_tokens(params.max_tokens);
        params.max_tokens = Some(max_tokens);
        let (bytes, messages) = match &prompt {
            Prompt::Text(t) => (t.len(), 1),
            Prompt::Chat(m) => (m.iter().map(|x| x.role.len() + x.content.len()).sum(), m.len()),
            Prompt::Tokens(t) => (t.len(), 1),
        };
        let (base, _) = parse_model_name(model);
        let price = billing.price(&base, caller.account.currency);
        let estimate = Billing::estimate(price, bytes, messages, max_tokens);
        let admission = match billing.admit(caller, estimate)? {
            Ok(a) => a,
            Err(Refusal::InsufficientCredit { available, needed, currency }) => {
                return Err(ApiError(
                    StatusCode::PAYMENT_REQUIRED,
                    format!(
                        "insufficient credit: this request can cost up to {} and your balance is {}. Add credits at {}/#billing",
                        format_money(needed, currency),
                        format_money(available, currency),
                        billing.config.public_url.trim_end_matches('/')
                    ),
                ))
            }
            Err(Refusal::TooManyRequests { limit }) => {
                return Err(ApiError(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("too many requests at once: this account can run {limit} at a time"),
                ))
            }
        };
        // If starting fails, dropping `admission` releases the hold.
        let (rx, cancel) = self.start(model, prompt, params).await?;
        Ok(Billing::meter(admission, model.to_string(), price, rx, cancel))
    }
}

/// An error rendered in OpenAI's error shape.
#[derive(Debug)]
pub struct ApiError(pub StatusCode, pub String);

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        let msg = format!("{e:#}");
        let code = if msg.contains("not found") || msg.contains("unknown adapter") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self(code, msg)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "message": self.1, "type": "invalid_request_error" } });
        (self.0, Json(body)).into_response()
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::parse_model_name;

    #[test]
    fn model_names() {
        assert_eq!(parse_model_name("qwen"), ("qwen".into(), None));
        assert_eq!(parse_model_name("qwen:latest"), ("qwen".into(), None));
        assert_eq!(parse_model_name("qwen@support"), ("qwen".into(), Some("support".into())));
    }
}
