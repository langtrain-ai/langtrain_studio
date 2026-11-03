//! llama.cpp backend (GGUF models, CPU/Metal/CUDA/Vulkan through ggml).

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::backend::{Backend, BatchToken, Token};
use crate::types::ChatMessage;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn global_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    let b = LlamaBackend::init().map_err(|e| anyhow!("llama.cpp init failed: {e}"))?;
    Ok(BACKEND.get_or_init(|| b))
}

#[derive(Debug, Clone)]
pub struct LlamaOptions {
    pub model_path: PathBuf,
    /// Adapter name -> GGUF LoRA file.
    pub adapters: BTreeMap<String, PathBuf>,
    pub parallel_slots: u32,
    pub context_per_slot: u32,
    pub n_batch: u32,
    pub threads: usize,
    pub threads_batch: usize,
    pub gpu_layers: u32,
    /// Quantize the KV cache to q8_0 (half the memory of f16).
    pub kv_q8: bool,
}

pub struct LlamaCppBackend {
    // Field order matters: the context borrows the model and must drop first.
    ctx: LlamaContext<'static>,
    adapters: BTreeMap<String, LlamaLoraAdapter>,
    template: Option<LlamaChatTemplate>,
    batch: LlamaBatch<'static>,
    n_vocab: usize,
    name: String,
    n_seq: u32,
    n_batch: u32,
    model: Box<LlamaModel>,
}

impl LlamaCppBackend {
    pub fn load(opts: &LlamaOptions) -> Result<Self> {
        let backend = global_backend()?;
        let mut params = LlamaModelParams::default().with_n_gpu_layers(opts.gpu_layers);
        if opts.gpu_layers == 0 {
            // Pin to the CPU: llama.cpp otherwise initializes Metal/GPU devices anyway.
            let cpu: Vec<usize> = llama_cpp_2::list_llama_ggml_backend_devices()
                .into_iter()
                .filter(|d| d.backend.eq_ignore_ascii_case("cpu"))
                .map(|d| d.index)
                .collect();
            params = params.with_devices(&cpu).map_err(|e| anyhow!("{e}"))?;
        }
        let mut model = Box::new(
            LlamaModel::load_from_file(backend, &opts.model_path, &params)
                .with_context(|| format!("loading {}", opts.model_path.display()))?,
        );

        // Adapters must be created before the context borrows the model.
        let mut adapters = BTreeMap::new();
        for (name, path) in &opts.adapters {
            match model.lora_adapter_init(path) {
                Ok(a) => {
                    adapters.insert(name.clone(), a);
                }
                Err(e) => tracing::warn!("skipping adapter {name} ({}): {e}", path.display()),
            }
        }

        // SAFETY: `model` is boxed (stable address) and outlives `ctx`, which is
        // declared first in the struct and therefore dropped first.
        let model_ref: &'static LlamaModel = unsafe { &*(model.as_ref() as *const LlamaModel) };
        let n_ctx = (opts.parallel_slots * opts.context_per_slot).min(model_ref.n_ctx_train() * opts.parallel_slots);
        let ctx_params = |kv_q8: bool| {
            let p = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(opts.n_batch)
                .with_n_ubatch(opts.n_batch)
                .with_n_seq_max(opts.parallel_slots)
                .with_kv_unified(true)
                .with_n_threads(opts.threads as i32)
                .with_n_threads_batch(opts.threads_batch as i32);
            if kv_q8 {
                p.with_type_k(KvCacheType::Q8_0).with_type_v(KvCacheType::Q8_0)
            } else {
                p
            }
        };
        let ctx = match model_ref.new_context(backend, ctx_params(opts.kv_q8)) {
            Ok(c) => c,
            // Quantized V cache needs flash attention; not every backend has it.
            Err(e) if opts.kv_q8 => {
                tracing::warn!("q8_0 KV cache unavailable ({e}); falling back to f16");
                model_ref.new_context(backend, ctx_params(false))?
            }
            Err(e) => return Err(e.into()),
        };

        let template = model_ref.chat_template(None).ok();
        let name = opts.model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
        Ok(Self {
            n_vocab: model_ref.n_vocab() as usize,
            batch: LlamaBatch::new(opts.n_batch as usize, opts.parallel_slots as i32),
            n_seq: opts.parallel_slots,
            n_batch: opts.n_batch,
            ctx,
            adapters,
            template,
            name,
            model,
        })
    }

    fn model(&self) -> &LlamaModel {
        &self.model
    }
}

/// ChatML, used when a GGUF ships without a chat template.
fn chatml(messages: &[ChatMessage]) -> String {
    let mut s: String = messages.iter().map(|m| format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content)).collect();
    s.push_str("<|im_start|>assistant\n");
    s
}

impl Backend for LlamaCppBackend {
    fn name(&self) -> String {
        format!("llama.cpp:{}", self.name)
    }
    fn n_ctx(&self) -> u32 {
        self.ctx.n_ctx()
    }
    fn n_seq_max(&self) -> u32 {
        self.n_seq
    }
    fn n_batch(&self) -> u32 {
        self.n_batch
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Vec<Token> {
        self.model().vocab().tokenize(text.as_bytes(), add_special, true).into_iter().map(|t| t.0).collect()
    }

    fn token_to_piece(&self, token: Token) -> Vec<u8> {
        self.model().vocab().token_to_piece(LlamaToken(token), false, None)
    }

    fn is_eog(&self, token: Token) -> bool {
        self.model().vocab().is_eog(LlamaToken(token))
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let Some(tmpl) = &self.template else { return Ok(chatml(messages)) };
        let msgs = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        match self.model().apply_chat_template(tmpl, &msgs, true) {
            Ok(s) => Ok(s),
            // llama.cpp only knows built-in templates; fall back for unknown Jinja ones.
            Err(e) => {
                tracing::debug!("chat template not supported by llama.cpp ({e}); using ChatML");
                Ok(chatml(messages))
            }
        }
    }

    fn decode(&mut self, batch: &[BatchToken]) -> Result<()> {
        self.batch.clear();
        for bt in batch {
            self.batch.add(LlamaToken(bt.token), bt.pos as i32, &[bt.seq as i32], bt.logits)?;
        }
        self.ctx.decode(&mut self.batch).map_err(|e| anyhow!("decode failed: {e}"))
    }

    fn logits(&self, batch_index: usize) -> &[f32] {
        let l = self.ctx.get_logits_ith(batch_index as i32);
        &l[..self.n_vocab.min(l.len())]
    }

    fn kv_remove(&mut self, seq: u32, from: u32) {
        let _ = self.ctx.clear_kv_cache_seq(Some(seq), Some(from), None);
    }

    fn kv_copy(&mut self, src: u32, dst: u32, len: u32) {
        let _ = self.ctx.copy_kv_cache_seq(src as i32, dst as i32, Some(0), Some(len));
    }

    fn adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    fn set_adapter(&mut self, adapter: Option<&str>) -> Result<()> {
        match adapter {
            Some(name) => {
                let a = self.adapters.get_mut(name).ok_or_else(|| anyhow!("unknown adapter {name}"))?;
                self.ctx.lora_adapter_set(a, 1.0).map_err(|e| anyhow!("{e}"))
            }
            None => match self.adapters.values_mut().next() {
                Some(a) => self.ctx.lora_adapter_remove(a).map_err(|e| anyhow!("{e}")),
                None => Ok(()),
            },
        }
    }
}

/// Read `general.architecture` etc. without loading weights.
pub fn is_gguf(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}
