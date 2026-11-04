//! Langtrain's own inference engine, in pure Rust.
//!
//! - `gguf`: zero-copy GGUF reader (memory-mapped)
//! - `quant`: repacked Q8_0/Q4_0 weights and int8 SIMD matmul (NEON, AVX2)
//! - `kv`: paged KV cache with copy-on-write prefix sharing
//! - `tokenizer`: byte-level BPE and Jinja chat templates from GGUF metadata
//! - `model`: Llama-family forward pass with unmerged multi-LoRA
//!
//! It plugs into the scheduler through [`langtrain_engine::Backend`], so
//! batching, prefix caching and adapter scheduling are shared with llama.cpp.

// GGUF blocks are byte runs; `chunks_exact(34)` reads clearer than `as_chunks::<34>()`.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod gguf;
pub mod kv;
pub mod model;
pub mod quant;
pub mod tokenizer;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use anyhow::{anyhow, Result};
use langtrain_engine::{Backend, BatchToken, ChatMessage, Token};

use crate::gguf::Gguf;
use crate::kv::KvCache;
use crate::model::{Lora, Model};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct NativeOptions {
    pub model_path: PathBuf,
    pub adapters: BTreeMap<String, PathBuf>,
    pub parallel_slots: u32,
    /// Total KV cells shared by all slots.
    pub kv_tokens: u32,
    pub n_batch: u32,
    pub threads: usize,
}

pub struct NativeBackend {
    model: Model,
    tok: Tokenizer,
    kv: KvCache,
    adapters: BTreeMap<String, Lora>,
    active: Option<String>,
    logits: Vec<f32>,
    /// Batch index -> row in `logits`.
    logit_row: Vec<Option<usize>>,
    opts: NativeOptions,
    name: String,
}

/// Whether the native engine can run this file (architecture and tensor types).
pub fn supports(path: &Path) -> Result<()> {
    let g = Gguf::open(path)?;
    let arch = g.architecture()?;
    if !matches!(arch, "llama" | "mistral" | "granite" | "qwen2") {
        anyhow::bail!("architecture '{arch}' not supported");
    }
    if g.str("tokenizer.ggml.model") != Some("gpt2") {
        anyhow::bail!("tokenizer not supported");
    }
    for (name, t) in &g.tensors {
        if t.dims.len() == 2 && t.dtype.size_of(t.n_elements()).is_none() {
            anyhow::bail!("tensor {name} has unsupported type {:?}", t.dtype);
        }
    }
    Ok(())
}

impl NativeBackend {
    pub fn load(opts: &NativeOptions) -> Result<Self> {
        static POOL: Once = Once::new();
        let threads = opts.threads.max(1);
        POOL.call_once(|| {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("lt-native-{i}")).build_global();
        });

        let t0 = std::time::Instant::now();
        let g = Gguf::open(&opts.model_path)?;
        let model = Model::load(&g)?;
        let tok = Tokenizer::from_gguf(&g)?;
        let c = &model.cfg;
        let kv = KvCache::new(c.n_layers, c.n_kv_heads * c.head_dim, opts.kv_tokens as usize);

        let mut adapters = BTreeMap::new();
        for (name, path) in &opts.adapters {
            match Lora::load(path, c) {
                Ok(l) => {
                    adapters.insert(name.clone(), l);
                }
                Err(e) => tracing::warn!("skipping adapter {name}: {e:#}"),
            }
        }
        tracing::info!(
            arch = %c.arch,
            layers = c.n_layers,
            weights_mb = model.weight_bytes() / (1 << 20),
            kv_mb = kv.bytes() / (1 << 20),
            load_ms = t0.elapsed().as_millis() as u64,
            "native engine loaded"
        );
        Ok(Self {
            name: opts.model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string(),
            model,
            tok,
            kv,
            adapters,
            active: None,
            logits: Vec::new(),
            logit_row: Vec::new(),
            opts: opts.clone(),
        })
    }
}

impl Backend for NativeBackend {
    fn name(&self) -> String {
        format!("native:{}", self.name)
    }
    fn n_ctx(&self) -> u32 {
        self.kv.capacity() as u32
    }
    fn n_seq_max(&self) -> u32 {
        self.opts.parallel_slots
    }
    fn n_batch(&self) -> u32 {
        self.opts.n_batch
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Vec<Token> {
        self.tok.encode(text, add_special)
    }
    fn token_to_piece(&self, token: Token) -> Vec<u8> {
        self.tok.decode_token(token)
    }
    fn is_eog(&self, token: Token) -> bool {
        self.tok.is_eog(token)
    }
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.tok.apply_chat_template(messages)
    }

    fn decode(&mut self, batch: &[BatchToken]) -> Result<()> {
        let tokens: Vec<i32> = batch.iter().map(|b| b.token).collect();
        let seqs: Vec<u32> = batch.iter().map(|b| b.seq).collect();
        let pos: Vec<usize> = batch.iter().map(|b| b.pos as usize).collect();
        let want: Vec<usize> = batch.iter().enumerate().filter(|(_, b)| b.logits).map(|(i, _)| i).collect();
        let lora = self.active.as_ref().and_then(|a| self.adapters.get(a));
        self.logits = self.model.forward(&mut self.kv, &tokens, &seqs, &pos, &want, lora)?;
        self.logit_row = vec![None; batch.len()];
        for (row, &i) in want.iter().enumerate() {
            self.logit_row[i] = Some(row);
        }
        Ok(())
    }

    fn logits(&self, batch_index: usize) -> &[f32] {
        let v = self.model.cfg.vocab;
        match self.logit_row.get(batch_index).copied().flatten() {
            Some(r) => &self.logits[r * v..(r + 1) * v],
            None => &[],
        }
    }

    fn kv_remove(&mut self, seq: u32, from: u32) {
        self.kv.truncate(seq, from as usize);
    }

    fn kv_copy(&mut self, src: u32, dst: u32, len: u32) {
        self.kv.share(src, dst, len as usize);
    }

    fn adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    fn set_adapter(&mut self, adapter: Option<&str>) -> Result<()> {
        if let Some(a) = adapter {
            if !self.adapters.contains_key(a) {
                return Err(anyhow!("unknown adapter {a}"));
            }
        }
        self.active = adapter.map(str::to_owned);
        Ok(())
    }
}
