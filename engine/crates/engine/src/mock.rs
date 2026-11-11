//! A deterministic byte-level backend for tests and UI development.
//!
//! It validates the KV bookkeeping the real backend relies on (contiguous
//! positions per sequence, cell budget, copies) and always answers with a
//! fixed reply, prefixed by the active adapter's name.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::backend::{Backend, BatchToken, Token};
use crate::types::ChatMessage;

pub const EOS: Token = 256;
const VOCAB: usize = 257;
pub const REPLY: &str = "Hello from Langtrain.";

pub struct MockBackend {
    n_ctx: u32,
    n_seq: u32,
    n_batch: u32,
    kv: HashMap<u32, Vec<Token>>,
    logits: Vec<Vec<f32>>,
    adapter: Option<String>,
    adapters: Vec<String>,
    /// Total tokens evaluated by `decode`, so tests can observe prefix reuse.
    pub evaluated: usize,
}

impl MockBackend {
    pub fn new(n_ctx: u32, n_seq: u32, n_batch: u32) -> Self {
        Self {
            n_ctx,
            n_seq,
            n_batch,
            kv: HashMap::new(),
            logits: Vec::new(),
            adapter: None,
            adapters: vec!["pirate".into()],
            evaluated: 0,
        }
    }

    fn reply(&self) -> Vec<u8> {
        match &self.adapter {
            Some(a) => format!("[{a}] {REPLY}").into_bytes(),
            None => REPLY.as_bytes().to_vec(),
        }
    }

    /// Next token: continue the reply from the longest suffix of `history` that
    /// is a prefix of it, then end with EOS.
    fn next_token(&self, history: &[Token]) -> Token {
        let reply = self.reply();
        let done = (1..=reply.len().min(history.len()))
            .rev()
            .find(|&k| history[history.len() - k..].iter().zip(&reply[..k]).all(|(&t, &b)| t == b as Token))
            .unwrap_or(0);
        reply.get(done).map(|&b| b as Token).unwrap_or(EOS)
    }
}

impl Backend for MockBackend {
    fn name(&self) -> String {
        "mock".into()
    }
    fn n_ctx(&self) -> u32 {
        self.n_ctx
    }
    fn n_seq_max(&self) -> u32 {
        self.n_seq
    }
    fn n_batch(&self) -> u32 {
        self.n_batch
    }

    fn tokenize(&self, text: &str, _add_special: bool) -> Vec<Token> {
        text.bytes().map(Token::from).collect()
    }
    fn token_to_piece(&self, token: Token) -> Vec<u8> {
        if token == EOS { Vec::new() } else { vec![token as u8] }
    }
    fn is_eog(&self, token: Token) -> bool {
        token == EOS
    }
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut s: String = messages.iter().map(|m| format!("<|{}|>{}\n", m.role, m.content)).collect();
        s.push_str("<|assistant|>");
        Ok(s)
    }

    fn decode(&mut self, batch: &[BatchToken]) -> Result<()> {
        if batch.len() > self.n_batch as usize {
            bail!("batch of {} exceeds n_batch {}", batch.len(), self.n_batch);
        }
        self.logits.clear();
        for bt in batch {
            if bt.seq >= self.n_seq {
                bail!("seq {} out of range", bt.seq);
            }
            let used: usize = self.kv.values().map(Vec::len).sum();
            if used >= self.n_ctx as usize {
                bail!("KV cache full");
            }
            let seq = self.kv.entry(bt.seq).or_default();
            if bt.pos as usize != seq.len() {
                bail!("seq {}: non-contiguous position {} (have {})", bt.seq, bt.pos, seq.len());
            }
            seq.push(bt.token);
            self.evaluated += 1;
        }
        for bt in batch {
            let mut row = vec![0.0; VOCAB];
            if bt.logits {
                let history = &self.kv[&bt.seq][..=bt.pos as usize];
                row[self.next_token(history) as usize] = 10.0;
            }
            self.logits.push(row);
        }
        Ok(())
    }

    fn logits(&self, batch_index: usize) -> &[f32] {
        &self.logits[batch_index]
    }

    fn kv_remove(&mut self, seq: u32, from: u32) {
        if let Some(v) = self.kv.get_mut(&seq) {
            v.truncate(from as usize);
        }
    }

    fn kv_copy(&mut self, src: u32, dst: u32, len: u32) {
        let prefix: Vec<Token> = self.kv.get(&src).map(|v| v[..len as usize].to_vec()).unwrap_or_default();
        self.kv.insert(dst, prefix);
    }

    fn adapters(&self) -> Vec<String> {
        self.adapters.clone()
    }

    fn set_adapter(&mut self, adapter: Option<&str>) -> Result<()> {
        if let Some(a) = adapter {
            if !self.adapters.iter().any(|x| x == a) {
                bail!("unknown adapter {a}");
            }
        }
        self.adapter = adapter.map(str::to_owned);
        Ok(())
    }
}
