//! Continuous-batching scheduler.
//!
//! One engine thread owns the backend. Each step it:
//! 1. admits queued requests onto free KV slots (reusing cached prefixes),
//! 2. picks one adapter group (llama.cpp applies LoRA per context, not per
//!    sequence, so a step only mixes sequences that share an adapter),
//! 3. packs one decode token per generating sequence, then fills the rest of
//!    the token budget with prefill chunks (chunked prefill keeps decode
//!    latency flat while long prompts are ingested),
//! 4. samples, streams text, and retires finished sequences.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc as std_mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::Serialize;

use crate::backend::{Backend, BatchToken, Token};
use crate::prefix::SlotCache;
use crate::sampling::Sampler;
use crate::types::{Event, FinishReason, GenerateRequest, Prompt, Usage};

/// Decode tokens a request must have room for before it is admitted.
const DECODE_HEADROOM: usize = 256;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Max prompt tokens one sequence may prefill per step.
    pub prefill_chunk: usize,
    /// Steps an adapter group may run before another waiting group gets a turn.
    pub adapter_quantum: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { prefill_chunk: 256, adapter_quantum: 16 }
    }
}

#[derive(Debug, Default)]
pub struct EngineMetrics {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub active: usize,
    pub queued: usize,
}

impl EngineMetrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub backend: String,
    pub n_ctx: u32,
    pub n_seq_max: u32,
    pub adapters: Vec<String>,
}

/// Handle to a running engine. Dropping it shuts the engine down and frees the model.
pub struct EngineHandle {
    tx: Option<std_mpsc::Sender<GenerateRequest>>,
    thread: Option<JoinHandle<()>>,
    pub info: EngineInfo,
    pub metrics: Arc<EngineMetrics>,
}

impl EngineHandle {
    /// Start an engine. `make_backend` runs on the engine thread, so backends
    /// need not be `Send` across their construction.
    pub fn spawn<F>(config: EngineConfig, make_backend: F) -> Result<Self>
    where
        F: FnOnce() -> Result<Box<dyn Backend>> + Send + 'static,
    {
        let (tx, rx) = std_mpsc::channel();
        let (ready_tx, ready_rx) = std_mpsc::channel();
        let metrics = Arc::new(EngineMetrics::default());
        let m = metrics.clone();
        let thread = std::thread::Builder::new()
            .name("langtrain-engine".into())
            .spawn(move || match make_backend() {
                Ok(backend) => {
                    let info = EngineInfo {
                        backend: backend.name(),
                        n_ctx: backend.n_ctx(),
                        n_seq_max: backend.n_seq_max(),
                        adapters: backend.adapters(),
                    };
                    let _ = ready_tx.send(Ok(info));
                    Engine::new(backend, config, m).run(rx);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })?;
        let info = ready_rx.recv().map_err(|_| anyhow!("engine thread exited during load"))??;
        Ok(Self { tx: Some(tx), thread: Some(thread), info, metrics })
    }

    pub fn submit(&self, req: GenerateRequest) -> Result<()> {
        self.metrics.queued.fetch_add(1, Ordering::Relaxed);
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("engine stopped"))?
            .send(req)
            .map_err(|_| anyhow!("engine stopped"))
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

enum Phase {
    Prefill,
    Decode,
}

struct Seq {
    slot: usize,
    adapter: Option<String>,
    tokens: Vec<Token>,
    n_prompt: usize,
    /// Tokens of `tokens` already evaluated into KV.
    n_past: usize,
    max_total: usize,
    sampler: Sampler,
    out: TextStream,
    events: tokio::sync::mpsc::UnboundedSender<Event>,
    usage: Usage,
    t_admit: Instant,
    t_first: Option<Instant>,
    finished: Option<FinishReason>,
}

impl Seq {
    fn phase(&self) -> Phase {
        if self.n_past < self.n_prompt { Phase::Prefill } else { Phase::Decode }
    }
}

struct Engine {
    backend: Box<dyn Backend>,
    config: EngineConfig,
    metrics: Arc<EngineMetrics>,
    slots: SlotCache,
    queue: VecDeque<(GenerateRequest, Vec<Token>, Instant)>,
    active: Vec<Seq>,
    current_adapter: Option<String>,
    adapter_steps: u32,
}

impl Engine {
    fn new(backend: Box<dyn Backend>, config: EngineConfig, metrics: Arc<EngineMetrics>) -> Self {
        let n = backend.n_seq_max() as usize;
        Self {
            backend,
            config,
            metrics,
            slots: SlotCache::new(n),
            queue: VecDeque::new(),
            active: Vec::new(),
            current_adapter: None,
            adapter_steps: 0,
        }
    }

    fn run(mut self, rx: std_mpsc::Receiver<GenerateRequest>) {
        loop {
            if self.active.is_empty() && self.queue.is_empty() {
                match rx.recv() {
                    Ok(req) => self.enqueue(req),
                    Err(_) => return,
                }
            }
            while let Ok(req) = rx.try_recv() {
                self.enqueue(req);
            }
            self.admit();
            if let Err(e) = self.step() {
                tracing::error!("engine step failed: {e:#}");
                for s in &mut self.active {
                    let _ = s.events.send(Event::Error(format!("{e:#}")));
                    s.finished = Some(FinishReason::Cancelled);
                }
            }
            self.retire();
            self.metrics.active.store(self.active.len(), Ordering::Relaxed);
            self.metrics.queued.store(self.queue.len(), Ordering::Relaxed);
        }
    }

    fn enqueue(&mut self, req: GenerateRequest) {
        let tokens = match &req.prompt {
            Prompt::Tokens(t) => Ok(t.clone()),
            Prompt::Text(t) => Ok(self.backend.tokenize(t, true)),
            Prompt::Chat(msgs) => self.backend.apply_chat_template(msgs).map(|t| self.backend.tokenize(&t, true)),
        };
        let fail = |req: &GenerateRequest, msg: String| {
            let _ = req.events.send(Event::Error(msg));
        };
        match tokens {
            Err(e) => fail(&req, format!("failed to render prompt: {e:#}")),
            Ok(t) if t.is_empty() => fail(&req, "empty prompt".into()),
            Ok(t) if t.len() + 1 > self.backend.n_ctx() as usize => fail(
                &req,
                format!("prompt has {} tokens; context window is {}", t.len(), self.backend.n_ctx()),
            ),
            Ok(_) if req.adapter.as_ref().is_some_and(|a| !self.backend.adapters().contains(a)) => {
                fail(&req, format!("unknown adapter '{}'", req.adapter.as_deref().unwrap_or_default()))
            }
            Ok(t) => self.queue.push_back((req, t, Instant::now())),
        }
    }

    /// KV cells in use: idle slots' cached prefixes plus live sequences.
    fn kv_in_use(&self) -> usize {
        self.slots.idle_cells() + self.active.iter().map(|s| s.n_past).sum::<usize>()
    }

    /// Admit queued requests FIFO while there are free slots and KV room.
    fn admit(&mut self) {
        let n_ctx = self.backend.n_ctx() as usize;
        while let Some((req, prompt, _)) = self.queue.front() {
            if req.events.is_closed() {
                self.queue.pop_front();
                continue;
            }
            let Some(place) = self.slots.place(prompt, req.adapter.as_deref()) else { break };

            // Admit when the prompt plus some decode headroom fits; idle
            // prefixes are evicted LRU to make room.
            let max_new = req.params.max_tokens.unwrap_or(usize::MAX).min(n_ctx - prompt.len());
            let need = prompt.len() - place.reuse + max_new.min(DECODE_HEADROOM);
            while self.kv_in_use() + need > n_ctx {
                match self.slots.evict_lru() {
                    Some(i) => self.backend.kv_remove(i as u32, 0),
                    None => break,
                }
            }
            if self.kv_in_use() + need > n_ctx && !self.active.is_empty() {
                break; // wait for running sequences to free KV
            }

            let (req, prompt, t_queued) = self.queue.pop_front().expect("front exists");
            // Eviction may have cleared the prefix we planned to reuse; re-plan.
            let place = self.slots.place(&prompt, req.adapter.as_deref()).expect("slot still free");
            let slot = place.slot as u32;
            match place.copy_from {
                Some((src, len)) => {
                    self.backend.kv_remove(slot, 0);
                    self.backend.kv_copy(src as u32, slot, len as u32);
                }
                None => self.backend.kv_remove(slot, place.reuse as u32),
            }
            self.slots.acquire(&place, &prompt, req.adapter.as_deref());

            self.metrics.requests.fetch_add(1, Ordering::Relaxed);
            self.metrics.prompt_tokens.fetch_add(prompt.len() as u64, Ordering::Relaxed);
            self.metrics.cached_tokens.fetch_add(place.reuse as u64, Ordering::Relaxed);

            let n_prompt = prompt.len();
            self.active.push(Seq {
                slot: place.slot,
                adapter: req.adapter,
                max_total: n_prompt + max_new,
                tokens: prompt,
                n_prompt,
                n_past: place.reuse,
                out: TextStream::new(req.params.stop.clone()),
                sampler: Sampler::new(req.params),
                events: req.events,
                usage: Usage {
                    prompt_tokens: n_prompt,
                    cached_tokens: place.reuse,
                    queue_ms: ms(t_queued.elapsed()),
                    ..Default::default()
                },
                t_admit: Instant::now(),
                t_first: None,
                finished: None,
            });
        }
    }

    /// Make room for `n` more KV cells: evict idle prefixes first, then end
    /// the youngest running sequence. Returns false if a sequence was ended.
    fn ensure_kv_room(&mut self, n: usize) -> bool {
        let n_ctx = self.backend.n_ctx() as usize;
        while self.kv_in_use() + n > n_ctx {
            if let Some(i) = self.slots.evict_lru() {
                self.backend.kv_remove(i as u32, 0);
                continue;
            }
            if let Some(victim) = self.active.iter_mut().filter(|s| s.finished.is_none()).max_by_key(|s| s.t_admit) {
                victim.finished = Some(FinishReason::Length);
            }
            return false;
        }
        true
    }

    /// Round-robin between adapter groups so one busy adapter cannot starve the others.
    fn pick_adapter(&mut self) -> Option<String> {
        let current_live = self.active.iter().any(|s| s.adapter == self.current_adapter);
        let others_waiting = self.active.iter().any(|s| s.adapter != self.current_adapter);
        if current_live && (self.adapter_steps < self.config.adapter_quantum || !others_waiting) {
            self.adapter_steps += 1;
            return self.current_adapter.clone();
        }
        // Next group: the one whose oldest sequence has waited longest.
        let next = self
            .active
            .iter()
            .filter(|s| !current_live || s.adapter != self.current_adapter)
            .min_by_key(|s| s.t_admit)
            .map(|s| s.adapter.clone())
            .unwrap_or(None);
        self.adapter_steps = 1;
        next
    }

    fn step(&mut self) -> Result<()> {
        if self.active.is_empty() {
            return Ok(());
        }
        for s in &mut self.active {
            if s.events.is_closed() {
                s.finished = Some(FinishReason::Cancelled);
            }
        }
        let adapter = self.pick_adapter();
        if adapter != self.current_adapter {
            self.backend.set_adapter(adapter.as_deref())?;
            self.current_adapter = adapter.clone();
        }

        let budget = self.backend.n_batch() as usize;
        let mut batch: Vec<BatchToken> = Vec::with_capacity(budget);
        // (seq index, batch index of its logits)
        let mut sample_at: Vec<(usize, usize)> = Vec::new();

        let in_group = |s: &Seq| s.finished.is_none() && s.adapter == adapter;
        for (i, s) in self.active.iter().enumerate() {
            if in_group(s) && matches!(s.phase(), Phase::Decode) && batch.len() < budget {
                sample_at.push((i, batch.len()));
                batch.push(BatchToken {
                    token: s.tokens[s.n_past],
                    pos: s.n_past as u32,
                    seq: s.slot as u32,
                    logits: true,
                });
            }
        }
        let mut prefill_advance: Vec<(usize, usize)> = Vec::new();
        for (i, s) in self.active.iter().enumerate() {
            let room = budget - batch.len();
            if room == 0 {
                break;
            }
            if !in_group(s) || !matches!(s.phase(), Phase::Prefill) {
                continue;
            }
            let n = (s.n_prompt - s.n_past).min(room).min(self.config.prefill_chunk);
            for k in 0..n {
                let pos = s.n_past + k;
                let last = pos + 1 == s.n_prompt;
                if last {
                    sample_at.push((i, batch.len()));
                }
                batch.push(BatchToken { token: s.tokens[pos], pos: pos as u32, seq: s.slot as u32, logits: last });
            }
            prefill_advance.push((i, n));
        }

        if batch.is_empty() || !self.ensure_kv_room(batch.len()) {
            return Ok(());
        }
        let t = Instant::now();
        self.backend.decode(&batch)?;
        let elapsed = ms(t.elapsed());

        for &(i, n) in &prefill_advance {
            let s = &mut self.active[i];
            s.n_past += n;
            s.usage.prefill_ms += elapsed;
        }
        for &(i, bi) in &sample_at {
            let s = &mut self.active[i];
            if matches!(s.phase(), Phase::Decode) && s.n_past < s.tokens.len() {
                // This was a decode step: the pending token is now in KV.
                s.n_past += 1;
                s.usage.decode_ms += elapsed;
            }
            let tok = s.sampler.sample(self.backend.logits(bi), &s.tokens[s.n_prompt..]);
            s.t_first.get_or_insert_with(Instant::now);
            self.metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);

            if self.backend.is_eog(tok) {
                s.finished = Some(FinishReason::Stop);
                continue;
            }
            s.tokens.push(tok);
            s.usage.completion_tokens += 1;
            let piece = self.backend.token_to_piece(tok);
            let (text, hit_stop) = s.out.push(&piece);
            if !text.is_empty() {
                let _ = s.events.send(Event::Text(text));
            }
            if hit_stop {
                s.finished = Some(FinishReason::Stop);
            } else if s.tokens.len() >= s.max_total {
                s.finished = Some(FinishReason::Length);
            }
        }
        Ok(())
    }

    fn retire(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let Some(reason) = self.active[i].finished else {
                i += 1;
                continue;
            };
            let mut s = self.active.swap_remove(i);
            if reason != FinishReason::Cancelled {
                let tail = s.out.finish();
                if !tail.is_empty() {
                    let _ = s.events.send(Event::Text(tail));
                }
            }
            let _ = s.events.send(Event::Done { reason, usage: s.usage.clone() });
            // Keep exactly what is in KV for future prefix reuse.
            self.slots.release(s.slot, &s.tokens[..s.n_past]);
            if s.n_past < s.tokens.len() {
                self.backend.kv_remove(s.slot as u32, s.n_past as u32);
            }
        }
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Turns token bytes into UTF-8 text, holding back output that might be the
/// start of a stop string or a split multi-byte character.
struct TextStream {
    bytes: Vec<u8>,
    pending: String,
    stop: Vec<String>,
}

impl TextStream {
    fn new(stop: Vec<String>) -> Self {
        Self { bytes: Vec::new(), pending: String::new(), stop: stop.into_iter().filter(|s| !s.is_empty()).collect() }
    }

    /// Returns `(text safe to emit, stop string hit)`.
    fn push(&mut self, piece: &[u8]) -> (String, bool) {
        self.bytes.extend_from_slice(piece);
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_some() => self.bytes.len(), // invalid, not just incomplete
            Err(e) => e.valid_up_to(),
        };
        let chunk: Vec<u8> = self.bytes.drain(..valid).collect();
        self.pending.push_str(&String::from_utf8_lossy(&chunk));

        if let Some(at) = self.stop.iter().filter_map(|s| self.pending.find(s.as_str())).min() {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            return (out, true);
        }
        // Hold back the longest suffix that is a prefix of some stop string.
        let hold = self
            .stop
            .iter()
            .flat_map(|s| (1..s.len()).rev().filter(move |&k| s.is_char_boundary(k)).map(move |k| &s[..k]))
            .filter(|p| self.pending.ends_with(p))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    fn finish(&mut self) -> String {
        let mut out = std::mem::take(&mut self.pending);
        out.push_str(&String::from_utf8_lossy(&std::mem::take(&mut self.bytes)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockBackend, REPLY};
    use crate::sampling::SamplingParams;
    use crate::types::ChatMessage;
    use tokio::sync::mpsc::unbounded_channel;

    fn engine(n_ctx: u32, n_seq: u32, n_batch: u32) -> EngineHandle {
        EngineHandle::spawn(EngineConfig { prefill_chunk: 32, adapter_quantum: 2 }, move || {
            Ok(Box::new(MockBackend::new(n_ctx, n_seq, n_batch)) as Box<dyn Backend>)
        })
        .unwrap()
    }

    fn run(handle: &EngineHandle, prompt: Prompt, params: SamplingParams, adapter: Option<&str>) -> (String, FinishReason, Usage) {
        let (tx, mut rx) = unbounded_channel();
        handle.submit(GenerateRequest { prompt, params, adapter: adapter.map(Into::into), events: tx }).unwrap();
        let mut text = String::new();
        loop {
            match rx.blocking_recv().expect("engine dropped request") {
                Event::Text(t) => text.push_str(&t),
                Event::Done { reason, usage } => return (text, reason, usage),
                Event::Error(e) => panic!("{e}"),
            }
        }
    }

    fn chat(s: &str) -> Prompt {
        Prompt::Chat(vec![ChatMessage::new("system", "You are helpful. ".repeat(8)), ChatMessage::new("user", s)])
    }

    #[test]
    fn generates_full_reply() {
        let h = engine(4096, 4, 64);
        let (text, reason, usage) = run(&h, chat("hi"), SamplingParams::greedy(), None);
        assert_eq!(text, REPLY);
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(usage.completion_tokens, REPLY.len());
    }

    #[test]
    fn max_tokens_and_stop_strings() {
        let h = engine(4096, 2, 64);
        let p = SamplingParams { max_tokens: Some(5), ..SamplingParams::greedy() };
        let (text, reason, _) = run(&h, chat("hi"), p, None);
        assert_eq!((text.as_str(), reason), ("Hello", FinishReason::Length));

        let p = SamplingParams { stop: vec!["Lang".into()], ..SamplingParams::greedy() };
        let (text, reason, _) = run(&h, chat("hi"), p, None);
        assert_eq!((text.as_str(), reason), ("Hello from ", FinishReason::Stop));
    }

    #[test]
    fn second_request_reuses_prefix() {
        let h = engine(4096, 2, 64);
        let (_, _, first) = run(&h, chat("first question"), SamplingParams::greedy(), None);
        assert_eq!(first.cached_tokens, 0);
        let (_, _, second) = run(&h, chat("second question"), SamplingParams::greedy(), None);
        // The shared system prompt and role tags come from the cache.
        assert!(second.cached_tokens > 100, "cached {}", second.cached_tokens);
    }

    #[test]
    fn concurrent_requests_with_mixed_adapters() {
        let h = engine(8192, 4, 48);
        let h = Arc::new(h);
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let h = h.clone();
                std::thread::spawn(move || {
                    let adapter = (i % 2 == 0).then_some("pirate");
                    let (text, reason, _) = run(&h, chat(&format!("question {i}")), SamplingParams::greedy(), adapter);
                    (adapter, text, reason)
                })
            })
            .collect();
        for t in threads {
            let (adapter, text, reason) = t.join().unwrap();
            let want = match adapter {
                Some(a) => format!("[{a}] {REPLY}"),
                None => REPLY.to_string(),
            };
            assert_eq!(text, want);
            assert_eq!(reason, FinishReason::Stop);
        }
    }

    #[test]
    fn long_prompt_is_chunked() {
        let h = engine(4096, 2, 16);
        let long = "x".repeat(1000);
        let (text, _, usage) = run(&h, Prompt::Text(long), SamplingParams::greedy(), None);
        assert_eq!(text, REPLY);
        assert_eq!(usage.prompt_tokens, 1000);
    }

    #[test]
    fn context_pressure_ends_youngest_sequence_gracefully() {
        // 600 cells: three ~170-token prompts fit, but not their full replies plus cached prefixes.
        let h = Arc::new(engine(600, 3, 64));
        let threads: Vec<_> = (0..3)
            .map(|i| {
                let h = h.clone();
                std::thread::spawn(move || run(&h, chat(&format!("q{i}")), SamplingParams::greedy(), None).1)
            })
            .collect();
        for t in threads {
            let reason = t.join().unwrap();
            assert!(matches!(reason, FinishReason::Stop | FinishReason::Length));
        }
    }

    #[test]
    fn rejects_prompt_larger_than_context() {
        let h = engine(64, 1, 16);
        let (tx, mut rx) = unbounded_channel();
        h.submit(GenerateRequest { prompt: Prompt::Text("y".repeat(100)), params: SamplingParams::greedy(), adapter: None, events: tx })
            .unwrap();
        assert!(matches!(rx.blocking_recv(), Some(Event::Error(_))));
    }

    #[test]
    fn text_stream_handles_split_utf8_and_partial_stop() {
        let mut s = TextStream::new(vec!["</end>".into()]);
        let euro = "€".as_bytes();
        assert_eq!(s.push(&euro[..1]), (String::new(), false));
        assert_eq!(s.push(&euro[1..]), ("€".into(), false));
        assert_eq!(s.push(b"ok</e"), ("ok".into(), false));
        assert_eq!(s.push(b"nd>tail"), (String::new(), true));
    }
}
