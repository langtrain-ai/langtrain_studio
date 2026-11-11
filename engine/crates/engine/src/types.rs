use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::backend::Token;
use crate::sampling::SamplingParams;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

#[derive(Debug, Clone)]
pub enum Prompt {
    /// Raw text, tokenized as-is (completion endpoints).
    Text(String),
    /// Chat messages, rendered with the model's chat template.
    Chat(Vec<ChatMessage>),
    Tokens(Vec<Token>),
}

#[derive(Debug)]
pub struct GenerateRequest {
    pub prompt: Prompt,
    pub params: SamplingParams,
    /// LoRA adapter to apply, by name. `None` serves the base model.
    pub adapter: Option<String>,
    pub events: mpsc::UnboundedSender<Event>,
    /// Set to stop generation early. Unlike dropping `events`, the engine
    /// still reports a final `Done` with usage, which billing needs.
    pub cancel: Arc<AtomicBool>,
}

impl GenerateRequest {
    pub fn new(prompt: Prompt, params: SamplingParams, adapter: Option<String>, events: mpsc::UnboundedSender<Event>) -> Self {
        Self { prompt, params, adapter, events, cancel: Arc::new(AtomicBool::new(false)) }
    }

    /// The client is gone or asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.events.is_closed() || self.cancel.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens served from the prefix cache instead of being recomputed.
    pub cached_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

impl Usage {
    pub fn decode_tokens_per_sec(&self) -> f64 {
        if self.decode_ms <= 0.0 {
            0.0
        } else {
            self.completion_tokens as f64 / (self.decode_ms / 1000.0)
        }
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    Text(String),
    Done { reason: FinishReason, usage: Usage },
    Error(String),
}
