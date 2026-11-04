//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures::stream::{self, Stream, StreamExt};
use langtrain_engine::sampling::SamplingParams;
use langtrain_engine::{ChatMessage, Event, FinishReason, Prompt, Usage};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::billing::Caller;
use crate::state::{unix_now, ApiError, AppState};

#[derive(Debug, Deserialize)]
pub struct InMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

/// `content` may be a string or an array of `{type: "text", text}` parts.
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CommonParams {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<Value>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<Stop>,
}

impl CommonParams {
    fn sampling(&mut self) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            temperature: self.temperature.unwrap_or(d.temperature),
            top_k: self.top_k.unwrap_or(d.top_k),
            top_p: self.top_p.unwrap_or(d.top_p),
            min_p: self.min_p.unwrap_or(d.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed,
            stop: self.stop.take().map(Stop::into_vec).unwrap_or_default(),
            ..d
        }
    }

    fn include_usage(&self) -> bool {
        self.stream_options.as_ref().and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false)
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<InMessage>,
    #[serde(flatten)]
    pub common: CommonParams,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(flatten)]
    pub common: CommonParams,
}

pub async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut data = Vec::new();
    for m in state.list_models() {
        data.push(json!({ "id": m.name, "object": "model", "created": 0, "owned_by": "local" }));
        for a in &m.adapters {
            data.push(json!({ "id": format!("{}@{a}", m.name), "object": "model", "created": 0, "owned_by": "local", "parent": m.name }));
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": u.cached_tokens },
    })
}

/// Drain an event channel into the full text.
pub async fn collect(mut rx: UnboundedReceiver<Event>) -> Result<(String, FinishReason, Usage), ApiError> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Text(t) => text.push_str(&t),
            Event::Done { reason, usage } => return Ok((text, reason, usage)),
            Event::Error(e) => return Err(ApiError::bad_request(e)),
        }
    }
    Err(ApiError(axum::http::StatusCode::INTERNAL_SERVER_ERROR, "engine stopped".into()))
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Json(mut req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let messages: Vec<ChatMessage> = req.messages.iter().map(|m| ChatMessage::new(&m.role, content_text(&m.content))).collect();
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let params = req.common.sampling();
    let rx = state.generate_for(caller.as_deref(), &req.common.model, Prompt::Chat(messages), params).await?;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = req.common.model.clone();

    if req.common.stream {
        let include_usage = req.common.include_usage();
        let chunk = move |delta: Value, finish: Option<&str>| {
            json!({ "id": id, "object": "chat.completion.chunk", "created": unix_now(), "model": model,
                    "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
        };
        let first = chunk(json!({ "role": "assistant", "content": "" }), None);
        let events = sse_stream(rx, first, move |ev| match ev {
            Event::Text(t) => vec![chunk(json!({ "content": t }), None)],
            Event::Done { reason, usage } => {
                let mut out = vec![chunk(json!({}), Some(reason.as_str()))];
                if include_usage {
                    let mut u = chunk(json!({}), None);
                    u["choices"] = json!([]);
                    u["usage"] = usage_json(&usage);
                    out.push(u);
                }
                out
            }
            Event::Error(e) => vec![json!({ "error": { "message": e } })],
        });
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }

    let (text, reason, usage) = collect(rx).await?;
    Ok(Json(json!({
        "id": id, "object": "chat.completion", "created": unix_now(), "model": req.common.model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": reason.as_str() }],
        "usage": usage_json(&usage),
        "timings": { "prefill_ms": usage.prefill_ms, "decode_tokens_per_sec": usage.decode_tokens_per_sec() },
    }))
    .into_response())
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    caller: Option<Extension<Caller>>,
    Json(mut req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let params = req.common.sampling();
    let rx = state.generate_for(caller.as_deref(), &req.common.model, Prompt::Text(req.prompt.clone()), params).await?;
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let model = req.common.model.clone();

    if req.common.stream {
        let chunk = move |text: &str, finish: Option<&str>| {
            json!({ "id": id, "object": "text_completion", "created": unix_now(), "model": model,
                    "choices": [{ "index": 0, "text": text, "finish_reason": finish }] })
        };
        let first = chunk("", None);
        let events = sse_stream(rx, first, move |ev| match ev {
            Event::Text(t) => vec![chunk(&t, None)],
            Event::Done { reason, .. } => vec![chunk("", Some(reason.as_str()))],
            Event::Error(e) => vec![json!({ "error": { "message": e } })],
        });
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }

    let (text, reason, usage) = collect(rx).await?;
    Ok(Json(json!({
        "id": id, "object": "text_completion", "created": unix_now(), "model": req.common.model,
        "choices": [{ "index": 0, "text": text, "finish_reason": reason.as_str() }],
        "usage": usage_json(&usage),
    }))
    .into_response())
}

/// SSE body: `first`, then mapped engine events, then `[DONE]`.
fn sse_stream<F>(rx: UnboundedReceiver<Event>, first: Value, map: F) -> impl Stream<Item = Result<SseEvent, Infallible>>
where
    F: Fn(Event) -> Vec<Value> + Send + 'static,
{
    let head = stream::once(async move { first });
    let body = stream::unfold((rx, false), |(mut rx, done)| async move {
        if done {
            return None;
        }
        let ev = rx.recv().await?;
        let finished = matches!(ev, Event::Done { .. } | Event::Error(_));
        Some((ev, (rx, finished)))
    })
    .flat_map(move |ev| stream::iter(map(ev)));
    head.chain(body)
        .map(|v| Ok(SseEvent::default().data(v.to_string())))
        .chain(stream::once(async { Ok(SseEvent::default().data("[DONE]")) }))
}
