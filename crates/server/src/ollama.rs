//! Ollama-compatible endpoints, so existing Ollama clients work unchanged.

use std::sync::Arc;
use std::time::SystemTime;

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, StreamExt};
use langtrain_engine::sampling::SamplingParams;
use langtrain_engine::{ChatMessage, Event, Prompt, Usage};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::openai::{collect, content_text, InMessage};
use crate::state::{ApiError, AppState};

fn rfc3339(t: SystemTime) -> String {
    // Seconds precision is all clients use.
    let secs = t.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Howard Hinnant's days-to-civil algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn now() -> String {
    rfc3339(SystemTime::now())
}

pub async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

pub async fn tags(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut models = Vec::new();
    for m in state.list_models() {
        let modified = m.modified.map(rfc3339).unwrap_or_else(now);
        let quant = m.name.rsplit(['-', '.']).next().filter(|q| q.starts_with('Q') || q.starts_with("F") || q.starts_with("BF")).unwrap_or("");
        let entry = |name: String| {
            json!({ "name": name, "model": name, "modified_at": modified, "size": m.size_bytes, "digest": "",
                    "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": quant } })
        };
        models.push(entry(m.name.clone()));
        for a in &m.adapters {
            models.push(entry(format!("{}@{a}", m.name)));
        }
    }
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<InMessage>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    pub system: Option<String>,
    #[serde(default)]
    pub raw: bool,
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

fn sampling(o: &Map<String, Value>) -> SamplingParams {
    let d = SamplingParams::default();
    let f = |k: &str, def: f32| o.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(def);
    let n = |k: &str| o.get(k).and_then(Value::as_i64);
    SamplingParams {
        // Ollama uses -1 for "no limit".
        max_tokens: n("num_predict").filter(|&v| v >= 0).map(|v| v as usize),
        temperature: f("temperature", d.temperature),
        top_k: n("top_k").map(|v| v.max(0) as usize).unwrap_or(d.top_k),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        repeat_penalty: f("repeat_penalty", d.repeat_penalty),
        repeat_last_n: n("repeat_last_n").map(|v| v.max(0) as usize).unwrap_or(d.repeat_last_n),
        presence_penalty: f("presence_penalty", 0.0),
        frequency_penalty: f("frequency_penalty", 0.0),
        seed: n("seed").map(|v| v as u64),
        stop: o
            .get("stop")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_owned)).collect())
            .unwrap_or_default(),
    }
}

fn final_fields(reason: &str, u: &Usage) -> Value {
    let ns = |ms: f64| (ms * 1e6) as u64;
    json!({
        "done": true,
        "done_reason": reason,
        "total_duration": ns(u.queue_ms + u.prefill_ms + u.decode_ms),
        "prompt_eval_count": u.prompt_tokens,
        "prompt_eval_duration": ns(u.prefill_ms),
        "eval_count": u.completion_tokens,
        "eval_duration": ns(u.decode_ms),
    })
}

/// Stream engine events as newline-delimited JSON. `piece` builds the
/// per-chunk payload (`message` for chat, `response` for generate).
fn ndjson<F>(rx: UnboundedReceiver<Event>, model: String, piece: F) -> Response
where
    F: Fn(&str) -> Value + Send + 'static,
{
    let lines = stream::unfold((rx, false), |(mut rx, done)| async move {
        if done {
            return None;
        }
        let ev = rx.recv().await?;
        let finished = matches!(ev, Event::Done { .. } | Event::Error(_));
        Some((ev, (rx, finished)))
    })
    .map(move |ev| {
        let mut v = match ev {
            Event::Text(t) => {
                let mut v = piece(&t);
                v["done"] = json!(false);
                v
            }
            Event::Done { reason, usage } => {
                let mut v = piece("");
                for (k, val) in final_fields(reason.as_str(), &usage).as_object().into_iter().flatten() {
                    v[k] = val.clone();
                }
                v
            }
            Event::Error(e) => json!({ "error": e }),
        };
        v["model"] = json!(model);
        v["created_at"] = json!(now());
        Ok::<_, std::convert::Infallible>(format!("{v}\n"))
    });
    ([(header::CONTENT_TYPE, "application/x-ndjson")], Body::from_stream(lines)).into_response()
}

pub async fn chat(State(state): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Result<Response, ApiError> {
    let messages: Vec<ChatMessage> = req.messages.iter().map(|m| ChatMessage::new(&m.role, content_text(&m.content))).collect();
    // An empty message list is Ollama's way to preload a model.
    if messages.is_empty() {
        let (base, _) = crate::state::parse_model_name(&req.model);
        state.engine(&base).await?;
        return Ok(Json(json!({ "model": req.model, "created_at": now(), "message": { "role": "assistant", "content": "" }, "done": true, "done_reason": "load" })).into_response());
    }
    let rx = state.generate(&req.model, Prompt::Chat(messages), sampling(&req.options)).await?;
    let msg = |t: &str| json!({ "message": { "role": "assistant", "content": t } });
    if req.stream.unwrap_or(true) {
        return Ok(ndjson(rx, req.model, msg));
    }
    let (text, reason, usage) = collect(rx).await?;
    let mut v = msg(&text);
    for (k, val) in final_fields(reason.as_str(), &usage).as_object().into_iter().flatten() {
        v[k] = val.clone();
    }
    v["model"] = json!(req.model);
    v["created_at"] = json!(now());
    Ok(Json(v).into_response())
}

pub async fn generate(State(state): State<Arc<AppState>>, Json(req): Json<GenerateRequest>) -> Result<Response, ApiError> {
    let prompt = if req.raw {
        Prompt::Text(req.prompt.clone())
    } else {
        let mut msgs = Vec::new();
        if let Some(s) = &req.system {
            msgs.push(ChatMessage::new("system", s));
        }
        msgs.push(ChatMessage::new("user", &req.prompt));
        Prompt::Chat(msgs)
    };
    let rx = state.generate(&req.model, prompt, sampling(&req.options)).await?;
    let resp = |t: &str| json!({ "response": t });
    if req.stream.unwrap_or(true) {
        return Ok(ndjson(rx, req.model, resp));
    }
    let (text, reason, usage) = collect(rx).await?;
    let mut v = resp(&text);
    for (k, val) in final_fields(reason.as_str(), &usage).as_object().into_iter().flatten() {
        v[k] = val.clone();
    }
    v["model"] = json!(req.model);
    v["created_at"] = json!(now());
    Ok(Json(v).into_response())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dates() {
        assert_eq!(super::civil_from_days(0), (1970, 1, 1));
        assert_eq!(super::civil_from_days(20_395), (2025, 11, 3));
    }
}
