use serde::{Deserialize, Serialize};
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::State;

/// State for the running llama.cpp server process.
pub struct InferenceState {
    pub process: Mutex<Option<Child>>,
    pub model_path: Mutex<Option<String>>,
    pub port: Mutex<u16>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            process: Mutex::new(None),
            model_path: Mutex::new(None),
            port: Mutex::new(11435),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    pub model: Option<String>,
    pub port: u16,
    pub endpoint: Option<String>,
}

/// Start the llama.cpp server as a sidecar process.
///
/// The server exposes an OpenAI-compatible API at `localhost:{port}/v1/chat/completions`.
/// This expects `llama-server` to be available as a Tauri sidecar or on PATH.
#[tauri::command]
pub fn start_inference_server(
    state: State<'_, InferenceState>,
    model_path: String,
    port: Option<u16>,
    context_size: Option<u32>,
    gpu_layers: Option<i32>,
) -> Result<ServerStatus, String> {
    let mut proc_guard = state.process.lock().map_err(|e| e.to_string())?;

    // Check if already running
    if proc_guard.is_some() {
        return Err("Inference server is already running. Stop it first.".to_string());
    }

    // Validate model path
    if !std::path::Path::new(&model_path).exists() {
        return Err(format!("Model file not found: {}", model_path));
    }

    let port = port.unwrap_or(11435);
    let ctx = context_size.unwrap_or(4096);
    let ngl = gpu_layers.unwrap_or(-1); // -1 = auto

    // Build command — try sidecar first, fallback to PATH
    let mut cmd = Command::new("llama-server");

    cmd.args([
        "--model", &model_path,
        "--port", &port.to_string(),
        "--ctx-size", &ctx.to_string(),
        "--n-gpu-layers", &ngl.to_string(),
        "--host", "127.0.0.1",
        // OpenAI-compatible API mode
        "--verbose",
    ]);

    // Suppress stdout/stderr to avoid blocking
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());

    let child = cmd.spawn().map_err(|e| {
        format!(
            "Failed to start llama-server. Make sure it's installed. Error: {}",
            e
        )
    })?;

    *proc_guard = Some(child);
    *state.model_path.lock().map_err(|e| e.to_string())? = Some(model_path.clone());
    *state.port.lock().map_err(|e| e.to_string())? = port;

    Ok(ServerStatus {
        running: true,
        model: Some(model_path),
        port,
        endpoint: Some(format!("http://127.0.0.1:{}/v1/chat/completions", port)),
    })
}

/// Stop the running inference server.
#[tauri::command]
pub fn stop_inference_server(state: State<'_, InferenceState>) -> Result<ServerStatus, String> {
    let mut proc_guard = state.process.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = proc_guard.take() {
        // Try graceful shutdown first
        let _ = child.kill();
        let _ = child.wait();
    }

    *state.model_path.lock().map_err(|e| e.to_string())? = None;

    let port = *state.port.lock().map_err(|e| e.to_string())?;

    Ok(ServerStatus {
        running: false,
        model: None,
        port,
        endpoint: None,
    })
}

/// Check if the inference server is running and healthy.
#[tauri::command]
pub fn inference_server_status(state: State<'_, InferenceState>) -> ServerStatus {
    let proc_guard = state.process.lock().unwrap_or_else(|e| e.into_inner());
    let model = state
        .model_path
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let port = *state.port.lock().unwrap_or_else(|e| e.into_inner());

    let running = proc_guard.is_some();

    ServerStatus {
        running,
        model,
        port,
        endpoint: if running {
            Some(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        } else {
            None
        },
    }
}
