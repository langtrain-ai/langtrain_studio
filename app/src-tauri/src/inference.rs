//! Local inference through the bundled Langtrain engine (`engine/` in this
//! repo), shipped inside the app as the `langtrain` sidecar on macOS, Windows
//! and Linux. It serves an OpenAI-compatible API on `127.0.0.1:{port}`
//! (`/v1/chat/completions`, `/v1/models`), the same endpoint the frontend has
//! always used, so chat, agents and the model picker work unchanged.
//!
//! The engine serves every `.gguf` in `~/.langtrain/models` by file name
//! (without `.gguf`), which is also where the model manager downloads to. A
//! model picked from elsewhere is linked into that directory first.
//!
//! If a `langtrain serve` is already answering on the port (for example one the
//! user started from a terminal), the app uses it instead of starting another.

use serde::{Deserialize, Serialize};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, State};

pub const DEFAULT_PORT: u16 = 11435;

/// The engine process, if this app started it, and what it is serving.
pub struct InferenceState {
    pub process: Mutex<Option<Child>>,
    pub model_path: Mutex<Option<String>>,
    pub model_name: Mutex<Option<String>>,
    pub port: Mutex<u16>,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self {
            process: Mutex::new(None),
            model_path: Mutex::new(None),
            model_name: Mutex::new(None),
            port: Mutex::new(DEFAULT_PORT),
        }
    }
}

impl InferenceState {
    /// Stops the engine if this app started it. Called on stop and on app exit.
    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.process.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    /// Path of the model file that was started (what the model list matches on).
    pub model: Option<String>,
    /// The name the engine knows the model by; send it as `model` in requests.
    pub model_name: Option<String>,
    pub port: u16,
    pub endpoint: Option<String>,
}

fn endpoint(port: u16) -> String {
    format!("http://127.0.0.1:{port}/v1/chat/completions")
}

fn is_listening(port: u16) -> bool {
    TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(200)).is_ok()
}

/// `~/.langtrain/models`: shared by the model manager and the engine.
fn engine_models_dir() -> PathBuf {
    crate::model_manager::models_dir()
}

/// The `langtrain` sidecar. Tauri installs `externalBin` entries next to the
/// app's own executable, without the target-triple suffix they are built with.
fn sidecar_path() -> PathBuf {
    let name = if cfg!(windows) { "langtrain.exe" } else { "langtrain" };
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(name)))
        .filter(|p| p.exists())
        // Development without a built sidecar: a `langtrain` on PATH.
        .unwrap_or_else(|| PathBuf::from(name))
}

/// Makes `model_path` visible to the engine under `~/.langtrain/models` and
/// returns the name it serves it as. Files already there are used in place;
/// others are hard-linked (same disk) or, failing that, symlinked. Never copied:
/// models are gigabytes.
fn ensure_in_models_dir(model_path: &Path) -> Result<String, String> {
    if model_path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")) != Some(true) {
        return Err(format!("{} is not a .gguf model file", model_path.display()));
    }
    let name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("Unusable model file name: {}", model_path.display()))?
        .to_string();

    let dir = engine_models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
    let target = dir.join(format!("{name}.gguf"));

    let same_file = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if target.exists() {
        if same_file(model_path, &target) {
            return Ok(name);
        }
        return Err(format!(
            "A different model named {name}.gguf is already in {}. Rename one of them.",
            dir.display()
        ));
    }

    if std::fs::hard_link(model_path, &target).is_ok() {
        return Ok(name);
    }
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(model_path, &target);
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(model_path, &target);
    linked.map(|_| name).map_err(|e| {
        format!(
            "Could not make {} available to the engine ({e}). Move it into {}.",
            model_path.display(),
            dir.display()
        )
    })
}

/// Where the engine's output goes, so a failed start can be diagnosed.
fn log_file() -> Option<std::fs::File> {
    let dir = engine_models_dir().parent()?.join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::OpenOptions::new().create(true).append(true).open(dir.join("studio-engine.log")).ok()
}

fn spawn_engine(app: &AppHandle, port: u16, context_size: Option<u32>) -> Result<Child, String> {
    let mut cmd = Command::new(sidecar_path());
    cmd.args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()]);
    if let Some(ctx) = context_size {
        cmd.env("LANGTRAIN_CTX", ctx.to_string());
    }
    // The fine-tuning package is bundled as a resource. The engine also looks
    // next to its binary, but Linux packages keep resources elsewhere.
    if let Ok(dir) = app.path().resource_dir() {
        let langtune = dir.join("langtune");
        if langtune.join("langtune").join("__init__.py").exists() {
            cmd.env("LANGTUNE_DIR", langtune);
        }
    }
    cmd.stdin(Stdio::null());
    match log_file().and_then(|f| f.try_clone().ok().map(|g| (f, g))) {
        Some((out, err)) => {
            cmd.stdout(out).stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.spawn().map_err(|e| format!("Failed to start the Langtrain engine ({}): {e}", sidecar_path().display()))
}

/// Start (or reuse) the engine and select `model_path` for chat.
///
/// `gpu_layers` is accepted for compatibility and ignored: the engine picks
/// Metal / CPU kernels itself.
#[tauri::command]
pub fn start_inference_server(
    app: AppHandle,
    state: State<'_, InferenceState>,
    model_path: String,
    port: Option<u16>,
    context_size: Option<u32>,
    gpu_layers: Option<i32>,
) -> Result<ServerStatus, String> {
    let _ = gpu_layers;
    let path = PathBuf::from(&model_path);
    if !path.exists() {
        return Err(format!("Model file not found: {model_path}"));
    }
    let name = ensure_in_models_dir(&path)?;
    let port = port.unwrap_or(DEFAULT_PORT);

    {
        let mut proc_guard = state.process.lock().map_err(|e| e.to_string())?;
        // Our engine exited (crash, killed): forget it so it can be restarted.
        if let Some(child) = proc_guard.as_mut() {
            if child.try_wait().ok().flatten().is_some() {
                *proc_guard = None;
            }
        }
        if proc_guard.is_none() && !is_listening(port) {
            let mut child = spawn_engine(&app, port, context_size)?;
            let started = Instant::now();
            while !is_listening(port) {
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(format!(
                        "The Langtrain engine exited during start-up ({status}). See ~/.langtrain/logs/studio-engine.log"
                    ));
                }
                if started.elapsed() > Duration::from_secs(30) {
                    let _ = child.kill();
                    return Err("The Langtrain engine did not start within 30 seconds.".into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            *proc_guard = Some(child);
        }
    }

    *state.model_path.lock().map_err(|e| e.to_string())? = Some(model_path.clone());
    *state.model_name.lock().map_err(|e| e.to_string())? = Some(name.clone());
    *state.port.lock().map_err(|e| e.to_string())? = port;

    Ok(ServerStatus {
        running: true,
        model: Some(model_path),
        model_name: Some(name),
        port,
        endpoint: Some(endpoint(port)),
    })
}

/// Stop the engine if this app started it, and clear the selected model.
#[tauri::command]
pub fn stop_inference_server(state: State<'_, InferenceState>) -> Result<ServerStatus, String> {
    state.shutdown();
    *state.model_path.lock().map_err(|e| e.to_string())? = None;
    *state.model_name.lock().map_err(|e| e.to_string())? = None;
    let port = *state.port.lock().map_err(|e| e.to_string())?;
    Ok(ServerStatus { running: false, model: None, model_name: None, port, endpoint: None })
}

/// Whether a model is selected and the engine is answering.
#[tauri::command]
pub fn inference_server_status(state: State<'_, InferenceState>) -> ServerStatus {
    let port = *state.port.lock().unwrap_or_else(|e| e.into_inner());
    let model = state.model_path.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let model_name = state.model_name.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let running = model_name.is_some() && is_listening(port);
    ServerStatus {
        running,
        model: if running { model } else { None },
        model_name: if running { model_name } else { None },
        port,
        endpoint: running.then(|| endpoint(port)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_gguf() {
        let dir = std::env::temp_dir().join("lt-inference-test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("weights.bin");
        std::fs::write(&f, b"x").unwrap();
        assert!(ensure_in_models_dir(&f).unwrap_err().contains("not a .gguf"));
    }

    #[test]
    fn endpoint_format() {
        assert_eq!(endpoint(11435), "http://127.0.0.1:11435/v1/chat/completions");
    }
}
