//! Local fine-tuning jobs, shaped like OpenAI's `/v1/fine_tuning/jobs` API.
//!
//! Each job runs the `langtune` trainer as a low-priority child process that
//! reports JSON-lines events on stdout. When it finishes, the GGUF LoRA
//! adapter is installed next to the base model and is servable immediately
//! as `base@suffix`, with no merge step.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::state::{parse_model_name, unix_now, ApiError, AppState, MOCK_MODEL};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub n_epochs: Option<u32>,
    pub learning_rate: Option<f64>,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<f64>,
    /// Train LoRA on the top N transformer layers only (enables the activation cache).
    pub train_layers: Option<u32>,
    pub max_seq_len: Option<u32>,
    /// `auto`, `cpu`, `mps` or `cuda`.
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub training_file: String,
    pub suffix: String,
    pub hyperparameters: Hyperparameters,
    pub status: JobStatus,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub fine_tuned_model: Option<String>,
    pub error: Option<String>,
    /// Latest progress report from the trainer (step, loss, tokens/s, ETA).
    pub progress: Option<Value>,
    #[serde(skip)]
    pub events: Vec<Value>,
}

#[derive(Default)]
pub struct Jobs {
    dir: PathBuf,
    jobs: Mutex<HashMap<String, Job>>,
    children: Mutex<HashMap<String, Child>>,
}

impl Jobs {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, ..Default::default() }
    }

    async fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        let mut jobs = self.jobs.lock().await;
        if let Some(j) = jobs.get_mut(id) {
            f(j);
            let _ = std::fs::write(self.dir.join(id).join("job.json"), serde_json::to_vec_pretty(j).unwrap_or_default());
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateJob {
    pub model: String,
    pub training_file: String,
    pub validation_file: Option<String>,
    pub suffix: Option<String>,
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect()
}

pub async fn create(State(state): State<Arc<AppState>>, Json(req): Json<CreateJob>) -> Result<Json<Job>, ApiError> {
    let (base, _) = parse_model_name(&req.model);
    if base == MOCK_MODEL {
        return Err(ApiError::bad_request("the mock model cannot be fine-tuned"));
    }
    let base_path = state.config.models_dir().join(format!("{base}.gguf"));
    if !base_path.exists() {
        return Err(ApiError(axum::http::StatusCode::NOT_FOUND, format!("model '{base}' not found")));
    }
    let train = PathBuf::from(&req.training_file);
    if !train.is_file() {
        return Err(ApiError::bad_request(format!("training_file '{}' does not exist", req.training_file)));
    }

    let id = format!("ftjob-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
    let suffix = sanitize(req.suffix.as_deref().unwrap_or(&id));
    let job_dir = state.jobs.dir.join(&id);
    std::fs::create_dir_all(&job_dir).map_err(anyhow::Error::from)?;

    let hp = &req.hyperparameters;
    let config = json!({
        "base_gguf": base_path,
        "train_file": train,
        "validation_file": req.validation_file,
        "out_dir": job_dir,
        "adapter_name": suffix,
        "epochs": hp.n_epochs.unwrap_or(3),
        "learning_rate": hp.learning_rate.unwrap_or(2e-4),
        "lora_rank": hp.lora_rank.unwrap_or(16),
        "lora_alpha": hp.lora_alpha.unwrap_or(32.0),
        "train_layers": hp.train_layers,
        "max_seq_len": hp.max_seq_len.unwrap_or(1024),
        "device": hp.device.clone().unwrap_or_else(|| state.config.profile.fine_tune.device.to_string()),
        "threads": state.config.hardware.physical_cores,
    });
    let config_path = job_dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap_or_default()).map_err(anyhow::Error::from)?;

    let job = Job {
        id: id.clone(),
        object: "fine_tuning.job",
        model: base.clone(),
        training_file: req.training_file,
        suffix,
        hyperparameters: req.hyperparameters,
        status: JobStatus::Queued,
        created_at: unix_now(),
        finished_at: None,
        fine_tuned_model: None,
        error: None,
        progress: None,
        events: Vec::new(),
    };
    state.jobs.jobs.lock().await.insert(id.clone(), job.clone());
    tokio::spawn(run_job(state.clone(), id, config_path));
    Ok(Json(job))
}

async fn run_job(state: Arc<AppState>, id: String, config_path: PathBuf) {
    let jobs = &state.jobs;
    let mut cmd = Command::new(&state.config.python);
    cmd.args(["-m", "langtune", "train", "--config"]).arg(&config_path);
    if let Some(dir) = &state.config.langtune_dir {
        let sep = if cfg!(windows) { ";" } else { ":" };
        let existing = std::env::var("PYTHONPATH").unwrap_or_default();
        cmd.env("PYTHONPATH", format!("{}{sep}{existing}", dir.display()));
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("could not start trainer ({}): {e}", state.config.python);
            jobs.update(&id, |j| {
                j.status = JobStatus::Failed;
                j.error = Some(msg);
                j.finished_at = Some(unix_now());
            })
            .await;
            return;
        }
    };
    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    jobs.children.lock().await.insert(id.clone(), child);
    jobs.update(&id, |j| j.status = JobStatus::Running).await;

    // Keep the stderr tail for error reports.
    let tail = Arc::new(Mutex::new(Vec::<String>::new()));
    let t = tail.clone();
    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(l)) = lines.next_line().await {
            let mut t = t.lock().await;
            t.push(l);
            if t.len() > 40 {
                t.remove(0);
            }
        }
    });

    let mut adapter: Option<PathBuf> = None;
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(ev) = serde_json::from_str::<Value>(&line) else { continue };
        if ev.get("type").and_then(Value::as_str) == Some("done") {
            adapter = ev.get("adapter_path").and_then(Value::as_str).map(PathBuf::from);
        }
        jobs.update(&id, |j| {
            if matches!(ev.get("type").and_then(Value::as_str), Some("progress" | "estimate")) {
                j.progress = Some(ev.clone());
            }
            j.events.push(ev);
        })
        .await;
    }
    let _ = stderr_task.await;

    let child = jobs.children.lock().await.remove(&id);
    let status = match child {
        Some(mut c) => c.wait().await.ok(),
        None => None, // cancelled: the child was taken and killed
    };
    let cancelled = jobs.jobs.lock().await.get(&id).is_some_and(|j| j.status == JobStatus::Cancelled);
    if cancelled {
        return;
    }

    let job = jobs.jobs.lock().await.get(&id).cloned();
    let Some(job) = job else { return };
    let result = match (status, adapter) {
        (Some(s), Some(path)) if s.success() => install_adapter(&state, &job, &path).await,
        (s, _) => {
            let tail = tail.lock().await.join("\n");
            Err(format!("trainer exited with {}: {tail}", s.map(|s| s.to_string()).unwrap_or_else(|| "unknown status".into())))
        }
    };
    jobs.update(&id, |j| {
        j.finished_at = Some(unix_now());
        match result {
            Ok(name) => {
                j.status = JobStatus::Succeeded;
                j.fine_tuned_model = Some(name);
            }
            Err(e) => {
                j.status = JobStatus::Failed;
                j.error = Some(e);
            }
        }
    })
    .await;
}

async fn install_adapter(state: &AppState, job: &Job, path: &std::path::Path) -> Result<String, String> {
    let dir = state.config.adapters_dir(&job.model);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::copy(path, dir.join(format!("{}.gguf", job.suffix))).map_err(|e| e.to_string())?;
    // Reload on next request so the new adapter is available.
    state.invalidate(&job.model).await;
    Ok(format!("{}@{}", job.model, job.suffix))
}

pub async fn list(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut jobs: Vec<Job> = state.jobs.jobs.lock().await.values().cloned().collect();
    jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(json!({ "object": "list", "data": jobs, "has_more": false }))
}

pub async fn get(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Job>, ApiError> {
    state.jobs.jobs.lock().await.get(&id).cloned().map(Json).ok_or_else(|| not_found(&id))
}

pub async fn events(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let jobs = state.jobs.jobs.lock().await;
    let job = jobs.get(&id).ok_or_else(|| not_found(&id))?;
    Ok(Json(json!({ "object": "list", "data": job.events })))
}

pub async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Job>, ApiError> {
    {
        let mut jobs = state.jobs.jobs.lock().await;
        let job = jobs.get_mut(&id).ok_or_else(|| not_found(&id))?;
        if matches!(job.status, JobStatus::Queued | JobStatus::Running) {
            job.status = JobStatus::Cancelled;
            job.finished_at = Some(unix_now());
        }
    }
    if let Some(mut child) = state.jobs.children.lock().await.remove(&id) {
        let _ = child.kill().await;
    }
    get(State(state), Path(id)).await
}

fn not_found(id: &str) -> ApiError {
    ApiError(axum::http::StatusCode::NOT_FOUND, format!("fine-tuning job '{id}' not found"))
}
