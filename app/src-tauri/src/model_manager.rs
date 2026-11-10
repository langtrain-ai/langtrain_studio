use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Metadata for a locally stored GGUF model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    pub filename: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_gb: f64,
    pub quantization: String,
    pub parameters: String,
}

/// Progress event emitted during downloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_name: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f64,
    pub speed_mbps: f64,
}

/// Get the default models directory (~/.langtrain/models).
pub fn models_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".langtrain").join("models")
}

/// Infer quantization level from filename.
fn infer_quantization(filename: &str) -> String {
    let lower = filename.to_lowercase();
    let quantizations = [
        "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
        "q4_0", "q4_1", "q4_k_s", "q4_k_m",
        "q5_0", "q5_1", "q5_k_s", "q5_k_m",
        "q6_k", "q8_0", "f16", "f32",
    ];
    for q in quantizations {
        if lower.contains(q) {
            return q.to_uppercase();
        }
    }
    "Unknown".to_string()
}

/// Infer parameter count from filename.
fn infer_parameters(filename: &str) -> String {
    let lower = filename.to_lowercase();
    let sizes = [
        ("70b", "70B"), ("34b", "34B"), ("13b", "13B"),
        ("8b", "8B"), ("7b", "7B"), ("3b", "3B"),
        ("1.5b", "1.5B"), ("1b", "1B"), ("0.5b", "0.5B"),
    ];
    for (pattern, label) in sizes {
        if lower.contains(pattern) {
            return label.to_string();
        }
    }
    "Unknown".to_string()
}

/// List all locally downloaded .gguf models.
#[tauri::command]
pub fn list_local_models() -> Vec<LocalModel> {
    let dir = models_dir();
    if !dir.exists() {
        let _ = fs::create_dir_all(&dir);
        return vec![];
    }

    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                if let Ok(metadata) = fs::metadata(&path) {
                    let filename = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let name = filename
                        .trim_end_matches(".gguf")
                        .replace('-', " ")
                        .replace('_', " ");

                    models.push(LocalModel {
                        name,
                        filename: filename.clone(),
                        path: path.to_string_lossy().to_string(),
                        size_bytes: metadata.len(),
                        size_gb: (metadata.len() as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
                        quantization: infer_quantization(&filename),
                        parameters: infer_parameters(&filename),
                    });
                }
            }
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Delete a local model by filename.
#[tauri::command]
pub fn delete_local_model(filename: String) -> Result<(), String> {
    let path = models_dir().join(&filename);
    if !path.exists() {
        return Err(format!("Model not found: {}", filename));
    }
    // Ensure the path is within the models directory (prevent path traversal)
    let canonical = path.canonicalize().map_err(|e| e.to_string())?;
    let models_canonical = models_dir().canonicalize().map_err(|e| e.to_string())?;
    if !canonical.starts_with(&models_canonical) {
        return Err("Invalid path".to_string());
    }
    fs::remove_file(canonical).map_err(|e| format!("Failed to delete: {}", e))
}

/// Download a GGUF model from a URL. Emits progress events via Tauri.
#[tauri::command]
pub async fn download_model(
    app: tauri::AppHandle,
    url: String,
    filename: String,
) -> Result<LocalModel, String> {
    use tauri::Emitter;

    let dir = models_dir();
    let _ = fs::create_dir_all(&dir);
    let dest = dir.join(&filename);

    // Check if already exists
    if dest.exists() {
        return Err(format!("Model already exists: {}", filename));
    }

    // Start download
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    let total_bytes = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let start = std::time::Instant::now();

    // Stream to file
    let mut file = fs::File::create(&dest).map_err(|e| format!("Cannot create file: {}", e))?;

    use std::io::Write;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("Write error: {}", e))?;

        downloaded += chunk.len() as u64;

        let elapsed = start.elapsed().as_secs_f64();
        let speed_mbps = if elapsed > 0.0 {
            (downloaded as f64 / 1_048_576.0) / elapsed
        } else {
            0.0
        };

        let progress = DownloadProgress {
            model_name: filename.clone(),
            downloaded_bytes: downloaded,
            total_bytes,
            percent: if total_bytes > 0 {
                (downloaded as f64 / total_bytes as f64 * 100.0).round()
            } else {
                0.0
            },
            speed_mbps: (speed_mbps * 10.0).round() / 10.0,
        };

        let _ = app.emit("download-progress", &progress);
    }

    // Build model info
    let model = LocalModel {
        name: filename
            .trim_end_matches(".gguf")
            .replace('-', " ")
            .replace('_', " "),
        filename: filename.clone(),
        path: dest.to_string_lossy().to_string(),
        size_bytes: downloaded,
        size_gb: (downloaded as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
        quantization: infer_quantization(&filename),
        parameters: infer_parameters(&filename),
    };

    let _ = app.emit("download-complete", &model);
    Ok(model)
}

/// Get the models directory path.
#[tauri::command]
pub fn get_models_dir() -> String {
    models_dir().to_string_lossy().to_string()
}
