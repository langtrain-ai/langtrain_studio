use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use tauri::Emitter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudModel {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub base_model: String,
    pub created_at: String,
    pub status: String,
    pub size_gb: f64,
    pub download_url: Option<String>,
    pub quantization: Option<String>,
    pub task_type: Option<String>,
}


fn get_langtrain_models_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or_else(|| "Cannot determine home directory".to_string())?;
    let models_dir = home.join(".langtrain").join("models");
    fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Failed to create models directory: {}", e))?;
    Ok(models_dir)
}

fn parse_cloud_model(v: &serde_json::Value) -> CloudModel {
    CloudModel {
        id: v["id"].as_str().unwrap_or("").to_string(),
        name: v["name"].as_str().unwrap_or("").to_string(),
        display_name: v["display_name"]
            .as_str()
            .or_else(|| v["displayName"].as_str())
            .unwrap_or_else(|| v["name"].as_str().unwrap_or(""))
            .to_string(),
        base_model: v["base_model"]
            .as_str()
            .or_else(|| v["baseModel"].as_str())
            .unwrap_or("")
            .to_string(),
        created_at: v["created_at"]
            .as_str()
            .or_else(|| v["createdAt"].as_str())
            .unwrap_or("")
            .to_string(),
        status: v["status"].as_str().unwrap_or("unknown").to_string(),
        size_gb: v["size_gb"]
            .as_f64()
            .or_else(|| v["sizeGb"].as_f64())
            .unwrap_or(0.0),
        download_url: v["download_url"]
            .as_str()
            .or_else(|| v["downloadUrl"].as_str())
            .map(|s| s.to_string()),
        quantization: v["quantization"]
            .as_str()
            .map(|s| s.to_string()),
        task_type: v["task_type"]
            .as_str()
            .or_else(|| v["taskType"].as_str())
            .map(|s| s.to_string()),
    }
}

#[tauri::command]
pub fn list_cloud_models(api_key: String) -> Result<Vec<CloudModel>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    let response = client
        .get("https://api.langtrain.xyz/api/v1/models")
        .query(&[("format", "gguf"), ("limit", "50")])
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .send()
        .map_err(|e| format!("Failed to reach Langtrain API: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(format!(
            "API request failed ({}): {}",
            status.as_u16(),
            body
        ));
    }

    let body: serde_json::Value = response
        .json()
        .map_err(|e| format!("Failed to parse API response: {}", e))?;

    // Try common envelope shapes: { data: [...] } or { models: [...] } or top-level array
    let items = if let Some(arr) = body.as_array() {
        arr.clone()
    } else if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        data.clone()
    } else if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
        models.clone()
    } else {
        return Err("Unexpected API response shape — could not locate model list".to_string());
    };

    let models: Vec<CloudModel> = items.iter().map(parse_cloud_model).collect();
    Ok(models)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub filename: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub percent: f64,
    pub done: bool,
}

#[tauri::command]
pub fn download_cloud_model(
    api_key: String,
    model_id: String,
    download_url: String,
    filename: String,
    window: tauri::Window,
) -> Result<String, String> {
    let models_dir = get_langtrain_models_dir()?;
    let dest_path = models_dir.join(&filename);

    if dest_path.exists() {
        return Ok(dest_path.to_string_lossy().to_string());
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1-hour cap for large models
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    let response = client
        .get(&download_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("Download request failed ({})", status.as_u16()));
    }

    let total_bytes = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let tmp_path = dest_path.with_extension("download");
    let mut file = fs::File::create(&tmp_path)
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    let mut bytes_downloaded: u64 = 0;
    let chunk_size: usize = 1024 * 256; // 256 KB chunks
    let mut buf = vec![0u8; chunk_size];
    let mut reader = response;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Error reading download stream: {}", e))?;

        if n == 0 {
            break;
        }

        file.write_all(&buf[..n])
            .map_err(|e| format!("Failed to write chunk: {}", e))?;

        bytes_downloaded += n as u64;

        let percent = if total_bytes > 0 {
            (bytes_downloaded as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };

        let progress = DownloadProgress {
            model_id: model_id.clone(),
            filename: filename.clone(),
            bytes_downloaded,
            total_bytes,
            percent,
            done: false,
        };

        // Emit progress event — frontend listens with listen("model_download_progress", ...)
        let _ = window.emit("model_download_progress", &progress);
    }

    // Flush and close before rename
    drop(file);

    fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to finalize downloaded file: {}", e))?;

    // Emit final done event
    let done_progress = DownloadProgress {
        model_id: model_id.clone(),
        filename: filename.clone(),
        bytes_downloaded,
        total_bytes,
        percent: 100.0,
        done: true,
    };
    let _ = window.emit("model_download_progress", &done_progress);

    Ok(dest_path.to_string_lossy().to_string())
}

#[tauri::command]
pub fn check_cloud_connection(api_key: String) -> Result<serde_json::Value, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    let response = client
        .get("https://api.langtrain.xyz/api/v1/me")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .send()
        .map_err(|e| format!("Failed to reach Langtrain API: {}", e))?;

    let status = response.status();
    let status_code = status.as_u16();

    let body: serde_json::Value = response
        .json()
        .unwrap_or_else(|_| serde_json::json!({}));

    if !status.is_success() {
        return Err(format!(
            "Connection check failed ({}): {}",
            status_code,
            body.get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error")
        ));
    }

    Ok(serde_json::json!({
        "connected": true,
        "status_code": status_code,
        "user": body,
    }))
}
