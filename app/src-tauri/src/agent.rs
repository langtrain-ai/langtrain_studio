use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    pub path: String,
    pub content: String,
    pub size: u64,
    pub encoding: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteResult {
    pub path: String,
    pub success: bool,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub is_symlink: bool,
    pub size: u64,
    pub extension: Option<String>,
    pub modified: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub success: bool,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub is_symlink: bool,
    pub size: u64,
    pub extension: Option<String>,
    pub modified: Option<String>,
    pub created: Option<String>,
    pub readonly: bool,
}

fn system_time_to_string(time: std::time::SystemTime) -> String {
    let duration = time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs().to_string()
}

#[tauri::command]
pub fn read_file(path: String) -> Result<FileContent, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("File not found: {}", path));
    }
    if !p.is_file() {
        return Err(format!("Path is not a file: {}", path));
    }

    let bytes = fs::read(&path).map_err(|e| format!("Failed to read file: {}", e))?;
    let size = bytes.len() as u64;

    match String::from_utf8(bytes.clone()) {
        Ok(content) => Ok(FileContent {
            path,
            content,
            size,
            encoding: "utf-8".to_string(),
        }),
        Err(_) => {
            // Return base64-like hex for binary files
            let hex = bytes
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();
            Ok(FileContent {
                path,
                content: hex,
                size,
                encoding: "binary-hex".to_string(),
            })
        }
    }
}

#[tauri::command]
pub fn write_file(path: String, content: String) -> Result<WriteResult, String> {
    let p = Path::new(&path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create parent directories: {}", e))?;
    }

    let bytes = content.as_bytes();
    fs::write(&path, bytes).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(WriteResult {
        path,
        success: true,
        bytes_written: bytes.len() as u64,
    })
}

#[tauri::command]
pub fn create_file(path: String, content: Option<String>) -> Result<WriteResult, String> {
    let p = Path::new(&path);
    if p.exists() {
        return Err(format!("File already exists: {}", path));
    }
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create parent directories: {}", e))?;
    }

    let body = content.unwrap_or_default();
    let bytes = body.as_bytes();
    fs::write(&path, bytes).map_err(|e| format!("Failed to create file: {}", e))?;

    Ok(WriteResult {
        path,
        success: true,
        bytes_written: bytes.len() as u64,
    })
}

#[tauri::command]
pub fn list_directory(path: String, show_hidden: Option<bool>) -> Result<Vec<DirEntry>, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("Directory not found: {}", path));
    }
    if !p.is_dir() {
        return Err(format!("Path is not a directory: {}", path));
    }

    let show_hidden = show_hidden.unwrap_or(false);
    let mut entries: Vec<DirEntry> = Vec::new();

    let read_dir = fs::read_dir(&path).map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry_result in read_dir {
        let entry = entry_result.map_err(|e| format!("Failed to read entry: {}", e))?;
        let name = entry.file_name().to_string_lossy().to_string();

        if !show_hidden && name.starts_with('.') {
            continue;
        }

        let entry_path = entry.path();
        let meta = entry.metadata().ok();
        let is_symlink = entry_path.symlink_metadata().map(|m| m.file_type().is_symlink()).unwrap_or(false);
        let is_dir = entry_path.is_dir();
        let is_file = entry_path.is_file();
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        let extension = entry_path
            .extension()
            .map(|e| e.to_string_lossy().to_string());
        let modified = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .map(system_time_to_string);

        entries.push(DirEntry {
            name,
            path: entry_path.to_string_lossy().to_string(),
            is_dir,
            is_file,
            is_symlink,
            size,
            extension,
            modified,
        });
    }

    entries.sort_by(|a, b| {
        b.is_dir
            .cmp(&a.is_dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    Ok(entries)
}

#[tauri::command]
pub fn delete_path(path: String) -> Result<bool, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("Path not found: {}", path));
    }

    if p.is_dir() {
        fs::remove_dir_all(&path).map_err(|e| format!("Failed to delete directory: {}", e))?;
    } else {
        fs::remove_file(&path).map_err(|e| format!("Failed to delete file: {}", e))?;
    }

    Ok(true)
}

#[tauri::command]
pub fn execute_shell(
    command: String,
    args: Vec<String>,
    working_dir: Option<String>,
    timeout_secs: Option<u64>,
) -> Result<ShellResult, String> {
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(30));

    let mut cmd = Command::new(&command);
    cmd.args(&args);

    if let Some(ref wd) = working_dir {
        let wd_path = Path::new(wd);
        if !wd_path.exists() {
            return Err(format!("Working directory not found: {}", wd));
        }
        cmd.current_dir(wd_path);
    }

    // Spawn and wait with timeout approximation via a thread
    let child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn command '{}': {}", command, e))?;

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = child.wait_with_output();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let exit_code = output.status.code().unwrap_or(-1);
            Ok(ShellResult {
                stdout,
                stderr,
                exit_code,
                success: output.status.success(),
                timed_out: false,
            })
        }
        Ok(Err(e)) => Err(format!("Command execution error: {}", e)),
        Err(_) => Ok(ShellResult {
            stdout: String::new(),
            stderr: format!("Command timed out after {} seconds", timeout.as_secs()),
            exit_code: -1,
            success: false,
            timed_out: true,
        }),
    }
}

#[tauri::command]
pub fn search_in_files(
    directory: String,
    pattern: String,
    file_extension: Option<String>,
    max_results: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    let dir_path = Path::new(&directory);
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(format!("Directory not found: {}", directory));
    }

    let max = max_results.unwrap_or(100);
    let mut results: Vec<SearchResult> = Vec::new();

    search_recursive(dir_path, &pattern, &file_extension, max, &mut results)?;

    Ok(results)
}

fn search_recursive(
    dir: &Path,
    pattern: &str,
    file_extension: &Option<String>,
    max: usize,
    results: &mut Vec<SearchResult>,
) -> Result<(), String> {
    if results.len() >= max {
        return Ok(());
    }

    let read_dir = fs::read_dir(dir).map_err(|e| format!("Cannot read dir: {}", e))?;

    for entry_result in read_dir {
        if results.len() >= max {
            break;
        }
        let entry = match entry_result {
            Ok(e) => e,
            Err(_) => continue,
        };
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files/dirs
        if name.starts_with('.') {
            continue;
        }

        if entry_path.is_dir() {
            search_recursive(&entry_path, pattern, file_extension, max, results)?;
        } else if entry_path.is_file() {
            // Filter by extension if specified
            if let Some(ref ext_filter) = file_extension {
                let file_ext = entry_path
                    .extension()
                    .map(|e| e.to_string_lossy().to_string())
                    .unwrap_or_default();
                let ext_filter_clean = ext_filter.trim_start_matches('.');
                if file_ext != ext_filter_clean {
                    continue;
                }
            }

            // Read and search file
            if let Ok(content) = fs::read_to_string(&entry_path) {
                for (line_idx, line) in content.lines().enumerate() {
                    if results.len() >= max {
                        break;
                    }
                    if let Some(match_start) = line.find(pattern) {
                        results.push(SearchResult {
                            file_path: entry_path.to_string_lossy().to_string(),
                            line_number: line_idx + 1,
                            line_content: line.to_string(),
                            match_start,
                            match_end: match_start + pattern.len(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

#[tauri::command]
pub fn get_file_info(path: String) -> Result<FileInfo, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("Path not found: {}", path));
    }

    let meta = fs::metadata(&path).map_err(|e| format!("Failed to get metadata: {}", e))?;
    let symlink_meta = p.symlink_metadata().ok();

    let is_symlink = symlink_meta
        .as_ref()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let extension = p.extension().map(|e| e.to_string_lossy().to_string());
    let modified = meta.modified().ok().map(system_time_to_string);
    let created = meta.created().ok().map(system_time_to_string);

    Ok(FileInfo {
        path,
        name,
        is_dir: meta.is_dir(),
        is_file: meta.is_file(),
        is_symlink,
        size: meta.len(),
        extension,
        modified,
        created,
        readonly: meta.permissions().readonly(),
    })
}

#[tauri::command]
pub fn move_or_rename(from: String, to: String) -> Result<bool, String> {
    let from_path = Path::new(&from);
    if !from_path.exists() {
        return Err(format!("Source path not found: {}", from));
    }

    let to_path = Path::new(&to);
    if let Some(parent) = to_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create destination directories: {}", e))?;
    }

    fs::rename(&from, &to).map_err(|e| format!("Failed to move/rename: {}", e))?;

    Ok(true)
}

#[tauri::command]
pub fn get_working_directory() -> Result<String, String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| format!("Failed to get working directory: {}", e))
}
