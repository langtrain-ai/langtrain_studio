use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use sysinfo::System;

/// Hardware profile returned to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemProfile {
    // CPU
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_arch: String,

    // Memory
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,

    // GPU
    pub gpu_name: String,
    pub gpu_vram_gb: f64,
    pub gpu_platform: GpuPlatform,
    pub has_gpu: bool,

    // Disk
    pub models_dir: String,
    pub disk_free_gb: f64,

    // Suitability
    pub can_run_inference: bool,
    pub can_finetune: bool,
    pub inference_note: String,
    pub finetune_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuPlatform {
    #[serde(rename = "nvidia")]
    Nvidia,
    #[serde(rename = "apple_silicon")]
    AppleSilicon,
    #[serde(rename = "amd")]
    Amd,
    #[serde(rename = "cpu")]
    CpuOnly,
}

/// Get the default models directory (~/.langtrain/models)
fn models_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".langtrain").join("models")
}

/// Detect GPU on macOS (Apple Silicon via sysctl)
#[cfg(target_os = "macos")]
fn detect_gpu() -> (String, f64, GpuPlatform, bool) {
    // Check for Apple Silicon
    let arch = std::env::consts::ARCH;
    if arch == "aarch64" {
        // Apple Silicon — unified memory, GPU shares system RAM
        let mut sys = System::new_all();
        sys.refresh_memory();
        let total_ram_gb = sys.total_memory() as f64 / 1_073_741_824.0;
        // Apple Silicon GPU can use ~75% of unified memory
        let gpu_vram = total_ram_gb * 0.75;
        let chip_name = get_apple_chip_name();
        return (chip_name, gpu_vram, GpuPlatform::AppleSilicon, true);
    }

    // Intel Mac — check for discrete GPU
    ("Integrated Graphics".to_string(), 0.0, GpuPlatform::CpuOnly, false)
}

#[cfg(target_os = "macos")]
fn get_apple_chip_name() -> String {
    use std::process::Command;
    let output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();
    match output {
        Ok(out) => {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if name.is_empty() { "Apple Silicon".to_string() } else { name }
        }
        Err(_) => "Apple Silicon".to_string(),
    }
}

/// Detect GPU on Linux (nvidia-smi)
#[cfg(target_os = "linux")]
fn detect_gpu() -> (String, f64, GpuPlatform, bool) {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = text.trim().split(',').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let vram_mb: f64 = parts[1].trim().parse().unwrap_or(0.0);
                return (name, vram_mb / 1024.0, GpuPlatform::Nvidia, true);
            }
            ("Unknown NVIDIA GPU".to_string(), 0.0, GpuPlatform::Nvidia, true)
        }
        _ => ("No GPU detected".to_string(), 0.0, GpuPlatform::CpuOnly, false),
    }
}

/// Detect GPU on Windows (nvidia-smi or DXGI fallback)
#[cfg(target_os = "windows")]
fn detect_gpu() -> (String, f64, GpuPlatform, bool) {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = text.trim().split(',').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let vram_mb: f64 = parts[1].trim().parse().unwrap_or(0.0);
                return (name, vram_mb / 1024.0, GpuPlatform::Nvidia, true);
            }
            ("Unknown NVIDIA GPU".to_string(), 0.0, GpuPlatform::Nvidia, true)
        }
        _ => ("No dedicated GPU detected".to_string(), 0.0, GpuPlatform::CpuOnly, false),
    }
}

/// Get free disk space for the models directory.
fn disk_free_gb() -> f64 {
    let dir = models_dir();
    // Ensure directory exists
    let _ = std::fs::create_dir_all(&dir);

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let path_c = std::ffi::CString::new(dir.to_string_lossy().as_bytes().to_vec())
            .unwrap_or_default();
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(path_c.as_ptr(), &mut stat) == 0 {
                return (stat.f_bavail as f64 * stat.f_frsize as f64) / 1_073_741_824.0;
            }
        }
        0.0
    }

    #[cfg(windows)]
    {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = OsStr::new(dir.to_string_lossy().as_ref())
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let mut free_bytes: u64 = 0;
        unsafe {
            winapi::um::fileapi::GetDiskFreeSpaceExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut free_bytes as *mut u64 as *mut _,
            );
        }
        free_bytes as f64 / 1_073_741_824.0
    }
}

/// Main detection function exposed as a Tauri command.
#[tauri::command]
pub fn detect_hardware() -> SystemProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_cores = sys.cpus().len();
    let cpu_arch = std::env::consts::ARCH.to_string();

    let ram_total_gb = sys.total_memory() as f64 / 1_073_741_824.0;
    let ram_available_gb = sys.available_memory() as f64 / 1_073_741_824.0;

    let (gpu_name, gpu_vram_gb, gpu_platform, has_gpu) = detect_gpu();

    let dir = models_dir();
    let disk_free = disk_free_gb();

    // Suitability assessment
    let can_run_inference = ram_total_gb >= 4.0 && disk_free >= 2.0;
    let can_finetune = has_gpu && gpu_vram_gb >= 6.0 && ram_total_gb >= 16.0;

    let inference_note = if can_run_inference {
        if ram_total_gb >= 16.0 {
            "Excellent — can run 7B+ models comfortably".to_string()
        } else if ram_total_gb >= 8.0 {
            "Good — can run 3B-7B quantized models".to_string()
        } else {
            "Limited — small models only (1B-3B)".to_string()
        }
    } else {
        "Insufficient resources for local inference".to_string()
    };

    let finetune_note = if can_finetune {
        if gpu_vram_gb >= 24.0 {
            format!("Excellent — {:.0}GB VRAM supports 13B+ QLoRA fine-tuning", gpu_vram_gb)
        } else if gpu_vram_gb >= 12.0 {
            format!("Good — {:.0}GB VRAM supports 7B QLoRA fine-tuning", gpu_vram_gb)
        } else {
            format!("Limited — {:.0}GB VRAM supports small model fine-tuning only", gpu_vram_gb)
        }
    } else if has_gpu {
        format!("GPU detected ({}) but insufficient VRAM ({:.0}GB). Need 6GB+ for fine-tuning.", gpu_name, gpu_vram_gb)
    } else {
        "No GPU detected. Fine-tuning requires a CUDA/Metal GPU with 6GB+ VRAM.".to_string()
    };

    SystemProfile {
        cpu_name,
        cpu_cores,
        cpu_arch,
        ram_total_gb: (ram_total_gb * 10.0).round() / 10.0,
        ram_available_gb: (ram_available_gb * 10.0).round() / 10.0,
        gpu_name,
        gpu_vram_gb: (gpu_vram_gb * 10.0).round() / 10.0,
        gpu_platform,
        has_gpu,
        models_dir: dir.to_string_lossy().to_string(),
        disk_free_gb: (disk_free * 10.0).round() / 10.0,
        can_run_inference,
        can_finetune,
        inference_note,
        finetune_note,
    }
}
