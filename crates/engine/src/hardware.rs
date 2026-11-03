//! Hardware detection and the auto-tuned runtime profile.

use serde::Serialize;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Serialize)]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub cpu_features: Vec<&'static str>,
    /// GPU backends compiled into this build.
    pub gpu_backends: Vec<&'static str>,
    pub apple_silicon: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Profile {
    pub threads: usize,
    pub threads_batch: usize,
    pub parallel_slots: u32,
    pub context_per_slot: u32,
    pub kv_cache_type: &'static str,
    pub gpu_layers: u32,
    /// Largest model (GGUF file size) that comfortably fits in memory.
    pub max_model_gb: f64,
    pub fine_tune: FineTuneAdvice,
}

#[derive(Debug, Clone, Serialize)]
pub struct FineTuneAdvice {
    pub device: &'static str,
    pub recommended_max_params_b: f64,
    pub note: String,
}

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

pub fn detect() -> Hardware {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::nothing().with_ram())
            .with_cpu(CpuRefreshKind::nothing()),
    )
    .ok();
    let arch = std::env::consts::ARCH.to_string();
    let os = std::env::consts::OS.to_string();
    let apple_silicon = os == "macos" && arch == "aarch64";
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (total, available) = sys.as_ref().map(|s| (s.total_memory(), s.available_memory())).unwrap_or((8 << 30, 4 << 30));
    Hardware {
        cpu_brand: sys
            .as_ref()
            .and_then(|s| s.cpus().first().map(|c| c.brand().trim().to_string()))
            .unwrap_or_default(),
        physical_cores: System::physical_core_count().unwrap_or(logical).max(1),
        logical_cores: logical,
        total_memory_gb: total as f64 / GB,
        available_memory_gb: available as f64 / GB,
        cpu_features: cpu_features(),
        gpu_backends: gpu_backends(),
        apple_silicon,
        os,
        arch,
    }
}

#[allow(clippy::vec_init_then_push)]
fn cpu_features() -> Vec<&'static str> {
    let mut f = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! check {
            ($($name:tt),*) => { $( if std::arch::is_x86_feature_detected!($name) { f.push($name); } )* };
        }
        check!("sse4.2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vnni", "avx512bf16", "avxvnni");
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! check {
            ($($name:tt),*) => { $( if std::arch::is_aarch64_feature_detected!($name) { f.push($name); } )* };
        }
        check!("neon", "dotprod", "fp16", "i8mm", "bf16", "sve");
    }
    f
}

#[allow(clippy::vec_init_then_push, unused_mut)]
fn gpu_backends() -> Vec<&'static str> {
    let mut b = Vec::new();
    #[cfg(feature = "metal")]
    b.push("metal");
    #[cfg(feature = "cuda")]
    b.push("cuda");
    #[cfg(feature = "vulkan")]
    b.push("vulkan");
    b
}

impl Hardware {
    pub fn has_gpu(&self) -> bool {
        !self.gpu_backends.is_empty()
    }

    /// Pick runtime settings for this machine.
    pub fn profile(&self) -> Profile {
        // Matmul-bound decode scales with physical cores; SMT siblings only add contention.
        let threads = self.physical_cores;
        let mem = self.total_memory_gb;
        // Leave room for the OS, the KV cache and the user's other apps.
        let budget = if self.apple_silicon { mem * 0.70 } else { mem * 0.60 };
        let max_model_gb = (budget - 1.5).max(0.5);

        let parallel_slots = match mem as u64 {
            0..=8 => 2,
            9..=16 => 4,
            17..=32 => 8,
            _ => 16,
        };

        let fine_tune = if self.apple_silicon {
            FineTuneAdvice {
                device: "mps",
                recommended_max_params_b: (mem / 4.0).clamp(1.0, 14.0),
                note: "Apple GPU via unified memory: much faster than the CPU path.".into(),
            }
        } else if self.gpu_backends.contains(&"cuda") {
            FineTuneAdvice { device: "cuda", recommended_max_params_b: 8.0, note: "NVIDIA GPU detected.".into() }
        } else {
            let fast_int8 = self.cpu_features.iter().any(|f| matches!(*f, "avx512vnni" | "avxvnni" | "i8mm" | "dotprod"));
            FineTuneAdvice {
                device: "cpu",
                recommended_max_params_b: if mem >= 32.0 { 4.0 } else if mem >= 16.0 { 1.5 } else { 0.5 },
                note: if fast_int8 {
                    "CPU has int8 dot-product instructions; LoRA on the top layers with the activation cache is practical.".into()
                } else {
                    "Older CPU: use models under 1B parameters and train only the top 4 to 8 layers.".into()
                },
            }
        };

        Profile {
            threads,
            threads_batch: self.logical_cores.max(threads),
            parallel_slots,
            context_per_slot: 4096,
            // q8_0 halves KV memory with negligible quality loss.
            kv_cache_type: "q8_0",
            gpu_layers: if self.has_gpu() { 999 } else { 0 },
            max_model_gb,
            fine_tune,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn detects_something_sane() {
        let hw = super::detect();
        assert!(hw.physical_cores >= 1);
        assert!(hw.total_memory_gb > 0.5);
        let p = hw.profile();
        assert!(p.threads >= 1 && p.parallel_slots >= 2);
    }
}
