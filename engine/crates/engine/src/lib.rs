//! Langtrain inference engine.
//!
//! The engine owns one loaded model and runs a continuous-batching loop on a
//! dedicated thread. Compute is delegated to a [`Backend`] (llama.cpp in
//! production, [`MockBackend`] in tests), so batching, prefix reuse, sampling
//! and multi-LoRA scheduling are backend-independent and unit-tested.

pub mod backend;
pub mod hardware;
pub mod mock;
pub mod prefix;
pub mod sampling;
pub mod scheduler;
pub mod types;

#[cfg(feature = "llama")]
pub mod llama;

pub use backend::{Backend, BatchToken, Token};
pub use mock::MockBackend;
pub use scheduler::{EngineConfig, EngineHandle, EngineMetrics};
pub use types::*;
