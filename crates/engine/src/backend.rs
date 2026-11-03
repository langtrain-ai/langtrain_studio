use anyhow::Result;

use crate::types::ChatMessage;

pub type Token = i32;

/// One token in a decode batch. Sequences map 1:1 to KV-cache slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchToken {
    pub token: Token,
    pub pos: u32,
    pub seq: u32,
    /// Whether logits are needed for this position.
    pub logits: bool,
}

/// The compute interface the scheduler drives.
///
/// The KV cache is unified: `n_ctx` cells are shared by up to `n_seq_max`
/// sequences, and `kv_copy` lets one sequence reuse another's prefix.
pub trait Backend: Send {
    fn name(&self) -> String;
    fn n_ctx(&self) -> u32;
    fn n_seq_max(&self) -> u32;
    /// Maximum tokens per `decode` call.
    fn n_batch(&self) -> u32;

    fn tokenize(&self, text: &str, add_special: bool) -> Vec<Token>;
    fn token_to_piece(&self, token: Token) -> Vec<u8>;
    fn is_eog(&self, token: Token) -> bool;
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String>;

    fn decode(&mut self, batch: &[BatchToken]) -> Result<()>;
    /// Logits for entry `batch_index` of the last decoded batch.
    fn logits(&self, batch_index: usize) -> &[f32];

    /// Drop the KV entries of `seq` from position `from` onwards.
    fn kv_remove(&mut self, seq: u32, from: u32);
    /// Make positions `[0, len)` of `src` visible to `dst`.
    fn kv_copy(&mut self, src: u32, dst: u32, len: u32);

    fn adapters(&self) -> Vec<String>;
    /// Activate an adapter for subsequent `decode` calls (`None` = base model).
    fn set_adapter(&mut self, adapter: Option<&str>) -> Result<()>;
}
