use serde::{Deserialize, Serialize};

/// Local-first limits and execution knobs.
///
/// Keep these conservative on a laptop. Cloud or GPU runs can override them via
/// config loading later without changing the scoring interfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Embedding vector width in scalar elements.
    pub embedding_dim: usize,
    /// Upper bound for all stored token vectors, across documents and modalities.
    pub max_resident_tokens: usize,
    /// Query batch size limit used by orchestration before entering core scoring.
    pub max_query_tokens: usize,
    /// Worker count requested by orchestration. The scoring implementation may
    /// decide how, or whether, to use it.
    pub worker_threads: usize,
}

impl EngineConfig {
    pub fn local_default(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            max_resident_tokens: 250_000,
            max_query_tokens: 512,
            worker_threads: 1,
        }
    }
}
