//! In-memory late-interaction retrieval with exact scalar MaxSim scoring.

pub mod config;
pub mod engine;
pub mod error;
pub mod layout;
#[cfg(feature = "python")]
pub mod python;
pub mod scoring;

pub use config::EngineConfig;
pub use engine::LateInteractionEngine;
pub use error::{LateDbError, Result};
pub use layout::{DocumentId, DocumentRange, EmbeddingDim, Modality, TokenMajorStore};
pub use scoring::{CandidateScore, MaxSimScorer, QueryBatch, Scorer};
