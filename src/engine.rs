use crate::config::EngineConfig;
use crate::error::{LateDbError, Result};
use crate::layout::{DocumentId, DocumentRange, EmbeddingDim, Modality, TokenMajorStore};
use crate::scoring::{CandidateScore, QueryBatch, Scorer};

/// Thin orchestration layer around storage and scoring.
pub struct LateInteractionEngine<S> {
    config: EngineConfig,
    store: TokenMajorStore,
    scorer: S,
}

impl<S: Scorer> LateInteractionEngine<S> {
    pub fn new(config: EngineConfig, scorer: S) -> Result<Self> {
        let store = TokenMajorStore::new(EmbeddingDim(config.embedding_dim))?;
        Ok(Self {
            config,
            store,
            scorer,
        })
    }

    pub fn store(&self) -> &TokenMajorStore {
        &self.store
    }

    pub fn push_document(
        &mut self,
        document_id: DocumentId,
        modality: Modality,
        flattened_embeddings: &[f32],
    ) -> Result<DocumentRange> {
        let token_count = flattened_embeddings.len() / self.config.embedding_dim;
        if token_count > self.config.max_resident_tokens {
            return Err(LateDbError::TokenLimitExceeded {
                token_count,
                max_tokens: self.config.max_resident_tokens,
            });
        }

        let projected_tokens = self
            .store
            .token_count()
            .checked_add(token_count)
            .ok_or(LateDbError::TokenCapacityExceeded)?;
        if projected_tokens > self.config.max_resident_tokens {
            return Err(LateDbError::TokenLimitExceeded {
                token_count: projected_tokens,
                max_tokens: self.config.max_resident_tokens,
            });
        }

        self.store
            .push_document(document_id, modality, flattened_embeddings)
    }

    /// Score the current in-memory corpus against a borrowed query batch.
    ///
    /// Inputs:
    /// - `query`: borrowed flat query embeddings and token count.
    /// - `out`: caller-owned output buffer, reused to avoid repeated allocation.
    ///
    /// Output:
    /// - `out` is populated by the scorer.
    ///
    /// Side effects:
    /// - Clears `out`; does not mutate corpus memory.
    pub fn score_into(&self, query: QueryBatch<'_>, out: &mut Vec<CandidateScore>) -> Result<()> {
        out.clear();
        if query.embedding_dim.0 != self.config.embedding_dim
            || query.query_token_count == 0
            || query
                .query_token_count
                .checked_mul(self.config.embedding_dim)
                != Some(query.flattened_embeddings.len())
        {
            return Err(LateDbError::InvalidQueryShape);
        }

        if query.query_token_count > self.config.max_query_tokens {
            return Err(LateDbError::TokenLimitExceeded {
                token_count: query.query_token_count,
                max_tokens: self.config.max_query_tokens,
            });
        }

        let result = self.scorer.score_candidates(
            query,
            self.store.scalar_buffer(),
            self.store.document_ranges(),
            out,
        );
        if result.is_err() {
            out.clear();
        }
        result
    }

    pub fn search_into(
        &self,
        query: QueryBatch<'_>,
        top_k: usize,
        out: &mut Vec<CandidateScore>,
    ) -> Result<()> {
        self.score_into(query, out)?;
        out.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.document_id.0.cmp(&right.document_id.0))
        });
        out.truncate(top_k);
        Ok(())
    }
}
