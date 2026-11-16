use crate::layout::{DocumentId, DocumentRange, EmbeddingDim};
use crate::{LateDbError, Result};

/// Query embeddings prepared by the caller.
///
/// Data boundary:
/// - `flattened_embeddings` is borrowed and must remain valid during scoring.
/// - `query_token_count` is measured in embedding vectors, not scalar elements.
/// - The scorer must not mutate query memory.
#[derive(Debug, Clone, Copy)]
pub struct QueryBatch<'a> {
    pub embedding_dim: EmbeddingDim,
    pub query_token_count: usize,
    pub flattened_embeddings: &'a [f32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateScore {
    pub document_id: DocumentId,
    pub score: f32,
}

/// Late-interaction scoring implementation.
///
/// Implementations decide the reduction order, similarity primitive, SIMD use,
/// parallelism, modality fusion, and numerical behavior.
pub trait Scorer {
    fn score_candidates(
        &self,
        query: QueryBatch<'_>,
        corpus_embeddings: &[f32],
        candidate_ranges: &[DocumentRange],
        out: &mut Vec<CandidateScore>,
    ) -> Result<()>;
}

/// Exact scalar sum-of-maximum-dot-products scoring over token vectors.
#[derive(Debug, Default)]
pub struct MaxSimScorer;

pub use MaxSimScorer as TodoScorer;

impl Scorer for MaxSimScorer {
    fn score_candidates(
        &self,
        query: QueryBatch<'_>,
        corpus_embeddings: &[f32],
        candidate_ranges: &[DocumentRange],
        out: &mut Vec<CandidateScore>,
    ) -> Result<()> {
        let embedding_dim = query.embedding_dim.0;
        if embedding_dim == 0
            || query.query_token_count == 0
            || query.query_token_count.checked_mul(embedding_dim)
                != Some(query.flattened_embeddings.len())
        {
            return Err(LateDbError::InvalidQueryShape);
        }
        if query
            .flattened_embeddings
            .iter()
            .chain(corpus_embeddings)
            .any(|value| !value.is_finite())
        {
            return Err(LateDbError::NonFiniteEmbedding);
        }
        out.clear();
        for range in candidate_ranges {
            if range.is_empty() {
                return Err(LateDbError::EmptyDocument);
            }
            let start = range
                .start_token
                .checked_mul(embedding_dim)
                .ok_or(LateDbError::DocumentRangeOutOfBounds)?;
            let end = range
                .end_token()
                .and_then(|token_end| token_end.checked_mul(embedding_dim))
                .ok_or(LateDbError::DocumentRangeOutOfBounds)?;
            let document_embeddings = corpus_embeddings
                .get(start..end)
                .ok_or(LateDbError::DocumentRangeOutOfBounds)?;
            let mut score = 0.0;
            for query_token in query.flattened_embeddings.chunks_exact(embedding_dim) {
                let mut maximum = f32::NEG_INFINITY;
                for document_token in document_embeddings.chunks_exact(embedding_dim) {
                    let similarity: f32 = query_token
                        .iter()
                        .zip(document_token)
                        .map(|(query_value, document_value)| query_value * document_value)
                        .sum();
                    if !similarity.is_finite() {
                        return Err(LateDbError::NonFiniteScore);
                    }
                    maximum = maximum.max(similarity);
                }
                score += maximum;
            }
            if !score.is_finite() {
                return Err(LateDbError::NonFiniteScore);
            }
            out.push(CandidateScore {
                document_id: range.document_id,
                score,
            });
        }
        Ok(())
    }
}
