use serde::{Deserialize, Serialize};

use crate::error::{LateDbError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingDim(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Other(u16),
}

/// Range descriptor into global token-major storage.
///
/// `start_token` and `token_len` are token indices, not scalar offsets. Scalar
/// offsets are derived at scoring boundaries using `embedding_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentRange {
    pub document_id: DocumentId,
    pub modality: Modality,
    pub start_token: usize,
    pub token_len: usize,
}

impl DocumentRange {
    pub fn end_token(&self) -> Option<usize> {
        self.start_token.checked_add(self.token_len)
    }

    pub fn is_empty(&self) -> bool {
        self.token_len == 0
    }
}

/// Global token-major embedding store.
///
/// Data boundary:
/// - `embeddings` owns a flat `[token0_dim0..token0_dimN, token1_dim0..]` buffer.
/// - `documents` stores ranges into `embeddings`, measured in tokens.
/// - No scoring, ranking, or optimization policy lives here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMajorStore {
    embedding_dim: EmbeddingDim,
    embeddings: Vec<f32>,
    documents: Vec<DocumentRange>,
}

impl TokenMajorStore {
    pub fn new(embedding_dim: EmbeddingDim) -> Result<Self> {
        if embedding_dim.0 == 0 {
            return Err(LateDbError::EmptyEmbeddingDim);
        }

        Ok(Self {
            embedding_dim,
            embeddings: Vec::new(),
            documents: Vec::new(),
        })
    }

    pub fn embedding_dim(&self) -> EmbeddingDim {
        self.embedding_dim
    }

    pub fn scalar_buffer(&self) -> &[f32] {
        &self.embeddings
    }

    pub fn document_ranges(&self) -> &[DocumentRange] {
        &self.documents
    }

    pub fn token_count(&self) -> usize {
        self.embeddings.len() / self.embedding_dim.0
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn document_embeddings(&self, range: DocumentRange) -> Result<&[f32]> {
        let start_scalar = range
            .start_token
            .checked_mul(self.embedding_dim.0)
            .ok_or(LateDbError::DocumentRangeOutOfBounds)?;
        let token_end = range
            .end_token()
            .ok_or(LateDbError::DocumentRangeOutOfBounds)?;
        let end_scalar = token_end
            .checked_mul(self.embedding_dim.0)
            .ok_or(LateDbError::DocumentRangeOutOfBounds)?;

        self.embeddings
            .get(start_scalar..end_scalar)
            .ok_or(LateDbError::DocumentRangeOutOfBounds)
    }

    /// Add already-flattened token embeddings for one document/modality.
    ///
    /// Inputs:
    /// - `document_id`: stable external identifier.
    /// - `modality`: source modality for the inserted token range.
    /// - `flattened_embeddings`: contiguous scalar data; length must be divisible
    ///   by `embedding_dim`.
    ///
    /// Output:
    /// - `DocumentRange` describing the inserted token span.
    ///
    /// Side effects:
    /// - Appends to the owned scalar buffer and document range table.
    pub fn push_document(
        &mut self,
        document_id: DocumentId,
        modality: Modality,
        flattened_embeddings: &[f32],
    ) -> Result<DocumentRange> {
        if flattened_embeddings.is_empty() {
            return Err(LateDbError::EmptyDocument);
        }
        if flattened_embeddings.len() % self.embedding_dim.0 != 0 {
            return Err(LateDbError::MisalignedEmbeddingBuffer {
                len: flattened_embeddings.len(),
                dim: self.embedding_dim.0,
            });
        }

        if flattened_embeddings.iter().any(|value| !value.is_finite()) {
            return Err(LateDbError::NonFiniteEmbedding);
        }
        if self
            .documents
            .iter()
            .any(|range| range.document_id == document_id)
        {
            return Err(LateDbError::DuplicateDocumentId(document_id.0));
        }

        let token_len = flattened_embeddings.len() / self.embedding_dim.0;
        let start_token = self.token_count();
        start_token
            .checked_add(token_len)
            .ok_or(LateDbError::TokenCapacityExceeded)?;

        let range = DocumentRange {
            document_id,
            modality,
            start_token,
            token_len,
        };

        self.embeddings.extend_from_slice(flattened_embeddings);
        self.documents.push(range);
        Ok(range)
    }
}
