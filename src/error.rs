use thiserror::Error;

pub type Result<T> = std::result::Result<T, LateDbError>;

#[derive(Debug, Error)]
pub enum LateDbError {
    #[error("embedding dimension must be non-zero")]
    EmptyEmbeddingDim,

    #[error("token count {token_count} exceeds configured max {max_tokens}")]
    TokenLimitExceeded {
        token_count: usize,
        max_tokens: usize,
    },

    #[error("embedding data length {len} is not divisible by dimension {dim}")]
    MisalignedEmbeddingBuffer { len: usize, dim: usize },

    #[error("document range is out of bounds")]
    DocumentRangeOutOfBounds,

    #[error("token storage would exceed configured or addressable capacity")]
    TokenCapacityExceeded,

    #[error("query shape is invalid")]
    InvalidQueryShape,

    #[error("document must contain at least one token")]
    EmptyDocument,

    #[error("document ID {0} already exists")]
    DuplicateDocumentId(u64),

    #[error("embeddings must contain only finite values")]
    NonFiniteEmbedding,

    #[error("scoring produced a non-finite result")]
    NonFiniteScore,
}
