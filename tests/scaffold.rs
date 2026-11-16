use late_db::scoring::{QueryBatch, TodoScorer};
use late_db::{
    DocumentId, EmbeddingDim, EngineConfig, LateDbError, LateInteractionEngine, Modality,
    TokenMajorStore,
};

#[test]
fn engine_rejects_zero_embedding_dimension() {
    let config = EngineConfig::local_default(0);
    match LateInteractionEngine::new(config, TodoScorer) {
        Err(LateDbError::EmptyEmbeddingDim) => {}
        Err(err) => panic!("expected EmptyEmbeddingDim, got {err:?}"),
        Ok(_) => panic!("expected EmptyEmbeddingDim, got Ok"),
    }
}

#[test]
fn engine_rejects_query_shape_mismatch() {
    let config = EngineConfig::local_default(4);
    let engine = LateInteractionEngine::new(config, TodoScorer).unwrap();
    let query = QueryBatch {
        embedding_dim: EmbeddingDim(4),
        query_token_count: 2,
        flattened_embeddings: &[0.0, 1.0, 2.0],
    };
    let mut out = Vec::new();
    let err = engine.score_into(query, &mut out).unwrap_err();
    assert!(matches!(err, LateDbError::InvalidQueryShape));
}

#[test]
fn token_major_store_appends_document_ranges() {
    let mut store = TokenMajorStore::new(EmbeddingDim(3)).unwrap();

    let first = store
        .push_document(
            DocumentId(10),
            Modality::Text,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
    let second = store
        .push_document(DocumentId(11), Modality::Image, &[7.0, 8.0, 9.0])
        .unwrap();

    assert_eq!(first.start_token, 0);
    assert_eq!(first.token_len, 2);
    assert_eq!(second.start_token, 2);
    assert_eq!(second.token_len, 1);
    assert_eq!(store.token_count(), 3);
    assert_eq!(store.document_count(), 2);
    assert_eq!(
        store.document_embeddings(first).unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn token_major_store_rejects_misaligned_embedding_buffers() {
    let mut store = TokenMajorStore::new(EmbeddingDim(4)).unwrap();
    let err = store
        .push_document(DocumentId(1), Modality::Text, &[1.0, 2.0, 3.0])
        .unwrap_err();

    assert!(matches!(
        err,
        LateDbError::MisalignedEmbeddingBuffer { len: 3, dim: 4 }
    ));
}

#[test]
fn engine_enforces_resident_token_limit_before_insert() {
    let config = EngineConfig {
        embedding_dim: 2,
        max_resident_tokens: 1,
        max_query_tokens: 4,
        worker_threads: 1,
    };
    let mut engine = LateInteractionEngine::new(config, TodoScorer).unwrap();
    let err = engine
        .push_document(DocumentId(1), Modality::Text, &[1.0, 2.0, 3.0, 4.0])
        .unwrap_err();

    assert!(matches!(
        err,
        LateDbError::TokenLimitExceeded {
            token_count: 2,
            max_tokens: 1
        }
    ));
    assert_eq!(engine.store().token_count(), 0);
}
