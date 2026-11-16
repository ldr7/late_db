use late_db::{
    CandidateScore, DocumentId, DocumentRange, EmbeddingDim, EngineConfig, LateDbError,
    LateInteractionEngine, MaxSimScorer, Modality, QueryBatch, Scorer,
};

fn query(embeddings: &[f32]) -> QueryBatch<'_> {
    QueryBatch {
        embedding_dim: EmbeddingDim(2),
        query_token_count: embeddings.len() / 2,
        flattened_embeddings: embeddings,
    }
}

#[test]
fn exact_maxsim_sums_each_query_tokens_best_document_match() {
    let mut engine =
        LateInteractionEngine::new(EngineConfig::local_default(2), MaxSimScorer).unwrap();
    engine
        .push_document(DocumentId(9), Modality::Image, &[1.0, 0.0, 0.0, 1.0])
        .unwrap();
    engine
        .push_document(DocumentId(2), Modality::Text, &[0.5, 0.5])
        .unwrap();
    let mut scores = Vec::new();
    engine
        .search_into(query(&[1.0, 0.0, 0.0, 1.0]), 5, &mut scores)
        .unwrap();
    assert_eq!(
        scores,
        vec![
            CandidateScore {
                document_id: DocumentId(9),
                score: 2.0,
            },
            CandidateScore {
                document_id: DocumentId(2),
                score: 1.0,
            },
        ]
    );
}

#[test]
fn negative_similarities_are_not_clamped_and_ties_use_document_id() {
    let mut engine =
        LateInteractionEngine::new(EngineConfig::local_default(2), MaxSimScorer).unwrap();
    for document_id in [9, 2, 6] {
        engine
            .push_document(DocumentId(document_id), Modality::Text, &[-1.0, -2.0])
            .unwrap();
    }
    let mut scores = Vec::new();
    engine
        .search_into(query(&[1.0, 1.0]), 2, &mut scores)
        .unwrap();
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].document_id, DocumentId(2));
    assert_eq!(scores[1].document_id, DocumentId(6));
    assert_eq!(scores[0].score, -3.0);
    engine
        .search_into(query(&[1.0, 1.0]), 0, &mut scores)
        .unwrap();
    assert!(scores.is_empty());
}

#[test]
fn insertion_rejects_empty_nonfinite_and_duplicate_documents_atomically() {
    let mut engine =
        LateInteractionEngine::new(EngineConfig::local_default(2), MaxSimScorer).unwrap();
    engine
        .push_document(DocumentId(1), Modality::Text, &[1.0, 2.0])
        .unwrap();
    assert!(matches!(
        engine.push_document(DocumentId(1), Modality::Text, &[3.0, 4.0]),
        Err(LateDbError::DuplicateDocumentId(1))
    ));
    assert!(matches!(
        engine.push_document(DocumentId(2), Modality::Text, &[]),
        Err(LateDbError::EmptyDocument)
    ));
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            engine.push_document(DocumentId(2), Modality::Text, &[invalid, 0.0]),
            Err(LateDbError::NonFiniteEmbedding)
        ));
    }
    assert_eq!(engine.store().document_count(), 1);
    assert_eq!(engine.store().token_count(), 1);
}

#[test]
fn invalid_queries_clear_previous_results_and_never_overflow() {
    let engine = LateInteractionEngine::new(EngineConfig::local_default(2), MaxSimScorer).unwrap();
    let mut scores = vec![CandidateScore {
        document_id: DocumentId(1),
        score: 1.0,
    }];
    for token_count in [0, usize::MAX] {
        let invalid = QueryBatch {
            embedding_dim: EmbeddingDim(2),
            query_token_count: token_count,
            flattened_embeddings: &[],
        };
        assert!(matches!(
            engine.score_into(invalid, &mut scores),
            Err(LateDbError::InvalidQueryShape)
        ));
        assert!(scores.is_empty());
    }
    assert!(matches!(
        engine.score_into(query(&[f32::NAN, 0.0]), &mut scores),
        Err(LateDbError::NonFiniteEmbedding)
    ));
}

#[test]
fn scoring_validates_direct_ranges_and_detects_arithmetic_overflow() {
    let mut scores = Vec::new();
    let invalid_range = DocumentRange {
        document_id: DocumentId(1),
        modality: Modality::Text,
        start_token: usize::MAX,
        token_len: 1,
    };
    assert!(matches!(
        MaxSimScorer.score_candidates(
            query(&[1.0, 0.0]),
            &[1.0, 0.0],
            &[invalid_range],
            &mut scores,
        ),
        Err(LateDbError::DocumentRangeOutOfBounds)
    ));
    let mut engine =
        LateInteractionEngine::new(EngineConfig::local_default(2), MaxSimScorer).unwrap();
    engine
        .push_document(DocumentId(1), Modality::Text, &[f32::MAX, 0.0])
        .unwrap();
    assert!(matches!(
        engine.search_into(query(&[2.0, 0.0]), 1, &mut scores),
        Err(LateDbError::NonFiniteScore)
    ));
    assert!(scores.is_empty());
}

#[test]
fn empty_corpus_returns_no_results_and_limits_are_enforced() {
    let config = EngineConfig {
        max_resident_tokens: 2,
        max_query_tokens: 1,
        ..EngineConfig::local_default(2)
    };
    let mut engine = LateInteractionEngine::new(config, MaxSimScorer).unwrap();
    let mut scores = Vec::new();
    engine
        .search_into(query(&[1.0, 0.0]), 5, &mut scores)
        .unwrap();
    assert!(scores.is_empty());
    engine
        .push_document(DocumentId(1), Modality::Text, &[1.0, 0.0])
        .unwrap();
    assert!(matches!(
        engine.push_document(DocumentId(2), Modality::Text, &[1.0, 0.0, 0.0, 1.0]),
        Err(LateDbError::TokenLimitExceeded { .. })
    ));
    assert_eq!(engine.store().document_count(), 1);
    assert!(matches!(
        engine.search_into(query(&[1.0, 0.0, 0.0, 1.0]), 1, &mut scores),
        Err(LateDbError::TokenLimitExceeded { .. })
    ));
}
