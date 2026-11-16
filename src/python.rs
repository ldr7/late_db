use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    DocumentId, EmbeddingDim, EngineConfig, LateInteractionEngine, MaxSimScorer, Modality,
    QueryBatch,
};

#[pyclass(name = "EngineConfig", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyEngineConfig {
    inner: EngineConfig,
}

#[pymethods]
impl PyEngineConfig {
    #[new]
    #[pyo3(signature = (embedding_dim, max_resident_tokens=250_000, max_query_tokens=512, worker_threads=1))]
    pub fn new(
        embedding_dim: usize,
        max_resident_tokens: usize,
        max_query_tokens: usize,
        worker_threads: usize,
    ) -> Self {
        Self {
            inner: EngineConfig {
                embedding_dim,
                max_resident_tokens,
                max_query_tokens,
                worker_threads,
            },
        }
    }

    #[staticmethod]
    pub fn local_default(embedding_dim: usize) -> Self {
        Self {
            inner: EngineConfig::local_default(embedding_dim),
        }
    }

    #[getter]
    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim
    }

    #[getter]
    pub fn max_resident_tokens(&self) -> usize {
        self.inner.max_resident_tokens
    }

    #[getter]
    pub fn max_query_tokens(&self) -> usize {
        self.inner.max_query_tokens
    }

    #[getter]
    pub fn worker_threads(&self) -> usize {
        self.inner.worker_threads
    }
}

/// An in-memory corpus scored using exact token-level MaxSim dot products.
#[pyclass(name = "LateInteractionEngine")]
pub struct PyLateInteractionEngine {
    inner: LateInteractionEngine<MaxSimScorer>,
}

fn flatten_embeddings(embeddings: Vec<Vec<f32>>, embedding_dim: usize) -> PyResult<Vec<f32>> {
    if embeddings.is_empty() || embeddings.iter().any(|token| token.len() != embedding_dim) {
        return Err(PyValueError::new_err(format!(
            "embeddings must contain at least one token with dimension {embedding_dim}"
        )));
    }
    Ok(embeddings.into_iter().flatten().collect())
}

#[pymethods]
impl PyLateInteractionEngine {
    #[new]
    #[pyo3(signature = (embedding_dim, max_resident_tokens=250_000, max_query_tokens=512))]
    pub fn new(
        embedding_dim: usize,
        max_resident_tokens: usize,
        max_query_tokens: usize,
    ) -> PyResult<Self> {
        let config = EngineConfig {
            embedding_dim,
            max_resident_tokens,
            max_query_tokens,
            worker_threads: 1,
        };
        let inner = LateInteractionEngine::new(config, MaxSimScorer)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Insert finite token vectors. Document IDs must be unique within the corpus.
    #[pyo3(signature = (document_id, embeddings, modality="text"))]
    pub fn add_document(
        &mut self,
        document_id: u64,
        embeddings: Vec<Vec<f32>>,
        modality: &str,
    ) -> PyResult<()> {
        let modality = match modality {
            "text" => Modality::Text,
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            _ => {
                return Err(PyValueError::new_err(
                    "modality must be 'text', 'image', or 'audio'",
                ));
            }
        };
        let flattened = flatten_embeddings(embeddings, self.embedding_dim())?;
        self.inner
            .push_document(DocumentId(document_id), modality, &flattened)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(())
    }

    /// Return (document_id, score) pairs ordered by score, then ascending document ID.
    #[pyo3(signature = (query_embeddings, top_k=5))]
    pub fn search(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        top_k: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let flattened = flatten_embeddings(query_embeddings, self.embedding_dim())?;
        let query = QueryBatch {
            embedding_dim: EmbeddingDim(self.embedding_dim()),
            query_token_count: flattened.len() / self.embedding_dim(),
            flattened_embeddings: &flattened,
        };
        let mut scores = Vec::new();
        self.inner
            .search_into(query, top_k, &mut scores)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(scores
            .into_iter()
            .map(|candidate| (candidate.document_id.0, candidate.score))
            .collect())
    }

    #[getter]
    pub fn document_count(&self) -> usize {
        self.inner.store().document_count()
    }

    #[getter]
    pub fn token_count(&self) -> usize {
        self.inner.store().token_count()
    }

    #[getter]
    pub fn embedding_dim(&self) -> usize {
        self.inner.store().embedding_dim().0
    }
}

#[pymodule(name = "_native")]
pub fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyEngineConfig>()?;
    m.add_class::<PyLateInteractionEngine>()?;
    Ok(())
}
