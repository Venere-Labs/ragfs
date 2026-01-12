//! Python wrapper for RAGFS embeddings.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_core::{Embedder, EmbeddingConfig};
use ragfs_embed::CandleEmbedder;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Local embeddings using GTE-small model (384 dimensions).
///
/// This embedder runs entirely locally with no API calls.
/// The model is downloaded on first use (~100MB).
///
/// Example:
///
/// ```python
/// embeddings = RagfsEmbeddings()
/// await embeddings.init()
/// vectors = await embeddings.embed_documents(["Hello", "World"])
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RagfsEmbeddings {
    embedder: Arc<RwLock<Option<CandleEmbedder>>>,
    model_path: PathBuf,
    batch_size: usize,
    normalize: bool,
}

#[pymethods]
impl RagfsEmbeddings {
    /// Create a new `RagfsEmbeddings` instance.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to store/load model files. Defaults to ~/.local/share/ragfs/models
    /// * `batch_size` - Batch size for embedding. Defaults to 32.
    /// * `normalize` - Whether to L2-normalize embeddings. Defaults to True.
    #[new]
    #[pyo3(signature = (model_path=None, batch_size=32, normalize=true))]
    fn new(model_path: Option<String>, batch_size: usize, normalize: bool) -> Self {
        let model_path = model_path.map(PathBuf::from).unwrap_or_else(|| {
            directories::ProjectDirs::from("", "", "ragfs")
                .map(|dirs| dirs.data_dir().join("models"))
                .unwrap_or_else(|| PathBuf::from(".ragfs/models"))
        });

        Self {
            embedder: Arc::new(RwLock::new(None)),
            model_path,
            batch_size,
            normalize,
        }
    }

    /// Initialize the embedder (downloads model if needed).
    ///
    /// This must be called before using `embed_documents` or `embed_query`.
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let model_path = self.model_path.clone();

        future_into_py(py, async move {
            let mut guard = embedder_lock.write().await;
            if guard.is_none() {
                let embedder = CandleEmbedder::new(model_path);
                embedder.init().await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to initialize embedder: {e}"))
                })?;
                *guard = Some(embedder);
            }
            Ok(())
        })
    }

    /// Embed a list of documents.
    ///
    /// Args:
    ///     texts: List of text strings to embed.
    ///
    /// Returns:
    ///     List of embedding vectors (each is a list of 384 floats).
    fn embed_documents<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let batch_size = self.batch_size;
        let normalize = self.normalize;

        future_into_py(py, async move {
            let guard = embedder_lock.read().await;
            let embedder = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Embedder not initialized. Call init() first.")
            })?;

            let config = EmbeddingConfig {
                normalize,
                instruction: None,
                batch_size,
            };

            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let results = embedder
                .embed_text(&text_refs, &config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Embedding failed: {e}")))?;

            let embeddings: Vec<Vec<f32>> = results.into_iter().map(|r| r.embedding).collect();
            Ok(embeddings)
        })
    }

    /// Embed a single query string.
    ///
    /// Args:
    ///     text: The query text to embed.
    ///
    /// Returns:
    ///     Embedding vector (list of 384 floats).
    fn embed_query<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let normalize = self.normalize;

        future_into_py(py, async move {
            let guard = embedder_lock.read().await;
            let embedder = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Embedder not initialized. Call init() first.")
            })?;

            let config = EmbeddingConfig {
                normalize,
                instruction: None,
                batch_size: 1,
            };

            let result = embedder
                .embed_query(&text, &config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Embedding failed: {e}")))?;

            Ok(result.embedding)
        })
    }

    /// Get the embedding dimension.
    #[getter]
    fn dimension(&self) -> usize {
        384 // GTE-small dimension
    }

    /// Get the model name.
    #[getter]
    fn model_name(&self) -> &str {
        "thenlper/gte-small"
    }

    /// Get the maximum number of tokens.
    #[getter]
    fn max_tokens(&self) -> usize {
        512
    }
}
