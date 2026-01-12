//! Python wrapper for RAGFS retriever.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_core::{Embedder, EmbeddingConfig, VectorStore};
use ragfs_embed::CandleEmbedder;
use ragfs_store::LanceStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::vectorstore::{Document, SearchResultPy};

/// Combined retriever with embeddings and vector store.
///
/// Provides a simple interface for semantic search without
/// manually managing embeddings and vector store.
///
/// Example:
///
/// ```python
/// retriever = RagfsRetriever("/path/to/db")
/// await retriever.init()
/// results = await retriever.get_relevant_documents("my query")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RagfsRetriever {
    embedder: Arc<RwLock<Option<CandleEmbedder>>>,
    store: Arc<LanceStore>,
    model_path: PathBuf,
    hybrid: bool,
    k: usize,
}

#[pymethods]
impl RagfsRetriever {
    /// Create a new retriever.
    ///
    /// Args:
    ///     db_path: Path to the LanceDB database.
    ///     model_path: Path to store model files. Defaults to ~/.local/share/ragfs/models.
    ///     hybrid: Enable hybrid search (vector + full-text). Defaults to True.
    ///     k: Number of results to return. Defaults to 4.
    ///     dimension: Embedding dimension. Defaults to 384.
    #[new]
    #[pyo3(signature = (db_path, model_path=None, hybrid=true, k=4, dimension=384))]
    fn new(
        db_path: String,
        model_path: Option<String>,
        hybrid: bool,
        k: usize,
        dimension: usize,
    ) -> Self {
        let model_path = model_path.map(PathBuf::from).unwrap_or_else(|| {
            directories::ProjectDirs::from("", "", "ragfs")
                .map(|dirs| dirs.data_dir().join("models"))
                .unwrap_or_else(|| PathBuf::from(".ragfs/models"))
        });

        let store = LanceStore::new(PathBuf::from(db_path), dimension);

        Self {
            embedder: Arc::new(RwLock::new(None)),
            store: Arc::new(store),
            model_path,
            hybrid,
            k,
        }
    }

    /// Initialize the retriever (downloads model and creates tables if needed).
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let store = self.store.clone();
        let model_path = self.model_path.clone();

        future_into_py(py, async move {
            // Initialize embedder
            {
                let mut guard = embedder_lock.write().await;
                if guard.is_none() {
                    let embedder = CandleEmbedder::new(model_path);
                    embedder.init().await.map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to initialize embedder: {e}"))
                    })?;
                    *guard = Some(embedder);
                }
            }

            // Initialize store
            store
                .init()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize store: {e}")))?;

            Ok(())
        })
    }

    /// Retrieve relevant documents for a query.
    ///
    /// Args:
    ///     query: The search query text.
    ///     k: Override the default k. If None, uses the instance default.
    ///
    /// Returns:
    ///     List of Document objects ranked by relevance.
    #[pyo3(signature = (query, k=None))]
    fn get_relevant_documents<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let store = self.store.clone();
        let hybrid = self.hybrid;
        let default_k = self.k;
        let k = k.unwrap_or(default_k);

        future_into_py(py, async move {
            // Embed the query
            let guard = embedder_lock.read().await;
            let embedder = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Retriever not initialized. Call init() first.")
            })?;

            let config = EmbeddingConfig::default();
            let embedding_output = embedder
                .embed_query(&query, &config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Embedding failed: {e}")))?;

            // Search
            let search_query = ragfs_core::SearchQuery {
                embedding: embedding_output.embedding,
                text: if hybrid { Some(query.clone()) } else { None },
                limit: k,
                filters: vec![],
                metric: ragfs_core::DistanceMetric::Cosine,
            };

            let results = if hybrid {
                store.hybrid_search(search_query).await
            } else {
                store.search(search_query).await
            }
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {e}")))?;

            // Convert to documents
            let documents: Vec<Document> = results
                .into_iter()
                .map(|r| {
                    let mut metadata: HashMap<String, String> = r.metadata;
                    metadata.insert(
                        "file_path".to_string(),
                        r.file_path.to_string_lossy().to_string(),
                    );
                    metadata.insert("score".to_string(), r.score.to_string());
                    metadata.insert("chunk_id".to_string(), r.chunk_id.to_string());

                    Document {
                        page_content: r.content,
                        metadata,
                    }
                })
                .collect();

            Ok(documents)
        })
    }

    /// Search with explicit control over hybrid mode.
    ///
    /// Args:
    ///     query: The search query text.
    ///     hybrid: Use hybrid search for this query.
    ///     k: Number of results.
    ///
    /// Returns:
    ///     List of SearchResult objects with score.
    #[pyo3(signature = (query, hybrid=None, k=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        hybrid: Option<bool>,
        k: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let store = self.store.clone();
        let use_hybrid = hybrid.unwrap_or(self.hybrid);
        let k = k.unwrap_or(self.k);

        future_into_py(py, async move {
            let guard = embedder_lock.read().await;
            let embedder = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Retriever not initialized. Call init() first.")
            })?;

            let config = EmbeddingConfig::default();
            let embedding_output = embedder
                .embed_query(&query, &config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Embedding failed: {e}")))?;

            let search_query = ragfs_core::SearchQuery {
                embedding: embedding_output.embedding,
                text: if use_hybrid {
                    Some(query.clone())
                } else {
                    None
                },
                limit: k,
                filters: vec![],
                metric: ragfs_core::DistanceMetric::Cosine,
            };

            let results = if use_hybrid {
                store.hybrid_search(search_query).await
            } else {
                store.search(search_query).await
            }
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {e}")))?;

            let py_results: Vec<SearchResultPy> = results
                .into_iter()
                .map(|r| {
                    let mut metadata = r.metadata;
                    metadata.insert(
                        "file_path".to_string(),
                        r.file_path.to_string_lossy().to_string(),
                    );

                    SearchResultPy {
                        document: Document {
                            page_content: r.content,
                            metadata,
                        },
                        score: r.score,
                        chunk_id: r.chunk_id.to_string(),
                    }
                })
                .collect();

            Ok(py_results)
        })
    }

    /// Get store statistics.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            let stats = store
                .stats()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get stats: {e}")))?;

            let result: HashMap<String, u64> = HashMap::from([
                ("total_chunks".to_string(), stats.total_chunks),
                ("total_files".to_string(), stats.total_files),
                ("index_size_bytes".to_string(), stats.index_size_bytes),
            ]);

            Ok(result)
        })
    }

    /// Check if the retriever is initialized.
    fn is_initialized<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();

        future_into_py(py, async move {
            let guard = embedder_lock.read().await;
            Ok(guard.is_some())
        })
    }

    /// Get the database path.
    #[getter]
    fn db_path(&self) -> String {
        self.store.db_path().to_string_lossy().to_string()
    }

    /// Get whether hybrid search is enabled.
    #[getter]
    fn hybrid(&self) -> bool {
        self.hybrid
    }

    /// Get the default k value.
    #[getter]
    fn k(&self) -> usize {
        self.k
    }
}
