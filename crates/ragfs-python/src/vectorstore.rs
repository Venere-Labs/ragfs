//! Python wrapper for RAGFS vector store.

use chrono::Utc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_core::{Chunk, ChunkMetadata, ContentType, DistanceMetric, SearchQuery, VectorStore};
use ragfs_store::LanceStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Document with content and metadata.
#[pyclass]
#[derive(Clone)]
pub struct Document {
    #[pyo3(get, set)]
    pub page_content: String,
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (page_content, metadata=None))]
    fn new(page_content: String, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            page_content,
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Document(page_content='{}...', metadata={:?})",
            &self.page_content.chars().take(50).collect::<String>(),
            self.metadata
        )
    }
}

/// Search result with document and score.
#[pyclass]
#[derive(Clone)]
pub struct SearchResultPy {
    #[pyo3(get)]
    pub document: Document,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub chunk_id: String,
}

#[pymethods]
impl SearchResultPy {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(score={:.4}, chunk_id='{}', content='{}...')",
            self.score,
            self.chunk_id,
            &self
                .document
                .page_content
                .chars()
                .take(50)
                .collect::<String>()
        )
    }
}

/// A chunk of content for vector storage.
///
/// Used to add documents programmatically to the vector store.
#[pyclass]
#[derive(Clone)]
pub struct PyChunk {
    /// Unique chunk identifier (UUID string)
    #[pyo3(get, set)]
    pub id: String,
    /// Parent file identifier (UUID string)
    #[pyo3(get, set)]
    pub file_id: String,
    /// Path to the source file
    #[pyo3(get, set)]
    pub file_path: String,
    /// The actual content
    #[pyo3(get, set)]
    pub content: String,
    /// Content type: "text", "code:python", "markdown", etc.
    #[pyo3(get, set)]
    pub content_type: String,
    /// MIME type of the source
    #[pyo3(get, set)]
    pub mime_type: Option<String>,
    /// Position index (0-indexed)
    #[pyo3(get, set)]
    pub chunk_index: u32,
    /// Start byte offset
    #[pyo3(get, set)]
    pub start_byte: u64,
    /// End byte offset
    #[pyo3(get, set)]
    pub end_byte: u64,
    /// Start line (optional)
    #[pyo3(get, set)]
    pub start_line: Option<u32>,
    /// End line (optional)
    #[pyo3(get, set)]
    pub end_line: Option<u32>,
    /// Embedding vector
    #[pyo3(get, set)]
    pub embedding: Option<Vec<f32>>,
    /// Additional metadata
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyChunk {
    #[new]
    #[pyo3(signature = (id, file_id, file_path, content, content_type="text".to_string(), mime_type=None, chunk_index=0, start_byte=0, end_byte=0, start_line=None, end_line=None, embedding=None, metadata=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        file_id: String,
        file_path: String,
        content: String,
        content_type: String,
        mime_type: Option<String>,
        chunk_index: u32,
        start_byte: u64,
        end_byte: u64,
        start_line: Option<u32>,
        end_line: Option<u32>,
        embedding: Option<Vec<f32>>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id,
            file_id,
            file_path,
            content,
            content_type,
            mime_type,
            chunk_index,
            start_byte,
            end_byte,
            start_line,
            end_line,
            embedding,
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyChunk(id='{}', file='{}', content='{}...')",
            self.id,
            self.file_path,
            &self.content.chars().take(50).collect::<String>()
        )
    }
}

impl PyChunk {
    /// Convert to ragfs_core::Chunk
    fn to_core_chunk(&self) -> Result<Chunk, String> {
        let id = Uuid::parse_str(&self.id)
            .map_err(|e| format!("Invalid chunk id '{}': {}", self.id, e))?;
        let file_id = Uuid::parse_str(&self.file_id)
            .map_err(|e| format!("Invalid file_id '{}': {}", self.file_id, e))?;

        // Parse content_type: "text", "code:python", "markdown", etc.
        let content_type = parse_content_type(&self.content_type);

        let line_range = match (self.start_line, self.end_line) {
            (Some(start), Some(end)) => Some(start..end),
            _ => None,
        };

        Ok(Chunk {
            id,
            file_id,
            file_path: PathBuf::from(&self.file_path),
            content: self.content.clone(),
            content_type,
            mime_type: self.mime_type.clone(),
            chunk_index: self.chunk_index,
            byte_range: self.start_byte..self.end_byte,
            line_range,
            parent_chunk_id: None,
            depth: 0,
            embedding: self.embedding.clone(),
            metadata: ChunkMetadata {
                embedding_model: None,
                indexed_at: Some(Utc::now()),
                token_count: None,
                extra: self.metadata.clone(),
            },
        })
    }
}

/// Parse content type string into ContentType enum.
fn parse_content_type(s: &str) -> ContentType {
    if s == "text" {
        ContentType::Text
    } else if s == "markdown" {
        ContentType::Markdown
    } else if s == "image_caption" {
        ContentType::ImageCaption
    } else if let Some(lang) = s.strip_prefix("code:") {
        ContentType::Code {
            language: lang.to_string(),
            symbol: None,
        }
    } else if let Some(page) = s.strip_prefix("pdf_page:") {
        ContentType::PdfPage {
            page_num: page.parse().unwrap_or(1),
        }
    } else {
        // Default to text
        ContentType::Text
    }
}

/// Local vector store using `LanceDB`.
///
/// Supports vector similarity search and hybrid search (vector + full-text).
///
/// Example:
///
/// ```python
/// store = RagfsVectorStore("/path/to/db")
/// await store.init()
/// results = await store.similarity_search(query_embedding, k=5)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RagfsVectorStore {
    store: Arc<LanceStore>,
    dimension: usize,
}

#[pymethods]
impl RagfsVectorStore {
    /// Create a new RagfsVectorStore.
    ///
    /// Args:
    ///     db_path: Path to the LanceDB database directory.
    ///     dimension: Embedding dimension. Defaults to 384 (GTE-small).
    #[new]
    #[pyo3(signature = (db_path, dimension=384))]
    fn new(db_path: String, dimension: usize) -> Self {
        let store = LanceStore::new(PathBuf::from(db_path), dimension);
        Self {
            store: Arc::new(store),
            dimension,
        }
    }

    /// Initialize the vector store (creates tables if needed).
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .init()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize store: {e}")))?;
            Ok(())
        })
    }

    /// Search for similar documents by embedding.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query embedding vector.
    /// * `k` - Number of results to return. Defaults to 4.
    ///
    /// # Returns
    ///
    /// List of `SearchResult` objects with document and score.
    #[pyo3(signature = (query_embedding, k=4))]
    fn similarity_search<'py>(
        &self,
        py: Python<'py>,
        query_embedding: Vec<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            let query = SearchQuery {
                embedding: query_embedding,
                text: None,
                limit: k,
                filters: vec![],
                metric: DistanceMetric::Cosine,
            };

            let results = store
                .search(query)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {e}")))?;

            let py_results: Vec<SearchResultPy> = results
                .into_iter()
                .map(|r| {
                    let mut metadata = r.metadata;
                    metadata.insert(
                        "file_path".to_string(),
                        r.file_path.to_string_lossy().to_string(),
                    );
                    metadata.insert("start_byte".to_string(), r.byte_range.start.to_string());
                    metadata.insert("end_byte".to_string(), r.byte_range.end.to_string());
                    if let Some(ref lr) = r.line_range {
                        metadata.insert("start_line".to_string(), lr.start.to_string());
                        metadata.insert("end_line".to_string(), lr.end.to_string());
                    }

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

    /// Hybrid search combining vector similarity and full-text search.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The text query for full-text search.
    /// * `query_embedding` - The query embedding vector.
    /// * `k` - Number of results to return. Defaults to 4.
    ///
    /// # Returns
    ///
    /// List of `SearchResult` objects with document and score.
    #[pyo3(signature = (query_text, query_embedding, k=4))]
    fn hybrid_search<'py>(
        &self,
        py: Python<'py>,
        query_text: String,
        query_embedding: Vec<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            let query = SearchQuery {
                embedding: query_embedding,
                text: Some(query_text),
                limit: k,
                filters: vec![],
                metric: DistanceMetric::Cosine,
            };

            let results = store
                .hybrid_search(query)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Hybrid search failed: {e}")))?;

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
    ///
    /// # Returns
    ///
    /// Dictionary with `total_chunks`, `total_files`, and `index_size_bytes`.
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

    /// Delete all chunks for a file path.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file to delete.
    fn delete_by_path<'py>(
        &self,
        py: Python<'py>,
        file_path: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .delete_by_file_path(&PathBuf::from(file_path))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Delete failed: {e}")))?;
            Ok(())
        })
    }

    /// Insert or update chunks in the vector store.
    ///
    /// # Arguments
    ///
    /// * `chunks` - List of `PyChunk` objects to upsert.
    ///
    /// # Returns
    ///
    /// Number of chunks upserted.
    fn upsert_chunks<'py>(
        &self,
        py: Python<'py>,
        chunks: Vec<PyChunk>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.store.clone();

        future_into_py(py, async move {
            // Convert PyChunks to core Chunks
            let core_chunks: Vec<Chunk> = chunks
                .into_iter()
                .map(|c| c.to_core_chunk())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid chunk: {e}")))?;

            let count = core_chunks.len();

            store
                .upsert_chunks(&core_chunks)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Upsert failed: {e}")))?;

            Ok(count)
        })
    }

    /// Get the embedding dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the database path.
    #[getter]
    fn db_path(&self) -> String {
        self.store.db_path().to_string_lossy().to_string()
    }
}
