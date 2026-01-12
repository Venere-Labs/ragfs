//! Python wrapper for RAGFS text splitter.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_chunker::{ChunkerRegistry, CodeChunker, FixedSizeChunker, SemanticChunker};
use ragfs_core::{ChunkConfig, ContentType, ExtractedContent, ContentMetadataInfo};
use std::sync::Arc;

use crate::vectorstore::Document;

/// Text splitter with multiple chunking strategies.
///
/// Strategies:
/// - "fixed": Token-based chunking with overlap
/// - "code": AST-aware chunking for source code
/// - "semantic": Structure-aware chunking on headings/paragraphs
/// - "auto": Automatically select based on content type
///
/// Example:
///
/// ```python
/// splitter = RagfsTextSplitter(chunk_size=512, chunk_overlap=64)
/// chunks = await splitter.split_text("Long text to split...")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RagfsTextSplitter {
    registry: Arc<ChunkerRegistry>,
    chunk_size: usize,
    chunk_overlap: usize,
    chunker_type: String,
}

#[pymethods]
impl RagfsTextSplitter {
    /// Create a new text splitter.
    ///
    /// Args:
    ///     chunk_size: Target chunk size in tokens. Defaults to 512.
    ///     chunk_overlap: Overlap between chunks in tokens. Defaults to 64.
    ///     chunker_type: Chunking strategy. Options: "auto", "fixed", "code", "semantic".
    #[new]
    #[pyo3(signature = (chunk_size=512, chunk_overlap=64, chunker_type="auto".to_string()))]
    fn new(chunk_size: usize, chunk_overlap: usize, chunker_type: String) -> Self {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());
        registry.register("code", CodeChunker::new());
        registry.register("semantic", SemanticChunker::new());
        registry.set_default("fixed");

        Self {
            registry: Arc::new(registry),
            chunk_size,
            chunk_overlap,
            chunker_type,
        }
    }

    /// Split text into chunks.
    ///
    /// Args:
    ///     text: The text to split.
    ///     language: Programming language hint for code chunking.
    ///
    /// Returns:
    ///     List of text chunks.
    #[pyo3(signature = (text, language=None))]
    fn split_text<'py>(
        &self,
        py: Python<'py>,
        text: String,
        language: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.registry.clone();
        let chunk_size = self.chunk_size;
        let chunk_overlap = self.chunk_overlap;
        let chunker_type = self.chunker_type.clone();

        future_into_py(py, async move {
            let content_type = if let Some(ref lang) = language {
                ContentType::Code {
                    language: lang.clone(),
                    symbol: None,
                }
            } else {
                ContentType::Text
            };

            let content = ExtractedContent {
                text: text.clone(),
                elements: vec![],
                images: vec![],
                metadata: ContentMetadataInfo::default(),
            };

            let config = ChunkConfig {
                target_size: chunk_size,
                max_size: chunk_size * 2,
                overlap: chunk_overlap,
                hierarchical: false,
                max_depth: 1,
            };

            // Select content type based on chunker_type or auto-detect
            let effective_content_type = if chunker_type == "auto" {
                content_type
            } else if chunker_type == "code" {
                ContentType::Code {
                    language: language.unwrap_or_else(|| "text".to_string()),
                    symbol: None,
                }
            } else if chunker_type == "semantic" {
                ContentType::Markdown
            } else {
                ContentType::Text
            };

            let chunker = registry.get_for_content_type(&effective_content_type).ok_or_else(|| {
                PyRuntimeError::new_err("No chunker available for content type".to_string())
            })?;

            let outputs = chunker.chunk(&content, &config).await.map_err(|e| {
                PyRuntimeError::new_err(format!("Chunking failed: {e}"))
            })?;

            let chunks: Vec<String> = outputs.into_iter().map(|o| o.content).collect();
            Ok(chunks)
        })
    }

    /// Split documents into smaller chunks.
    ///
    /// Args:
    ///     documents: List of Document objects to split.
    ///
    /// Returns:
    ///     List of chunked Document objects with preserved metadata.
    fn split_documents<'py>(
        &self,
        py: Python<'py>,
        documents: Vec<Document>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.registry.clone();
        let chunk_size = self.chunk_size;
        let chunk_overlap = self.chunk_overlap;
        let chunker_type = self.chunker_type.clone();

        future_into_py(py, async move {
            let mut result_docs = Vec::new();

            for doc in documents {
                let content = ExtractedContent {
                    text: doc.page_content.clone(),
                    elements: vec![],
                    images: vec![],
                    metadata: ContentMetadataInfo::default(),
                };

                let config = ChunkConfig {
                    target_size: chunk_size,
                    max_size: chunk_size * 2,
                    overlap: chunk_overlap,
                    hierarchical: false,
                    max_depth: 1,
                };

                // Detect language from metadata
                let language = doc.metadata.get("language").cloned();
                let content_type = if let Some(ref lang) = language {
                    ContentType::Code {
                        language: lang.clone(),
                        symbol: None,
                    }
                } else {
                    ContentType::Text
                };

                let effective_content_type = if chunker_type == "auto" {
                    content_type
                } else if chunker_type == "code" {
                    ContentType::Code {
                        language: language.unwrap_or_else(|| "text".to_string()),
                        symbol: None,
                    }
                } else if chunker_type == "semantic" {
                    ContentType::Markdown
                } else {
                    ContentType::Text
                };

                let chunker = registry.get_for_content_type(&effective_content_type).ok_or_else(|| {
                    PyRuntimeError::new_err("No chunker available for content type".to_string())
                })?;

                let outputs = chunker.chunk(&content, &config).await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Chunking failed: {e}"))
                })?;

                for (i, output) in outputs.into_iter().enumerate() {
                    let mut metadata = doc.metadata.clone();
                    metadata.insert("chunk_index".to_string(), i.to_string());
                    metadata.insert("start_byte".to_string(), output.byte_range.start.to_string());
                    metadata.insert("end_byte".to_string(), output.byte_range.end.to_string());
                    if let Some(ref lr) = output.line_range {
                        metadata.insert("start_line".to_string(), lr.start.to_string());
                        metadata.insert("end_line".to_string(), lr.end.to_string());
                    }

                    result_docs.push(Document {
                        page_content: output.content,
                        metadata,
                    });
                }
            }

            Ok(result_docs)
        })
    }

    /// Get the configured chunk size.
    #[getter]
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the configured chunk overlap.
    #[getter]
    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    /// Get the chunker type.
    #[getter]
    fn chunker_type(&self) -> &str {
        &self.chunker_type
    }
}
