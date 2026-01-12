//! Python wrapper for RAGFS document loader.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_core::ContentExtractor;
use ragfs_extract::{ExtractorRegistry, ImageExtractor, PdfExtractor, TextExtractor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::vectorstore::Document;

/// Document loader that extracts content from various file formats.
///
/// Supports:
/// - Text files (40+ formats: .txt, .md, .rs, .py, .js, .json, .yaml, etc.)
/// - PDF files (with embedded image extraction)
/// - Images (metadata extraction)
///
/// Example:
///
/// ```python
/// loader = RagfsDocumentLoader()
/// documents = await loader.load("/path/to/file.pdf")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RagfsDocumentLoader {
    registry: Arc<ExtractorRegistry>,
    supported_mimes: Vec<String>,
}

#[pymethods]
impl RagfsDocumentLoader {
    /// Create a new document loader.
    ///
    /// Args:
    ///     extractors: List of extractors to enable. Defaults to all.
    ///                 Options: "text", "pdf", "image"
    #[new]
    #[pyo3(signature = (extractors=None))]
    fn new(extractors: Option<Vec<String>>) -> Self {
        let mut registry = ExtractorRegistry::new();
        let mut supported_mimes = Vec::new();

        let enabled = extractors.unwrap_or_else(|| {
            vec!["text".to_string(), "pdf".to_string(), "image".to_string()]
        });

        for ext in enabled {
            match ext.as_str() {
                "text" => {
                    let extractor = TextExtractor::new();
                    supported_mimes.extend(extractor.supported_types().iter().map(|s| (*s).to_string()));
                    registry.register("text", extractor);
                }
                "pdf" => {
                    let extractor = PdfExtractor::new();
                    supported_mimes.extend(extractor.supported_types().iter().map(|s| (*s).to_string()));
                    registry.register("pdf", extractor);
                }
                "image" => {
                    let extractor = ImageExtractor::new();
                    supported_mimes.extend(extractor.supported_types().iter().map(|s| (*s).to_string()));
                    registry.register("image", extractor);
                }
                _ => {} // Ignore unknown extractors
            }
        }

        Self {
            registry: Arc::new(registry),
            supported_mimes,
        }
    }

    /// Load documents from a file or directory.
    ///
    /// Args:
    ///     path: Path to file or directory to load.
    ///
    /// Returns:
    ///     List of Document objects with `page_content` and metadata.
    fn load<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.registry.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let mut documents = Vec::new();

            if path.is_file() {
                let docs = load_file(&registry, &path).await?;
                documents.extend(docs);
            } else if path.is_dir() {
                let entries = tokio::fs::read_dir(&path).await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to read directory: {e}"))
                })?;

                let mut entries = entries;
                while let Some(entry) = entries.next_entry().await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to read entry: {e}"))
                })? {
                    let entry_path = entry.path();
                    if entry_path.is_file()
                        && let Ok(docs) = load_file(&registry, &entry_path).await
                    {
                        documents.extend(docs);
                    }
                }
            } else {
                return Err(PyRuntimeError::new_err(format!(
                    "Path does not exist: {}",
                    path.display()
                )));
            }

            Ok(documents)
        })
    }

    /// Check if a file can be loaded.
    ///
    /// Args:
    ///     path: Path to check.
    ///
    /// Returns:
    ///     True if the file format is supported.
    fn can_load(&self, path: String) -> bool {
        let path = PathBuf::from(path);
        let mime = mime_guess::from_path(&path)
            .first_or_octet_stream()
            .to_string();
        self.registry.get_for_file(&path, &mime).is_some()
    }

    /// Get list of supported MIME types.
    fn supported_types(&self) -> Vec<String> {
        self.supported_mimes.clone()
    }
}

async fn load_file(
    registry: &ExtractorRegistry,
    path: &PathBuf,
) -> PyResult<Vec<Document>> {
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();

    let extractor = registry.get_for_file(path, &mime).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "No extractor available for: {} ({})",
            path.display(),
            mime
        ))
    })?;

    let content = extractor.extract(path).await.map_err(|e| {
        PyRuntimeError::new_err(format!("Extraction failed: {e}"))
    })?;

    let mut documents = Vec::new();
    let mut metadata = HashMap::new();

    metadata.insert("source".to_string(), path.to_string_lossy().to_string());
    metadata.insert("mime_type".to_string(), mime);

    if let Some(title) = content.metadata.title {
        metadata.insert("title".to_string(), title);
    }
    if let Some(author) = content.metadata.author {
        metadata.insert("author".to_string(), author);
    }
    if let Some(page_count) = content.metadata.page_count {
        metadata.insert("page_count".to_string(), page_count.to_string());
    }

    // Main document
    if !content.text.is_empty() {
        documents.push(Document {
            page_content: content.text,
            metadata: metadata.clone(),
        });
    }

    // Additional elements (headings, code blocks, etc.)
    for element in content.elements {
        let (text, element_type) = match element {
            ragfs_core::ContentElement::Heading { text, level, .. } => {
                (text, format!("heading_{level}"))
            }
            ragfs_core::ContentElement::CodeBlock { code, language, .. } => {
                let lang = language.unwrap_or_else(|| "unknown".to_string());
                (code, format!("code_{lang}"))
            }
            ragfs_core::ContentElement::Paragraph { text, .. } => {
                (text, "paragraph".to_string())
            }
            ragfs_core::ContentElement::List { items, ordered, .. } => {
                let list_type = if ordered { "ordered_list" } else { "unordered_list" };
                (items.join("\n"), list_type.to_string())
            }
            ragfs_core::ContentElement::Table { headers, rows, .. } => {
                let mut table_text = headers.join(" | ");
                for row in rows {
                    table_text.push('\n');
                    table_text.push_str(&row.join(" | "));
                }
                (table_text, "table".to_string())
            }
        };

        if !text.is_empty() {
            let mut elem_metadata = metadata.clone();
            elem_metadata.insert("element_type".to_string(), element_type);
            documents.push(Document {
                page_content: text,
                metadata: elem_metadata,
            });
        }
    }

    Ok(documents)
}
