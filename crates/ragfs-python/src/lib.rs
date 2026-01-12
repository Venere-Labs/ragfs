//! Python bindings for RAGFS - local semantic search and RAG pipeline.
#![allow(clippy::doc_markdown)]
#![allow(clippy::map_unwrap_or)]
//!
//! This crate provides Python bindings via `PyO3` for the core RAGFS components:
//!
//! - [`RagfsEmbeddings`]: Local embeddings using GTE-small (384 dimensions)
//! - [`RagfsVectorStore`]: Vector storage with `LanceDB`
//! - [`RagfsDocumentLoader`]: Multi-format document extraction
//! - [`RagfsTextSplitter`]: Code-aware and semantic text chunking
//! - [`RagfsRetriever`]: Combined embeddings + search
//!
//! # Quick Start
//!
//! ```python
//! from ragfs import RagfsRetriever
//!
//! # Initialize retriever
//! retriever = RagfsRetriever("/path/to/db")
//! await retriever.init()
//!
//! # Search
//! documents = await retriever.get_relevant_documents("my query")
//! for doc in documents:
//!     print(f"{doc.metadata['file_path']}: {doc.page_content[:100]}")
//! ```

mod document_loader;
mod embeddings;
mod retriever;
mod text_splitter;
mod vectorstore;

use pyo3::prelude::*;

pub use document_loader::RagfsDocumentLoader;
pub use embeddings::RagfsEmbeddings;
pub use retriever::RagfsRetriever;
pub use text_splitter::RagfsTextSplitter;
pub use vectorstore::{Document, PyChunk, RagfsVectorStore, SearchResultPy};

/// RAGFS Python module.
///
/// Provides local semantic search and RAG pipeline components.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<Document>()?;
    m.add_class::<SearchResultPy>()?;
    m.add_class::<PyChunk>()?;

    // Components
    m.add_class::<RagfsEmbeddings>()?;
    m.add_class::<RagfsVectorStore>()?;
    m.add_class::<RagfsDocumentLoader>()?;
    m.add_class::<RagfsTextSplitter>()?;
    m.add_class::<RagfsRetriever>()?;

    Ok(())
}
