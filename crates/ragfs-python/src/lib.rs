//! Python bindings for RAGFS - local semantic search and RAG pipeline.
#![allow(clippy::doc_markdown)]
#![allow(clippy::map_unwrap_or)]
//!
//! This crate provides Python bindings via `PyO3` for the core RAGFS components:
//!
//! ## Core Components
//! - [`RagfsEmbeddings`]: Local embeddings using GTE-small (384 dimensions)
//! - [`RagfsVectorStore`]: Vector storage with `LanceDB`
//! - [`RagfsDocumentLoader`]: Multi-format document extraction
//! - [`RagfsTextSplitter`]: Code-aware and semantic text chunking
//! - [`RagfsRetriever`]: Combined embeddings + search
//!
//! ## FUSE Capabilities (AI Agent Operations)
//! - [`RagfsSafetyManager`]: Soft delete, audit history, undo support
//! - [`RagfsSemanticManager`]: AI-powered file organization with Propose-Review-Apply pattern
//! - [`RagfsOpsManager`]: Structured file operations with JSON feedback
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
//!
//! # Safety Layer Example
//!
//! ```python
//! from ragfs import RagfsSafetyManager
//!
//! safety = RagfsSafetyManager("/path/to/source")
//! entry = await safety.delete_to_trash("/path/to/file.txt")
//! # File is in trash, can be restored
//! await safety.restore_from_trash(entry.id)
//! ```
//!
//! # Semantic Operations Example
//!
//! ```python
//! from ragfs import RagfsSemanticManager, OrganizeRequest, OrganizeStrategy
//!
//! semantic = RagfsSemanticManager("/path/to/source", "/path/to/db")
//! await semantic.init()
//!
//! # Create organization plan (NOT executed until approved)
//! request = OrganizeRequest("./docs", OrganizeStrategy.by_topic())
//! plan = await semantic.create_organize_plan(request)
//!
//! # Review and approve
//! for action in plan.actions:
//!     print(f"{action.action} - {action.reason}")
//! result = await semantic.approve_plan(plan.id)
//! ```

mod document_loader;
mod embeddings;
mod ops;
mod retriever;
mod safety;
mod semantic;
mod text_splitter;
mod vectorstore;

use pyo3::prelude::*;

pub use document_loader::RagfsDocumentLoader;
pub use embeddings::RagfsEmbeddings;
pub use ops::{PyBatchResult, PyOperation, PyOperationResult, RagfsOpsManager};
pub use retriever::RagfsRetriever;
pub use safety::{PyHistoryEntry, PyHistoryOperation, PyTrashEntry, RagfsSafetyManager};
pub use semantic::{
    PyCleanupAnalysis, PyCleanupCandidate, PyDuplicateEntry, PyDuplicateGroup, PyDuplicateGroups,
    PyOrganizeRequest, PyOrganizeStrategy, PyPlanAction, PyPlanImpact, PySemanticPlan,
    PySimilarFile, PySimilarFilesResult, RagfsSemanticManager,
};
pub use text_splitter::RagfsTextSplitter;
pub use vectorstore::{Document, PyChunk, RagfsVectorStore, SearchResultPy};

/// RAGFS Python module.
///
/// Provides local semantic search and RAG pipeline components,
/// plus AI agent file operations (safety, semantic, ops managers).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<Document>()?;
    m.add_class::<SearchResultPy>()?;
    m.add_class::<PyChunk>()?;

    // Core Components
    m.add_class::<RagfsEmbeddings>()?;
    m.add_class::<RagfsVectorStore>()?;
    m.add_class::<RagfsDocumentLoader>()?;
    m.add_class::<RagfsTextSplitter>()?;
    m.add_class::<RagfsRetriever>()?;

    // Safety Layer (soft delete, history, undo)
    m.add_class::<RagfsSafetyManager>()?;
    m.add_class::<PyTrashEntry>()?;
    m.add_class::<PyHistoryEntry>()?;
    m.add_class::<PyHistoryOperation>()?;

    // Semantic Operations (AI-powered file organization)
    m.add_class::<RagfsSemanticManager>()?;
    m.add_class::<PyOrganizeStrategy>()?;
    m.add_class::<PyOrganizeRequest>()?;
    m.add_class::<PySemanticPlan>()?;
    m.add_class::<PyPlanAction>()?;
    m.add_class::<PyPlanImpact>()?;
    m.add_class::<PySimilarFile>()?;
    m.add_class::<PySimilarFilesResult>()?;
    m.add_class::<PyDuplicateEntry>()?;
    m.add_class::<PyDuplicateGroup>()?;
    m.add_class::<PyDuplicateGroups>()?;
    m.add_class::<PyCleanupCandidate>()?;
    m.add_class::<PyCleanupAnalysis>()?;

    // Operations Manager (structured file ops with JSON feedback)
    m.add_class::<RagfsOpsManager>()?;
    m.add_class::<PyOperation>()?;
    m.add_class::<PyOperationResult>()?;
    m.add_class::<PyBatchResult>()?;

    Ok(())
}
