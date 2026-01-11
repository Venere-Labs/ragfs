//! # ragfs-core
//!
//! Core types and traits for the RAGFS (Retrieval-Augmented Generation `FileSystem`) project.
//!
//! This crate provides the foundational abstractions used throughout RAGFS:
//!
//! - **Content Extraction**: [`ContentExtractor`] trait for extracting text from files
//! - **Document Chunking**: [`Chunker`] trait for splitting content into searchable chunks
//! - **Embedding Generation**: [`Embedder`] trait for converting text to vector embeddings
//! - **Vector Storage**: [`VectorStore`] trait for storing and searching embeddings
//! - **Indexing Coordination**: [`Indexer`] trait for managing the indexing pipeline
//!
//! ## Architecture
//!
//! The crate is organized around a pipeline pattern:
//!
//! ```text
//! File → ContentExtractor → Chunker → Embedder → VectorStore
//!                                                     ↓
//!                                              SearchQuery → SearchResult
//! ```
//!
//! ## Key Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`FileRecord`] | Metadata about an indexed file |
//! | [`Chunk`] | A segment of content with its embedding |
//! | [`ExtractedContent`] | Raw content extracted from a file |
//! | [`SearchQuery`] | Parameters for a vector search |
//! | [`SearchResult`] | A matching chunk with similarity score |
//!
//! ## Key Traits
//!
//! | Trait | Purpose |
//! |-------|---------|
//! | [`ContentExtractor`] | Extract text and metadata from files |
//! | [`Chunker`] | Split extracted content into chunks |
//! | [`Embedder`] | Generate vector embeddings |
//! | [`VectorStore`] | Store and search vector embeddings |
//! | [`Indexer`] | Coordinate the indexing pipeline |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ragfs_core::{ContentExtractor, Chunker, Embedder, VectorStore};
//! use ragfs_core::{ChunkConfig, EmbeddingConfig, SearchQuery};
//!
//! // Components implement these traits
//! async fn index_file(
//!     extractor: &impl ContentExtractor,
//!     chunker: &impl Chunker,
//!     embedder: &impl Embedder,
//!     store: &impl VectorStore,
//!     path: &Path,
//! ) -> Result<(), Error> {
//!     // 1. Extract content
//!     let content = extractor.extract(path).await?;
//!
//!     // 2. Chunk the content
//!     let chunks = chunker.chunk(&content, &ChunkConfig::default()).await?;
//!
//!     // 3. Generate embeddings
//!     let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
//!     let embeddings = embedder.embed_text(&texts, &EmbeddingConfig::default()).await?;
//!
//!     // 4. Store in vector database
//!     // ... create Chunk structs with embeddings and store
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! This crate has no optional features.
//!
//! ## Related Crates
//!
//! - `ragfs-extract`: Content extraction implementations
//! - `ragfs-chunker`: Chunking strategy implementations
//! - `ragfs-embed`: Embedding generation with Candle
//! - `ragfs-store`: `LanceDB` vector storage implementation
//! - `ragfs-index`: Indexing pipeline coordination
//! - `ragfs-query`: Query parsing and execution

pub mod error;
pub mod traits;
pub mod types;

pub use error::{ChunkError, EmbedError, Error, ExtractError, Result, StoreError};
pub use traits::*;
pub use types::*;
