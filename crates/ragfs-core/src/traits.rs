//! Core traits for RAGFS components.
//!
//! This module defines the trait interfaces that all RAGFS components implement:
//!
//! - [`ContentExtractor`]: Extract content from files
//! - [`Chunker`]: Split content into chunks
//! - [`Embedder`]: Generate vector embeddings
//! - [`VectorStore`]: Store and search vectors
//! - [`Indexer`]: Coordinate the indexing pipeline
//!
//! These traits enable a pluggable architecture where different implementations
//! can be swapped without changing the rest of the system.

use async_trait::async_trait;
use std::path::Path;

use crate::error::{ChunkError, EmbedError, ExtractError, StoreError};
use crate::types::{
    Chunk, ChunkConfig, ChunkOutput, ContentType, EmbeddingConfig, EmbeddingOutput,
    ExtractedContent, FileRecord, IndexStats, Modality, SearchQuery, SearchResult, StoreStats,
};

// ============================================================================
// Content Extraction
// ============================================================================

/// Trait for extracting content from files.
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Returns the MIME types this extractor can handle.
    fn supported_types(&self) -> &[&str];

    /// Check if this extractor can handle the given file.
    fn can_extract(&self, path: &Path, mime_type: &str) -> bool {
        self.supported_types().contains(&mime_type) || self.can_extract_by_extension(path)
    }

    /// Check if extractor can handle based on file extension.
    fn can_extract_by_extension(&self, _path: &Path) -> bool {
        false
    }

    /// Extract content from a file.
    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError>;

    /// Extract content from bytes (for embedded content).
    async fn extract_bytes(
        &self,
        _data: &[u8],
        _mime_type: &str,
    ) -> Result<ExtractedContent, ExtractError> {
        Err(ExtractError::UnsupportedType(
            "byte extraction not supported".to_string(),
        ))
    }
}

// ============================================================================
// Chunking
// ============================================================================

/// Trait for splitting content into chunks.
#[async_trait]
pub trait Chunker: Send + Sync {
    /// Name of this chunking strategy.
    fn name(&self) -> &str;

    /// Content types this chunker is designed for.
    fn content_types(&self) -> &[&str];

    /// Check if this chunker can handle the given content type.
    fn can_chunk(&self, content_type: &ContentType) -> bool;

    /// Chunk the extracted content.
    async fn chunk(
        &self,
        content: &ExtractedContent,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError>;
}

// ============================================================================
// Embedding
// ============================================================================

/// Trait for generating embeddings.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Model name/identifier.
    fn model_name(&self) -> &str;

    /// Embedding dimension.
    fn dimension(&self) -> usize;

    /// Maximum tokens per input.
    fn max_tokens(&self) -> usize;

    /// Supported modalities.
    fn modalities(&self) -> &[Modality];

    /// Embed text content.
    async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError>;

    /// Embed image content.
    async fn embed_image(
        &self,
        _image_data: &[u8],
        _config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        Err(EmbedError::ModalityNotSupported(Modality::Image))
    }

    /// Embed a query (may use different instruction).
    async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        let results = self.embed_text(&[query], config).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Inference("empty embedding result".to_string()))
    }
}

// ============================================================================
// Vector Storage
// ============================================================================

/// Trait for vector storage and search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Initialize the store.
    async fn init(&self) -> Result<(), StoreError>;

    /// Insert or update chunks.
    async fn upsert_chunks(&self, chunks: &[Chunk]) -> Result<(), StoreError>;

    /// Search for similar chunks.
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError>;

    /// Hybrid search (vector + full-text).
    async fn hybrid_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError>;

    /// Delete all chunks for a file.
    async fn delete_by_file_path(&self, path: &Path) -> Result<u64, StoreError>;

    /// Update file path for all chunks (for renames).
    async fn update_file_path(&self, from: &Path, to: &Path) -> Result<u64, StoreError>;

    /// Get all chunks for a file.
    async fn get_chunks_for_file(&self, path: &Path) -> Result<Vec<Chunk>, StoreError>;

    /// Get file record.
    async fn get_file(&self, path: &Path) -> Result<Option<FileRecord>, StoreError>;

    /// Upsert file record.
    async fn upsert_file(&self, record: &FileRecord) -> Result<(), StoreError>;

    /// Get store statistics.
    async fn stats(&self) -> Result<StoreStats, StoreError>;
}

// ============================================================================
// Indexer
// ============================================================================

/// Trait for file indexing coordination.
#[async_trait]
pub trait Indexer: Send + Sync {
    /// Start watching a directory for changes.
    async fn watch(&self, path: &Path) -> Result<(), crate::Error>;

    /// Stop watching.
    async fn stop(&self) -> Result<(), crate::Error>;

    /// Manually trigger indexing of a path.
    async fn index(&self, path: &Path, force: bool) -> Result<(), crate::Error>;

    /// Get current index statistics.
    async fn stats(&self) -> Result<IndexStats, crate::Error>;

    /// Check if a file needs re-indexing.
    async fn needs_reindex(&self, path: &Path) -> Result<bool, crate::Error>;
}
