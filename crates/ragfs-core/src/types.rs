//! Core types for RAGFS.
//!
//! This module contains all shared data structures used across RAGFS:
//!
//! ## File Management
//! - [`FileRecord`]: Metadata about an indexed file
//! - [`FileStatus`]: Current indexing state of a file
//! - [`FileEvent`]: File system events for the watcher
//!
//! ## Content Chunks
//! - [`Chunk`]: A segment of content with its embedding
//! - [`ContentType`]: Type classification for chunk content
//! - [`ChunkConfig`]: Configuration for chunking behavior
//!
//! ## Extraction
//! - [`ExtractedContent`]: Content extracted from a file
//! - [`ContentElement`]: Structural elements (headings, paragraphs, etc.)
//!
//! ## Embeddings
//! - [`Modality`]: Supported embedding modalities (text, image, audio)
//! - [`EmbeddingConfig`]: Configuration for embedding generation
//! - [`EmbeddingOutput`]: Result of embedding a text
//!
//! ## Search
//! - [`SearchQuery`]: Parameters for a vector search
//! - [`SearchResult`]: A matching chunk with similarity score
//! - [`SearchFilter`]: Filters to narrow search results
//! - [`DistanceMetric`]: Vector distance calculation method

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use uuid::Uuid;

// ============================================================================
// File Records
// ============================================================================

/// Metadata about an indexed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// Unique file identifier
    pub id: Uuid,
    /// Absolute path to the file
    pub path: PathBuf,
    /// File size in bytes
    pub size_bytes: u64,
    /// MIME type
    pub mime_type: String,
    /// Content hash for change detection (blake3)
    pub content_hash: String,
    /// Last modification time
    pub modified_at: DateTime<Utc>,
    /// When the file was indexed (None if not yet indexed)
    pub indexed_at: Option<DateTime<Utc>>,
    /// Number of chunks produced
    pub chunk_count: u32,
    /// Current indexing status
    pub status: FileStatus,
    /// Error message if status is Error
    pub error_message: Option<String>,
}

/// File indexing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStatus {
    /// Waiting to be indexed
    Pending,
    /// Currently being indexed
    Indexing,
    /// Successfully indexed
    Indexed,
    /// Indexing failed
    Error,
    /// File was deleted
    Deleted,
}

// ============================================================================
// Chunks
// ============================================================================

/// A chunk of content from a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: Uuid,
    /// Parent file identifier
    pub file_id: Uuid,
    /// Path to the source file
    pub file_path: PathBuf,
    /// The actual content
    pub content: String,
    /// Type of content
    pub content_type: ContentType,
    /// MIME type of the source file
    pub mime_type: Option<String>,
    /// Position in file (0-indexed)
    pub chunk_index: u32,
    /// Byte range in source file
    pub byte_range: Range<u64>,
    /// Line range (if applicable)
    pub line_range: Option<Range<u32>>,
    /// Parent chunk ID (for hierarchical chunking)
    pub parent_chunk_id: Option<Uuid>,
    /// Depth in hierarchy (0 = root)
    pub depth: u8,
    /// Embedding vector (if computed)
    pub embedding: Option<Vec<f32>>,
    /// Additional metadata
    pub metadata: ChunkMetadata,
}

/// Type of chunk content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentType {
    /// Plain text content
    Text,
    /// Source code
    Code {
        /// Programming language
        language: String,
        /// Code symbol information
        symbol: Option<CodeSymbol>,
    },
    /// Caption for an image
    ImageCaption,
    /// Content from a PDF page
    PdfPage {
        /// Page number (1-indexed)
        page_num: u32,
    },
    /// Markdown content
    Markdown,
}

/// Code symbol information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSymbol {
    /// Type of symbol
    pub kind: SymbolKind,
    /// Symbol name
    pub name: String,
}

/// Types of code symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Module,
    Constant,
    Variable,
    Interface,
    Trait,
}

/// Metadata associated with a chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Embedding model used
    pub embedding_model: Option<String>,
    /// When chunk was indexed
    pub indexed_at: Option<DateTime<Utc>>,
    /// Token count (approximate)
    pub token_count: Option<usize>,
    /// Additional key-value metadata
    #[serde(flatten)]
    pub extra: HashMap<String, String>,
}

// ============================================================================
// Extraction
// ============================================================================

/// Content extracted from a file.
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Main text content
    pub text: String,
    /// Structured elements
    pub elements: Vec<ContentElement>,
    /// Extracted images
    pub images: Vec<ExtractedImage>,
    /// File-level metadata
    pub metadata: ContentMetadataInfo,
}

/// A structural element in extracted content.
#[derive(Debug, Clone)]
pub enum ContentElement {
    Heading {
        level: u8,
        text: String,
        byte_offset: u64,
    },
    Paragraph {
        text: String,
        byte_offset: u64,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
        byte_offset: u64,
    },
    List {
        items: Vec<String>,
        ordered: bool,
        byte_offset: u64,
    },
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        byte_offset: u64,
    },
}

/// An image extracted from a document.
#[derive(Debug, Clone)]
pub struct ExtractedImage {
    /// Raw image data
    pub data: Vec<u8>,
    /// MIME type
    pub mime_type: String,
    /// Caption if available
    pub caption: Option<String>,
    /// Page number (for PDFs)
    pub page: Option<u32>,
}

/// Metadata extracted from file content.
#[derive(Debug, Clone, Default)]
pub struct ContentMetadataInfo {
    /// Document title
    pub title: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Page count (for PDFs)
    pub page_count: Option<u32>,
    /// Creation date
    pub created_at: Option<DateTime<Utc>>,
}

// ============================================================================
// Chunking
// ============================================================================

/// Configuration for chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target chunk size in tokens
    pub target_size: usize,
    /// Maximum chunk size in tokens
    pub max_size: usize,
    /// Overlap between chunks in tokens
    pub overlap: usize,
    /// Enable hierarchical chunking
    pub hierarchical: bool,
    /// Maximum hierarchy depth
    pub max_depth: u8,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 512,
            max_size: 1024,
            overlap: 64,
            hierarchical: true,
            max_depth: 2,
        }
    }
}

/// Output from a chunker.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    /// Chunk content
    pub content: String,
    /// Byte range in source
    pub byte_range: Range<u64>,
    /// Line range if applicable
    pub line_range: Option<Range<u32>>,
    /// Index of parent chunk (in output array)
    pub parent_index: Option<usize>,
    /// Depth in hierarchy
    pub depth: u8,
    /// Additional metadata
    pub metadata: ChunkOutputMetadata,
}

/// Metadata for chunk output.
#[derive(Debug, Clone, Default)]
pub struct ChunkOutputMetadata {
    /// Symbol type (for code)
    pub symbol_type: Option<String>,
    /// Symbol name (for code)
    pub symbol_name: Option<String>,
    /// Programming language
    pub language: Option<String>,
}

// ============================================================================
// Embedding
// ============================================================================

/// Supported modalities for embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
}

/// Configuration for embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Instruction prefix for models that support it
    pub instruction: Option<String>,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            instruction: None,
            batch_size: 32,
        }
    }
}

/// Output from embedding.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Number of tokens in input
    pub token_count: usize,
}

// ============================================================================
// Search
// ============================================================================

/// A search query.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// Optional text for hybrid search
    pub text: Option<String>,
    /// Maximum results to return
    pub limit: usize,
    /// Search filters
    pub filters: Vec<SearchFilter>,
    /// Distance metric
    pub metric: DistanceMetric,
}

/// Search filters.
#[derive(Debug, Clone)]
pub enum SearchFilter {
    /// Match files with path prefix
    PathPrefix(String),
    /// Match files by glob pattern
    PathGlob(String),
    /// Match by MIME type
    MimeType(String),
    /// Match by programming language
    Language(String),
    /// Files modified after date
    ModifiedAfter(DateTime<Utc>),
    /// Files modified before date
    ModifiedBefore(DateTime<Utc>),
    /// Minimum hierarchy depth
    MinDepth(u8),
    /// Maximum hierarchy depth
    MaxDepth(u8),
}

/// Distance metric for vector search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    #[default]
    Cosine,
    L2,
    Dot,
}

/// A search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Chunk ID
    pub chunk_id: Uuid,
    /// File path
    pub file_path: PathBuf,
    /// Chunk content
    pub content: String,
    /// Similarity score
    pub score: f32,
    /// Byte range in file
    pub byte_range: Range<u64>,
    /// Line range if available
    pub line_range: Option<Range<u32>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Vector store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    /// Total number of chunks
    pub total_chunks: u64,
    /// Total number of files
    pub total_files: u64,
    /// Index size in bytes
    pub index_size_bytes: u64,
    /// Last update time
    pub last_updated: Option<DateTime<Utc>>,
}

// ============================================================================
// Index Status
// ============================================================================

/// Overall index statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total files tracked
    pub total_files: u64,
    /// Successfully indexed files
    pub indexed_files: u64,
    /// Files pending indexing
    pub pending_files: u64,
    /// Files with errors
    pub error_files: u64,
    /// Total chunks stored
    pub total_chunks: u64,
    /// Last update time
    pub last_update: Option<DateTime<Utc>>,
}

// ============================================================================
// File Events
// ============================================================================

/// File system event for indexing.
#[derive(Debug, Clone)]
pub enum FileEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
    Renamed { from: PathBuf, to: PathBuf },
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== FileRecord Tests ====================

    #[test]
    fn test_file_record_serialization() {
        let record = FileRecord {
            id: Uuid::new_v4(),
            path: PathBuf::from("/test/file.txt"),
            size_bytes: 1024,
            mime_type: "text/plain".to_string(),
            content_hash: "abc123".to_string(),
            modified_at: Utc::now(),
            indexed_at: Some(Utc::now()),
            chunk_count: 5,
            status: FileStatus::Indexed,
            error_message: None,
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: FileRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.id, deserialized.id);
        assert_eq!(record.path, deserialized.path);
        assert_eq!(record.size_bytes, deserialized.size_bytes);
        assert_eq!(record.status, deserialized.status);
    }

    #[test]
    fn test_file_status_serialization() {
        assert_eq!(
            serde_json::to_string(&FileStatus::Pending).unwrap(),
            "\"pending\""
        );
        assert_eq!(
            serde_json::to_string(&FileStatus::Indexed).unwrap(),
            "\"indexed\""
        );
        assert_eq!(
            serde_json::to_string(&FileStatus::Error).unwrap(),
            "\"error\""
        );
    }

    #[test]
    fn test_file_status_equality() {
        assert_eq!(FileStatus::Pending, FileStatus::Pending);
        assert_ne!(FileStatus::Pending, FileStatus::Indexed);
    }

    // ==================== Chunk Tests ====================

    #[test]
    fn test_chunk_serialization() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            file_id: Uuid::new_v4(),
            file_path: PathBuf::from("/test/file.rs"),
            content: "fn main() {}".to_string(),
            content_type: ContentType::Code {
                language: "rust".to_string(),
                symbol: Some(CodeSymbol {
                    kind: SymbolKind::Function,
                    name: "main".to_string(),
                }),
            },
            mime_type: Some("text/x-rust".to_string()),
            chunk_index: 0,
            byte_range: 0..12,
            line_range: Some(0..1),
            parent_chunk_id: None,
            depth: 0,
            embedding: None,
            metadata: ChunkMetadata::default(),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();

        assert_eq!(chunk.id, deserialized.id);
        assert_eq!(chunk.content, deserialized.content);
    }

    #[test]
    fn test_content_type_text() {
        let ct = ContentType::Text;
        let json = serde_json::to_string(&ct).unwrap();
        assert!(json.contains("\"type\":\"text\""));
    }

    #[test]
    fn test_content_type_code() {
        let ct = ContentType::Code {
            language: "python".to_string(),
            symbol: None,
        };
        let json = serde_json::to_string(&ct).unwrap();
        assert!(json.contains("\"type\":\"code\""));
        assert!(json.contains("\"language\":\"python\""));
    }

    #[test]
    fn test_content_type_pdf_page() {
        let ct = ContentType::PdfPage { page_num: 5 };
        let json = serde_json::to_string(&ct).unwrap();
        assert!(json.contains("\"type\":\"pdf_page\""));
        assert!(json.contains("\"page_num\":5"));
    }

    #[test]
    fn test_content_type_markdown() {
        let ct = ContentType::Markdown;
        let json = serde_json::to_string(&ct).unwrap();
        assert!(json.contains("\"type\":\"markdown\""));
    }

    #[test]
    fn test_symbol_kind_serialization() {
        assert_eq!(
            serde_json::to_string(&SymbolKind::Function).unwrap(),
            "\"function\""
        );
        assert_eq!(
            serde_json::to_string(&SymbolKind::Struct).unwrap(),
            "\"struct\""
        );
        assert_eq!(
            serde_json::to_string(&SymbolKind::Trait).unwrap(),
            "\"trait\""
        );
    }

    // ==================== ChunkConfig Tests ====================

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.target_size, 512);
        assert_eq!(config.max_size, 1024);
        assert_eq!(config.overlap, 64);
        assert!(config.hierarchical);
        assert_eq!(config.max_depth, 2);
    }

    #[test]
    fn test_chunk_config_serialization() {
        let config = ChunkConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ChunkConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.target_size, deserialized.target_size);
        assert_eq!(config.max_size, deserialized.max_size);
    }

    // ==================== EmbeddingConfig Tests ====================

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert!(config.normalize);
        assert!(config.instruction.is_none());
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_embedding_config_serialization() {
        let config = EmbeddingConfig {
            normalize: false,
            instruction: Some("Search: ".to_string()),
            batch_size: 16,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EmbeddingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.normalize, deserialized.normalize);
        assert_eq!(config.instruction, deserialized.instruction);
        assert_eq!(config.batch_size, deserialized.batch_size);
    }

    // ==================== Modality Tests ====================

    #[test]
    fn test_modality_serialization() {
        assert_eq!(serde_json::to_string(&Modality::Text).unwrap(), "\"text\"");
        assert_eq!(
            serde_json::to_string(&Modality::Image).unwrap(),
            "\"image\""
        );
        assert_eq!(
            serde_json::to_string(&Modality::Audio).unwrap(),
            "\"audio\""
        );
    }

    #[test]
    fn test_modality_equality() {
        assert_eq!(Modality::Text, Modality::Text);
        assert_ne!(Modality::Text, Modality::Image);
    }

    // ==================== DistanceMetric Tests ====================

    #[test]
    fn test_distance_metric_default() {
        let metric = DistanceMetric::default();
        assert_eq!(metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_distance_metric_serialization() {
        assert_eq!(
            serde_json::to_string(&DistanceMetric::Cosine).unwrap(),
            "\"cosine\""
        );
        assert_eq!(serde_json::to_string(&DistanceMetric::L2).unwrap(), "\"l2\"");
        assert_eq!(
            serde_json::to_string(&DistanceMetric::Dot).unwrap(),
            "\"dot\""
        );
    }

    // ==================== SearchResult Tests ====================

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            chunk_id: Uuid::new_v4(),
            file_path: PathBuf::from("/test/file.txt"),
            content: "Test content".to_string(),
            score: 0.95,
            byte_range: 0..12,
            line_range: Some(0..1),
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.chunk_id, deserialized.chunk_id);
        assert_eq!(result.score, deserialized.score);
        assert_eq!(result.content, deserialized.content);
    }

    // ==================== StoreStats Tests ====================

    #[test]
    fn test_store_stats_serialization() {
        let stats = StoreStats {
            total_chunks: 100,
            total_files: 10,
            index_size_bytes: 1024 * 1024,
            last_updated: Some(Utc::now()),
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: StoreStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.total_chunks, deserialized.total_chunks);
        assert_eq!(stats.total_files, deserialized.total_files);
    }

    // ==================== IndexStats Tests ====================

    #[test]
    fn test_index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.indexed_files, 0);
        assert_eq!(stats.pending_files, 0);
        assert_eq!(stats.error_files, 0);
        assert_eq!(stats.total_chunks, 0);
        assert!(stats.last_update.is_none());
    }

    #[test]
    fn test_index_stats_serialization() {
        let stats = IndexStats {
            total_files: 50,
            indexed_files: 45,
            pending_files: 3,
            error_files: 2,
            total_chunks: 500,
            last_update: Some(Utc::now()),
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: IndexStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.total_files, deserialized.total_files);
        assert_eq!(stats.indexed_files, deserialized.indexed_files);
    }

    // ==================== ChunkMetadata Tests ====================

    #[test]
    fn test_chunk_metadata_default() {
        let meta = ChunkMetadata::default();
        assert!(meta.embedding_model.is_none());
        assert!(meta.indexed_at.is_none());
        assert!(meta.token_count.is_none());
        assert!(meta.extra.is_empty());
    }

    // ==================== ChunkOutput Tests ====================

    #[test]
    fn test_chunk_output_metadata_default() {
        let meta = ChunkOutputMetadata::default();
        assert!(meta.symbol_type.is_none());
        assert!(meta.symbol_name.is_none());
        assert!(meta.language.is_none());
    }

    // ==================== ContentMetadataInfo Tests ====================

    #[test]
    fn test_content_metadata_info_default() {
        let meta = ContentMetadataInfo::default();
        assert!(meta.title.is_none());
        assert!(meta.author.is_none());
        assert!(meta.language.is_none());
        assert!(meta.page_count.is_none());
        assert!(meta.created_at.is_none());
    }

    // ==================== FileEvent Tests ====================

    #[test]
    fn test_file_event_created() {
        let event = FileEvent::Created(PathBuf::from("/test/new.txt"));
        match event {
            FileEvent::Created(path) => assert_eq!(path, PathBuf::from("/test/new.txt")),
            _ => panic!("Expected Created event"),
        }
    }

    #[test]
    fn test_file_event_modified() {
        let event = FileEvent::Modified(PathBuf::from("/test/changed.txt"));
        match event {
            FileEvent::Modified(path) => assert_eq!(path, PathBuf::from("/test/changed.txt")),
            _ => panic!("Expected Modified event"),
        }
    }

    #[test]
    fn test_file_event_deleted() {
        let event = FileEvent::Deleted(PathBuf::from("/test/removed.txt"));
        match event {
            FileEvent::Deleted(path) => assert_eq!(path, PathBuf::from("/test/removed.txt")),
            _ => panic!("Expected Deleted event"),
        }
    }

    #[test]
    fn test_file_event_renamed() {
        let event = FileEvent::Renamed {
            from: PathBuf::from("/test/old.txt"),
            to: PathBuf::from("/test/new.txt"),
        };
        match event {
            FileEvent::Renamed { from, to } => {
                assert_eq!(from, PathBuf::from("/test/old.txt"));
                assert_eq!(to, PathBuf::from("/test/new.txt"));
            }
            _ => panic!("Expected Renamed event"),
        }
    }
}
