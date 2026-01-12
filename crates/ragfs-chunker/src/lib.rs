//! # ragfs-chunker
//!
//! Document chunking strategies for the RAGFS indexing pipeline.
//!
//! This crate splits [`ExtractedContent`](ragfs_core::ExtractedContent) into smaller
//! chunks suitable for embedding. Different strategies optimize for different content types.
//!
//! ## Chunking Strategies
//!
//! | Chunker | Best For | Method |
//! |---------|----------|--------|
//! | [`FixedSizeChunker`] | General text | Token-based splitting with overlap |
//! | [`CodeChunker`] | Source code | AST-aware splitting via tree-sitter |
//! | [`SemanticChunker`] | Documents | Structure-aware (headings, paragraphs) |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ragfs_chunker::{ChunkerRegistry, FixedSizeChunker, CodeChunker, SemanticChunker};
//! use ragfs_core::ChunkConfig;
//!
//! // Create a registry with all chunkers
//! let mut registry = ChunkerRegistry::new();
//! registry.register("fixed", FixedSizeChunker::new());
//! registry.register("code", CodeChunker::new());
//! registry.register("semantic", SemanticChunker::new());
//! registry.set_default("fixed");
//!
//! // Configure chunking parameters
//! let config = ChunkConfig {
//!     target_size: 512,    // Target tokens per chunk
//!     max_size: 1024,      // Maximum tokens
//!     overlap: 64,         // Overlap between chunks
//!     hierarchical: true,  // Enable parent/child relationships
//!     max_depth: 2,        // Maximum hierarchy depth
//! };
//!
//! // Chunk content
//! let chunks = registry.chunk(&content, &content_type, &config).await?;
//! ```
//!
//! ## Fixed-Size Chunking
//!
//! The [`FixedSizeChunker`] splits text into chunks of approximately equal size:
//!
//! - Token-based sizing (not character-based)
//! - Configurable overlap for context preservation
//! - Smart break detection (prefers newlines, sentence boundaries)
//!
//! ## Code-Aware Chunking
//!
//! The [`CodeChunker`] uses tree-sitter for syntax-aware splitting:
//!
//! - Respects function/class boundaries
//! - Preserves complete code constructs
//! - Supports Rust, Python, JavaScript, TypeScript, Go, Java, and more
//!
//! ## Semantic Chunking
//!
//! The [`SemanticChunker`] understands document structure:
//!
//! - Splits on headings and sections
//! - Preserves paragraph integrity
//! - Maintains hierarchical relationships
//!
//! ## Components
//!
//! | Type | Description |
//! |------|-------------|
//! | [`ChunkerRegistry`] | Routes content to appropriate chunkers |
//! | [`FixedSizeChunker`] | Token-based chunking with overlap |
//! | [`CodeChunker`] | AST-aware code chunking |
//! | [`SemanticChunker`] | Document structure-aware chunking |

pub mod code;
pub mod fixed;
pub mod registry;
pub mod semantic;

pub use code::CodeChunker;
pub use fixed::FixedSizeChunker;
pub use registry::ChunkerRegistry;
pub use semantic::SemanticChunker;
