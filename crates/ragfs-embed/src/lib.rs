//! # ragfs-embed
//!
//! Local embedding generation for RAGFS using the Candle ML framework.
//!
//! This crate provides offline, privacy-preserving vector embeddings without external APIs.
//! Embeddings are generated using the `gte-small` model from Hugging Face.
//!
//! ## Features
//!
//! - **Local-first**: All computation happens on your machine
//! - **Offline capable**: Works without internet after initial model download
//! - **No API costs**: No rate limits or usage fees
//! - **Concurrent**: Thread pool for parallel embedding generation
//! - **Cached**: LRU cache to avoid redundant computations
//!
//! ## Cargo Features
//!
//! - `candle` (default): Enables the Candle ML stack for real embeddings
//! - Without `candle`: Only `NoopEmbedder` is available (for testing/development)
//!
//! ## Model Details
//!
//! | Property | Value |
//! |----------|-------|
//! | Model | `thenlper/gte-small` |
//! | Dimension | 384 |
//! | Max tokens | 512 |
//! | Architecture | BERT-based |
//! | Size | ~100MB |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ragfs_embed::{CandleEmbedder, EmbedderPool, EmbeddingCache};
//! use ragfs_core::{Embedder, EmbeddingConfig};
//! use std::sync::Arc;
//!
//! // Create and initialize the embedder
//! let embedder = CandleEmbedder::new("~/.local/share/ragfs/models".into());
//! embedder.init().await?;  // Downloads model on first run
//!
//! // Wrap with a thread pool for concurrency
//! let pool = EmbedderPool::new(Arc::new(embedder), 4);
//!
//! // Embed documents
//! let config = EmbeddingConfig::default();
//! let texts = vec!["Hello world", "Machine learning"];
//! let embeddings = pool.embed_batch(&texts, &config).await?;
//! // Each embedding is a Vec<f32> with 384 dimensions
//! ```
//!
//! ## Caching
//!
//! Use [`EmbeddingCache`] to avoid recomputing embeddings for identical text:
//!
//! ```rust,ignore
//! use ragfs_embed::EmbeddingCache;
//!
//! // Create a cache with default capacity (10,000 entries)
//! let cache = EmbeddingCache::new(embedder);
//!
//! // Or with custom capacity
//! let cache = EmbeddingCache::with_capacity(embedder, 50_000);
//!
//! // Embeddings are cached by content hash
//! let result = cache.embed_text(&["Hello"], &config).await?;
//! ```
//!
//! ## Components
//!
//! | Type | Description |
//! |------|-------------|
//! | [`CandleEmbedder`] | Transformer-based embeddings using `gte-small` (requires `candle` feature) |
//! | [`EmbeddingCache`] | LRU cache for embedding results (requires `candle` feature) |
//! | [`EmbedderPool`] | Concurrent embedding with semaphore limiting (always available) |
//! | [`NoopEmbedder`] | No-op embedder for testing (always available) |

// Candle-based modules (optional)
#[cfg(feature = "candle")]
pub mod cache;
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub use cache::EmbeddingCache;
#[cfg(feature = "candle")]
pub use candle::CandleEmbedder;

// Always available modules
pub mod noop;
pub mod pool;

pub use noop::NoopEmbedder;
pub use pool::EmbedderPool;
