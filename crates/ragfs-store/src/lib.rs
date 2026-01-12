//! Vector storage layer for RAGFS.
//!
//! This crate provides storage backends for RAGFS, implementing the
//! [`VectorStore`](ragfs_core::VectorStore) trait.
//!
//! ## Cargo Features
//!
//! - `lancedb` (default): Enables the `LanceDB` backend for production use
//! - Without `lancedb`: Only `MemoryStore` is available (for testing/development)
//!
//! ## Backends
//!
//! | Backend | Description |
//! |---------|-------------|
//! | [`LanceStore`] | Production backend using `LanceDB` (requires `lancedb` feature) |
//! | [`MemoryStore`] | In-memory backend for testing (always available) |
//!
//! ## `LanceDB` Features
//!
//! When the `lancedb` feature is enabled:
//!
//! - **Vector Search**: Fast approximate nearest neighbor search using HNSW
//! - **Hybrid Search**: Combined FTS and vector search for better relevance
//! - **Full CRUD**: Create, read, update, delete operations for chunks and files
//! - **Automatic Indexing**: Creates vector and FTS indices automatically
//!
//! ## Example
//!
//! ```rust,ignore
//! use ragfs_store::LanceStore;
//! use ragfs_core::VectorStore;
//!
//! // Create and initialize store
//! let store = LanceStore::new("path/to/db.lance".into(), 384);
//! store.init().await?;
//!
//! // Store chunks
//! store.upsert_chunks(&chunks).await?;
//!
//! // Search
//! let results = store.search(query).await?;
//! ```

// LanceDB-based modules (optional)
#[cfg(feature = "lancedb")]
pub mod lancedb;
#[cfg(feature = "lancedb")]
pub mod schema;

#[cfg(feature = "lancedb")]
pub use lancedb::LanceStore;

// In-memory store (always available for testing)
pub mod memory;
pub use memory::MemoryStore;
