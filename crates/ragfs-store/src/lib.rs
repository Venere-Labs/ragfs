//! Vector storage layer for RAGFS using LanceDB.
//!
//! This crate provides the storage backend for RAGFS, implementing the
//! [`VectorStore`](ragfs_core::VectorStore) trait using LanceDB as the
//! underlying database.
//!
//! # Features
//!
//! - **Vector Search**: Fast approximate nearest neighbor search using HNSW
//! - **Hybrid Search**: Combined FTS and vector search for better relevance
//! - **Full CRUD**: Create, read, update, delete operations for chunks and files
//! - **Automatic Indexing**: Creates vector and FTS indices automatically
//!
//! # Example
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

pub mod lancedb;
pub mod schema;

pub use lancedb::LanceStore;
