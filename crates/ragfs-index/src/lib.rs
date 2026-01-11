//! File indexing engine for RAGFS.
//!
//! This crate provides the indexing pipeline that processes files through:
//! extraction → chunking → embedding → storage.
//!
//! # Components
//!
//! - [`IndexerService`]: Main service that coordinates the indexing pipeline
//! - [`FileWatcher`]: Monitors directories for file changes
//! - [`IndexerConfig`]: Configuration for the indexer
//! - [`IndexUpdate`]: Events emitted during indexing
//!
//! # Example
//!
//! ```rust,ignore
//! use ragfs_index::{IndexerService, IndexerConfig};
//!
//! let indexer = IndexerService::new(
//!     source_path,
//!     store,
//!     extractors,
//!     chunkers,
//!     embedder,
//!     IndexerConfig::default(),
//! );
//!
//! // Subscribe to updates
//! let mut updates = indexer.subscribe();
//!
//! // Start indexing
//! indexer.start().await?;
//!
//! // Process updates
//! while let Ok(update) = updates.recv().await {
//!     match update {
//!         IndexUpdate::FileIndexed { path, chunk_count } => { /* ... */ }
//!         IndexUpdate::FileError { path, error } => { /* ... */ }
//!         _ => {}
//!     }
//! }
//! ```

pub mod indexer;
pub mod watcher;

pub use indexer::{IndexerConfig, IndexerService, IndexUpdate};
pub use watcher::FileWatcher;
