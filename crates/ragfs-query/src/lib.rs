//! # ragfs-query
//!
//! Query parsing and execution for RAGFS semantic search.
//!
//! This crate handles the query pipeline: parsing natural language queries,
//! generating query embeddings, and executing similarity searches against the vector store.
//!
//! ## Features
//!
//! - **Natural language queries**: Search with plain text like "authentication logic"
//! - **Semantic matching**: Uses vector similarity (cosine distance)
//! - **Hybrid search**: Combine vector similarity with full-text search
//! - **Filtering**: Narrow results by path, MIME type, or date range
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ragfs_query::QueryExecutor;
//! use ragfs_core::{VectorStore, Embedder};
//! use std::sync::Arc;
//!
//! // Create the query executor
//! let executor = QueryExecutor::new(
//!     store,      // Arc<dyn VectorStore>
//!     embedder,   // Arc<dyn Embedder>
//!     10,         // Default result limit
//!     true,       // Enable hybrid search
//! );
//!
//! // Execute a semantic query
//! let results = executor.execute("error handling implementation").await?;
//!
//! for result in results {
//!     println!("{}: {:.3}", result.file_path.display(), result.score);
//!     println!("  {}", result.content);
//! }
//! ```
//!
//! ## Query Pipeline
//!
//! 1. **Parse**: Extract query text and optional filters
//! 2. **Embed**: Convert query text to a 384-dimensional vector
//! 3. **Search**: Find similar chunks using ANN (approximate nearest neighbors)
//! 4. **Rank**: Order results by similarity score
//! 5. **Return**: Provide results with content, scores, and locations
//!
//! ## Hybrid Search
//!
//! When enabled, combines:
//! - **Vector similarity**: Semantic matching via embeddings
//! - **Full-text search**: Keyword matching for precision
//!
//! Results are fused using reciprocal rank fusion (RRF).
//!
//! ## Components
//!
//! | Type | Description |
//! |------|-------------|
//! | [`QueryExecutor`] | Executes semantic queries against the index |
//! | [`QueryParser`] | Parses query strings with optional filters |

pub mod executor;
pub mod parser;

pub use executor::QueryExecutor;
pub use parser::QueryParser;
