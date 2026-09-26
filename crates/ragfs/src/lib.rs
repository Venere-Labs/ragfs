//! Library surface for the RAGFS CLI.
//!
//! The binary (`main.rs`) uses these types to load `config.toml` and apply it
//! to the indexer, query executor, and embedder.

pub mod config;

pub use config::{Config, ConfigError, SUPPORTED_EMBEDDING_MODEL, resolve_supported_model};
