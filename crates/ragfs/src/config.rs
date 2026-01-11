//! Configuration handling for RAGFS.
//!
//! These structs are prepared for TOML config file loading (future feature).

#![allow(dead_code)]

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Mount configuration
    #[serde(default)]
    pub mount: MountConfig,

    /// Index configuration
    #[serde(default)]
    pub index: IndexConfig,

    /// Embedding configuration
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// Chunking configuration
    #[serde(default)]
    pub chunking: ChunkingConfig,

    /// Query configuration
    #[serde(default)]
    pub query: QueryConfig,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Mount-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MountConfig {
    /// Allow other users to access the mount
    #[serde(default)]
    pub allow_other: bool,
}

/// Index-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// File patterns to include
    #[serde(default = "default_include")]
    pub include: Vec<String>,

    /// File patterns to exclude
    #[serde(default = "default_exclude")]
    pub exclude: Vec<String>,

    /// Maximum file size to index (bytes)
    #[serde(default = "default_max_file_size")]
    pub max_file_size: u64,

    /// Debounce duration for file watcher (ms)
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

fn default_include() -> Vec<String> {
    vec!["**/*".to_string()]
}

fn default_exclude() -> Vec<String> {
    vec![
        "**/node_modules/**".to_string(),
        "**/.git/**".to_string(),
        "**/target/**".to_string(),
        "**/__pycache__/**".to_string(),
        "**/*.pyc".to_string(),
        "**/.venv/**".to_string(),
        "**/venv/**".to_string(),
        "**/.env".to_string(),
    ]
}

fn default_max_file_size() -> u64 {
    52_428_800 // 50MB
}

fn default_debounce_ms() -> u64 {
    500
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            include: default_include(),
            exclude: default_exclude(),
            max_file_size: default_max_file_size(),
            debounce_ms: default_debounce_ms(),
        }
    }
}

/// Embedding-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model to use
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Batch size for embedding
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Use GPU if available
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,

    /// Max concurrent embedding jobs
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

fn default_embedding_model() -> String {
    "jina-embeddings-v3".to_string()
}

fn default_batch_size() -> usize {
    32
}

fn default_use_gpu() -> bool {
    true
}

fn default_max_concurrent() -> usize {
    4
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            batch_size: default_batch_size(),
            use_gpu: default_use_gpu(),
            max_concurrent: default_max_concurrent(),
        }
    }
}

/// Chunking-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target chunk size (tokens)
    #[serde(default = "default_target_size")]
    pub target_size: usize,

    /// Maximum chunk size (tokens)
    #[serde(default = "default_max_size")]
    pub max_size: usize,

    /// Overlap between chunks (tokens)
    #[serde(default = "default_overlap")]
    pub overlap: usize,

    /// Enable hierarchical chunking
    #[serde(default = "default_hierarchical")]
    pub hierarchical: bool,

    /// Maximum hierarchy depth
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
}

fn default_target_size() -> usize {
    512
}

fn default_max_size() -> usize {
    1024
}

fn default_overlap() -> usize {
    64
}

fn default_hierarchical() -> bool {
    true
}

fn default_max_depth() -> u8 {
    2
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_size: default_target_size(),
            max_size: default_max_size(),
            overlap: default_overlap(),
            hierarchical: default_hierarchical(),
            max_depth: default_max_depth(),
        }
    }
}

/// Query-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Default result limit
    #[serde(default = "default_limit")]
    pub default_limit: usize,

    /// Maximum result limit
    #[serde(default = "default_max_limit")]
    pub max_limit: usize,

    /// Enable hybrid search (vector + FTS)
    #[serde(default = "default_hybrid")]
    pub hybrid: bool,

    /// Enable reranking
    #[serde(default)]
    pub rerank: bool,
}

fn default_limit() -> usize {
    10
}

fn default_max_limit() -> usize {
    100
}

fn default_hybrid() -> bool {
    true
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            default_limit: default_limit(),
            max_limit: default_max_limit(),
            hybrid: default_hybrid(),
            rerank: false,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log file path (optional)
    pub file: Option<PathBuf>,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
        }
    }
}

/// Get the XDG data directory for RAGFS.
pub fn data_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RAGFS_DATA_DIR") {
        return Some(PathBuf::from(dir));
    }

    ProjectDirs::from("", "", "ragfs").map(|dirs| dirs.data_dir().to_path_buf())
}

/// Get the XDG config directory for RAGFS.
pub fn config_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RAGFS_CONFIG_DIR") {
        return Some(PathBuf::from(dir));
    }

    ProjectDirs::from("", "", "ragfs").map(|dirs| dirs.config_dir().to_path_buf())
}

/// Get the XDG cache directory for RAGFS.
pub fn cache_dir() -> Option<PathBuf> {
    ProjectDirs::from("", "", "ragfs").map(|dirs| dirs.cache_dir().to_path_buf())
}
