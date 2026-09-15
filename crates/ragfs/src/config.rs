//! Configuration handling for RAGFS.
//!
//! Supports loading from TOML config files with CLI override.
//!
//! ## Config File Location
//!
//! - Default: `~/.config/ragfs/config.toml`
//! - Override with: `RAGFS_CONFIG_DIR` environment variable
//! - Override with: `--config /path/to/config.toml` CLI flag
//!
//! ## Precedence
//!
//! Settings are applied in this order (later overrides earlier):
//! 1. Built-in defaults
//! 2. Config file (`config.toml`)
//! 3. CLI arguments

use directories::ProjectDirs;
use ragfs_core::{ChunkConfig, EmbeddingConfig as CoreEmbeddingConfig};
use ragfs_index::IndexerConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Error type for configuration loading.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to read the config file.
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the config file.
    #[error("failed to parse config file: {0}")]
    Parse(#[from] toml::de::Error),

    /// Embedding model is not implemented.
    #[error(
        "unsupported embedding model '{0}': RAGFS currently supports only 'thenlper/gte-small' (alias: 'gte-small')"
    )]
    UnsupportedModel(String),
}

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

impl Config {
    /// Load configuration from the default config file.
    ///
    /// Returns `Ok(Config::default())` if the config file doesn't exist.
    /// Returns an error if the file exists but is invalid.
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(config_dir().map(|d| d.join("config.toml")))
    }

    /// Load configuration from a specific path.
    ///
    /// If `path` is `None`, returns defaults.
    /// If the file doesn't exist, returns defaults.
    /// If the file exists but is invalid, returns an error.
    pub fn load_from(path: Option<PathBuf>) -> Result<Self, ConfigError> {
        let Some(path) = path else {
            return Ok(Self::default());
        };

        if !path.exists() {
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(&path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Generate a sample config file content.
    pub fn sample_toml() -> String {
        toml::to_string_pretty(&Config::default())
            .unwrap_or_else(|_| "# Failed to generate sample".to_string())
    }

    /// Get the config file path.
    pub fn config_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("config.toml"))
    }

    /// Convert this config into an [`IndexerConfig`].
    ///
    /// `force` comes from the CLI (`ragfs index --force`) and skips the
    /// content-hash short-circuit so every eligible file is reindexed.
    pub fn to_indexer_config(&self, force: bool) -> IndexerConfig {
        IndexerConfig {
            chunk_config: self.chunk_config(),
            embed_config: self.core_embedding_config(),
            include_patterns: self.index.include.clone(),
            exclude_patterns: self.index.exclude.clone(),
            debounce_ms: self.index.debounce_ms,
            max_file_size: self.index.max_file_size,
            force,
        }
    }

    /// Chunk settings applied to the indexer.
    pub fn chunk_config(&self) -> ChunkConfig {
        ChunkConfig {
            target_size: self.chunking.target_size,
            max_size: self.chunking.max_size,
            overlap: self.chunking.overlap,
            hierarchical: self.chunking.hierarchical,
            max_depth: self.chunking.max_depth,
        }
    }

    /// Core embedding batch settings (normalize stays on for gte-small).
    pub fn core_embedding_config(&self) -> CoreEmbeddingConfig {
        CoreEmbeddingConfig {
            normalize: true,
            instruction: None,
            batch_size: self.embedding.batch_size.max(1),
        }
    }

    /// Canonical model id, or a clear error if the configured model is not implemented.
    pub fn resolve_embedding_model(&self) -> Result<&'static str, ConfigError> {
        resolve_supported_model(&self.embedding.model)
    }

    /// Worker-pool size for the embedder.
    pub fn embedder_pool_size(&self) -> usize {
        self.embedding.max_concurrent.max(1)
    }

    /// Result limit: CLI `--limit` overrides `[query].default_limit`, then clamped to `max_limit`.
    pub fn query_limit(&self, cli_limit: Option<usize>) -> usize {
        cli_limit
            .unwrap_or(self.query.default_limit)
            .min(self.query.max_limit)
            .max(1)
    }

    /// Hybrid search: CLI `--hybrid` forces on; otherwise `[query].hybrid` is used.
    pub fn query_hybrid(&self, cli_hybrid: bool) -> bool {
        cli_hybrid || self.query.hybrid
    }
}

    /// Hugging Face id of the only implemented embedding model.
pub const SUPPORTED_EMBEDDING_MODEL: &str = "thenlper/gte-small";

    /// Resolve a user-facing model name to the implemented Hugging Face id.
pub fn resolve_supported_model(model: &str) -> Result<&'static str, ConfigError> {
    match model.trim() {
        "thenlper/gte-small" | "gte-small" => Ok(SUPPORTED_EMBEDDING_MODEL),
        other => Err(ConfigError::UnsupportedModel(other.to_string())),
    }
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
    SUPPORTED_EMBEDDING_MODEL.to_string()
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
#[allow(dead_code)]
pub fn cache_dir() -> Option<PathBuf> {
    ProjectDirs::from("", "", "ragfs").map(|dirs| dirs.cache_dir().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml: &str) -> Config {
        toml::from_str(toml).expect("valid test config")
    }

    #[test]
    fn default_model_is_gte_small() {
        let config = Config::default();
        assert_eq!(config.embedding.model, SUPPORTED_EMBEDDING_MODEL);
        assert_eq!(
            config.resolve_embedding_model().unwrap(),
            SUPPORTED_EMBEDDING_MODEL
        );
        assert!(config.query.hybrid);
        let sample = Config::sample_toml();
        assert!(
            sample.contains("thenlper/gte-small"),
            "sample config should advertise the implemented model: {sample}"
        );
        assert!(
            !sample.contains("jina-embeddings-v3"),
            "sample config must not default to an unimplemented model"
        );
    }

    #[test]
    fn toml_overrides_drive_indexer_config() {
        let config = parse(
            r#"
            [index]
            include = ["**/*.rs", "**/*.md"]
            exclude = ["**/secrets/**", "**/*.bin"]
            max_file_size = 4096
            debounce_ms = 250

            [chunking]
            target_size = 128
            max_size = 256
            overlap = 16
            hierarchical = false
            max_depth = 1

            [embedding]
            model = "gte-small"
            batch_size = 8
            use_gpu = false
            max_concurrent = 2
            "#,
        );

        let indexer = config.to_indexer_config(true);
        assert_eq!(
            indexer.include_patterns,
            vec!["**/*.rs".to_string(), "**/*.md".to_string()]
        );
        assert_eq!(
            indexer.exclude_patterns,
            vec!["**/secrets/**".to_string(), "**/*.bin".to_string()]
        );
        assert_eq!(indexer.max_file_size, 4096);
        assert_eq!(indexer.debounce_ms, 250);
        assert!(indexer.force);
        assert_eq!(indexer.chunk_config.target_size, 128);
        assert_eq!(indexer.chunk_config.max_size, 256);
        assert_eq!(indexer.chunk_config.overlap, 16);
        assert!(!indexer.chunk_config.hierarchical);
        assert_eq!(indexer.chunk_config.max_depth, 1);
        assert_eq!(indexer.embed_config.batch_size, 8);
        assert!(!config.embedding.use_gpu);
        assert_eq!(config.embedder_pool_size(), 2);
        assert_eq!(
            config.resolve_embedding_model().unwrap(),
            SUPPORTED_EMBEDDING_MODEL
        );
    }

    #[test]
    fn toml_overrides_drive_query_options() {
        let config = parse(
            r"
            [query]
            default_limit = 3
            max_limit = 7
            hybrid = false
            ",
        );

        assert!(!config.query.hybrid);
        assert!(!config.query_hybrid(false));
        assert!(
            config.query_hybrid(true),
            "--hybrid must force hybrid on when config disables it"
        );
        assert_eq!(config.query_limit(None), 3);
        assert_eq!(config.query_limit(Some(20)), 7, "CLI limit is clamped");
        assert_eq!(config.query_limit(Some(5)), 5);
    }

    #[test]
    fn unsupported_model_is_a_clear_error() {
        let config = parse(
            r#"
            [embedding]
            model = "jina-embeddings-v3"
            "#,
        );

        let err = config.resolve_embedding_model().unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("jina-embeddings-v3"),
            "error should name the requested model: {message}"
        );
        assert!(
            message.contains("thenlper/gte-small"),
            "error should name the supported model: {message}"
        );
    }

    #[test]
    fn default_indexer_mapping_uses_config_excludes() {
        let indexer = Config::default().to_indexer_config(false);
        assert!(!indexer.force);
        assert_eq!(indexer.max_file_size, default_max_file_size());
        assert_eq!(indexer.debounce_ms, default_debounce_ms());
        assert!(
            indexer
                .exclude_patterns
                .contains(&"**/node_modules/**".to_string())
        );
        assert!(indexer.include_patterns.contains(&"**/*".to_string()));
    }

    #[test]
    fn load_from_missing_file_is_defaults() {
        let config = Config::load_from(Some(PathBuf::from("/no/such/ragfs-config.toml"))).unwrap();
        assert_eq!(config.embedding.model, SUPPORTED_EMBEDDING_MODEL);
        assert!(config.query.hybrid);
    }
}
