//! # RAGFS CLI
//!
//! Command-line interface for RAGFS (Retrieval-Augmented Generation `FileSystem`).
//!
//! RAGFS enables semantic search over your files using vector embeddings.
//! This binary provides commands for indexing, querying, mounting, and
//! managing the search index.
//!
//! ## Commands
//!
//! - `ragfs index <PATH>` - Index a directory for semantic search
//! - `ragfs query <PATH> <QUERY>` - Search indexed content
//! - `ragfs mount <SOURCE> <MOUNTPOINT>` - Mount as FUSE filesystem
//! - `ragfs status <PATH>` - Show index statistics
//!
//! ## Examples
//!
//! ```bash
//! # Index a directory
//! ragfs index ~/Documents
//!
//! # Search for content
//! ragfs query ~/Documents "machine learning implementation"
//!
//! # Get JSON output
//! ragfs query ~/Documents "auth" --format json
//! ```
//!
//! See the [User Guide](../docs/USER_GUIDE.md) for detailed usage.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ragfs_chunker::{ChunkerRegistry, CodeChunker, FixedSizeChunker, SemanticChunker};
use ragfs_core::{ChunkConfig, Embedder, EmbeddingConfig, Indexer, VectorStore};
#[cfg(feature = "candle")]
use ragfs_embed::CandleEmbedder;
use ragfs_embed::EmbedderPool;
use ragfs_extract::{ExtractorRegistry, ImageExtractor, PdfExtractor, TextExtractor};
use ragfs_index::{IndexerConfig, IndexerService};
use ragfs_query::QueryExecutor;
#[cfg(feature = "lancedb")]
use ragfs_store::LanceStore;
use serde::Serialize;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

mod config;

use config::{Config, data_dir};

/// Embedding dimension for gte-small model.
const EMBEDDING_DIM: usize = 384;

#[derive(Parser)]
#[command(name = "ragfs")]
#[command(about = "A FUSE filesystem for RAG architectures")]
#[command(version)]
struct Cli {
    /// Path to config file (default: ~/.config/ragfs/config.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Mount a directory as a RAGFS filesystem
    Mount {
        /// Source directory to index
        source: PathBuf,

        /// Mount point
        mountpoint: PathBuf,

        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,

        /// Allow other users to access the mount
        #[arg(long)]
        allow_other: bool,
    },

    /// Index a directory (without mounting)
    Index {
        /// Directory to index
        path: PathBuf,

        /// Force reindexing of all files
        #[arg(short, long)]
        force: bool,

        /// Watch for changes after initial indexing
        #[arg(short, long)]
        watch: bool,
    },

    /// Query the index
    Query {
        /// Path to indexed directory
        path: PathBuf,

        /// Query string
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show index status
    Status {
        /// Path to indexed directory
        path: PathBuf,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Print sample configuration file
    Init,
    /// Show config file path
    Path,
}

/// Output structure for query results.
#[derive(Serialize)]
struct QueryOutput {
    query: String,
    results: Vec<ResultItem>,
}

#[derive(Serialize)]
struct ResultItem {
    file: String,
    score: f32,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    lines: Option<String>,
}

/// Output structure for status.
#[derive(Serialize)]
struct StatusOutput {
    path: String,
    total_files: u64,
    total_chunks: u64,
    index_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_updated: Option<String>,
}

/// Get the database path for a given source directory.
fn get_db_path(source: &Path) -> Result<PathBuf> {
    let data = data_dir().context("Failed to get data directory")?;
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];
    Ok(data.join("indices").join(hash_str).join("index.lance"))
}

/// Get the PID file path for a mount.
///
/// Uses `$XDG_RUNTIME_DIR/ragfs/` if available, otherwise falls back to
/// `$XDG_CACHE_HOME/ragfs/run/`.
fn get_pid_path(source: &Path) -> Result<PathBuf> {
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];

    // Try XDG_RUNTIME_DIR first (e.g., /run/user/1000)
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(runtime_dir).join("ragfs");
        std::fs::create_dir_all(&dir).ok();
        return Ok(dir.join(format!("{hash_str}.pid")));
    }

    // Fallback to cache directory
    let dirs = directories::ProjectDirs::from("", "", "ragfs")
        .context("Failed to get project directories")?;
    let dir = dirs.cache_dir().join("run");
    std::fs::create_dir_all(&dir).context("Failed to create PID directory")?;
    Ok(dir.join(format!("{hash_str}.pid")))
}

/// Get the log file path for daemon output.
fn get_log_path(source: &Path) -> Result<PathBuf> {
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let hash_str = &hash.to_hex()[..16];

    let dirs = directories::ProjectDirs::from("", "", "ragfs")
        .context("Failed to get project directories")?;
    let dir = dirs.cache_dir().join("logs");
    std::fs::create_dir_all(&dir).context("Failed to create log directory")?;
    Ok(dir.join(format!("{hash_str}.log")))
}

/// Create the standard component stack.
async fn create_components(
    source: PathBuf,
) -> Result<(
    Arc<LanceStore>,
    Arc<ExtractorRegistry>,
    Arc<ChunkerRegistry>,
    Arc<EmbedderPool>,
)> {
    // Create store
    let db_path = get_db_path(&source)?;
    let store = Arc::new(LanceStore::new(db_path, EMBEDDING_DIM));

    // Create extractor registry
    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());
    extractors.register("pdf", PdfExtractor::new());
    extractors.register("image", ImageExtractor::new());
    let extractors = Arc::new(extractors);

    // Create chunker registry
    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.register("code", CodeChunker::new());
    chunkers.register("semantic", SemanticChunker::new());
    chunkers.set_default("fixed");
    let chunkers = Arc::new(chunkers);

    // Create embedder
    let cache_dir = data_dir()
        .context("Failed to get data directory")?
        .join("models");
    let embedder = CandleEmbedder::new(cache_dir);

    // Initialize embedder (downloads model if needed)
    info!("Initializing embedder (this may download the model on first run)...");
    embedder
        .init()
        .await
        .context("Failed to initialize embedder")?;

    let embedder_pool = Arc::new(EmbedderPool::new(
        Arc::new(embedder) as Arc<dyn Embedder>,
        4,
    ));

    Ok((store, extractors, chunkers, embedder_pool))
}

/// Detach the process into the background.
///
/// Replaces the unmaintained `daemonize` crate (RUSTSEC-2025-0069) with the
/// `nix` primitives. It must run *before* the Tokio runtime is built: forking a
/// process that already owns worker threads is not safe.
///
/// `source` must already be canonicalized, because the pid file path is derived
/// from it and must match the one `run` computes for cleanup.
// The only unsafe operation is the documented `fork()` below; there is no safe
// wrapper for it in std. See the SAFETY comment at the call site.
#[allow(unsafe_code)]
fn daemonize_process(source: &Path) -> Result<()> {
    let pid_path = get_pid_path(source)?;
    let log_path = get_log_path(source)?;

    // Last lines the user sees: after the fork below, stdout is the log file.
    println!("Mounting in background...");
    println!("PID file: {}", pid_path.display());
    println!("Log file: {}", log_path.display());
    println!("Unmount: fusermount -u <mountpoint>");

    // Opened once and shared by stdout and stderr. The previous code called
    // `File::create` twice on the same path, so the second open truncated the
    // first and any early output was lost.
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("Failed to open log file {}", log_path.display()))?;

    // SAFETY: no threads exist yet, because the Tokio runtime is built by the
    // caller only after this returns. Everything executed between fork and the
    // end of this function (setsid, chdir, dup2, write) is async-signal-safe.
    match unsafe { nix::unistd::fork() }.context("Failed to fork into the background")? {
        // The parent exits immediately so the shell prompt comes back.
        nix::unistd::ForkResult::Parent { .. } => std::process::exit(0),
        nix::unistd::ForkResult::Child => {}
    }

    nix::unistd::setsid().context("Failed to start a new session")?;
    std::env::set_current_dir("/").context("Failed to change directory to /")?;

    let fd = log.as_raw_fd();
    nix::unistd::dup2(fd, nix::libc::STDOUT_FILENO).context("Failed to redirect stdout")?;
    nix::unistd::dup2(fd, nix::libc::STDERR_FILENO).context("Failed to redirect stderr")?;

    if fd > nix::libc::STDERR_FILENO {
        // stdout and stderr hold their own descriptions of the file now.
        drop(log);
    } else {
        // The kernel handed us fd 1 or 2: closing it would close stdio.
        std::mem::forget(log);
    }

    std::fs::write(&pid_path, format!("{}\n", nix::unistd::getpid()))
        .with_context(|| format!("Failed to write pid file {}", pid_path.display()))?;

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    // Detach before the runtime exists, so the model download, the indexer and
    // the FUSE session all live in the background process.
    if let Commands::Mount {
        source,
        foreground: false,
        ..
    } = &cli.command
    {
        if !source.exists() {
            anyhow::bail!("Source directory does not exist: {}", source.display());
        }
        let canonical = source
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize source path {}", source.display()))?;
        daemonize_process(&canonical)?;
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build the Tokio runtime")?
        .block_on(run(cli))
}

/// Execute the selected command.
async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Mount {
            source,
            mountpoint,
            foreground,
            allow_other,
        } => {
            info!("Mounting {:?} at {:?}", source, mountpoint);

            // Verify paths exist
            if !source.exists() {
                anyhow::bail!("Source directory does not exist: {}", source.display());
            }
            if !mountpoint.exists() {
                anyhow::bail!("Mount point does not exist: {}", mountpoint.display());
            }

            let source = source.canonicalize()?;

            // Create components for RAG functionality
            let (store, extractors, chunkers, embedder_pool) =
                create_components(source.clone()).await?;

            // Initialize store
            store.init().await.context("Failed to initialize store")?;

            // Create indexer for reindex requests
            let indexer_config = IndexerConfig {
                chunk_config: ChunkConfig::default(),
                embed_config: EmbeddingConfig::default(),
                ..Default::default()
            };

            let indexer = Arc::new(IndexerService::new(
                source.clone(),
                store.clone() as Arc<dyn VectorStore>,
                extractors,
                chunkers,
                embedder_pool.clone(),
                indexer_config,
            ));

            // Create channel for reindex requests
            let (reindex_tx, mut reindex_rx) = tokio::sync::mpsc::channel::<PathBuf>(32);

            // Spawn reindex handler task
            let indexer_clone = indexer.clone();
            let reindex_handler = tokio::spawn(async move {
                while let Some(path) = reindex_rx.recv().await {
                    info!("Processing reindex request for: {:?}", path);
                    match indexer_clone.reindex_path(&path).await {
                        Ok(()) => {
                            info!("Successfully reindexed: {:?}", path);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to reindex {:?}: {}", path, e);
                        }
                    }
                }
            });

            // Get runtime handle for FUSE (which runs in blocking context)
            let runtime = tokio::runtime::Handle::current();

            // Create filesystem with RAG capabilities
            let fs = ragfs_fuse::RagFs::with_rag(
                source.clone(),
                store as Arc<dyn VectorStore>,
                embedder_pool.document_embedder(),
                runtime,
                Some(reindex_tx),
            );

            // Build mount options
            let mut options = vec![
                fuser::MountOption::FSName("ragfs".to_string()),
                fuser::MountOption::AutoUnmount,
                fuser::MountOption::DefaultPermissions,
            ];

            if allow_other {
                options.push(fuser::MountOption::AllowOther);
            }

            // Mount
            // Mount. When `foreground` is false the process was already
            // detached by `daemonize_process` before this runtime existed.
            if foreground {
                info!("Running in foreground (Ctrl+C to unmount)");
                info!("Try: cat {:?}/.ragfs/.index", mountpoint);
                info!(
                    "Reindex: echo 'path/to/file' > {:?}/.ragfs/.reindex",
                    mountpoint
                );
            } else {
                info!("Running in the background (pid {})", std::process::id());
            }

            fuser::mount2(fs, &mountpoint, &options)?;

            // Cleanup reindex handler on unmount
            reindex_handler.abort();

            // `mount2` returns only once the filesystem is unmounted, so the
            // pid file is stale from here on.
            if !foreground {
                if let Ok(pid_path) = get_pid_path(&source) {
                    let _ = std::fs::remove_file(pid_path);
                }
            }
        }

        Commands::Index { path, force, watch } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;
            info!("Indexing {:?} (force={})", path, force);

            let (store, extractors, chunkers, embedder) = create_components(path.clone()).await?;

            // Create indexer config
            let config = IndexerConfig {
                chunk_config: ChunkConfig::default(),
                embed_config: EmbeddingConfig::default(),
                ..Default::default()
            };

            // Create indexer
            let indexer = IndexerService::new(
                path.clone(),
                store.clone() as Arc<dyn VectorStore>,
                extractors,
                chunkers,
                embedder,
                config,
            );

            // Subscribe to updates for progress
            let mut updates = indexer.subscribe();

            // Spawn progress reporter
            let progress_handle = tokio::spawn(async move {
                let mut indexed = 0u64;
                let mut errors = 0u64;
                while let Ok(update) = updates.recv().await {
                    match update {
                        ragfs_index::IndexUpdate::FileIndexed { path, chunk_count } => {
                            indexed += 1;
                            info!("Indexed: {:?} ({} chunks)", path, chunk_count);
                        }
                        ragfs_index::IndexUpdate::FileError { path, error } => {
                            errors += 1;
                            tracing::warn!("Error: {:?}: {}", path, error);
                        }
                        ragfs_index::IndexUpdate::IndexingStarted { .. } => {}
                        ragfs_index::IndexUpdate::FileRemoved { .. } => {}
                    }
                }
                (indexed, errors)
            });

            // Start indexer
            indexer.start().await.context("Failed to start indexer")?;

            if watch {
                info!("Watching for changes. Press Ctrl+C to stop.");
                // Wait indefinitely
                tokio::signal::ctrl_c()
                    .await
                    .context("Failed to wait for Ctrl+C")?;
                indexer.stop().await?;
            } else {
                // Give some time for initial indexing
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

                // Check stats
                let stats = store.stats().await?;
                info!(
                    "Indexing complete: {} files, {} chunks",
                    stats.total_files, stats.total_chunks
                );
            }

            drop(progress_handle);
        }

        Commands::Query { path, query, limit } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;

            // Check if index exists
            let db_path = get_db_path(&path)?;
            if !db_path.exists() {
                anyhow::bail!(
                    "Index not found for {}. Run 'ragfs index {}' first.",
                    path.display(),
                    path.display()
                );
            }

            let (store, _extractors, _chunkers, embedder) = create_components(path).await?;

            // Initialize store
            store.init().await.context("Failed to initialize store")?;

            // Create query executor
            let executor = QueryExecutor::new(
                store as Arc<dyn VectorStore>,
                embedder.document_embedder(),
                limit,
                false, // hybrid search
            );

            // Execute query
            let results = executor
                .execute(&query)
                .await
                .context("Query execution failed")?;

            // Output results
            match cli.format {
                OutputFormat::Json => {
                    let output = QueryOutput {
                        query: query.clone(),
                        results: results
                            .iter()
                            .map(|r| ResultItem {
                                file: r.file_path.to_string_lossy().to_string(),
                                score: r.score,
                                content: truncate(&r.content, 200),
                                lines: r
                                    .line_range
                                    .as_ref()
                                    .map(|l| format!("{}:{}", l.start, l.end)),
                            })
                            .collect(),
                    };
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Text => {
                    println!("Query: {query}\n");
                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        for (i, result) in results.iter().enumerate() {
                            println!(
                                "{}. {} (score: {:.3})",
                                i + 1,
                                result.file_path.display(),
                                result.score
                            );
                            if let Some(ref lines) = result.line_range {
                                println!("   Lines: {}-{}", lines.start, lines.end);
                            }
                            println!("   {}", truncate(&result.content, 100));
                            println!();
                        }
                    }
                }
            }
        }

        Commands::Status { path } => {
            if !path.exists() {
                anyhow::bail!("Directory does not exist: {}", path.display());
            }

            let path = path.canonicalize()?;
            let db_path = get_db_path(&path)?;

            if !db_path.exists() {
                match cli.format {
                    OutputFormat::Json => {
                        println!(r#"{{"error": "Index not found"}}"#);
                    }
                    OutputFormat::Text => {
                        println!("Index not found for {}", path.display());
                        println!("Run 'ragfs index {}' to create it.", path.display());
                    }
                }
                return Ok(());
            }

            let store = LanceStore::new(db_path, EMBEDDING_DIM);
            store.init().await.context("Failed to initialize store")?;

            let stats = store.stats().await?;

            match cli.format {
                OutputFormat::Json => {
                    let output = StatusOutput {
                        path: path.to_string_lossy().to_string(),
                        total_files: stats.total_files,
                        total_chunks: stats.total_chunks,
                        index_size_bytes: stats.index_size_bytes,
                        last_updated: stats.last_updated.map(|t| t.to_rfc3339()),
                    };
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                OutputFormat::Text => {
                    println!("Index Status for {}", path.display());
                    println!("  Files:  {}", stats.total_files);
                    println!("  Chunks: {}", stats.total_chunks);
                    if let Some(last) = stats.last_updated {
                        println!("  Updated: {}", last.format("%Y-%m-%d %H:%M:%S"));
                    }
                }
            }
        }

        Commands::Config { action } => {
            // Load config from file or CLI-specified path
            let config = if let Some(ref path) = cli.config {
                Config::load_from(Some(path.clone()))
                    .context(format!("Failed to load config from {}", path.display()))?
            } else {
                Config::load().context("Failed to load config")?
            };

            match action {
                ConfigAction::Show => match cli.format {
                    OutputFormat::Json => {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&config)
                                .context("Failed to serialize config")?
                        );
                    }
                    OutputFormat::Text => {
                        println!(
                            "{}",
                            toml::to_string_pretty(&config)
                                .context("Failed to serialize config")?
                        );
                    }
                },
                ConfigAction::Init => {
                    println!("{}", Config::sample_toml());
                }
                ConfigAction::Path => {
                    if let Some(path) = Config::config_path() {
                        println!("{}", path.display());
                    } else {
                        println!("Could not determine config directory");
                    }
                }
            }
        }
    }

    Ok(())
}

/// Truncate a string to max length, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
