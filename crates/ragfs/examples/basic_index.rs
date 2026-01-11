//! Basic example: Indexing a directory
//!
//! This example demonstrates how to programmatically index a directory
//! using the RAGFS components.
//!
//! Run with:
//! ```bash
//! cargo run --example basic_index -- /path/to/directory
//! ```

use anyhow::{Context, Result};
use ragfs_chunker::{ChunkerRegistry, FixedSizeChunker};
use ragfs_core::{ChunkConfig, Embedder, EmbeddingConfig, VectorStore};
use ragfs_embed::{CandleEmbedder, EmbedderPool};
use ragfs_extract::{ExtractorRegistry, TextExtractor};
use ragfs_index::{IndexerConfig, IndexerService};
use ragfs_store::LanceStore;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

/// Embedding dimension for the gte-small model.
const EMBEDDING_DIM: usize = 384;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <directory>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} ./src", args[0]);
        std::process::exit(1);
    }

    let source = PathBuf::from(&args[1]);
    if !source.exists() {
        anyhow::bail!("Directory does not exist: {}", source.display());
    }

    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting indexing of {:?}", source);

    // Create components
    let (store, extractors, chunkers, embedder) = create_components(&source).await?;

    // Create indexer configuration
    let config = IndexerConfig {
        chunk_config: ChunkConfig::default(),
        embed_config: EmbeddingConfig::default(),
        ..Default::default()
    };

    // Create the indexer service
    let indexer = IndexerService::new(
        source.clone(),
        store.clone() as Arc<dyn VectorStore>,
        extractors,
        chunkers,
        embedder,
        config,
    );

    // Subscribe to indexing updates
    let mut updates = indexer.subscribe();

    // Spawn a task to log progress
    let progress_task = tokio::spawn(async move {
        let mut indexed_count = 0u64;
        let mut error_count = 0u64;

        while let Ok(update) = updates.recv().await {
            match update {
                ragfs_index::IndexUpdate::FileIndexed { path, chunk_count } => {
                    indexed_count += 1;
                    info!("Indexed: {:?} ({} chunks)", path, chunk_count);
                }
                ragfs_index::IndexUpdate::FileError { path, error } => {
                    error_count += 1;
                    tracing::warn!("Error indexing {:?}: {}", path, error);
                }
                ragfs_index::IndexUpdate::IndexingStarted { path } => {
                    info!("Starting indexing at {:?}", path);
                }
                ragfs_index::IndexUpdate::FileRemoved { path } => {
                    info!("Removed: {:?}", path);
                }
            }
        }

        (indexed_count, error_count)
    });

    // Start indexing
    indexer.start().await.context("Failed to start indexer")?;

    // Wait for initial indexing to complete
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Get statistics
    let stats = store.stats().await?;
    info!(
        "Indexing complete: {} files, {} chunks",
        stats.total_files, stats.total_chunks
    );

    // Cleanup
    drop(progress_task);

    Ok(())
}

/// Create the standard component stack for indexing.
async fn create_components(
    source: &PathBuf,
) -> Result<(
    Arc<LanceStore>,
    Arc<ExtractorRegistry>,
    Arc<ChunkerRegistry>,
    Arc<EmbedderPool>,
)> {
    // Create a temporary database path for this example
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let db_path = std::env::temp_dir()
        .join("ragfs_example")
        .join(&hash.to_hex()[..16])
        .join("index.lance");

    info!("Using database at {:?}", db_path);

    // Create vector store
    let store = Arc::new(LanceStore::new(db_path, EMBEDDING_DIM));

    // Create extractor registry with text extractor
    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());
    let extractors = Arc::new(extractors);

    // Create chunker registry with fixed-size chunker
    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");
    let chunkers = Arc::new(chunkers);

    // Create embedder (downloads model on first run)
    let cache_dir = std::env::temp_dir().join("ragfs_example").join("models");
    let embedder = CandleEmbedder::new(cache_dir);

    info!("Initializing embedder (this may download the model on first run)...");
    embedder
        .init()
        .await
        .context("Failed to initialize embedder")?;

    // Wrap embedder in a pool for concurrent embedding
    let embedder_pool = Arc::new(EmbedderPool::new(
        Arc::new(embedder) as Arc<dyn Embedder>,
        4,
    ));

    Ok((store, extractors, chunkers, embedder_pool))
}
