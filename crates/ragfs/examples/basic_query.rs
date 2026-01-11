//! Basic example: Querying an index
//!
//! This example demonstrates how to programmatically query an indexed directory.
//! The directory must be indexed first using `basic_index` or the CLI.
//!
//! Run with:
//! ```bash
//! cargo run --example basic_query -- /path/to/indexed/directory "your query"
//! ```

use anyhow::{Context, Result};
use ragfs_core::{Embedder, VectorStore};
use ragfs_embed::{CandleEmbedder, EmbedderPool};
use ragfs_query::QueryExecutor;
use ragfs_store::LanceStore;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Embedding dimension for the gte-small model.
const EMBEDDING_DIM: usize = 384;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <directory> <query>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} ./src \"error handling\"", args[0]);
        std::process::exit(1);
    }

    let source = PathBuf::from(&args[1]);
    let query = &args[2];

    if !source.exists() {
        anyhow::bail!("Directory does not exist: {:?}", source);
    }

    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Querying index for {:?}", source);
    info!("Query: {}", query);

    // Get database path (same logic as in basic_index)
    let hash = blake3::hash(source.to_string_lossy().as_bytes());
    let db_path = std::env::temp_dir()
        .join("ragfs_example")
        .join(&hash.to_hex()[..16])
        .join("index.lance");

    if !db_path.exists() {
        anyhow::bail!(
            "Index not found at {:?}. Run basic_index first:\n  cargo run --example basic_index -- {:?}",
            db_path,
            source
        );
    }

    // Create vector store and initialize
    let store = Arc::new(LanceStore::new(db_path, EMBEDDING_DIM));
    store.init().await.context("Failed to initialize store")?;

    // Create embedder for query embedding
    let cache_dir = std::env::temp_dir().join("ragfs_example").join("models");
    let embedder = CandleEmbedder::new(cache_dir);
    embedder
        .init()
        .await
        .context("Failed to initialize embedder")?;

    let embedder_pool = Arc::new(EmbedderPool::new(Arc::new(embedder) as Arc<dyn Embedder>, 4));

    // Create query executor
    let executor = QueryExecutor::new(
        store as Arc<dyn VectorStore>,
        embedder_pool.document_embedder(),
        10,    // limit
        false, // hybrid search
    );

    // Execute the query
    info!("Executing semantic search...");
    let results = executor
        .execute(query)
        .await
        .context("Query execution failed")?;

    // Display results
    println!("\n{} results found:\n", results.len());

    if results.is_empty() {
        println!("No matching results.");
        println!("\nTips:");
        println!("  - Make sure the directory is indexed");
        println!("  - Try a different query");
        println!("  - Check if the files contain relevant content");
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

            // Truncate content for display
            let content = result.content.replace('\n', " ");
            let display_content = if content.len() > 100 {
                format!("{}...", &content[..100])
            } else {
                content
            };
            println!("   {}\n", display_content);
        }
    }

    Ok(())
}
