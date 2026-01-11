//! Integration tests for the full RAGFS pipeline.
//!
//! Tests the complete flow: extract → chunk → embed → store → search.

use async_trait::async_trait;
use ragfs_chunker::{ChunkerRegistry, FixedSizeChunker};
use ragfs_core::{
    Chunk, ChunkConfig, ChunkMetadata, ContentType, DistanceMetric, EmbedError, Embedder,
    EmbeddingConfig, EmbeddingOutput, Modality, SearchQuery, VectorStore,
};
use ragfs_extract::{ExtractorRegistry, TextExtractor};
use ragfs_store::LanceStore;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

const TEST_DIM: usize = 384;

/// Mock embedder for testing (avoids model download).
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    fn model_name(&self) -> &'static str {
        "mock-embedder"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        512
    }

    fn modalities(&self) -> &[Modality] {
        &[Modality::Text]
    }

    async fn embed_text(
        &self,
        texts: &[&str],
        _config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        // Generate deterministic embeddings based on text content
        Ok(texts
            .iter()
            .map(|text| {
                let hash = blake3::hash(text.as_bytes());
                let bytes = hash.as_bytes();
                let embedding: Vec<f32> = (0..self.dimension)
                    .map(|i| {
                        let byte_idx = i % 32;
                        (f32::from(bytes[byte_idx]) / 255.0) - 0.5
                    })
                    .collect();
                EmbeddingOutput {
                    embedding,
                    token_count: text.split_whitespace().count(),
                }
            })
            .collect())
    }

    async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        let results = self.embed_text(&[query], config).await?;
        Ok(results.into_iter().next().unwrap())
    }
}

/// Create a chunk from chunk output and embedding.
fn create_chunk(
    file_path: &std::path::Path,
    chunk_output: &ragfs_core::ChunkOutput,
    embedding: Vec<f32>,
    chunk_index: u32,
    token_count: usize,
) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        file_path: file_path.to_path_buf(),
        content: chunk_output.content.clone(),
        content_type: ContentType::Text,
        mime_type: Some("text/plain".to_string()),
        chunk_index,
        byte_range: chunk_output.byte_range.clone(),
        line_range: chunk_output.line_range.clone(),
        parent_chunk_id: None,
        depth: chunk_output.depth,
        embedding: Some(embedding),
        metadata: ChunkMetadata {
            embedding_model: Some("mock-embedder".to_string()),
            indexed_at: Some(chrono::Utc::now()),
            token_count: Some(token_count),
            extra: HashMap::new(),
        },
    }
}

#[tokio::test]
async fn test_full_pipeline_extract_chunk_embed_store_search() {
    // Setup temporary directories
    let source_dir = tempdir().unwrap();
    let db_dir = tempdir().unwrap();

    // Create test files
    let test_content_1 = "This is a document about machine learning and neural networks. \
        Neural networks are a subset of machine learning algorithms. \
        They are inspired by the structure of the human brain.";
    let test_content_2 = "This document discusses database systems and SQL. \
        SQL is used for querying relational databases. \
        PostgreSQL and MySQL are popular database systems.";
    let test_content_3 = "Authentication and authorization are important security concepts. \
        OAuth2 is a popular authentication protocol. \
        JWT tokens are often used for API authentication.";

    let file1 = source_dir.path().join("ml.txt");
    let file2 = source_dir.path().join("database.txt");
    let file3 = source_dir.path().join("security.txt");

    std::fs::write(&file1, test_content_1).unwrap();
    std::fs::write(&file2, test_content_2).unwrap();
    std::fs::write(&file3, test_content_3).unwrap();

    // Create components
    let store = Arc::new(LanceStore::new(db_dir.path().join("test.lance"), TEST_DIM));
    store.init().await.unwrap();

    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());

    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");

    let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
    let config = ChunkConfig {
        target_size: 200,
        max_size: 400,
        overlap: 50,
        hierarchical: false,
        max_depth: 1,
    };
    let embed_config = EmbeddingConfig::default();

    // Process each file through the pipeline
    for file_path in [&file1, &file2, &file3] {
        // 1. Extract
        let content = extractors.extract(file_path, "text/plain").await.unwrap();

        // 2. Chunk
        let chunk_outputs = chunkers
            .chunk(&content, &ContentType::Text, &config)
            .await
            .unwrap();

        // 3. Embed and create Chunks
        let mut chunks = Vec::new();
        for (i, chunk_output) in chunk_outputs.iter().enumerate() {
            let embed_result = embedder
                .embed_text(&[&chunk_output.content], &embed_config)
                .await
                .unwrap();

            chunks.push(create_chunk(
                file_path,
                chunk_output,
                embed_result[0].embedding.clone(),
                i as u32,
                embed_result[0].token_count,
            ));
        }

        // 4. Store
        store.upsert_chunks(&chunks).await.unwrap();
    }

    // Verify stats
    let stats = store.stats().await.unwrap();
    assert!(stats.total_chunks > 0, "Should have stored chunks");
    assert!(stats.index_size_bytes > 0, "Index should have size");

    // 5. Search for "machine learning"
    let query_embedding = embedder
        .embed_query("machine learning neural networks", &embed_config)
        .await
        .unwrap();

    let results = store
        .search(SearchQuery {
            embedding: query_embedding.embedding,
            text: Some("machine learning".to_string()),
            limit: 5,
            metric: DistanceMetric::Cosine,
            filters: vec![],
        })
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should find results for ML query");

    // The first result should be from ml.txt (best semantic match)
    let top_result = &results[0];
    assert!(
        top_result.file_path.to_string_lossy().contains("ml.txt"),
        "Top result should be from ml.txt, got {:?}",
        top_result.file_path
    );

    // 6. Search for "database SQL"
    let query_embedding = embedder
        .embed_query("database SQL PostgreSQL", &embed_config)
        .await
        .unwrap();

    let results = store
        .search(SearchQuery {
            embedding: query_embedding.embedding,
            text: Some("database".to_string()),
            limit: 5,
            metric: DistanceMetric::Cosine,
            filters: vec![],
        })
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should find results for DB query");
    let top_result = &results[0];
    assert!(
        top_result
            .file_path
            .to_string_lossy()
            .contains("database.txt"),
        "Top result should be from database.txt, got {:?}",
        top_result.file_path
    );

    // 7. Search for "authentication"
    let query_embedding = embedder
        .embed_query("authentication OAuth JWT", &embed_config)
        .await
        .unwrap();

    let results = store
        .search(SearchQuery {
            embedding: query_embedding.embedding,
            text: Some("authentication".to_string()),
            limit: 5,
            metric: DistanceMetric::Cosine,
            filters: vec![],
        })
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should find results for auth query");
    let top_result = &results[0];
    assert!(
        top_result
            .file_path
            .to_string_lossy()
            .contains("security.txt"),
        "Top result should be from security.txt, got {:?}",
        top_result.file_path
    );
}

#[tokio::test]
async fn test_pipeline_delete_and_reindex() {
    let source_dir = tempdir().unwrap();
    let db_dir = tempdir().unwrap();

    // Create test file
    let file_path = source_dir.path().join("test.txt");
    std::fs::write(&file_path, "Initial content about Rust programming").unwrap();

    // Setup components
    let store = Arc::new(LanceStore::new(db_dir.path().join("test.lance"), TEST_DIM));
    store.init().await.unwrap();

    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());

    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");

    let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
    let config = ChunkConfig::default();
    let embed_config = EmbeddingConfig::default();

    // Index initial content
    let content = extractors.extract(&file_path, "text/plain").await.unwrap();
    let chunk_outputs = chunkers
        .chunk(&content, &ContentType::Text, &config)
        .await
        .unwrap();

    let mut chunks = Vec::new();
    for (i, chunk_output) in chunk_outputs.iter().enumerate() {
        let embed_result = embedder
            .embed_text(&[&chunk_output.content], &embed_config)
            .await
            .unwrap();

        chunks.push(create_chunk(
            &file_path,
            chunk_output,
            embed_result[0].embedding.clone(),
            i as u32,
            embed_result[0].token_count,
        ));
    }
    store.upsert_chunks(&chunks).await.unwrap();

    // Verify initial indexing
    let stats = store.stats().await.unwrap();
    let initial_chunks = stats.total_chunks;
    assert!(initial_chunks > 0, "Should have indexed chunks");

    // Delete chunks for file
    let deleted = store.delete_by_file_path(&file_path).await.unwrap();
    assert_eq!(deleted, initial_chunks, "Should delete all chunks");

    // Verify deletion
    let stats = store.stats().await.unwrap();
    assert_eq!(
        stats.total_chunks, 0,
        "Should have no chunks after deletion"
    );

    // Update file content and reindex
    std::fs::write(
        &file_path,
        "Updated content about Python and machine learning",
    )
    .unwrap();

    let content = extractors.extract(&file_path, "text/plain").await.unwrap();
    let chunk_outputs = chunkers
        .chunk(&content, &ContentType::Text, &config)
        .await
        .unwrap();

    let mut new_chunks = Vec::new();
    for (i, chunk_output) in chunk_outputs.iter().enumerate() {
        let embed_result = embedder
            .embed_text(&[&chunk_output.content], &embed_config)
            .await
            .unwrap();

        new_chunks.push(create_chunk(
            &file_path,
            chunk_output,
            embed_result[0].embedding.clone(),
            i as u32,
            embed_result[0].token_count,
        ));
    }
    store.upsert_chunks(&new_chunks).await.unwrap();

    // Verify reindexing
    let stats = store.stats().await.unwrap();
    assert!(stats.total_chunks > 0, "Should have chunks after reindex");

    // Search for new content
    let query_embedding = embedder
        .embed_query("Python machine learning", &embed_config)
        .await
        .unwrap();

    let results = store
        .search(SearchQuery {
            embedding: query_embedding.embedding,
            text: Some("Python".to_string()),
            limit: 5,
            metric: DistanceMetric::Cosine,
            filters: vec![],
        })
        .await
        .unwrap();

    assert!(!results.is_empty(), "Should find Python content");
    assert!(
        results[0].content.contains("Python"),
        "Content should contain Python"
    );
}

#[tokio::test]
async fn test_pipeline_hybrid_search() {
    let source_dir = tempdir().unwrap();
    let db_dir = tempdir().unwrap();

    // Create files with specific keywords
    let file1 = source_dir.path().join("rust.txt");
    let file2 = source_dir.path().join("python.txt");

    std::fs::write(&file1, "Rust is a systems programming language focused on safety and performance. Memory safety without garbage collection.").unwrap();
    std::fs::write(&file2, "Python is a high-level programming language known for readability. It has dynamic typing and automatic memory management.").unwrap();

    // Setup
    let store = Arc::new(LanceStore::new(db_dir.path().join("test.lance"), TEST_DIM));
    store.init().await.unwrap();

    let mut extractors = ExtractorRegistry::new();
    extractors.register("text", TextExtractor::new());

    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");

    let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
    let config = ChunkConfig::default();
    let embed_config = EmbeddingConfig::default();

    // Index both files
    for file_path in [&file1, &file2] {
        let content = extractors.extract(file_path, "text/plain").await.unwrap();
        let chunk_outputs = chunkers
            .chunk(&content, &ContentType::Text, &config)
            .await
            .unwrap();

        let mut chunks = Vec::new();
        for (i, chunk_output) in chunk_outputs.iter().enumerate() {
            let embed_result = embedder
                .embed_text(&[&chunk_output.content], &embed_config)
                .await
                .unwrap();

            chunks.push(create_chunk(
                file_path,
                chunk_output,
                embed_result[0].embedding.clone(),
                i as u32,
                embed_result[0].token_count,
            ));
        }
        store.upsert_chunks(&chunks).await.unwrap();
    }

    // Hybrid search for "memory safety"
    let query_embedding = embedder
        .embed_query("memory safety garbage collection", &embed_config)
        .await
        .unwrap();

    let results = store
        .hybrid_search(SearchQuery {
            embedding: query_embedding.embedding,
            text: Some("memory safety".to_string()),
            limit: 5,
            metric: DistanceMetric::Cosine,
            filters: vec![],
        })
        .await
        .unwrap();

    // Hybrid search should return results
    assert!(!results.is_empty(), "Should find results");

    // At least one result should contain our search terms
    // (Note: with mock embeddings, exact ranking may vary)
    let has_memory_safety = results.iter().any(|r| r.content.contains("Memory safety"));
    assert!(
        has_memory_safety,
        "Hybrid search should find content containing 'Memory safety'"
    );

    // Verify results are from our indexed files
    for result in &results {
        let path_str = result.file_path.to_string_lossy();
        assert!(
            path_str.contains("rust.txt") || path_str.contains("python.txt"),
            "Result should be from indexed files, got {:?}",
            result.file_path
        );
    }
}
