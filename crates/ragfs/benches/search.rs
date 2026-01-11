//! Benchmarks for vector search latency.
//!
//! Measures search latency (p50, p95, p99) across different index sizes.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ragfs_core::{Chunk, ChunkMetadata, ContentType, DistanceMetric, SearchQuery, VectorStore};
use ragfs_store::LanceStore;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

const EMBEDDING_DIM: usize = 384;

/// Create a random embedding vector.
fn create_random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let base = hasher.finish();

    (0..dim)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (base + i as u64).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Create a test chunk with random embedding.
fn create_test_chunk(file_path: &PathBuf, content: &str, chunk_index: u32, seed: u64) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        file_path: file_path.clone(),
        content: content.to_string(),
        content_type: ContentType::Text,
        mime_type: Some("text/plain".to_string()),
        chunk_index,
        byte_range: 0..content.len() as u64,
        line_range: Some(0..1),
        parent_chunk_id: None,
        depth: 0,
        embedding: Some(create_random_embedding(EMBEDDING_DIM, seed)),
        metadata: ChunkMetadata::default(),
    }
}

/// Populate store with test chunks.
async fn populate_store(store: &LanceStore, chunk_count: usize) {
    let chunks: Vec<Chunk> = (0..chunk_count)
        .map(|i| {
            let file_path = PathBuf::from(format!("/test/file_{}.txt", i / 10));
            let content =
                format!("Test content for chunk number {i} with some additional text for variety.");
            create_test_chunk(&file_path, &content, (i % 10) as u32, i as u64)
        })
        .collect();

    // Insert in batches of 100
    for chunk_batch in chunks.chunks(100) {
        store.upsert_chunks(chunk_batch).await.unwrap();
    }
}

fn search_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("search");

    // Benchmark different index sizes
    for chunk_count in &[100, 1_000, 10_000] {
        // Skip large benchmarks in CI
        if *chunk_count > 1_000 && std::env::var("CI").is_ok() {
            continue;
        }

        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("bench.lance");
        let store = LanceStore::new(db_path, EMBEDDING_DIM);

        // Initialize and populate
        rt.block_on(async {
            store.init().await.unwrap();
            populate_store(&store, *chunk_count).await;
        });

        let store = Arc::new(store);
        let query_embedding = create_random_embedding(EMBEDDING_DIM, 12345);

        group.bench_with_input(
            BenchmarkId::new("vector_search", format!("{chunk_count}_chunks")),
            chunk_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let store = store.clone();
                    let query = SearchQuery {
                        text: Some("test content".to_string()),
                        embedding: query_embedding.clone(),
                        limit: 10,
                        filters: vec![],
                        metric: DistanceMetric::Cosine,
                    };
                    black_box(store.search(query).await)
                });
            },
        );

        // Benchmark hybrid search if available
        group.bench_with_input(
            BenchmarkId::new("hybrid_search", format!("{chunk_count}_chunks")),
            chunk_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let store = store.clone();
                    let query = SearchQuery {
                        text: Some("test content".to_string()),
                        embedding: query_embedding.clone(),
                        limit: 10,
                        filters: vec![],
                        metric: DistanceMetric::Cosine,
                    };
                    black_box(store.hybrid_search(query).await)
                });
            },
        );

        // Benchmark with different result limits
        for limit in &[5, 10, 25, 50] {
            if *chunk_count < 10_000 {
                continue; // Only benchmark limits on larger indices
            }

            group.bench_with_input(
                BenchmarkId::new("limit", format!("top_{limit}")),
                limit,
                |b, limit| {
                    b.to_async(&rt).iter(|| async {
                        let store = store.clone();
                        let query = SearchQuery {
                            text: Some("test content".to_string()),
                            embedding: query_embedding.clone(),
                            limit: *limit,
                            filters: vec![],
                            metric: DistanceMetric::Cosine,
                        };
                        black_box(store.search(query).await)
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, search_benchmark);
criterion_main!(benches);
