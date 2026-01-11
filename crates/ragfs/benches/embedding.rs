//! Benchmarks for embedding generation.
//!
//! Measures embedding throughput in tokens/second for various batch sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ragfs_core::{Embedder, EmbeddingConfig};
use ragfs_embed::CandleEmbedder;
use std::sync::Arc;

/// Sample texts of varying lengths for benchmarking.
const SHORT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const MEDIUM_TEXT: &str = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.";
const LONG_TEXT: &str = "Large language models (LLMs) are advanced artificial intelligence systems trained on vast amounts of text data. These models, such as GPT-4 and Claude, can understand and generate human-like text, making them useful for a wide range of applications including question answering, content generation, code writing, and much more. The training process involves learning patterns, relationships, and contextual understanding from billions of text samples, enabling these models to perform complex language tasks with remarkable accuracy.";

fn create_test_texts(count: usize, text: &str) -> Vec<String> {
    (0..count).map(|_| text.to_string()).collect()
}

fn embedding_benchmark(c: &mut Criterion) {
    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Get cache directory
    let cache_dir = std::env::temp_dir().join("ragfs_bench_models");

    // Create embedder
    let embedder = rt.block_on(async {
        let embedder = CandleEmbedder::new(cache_dir);
        // Initialize (may download model on first run)
        if let Err(e) = embedder.init().await {
            eprintln!("Warning: Could not initialize embedder: {}", e);
            return None;
        }
        Some(Arc::new(embedder) as Arc<dyn Embedder>)
    });

    let Some(embedder) = embedder else {
        eprintln!("Skipping embedding benchmarks: embedder not available");
        return;
    };

    let config = EmbeddingConfig::default();

    let mut group = c.benchmark_group("embedding");

    // Benchmark different batch sizes with short texts
    for batch_size in [1, 10, 50, 100].iter() {
        let texts = create_test_texts(*batch_size, SHORT_TEXT);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Estimate token count (rough approximation: ~4 chars per token)
        let total_tokens = texts.iter().map(|t| t.len() / 4).sum::<usize>() as u64;

        group.throughput(Throughput::Elements(total_tokens));
        group.bench_with_input(
            BenchmarkId::new("short_text", batch_size),
            &text_refs,
            |b, texts| {
                b.to_async(&rt).iter(|| async {
                    let embedder = embedder.clone();
                    let texts = texts.clone();
                    let config = config.clone();
                    black_box(embedder.embed_text(&texts, &config).await)
                });
            },
        );
    }

    // Benchmark different text lengths
    for (name, text) in [
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        let texts = create_test_texts(10, text);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let total_tokens = texts.iter().map(|t| t.len() / 4).sum::<usize>() as u64;

        group.throughput(Throughput::Elements(total_tokens));
        group.bench_with_input(
            BenchmarkId::new("text_length", name),
            &text_refs,
            |b, texts| {
                b.to_async(&rt).iter(|| async {
                    let embedder = embedder.clone();
                    let texts = texts.clone();
                    let config = config.clone();
                    black_box(embedder.embed_text(&texts, &config).await)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, embedding_benchmark);
criterion_main!(benches);
