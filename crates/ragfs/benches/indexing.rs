//! Benchmarks for the indexing pipeline.
//!
//! Measures throughput of extraction, chunking, and the full pipeline.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ragfs_chunker::{ChunkerRegistry, FixedSizeChunker};
use ragfs_core::{ChunkConfig, ContentMetadataInfo, ContentType, ExtractedContent};
use ragfs_extract::ExtractorRegistry;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

/// Sample document content for benchmarking.
const SAMPLE_DOC: &str = r#"
# Introduction to Machine Learning

Machine learning (ML) is a subset of artificial intelligence (AI) that provides systems the ability
to automatically learn and improve from experience without being explicitly programmed.

## Key Concepts

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data. The model makes predictions
based on input features and compares them with known outputs to improve accuracy.

### Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or
intrinsic structures in the input data.

### Reinforcement Learning

Reinforcement learning involves an agent that learns to make decisions by taking actions in an
environment to maximize cumulative reward.

## Applications

- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles

## Conclusion

Machine learning continues to evolve rapidly, with new algorithms and applications emerging regularly.
Understanding the fundamentals helps in choosing the right approach for specific problems.
"#;

/// Generate test content of specified size (in KB).
fn generate_content(size_kb: usize) -> String {
    let base = SAMPLE_DOC;
    let repetitions = (size_kb * 1024) / base.len() + 1;
    base.repeat(repetitions)
}

/// Create test files in the given directory.
fn create_test_files(dir: &Path, file_count: usize, size_kb: usize) {
    let content = generate_content(size_kb);
    for i in 0..file_count {
        let file_path = dir.join(format!("doc_{}.md", i));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }
}

fn extraction_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let temp_dir = tempdir().unwrap();
    create_test_files(temp_dir.path(), 1, 10);
    let _test_file = temp_dir.path().join("doc_0.md");

    let extractors = ExtractorRegistry::new();

    let mut group = c.benchmark_group("extraction");

    // Benchmark text extraction
    for size_kb in [1, 10, 100].iter() {
        let content = generate_content(*size_kb);
        let file_path = temp_dir.path().join(format!("test_{}.md", size_kb));
        std::fs::write(&file_path, &content).unwrap();

        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("text_extract", format!("{}kb", size_kb)),
            &file_path,
            |b, path| {
                b.to_async(&rt).iter(|| async {
                    black_box(extractors.extract(path, "text/markdown").await)
                });
            },
        );
    }

    group.finish();
}

fn chunking_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");
    let chunkers = Arc::new(chunkers);

    let mut group = c.benchmark_group("chunking");

    // Benchmark different content sizes
    for size_kb in [1, 10, 50].iter() {
        let content = generate_content(*size_kb);
        let extracted = ExtractedContent {
            text: content.clone(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let config = ChunkConfig::default();

        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("fixed_chunker", format!("{}kb", size_kb)),
            &extracted,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    black_box(chunkers.chunk(content, &ContentType::Text, &config).await)
                });
            },
        );
    }

    // Benchmark different chunk sizes
    let content = generate_content(10);
    let extracted = ExtractedContent {
        text: content.clone(),
        elements: vec![],
        images: vec![],
        metadata: ContentMetadataInfo::default(),
    };

    for target_size in [256, 512, 1024, 2048].iter() {
        let config = ChunkConfig {
            target_size: *target_size,
            max_size: *target_size * 2,
            overlap: *target_size / 8,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("chunk_size", format!("{}_tokens", target_size)),
            &extracted,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    black_box(chunkers.chunk(content, &ContentType::Text, &config).await)
                });
            },
        );
    }

    group.finish();
}

fn pipeline_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let extractors = Arc::new(ExtractorRegistry::new());
    let mut chunkers = ChunkerRegistry::new();
    chunkers.register("fixed", FixedSizeChunker::new());
    chunkers.set_default("fixed");
    let chunkers = Arc::new(chunkers);

    let temp_dir = tempdir().unwrap();

    let mut group = c.benchmark_group("pipeline");

    // Benchmark extract + chunk pipeline
    for size_kb in [1, 10, 50].iter() {
        let content = generate_content(*size_kb);
        let file_path = temp_dir.path().join(format!("pipeline_{}.md", size_kb));
        std::fs::write(&file_path, &content).unwrap();

        let config = ChunkConfig::default();

        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("extract_and_chunk", format!("{}kb", size_kb)),
            &file_path,
            |b, path| {
                let extractors = extractors.clone();
                let chunkers = chunkers.clone();
                let config = config.clone();

                b.to_async(&rt).iter(|| async {
                    let extracted = extractors.extract(path, "text/markdown").await.unwrap();
                    black_box(
                        chunkers
                            .chunk(&extracted, &ContentType::Text, &config)
                            .await,
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, extraction_benchmark, chunking_benchmark, pipeline_benchmark);
criterion_main!(benches);
