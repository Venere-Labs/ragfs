# RAGFS API Reference

This document provides a reference for the public APIs of RAGFS crates.

## Table of Contents

- [ragfs-core](#ragfs-core) - Core types and traits
- [ragfs-store](#ragfs-store) - Vector storage (LanceDB)
- [ragfs-embed](#ragfs-embed) - Embedding generation
- [ragfs-extract](#ragfs-extract) - Content extraction
- [ragfs-chunker](#ragfs-chunker) - Text chunking
- [ragfs-index](#ragfs-index) - Indexing pipeline
- [ragfs-query](#ragfs-query) - Query execution
- [ragfs-fuse](#ragfs-fuse) - FUSE filesystem

---

## ragfs-core

Core types, traits, and error definitions used across all RAGFS crates.

### Key Traits

#### `VectorStore`

Storage backend for chunks and file records.

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Initialize the store (create tables, indices).
    async fn init(&self) -> Result<(), StoreError>;

    /// Insert or update chunks.
    async fn upsert_chunks(&self, chunks: &[Chunk]) -> Result<(), StoreError>;

    /// Search for similar chunks.
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError>;

    /// Hybrid search (vector + FTS).
    async fn hybrid_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError>;

    /// Delete chunks by file path.
    async fn delete_by_file_path(&self, path: &Path) -> Result<(), StoreError>;

    /// Get file record.
    async fn get_file(&self, path: &Path) -> Result<Option<FileRecord>, StoreError>;

    /// Upsert file record.
    async fn upsert_file(&self, record: &FileRecord) -> Result<(), StoreError>;

    /// Get store statistics.
    async fn stats(&self) -> Result<StoreStats, StoreError>;
}
```

#### `Embedder`

Interface for embedding generation.

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the maximum token limit.
    fn max_tokens(&self) -> usize;

    /// Get supported modalities.
    fn modalities(&self) -> &[Modality];

    /// Embed multiple texts.
    async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError>;

    /// Embed a single query (may use different prompt).
    async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError>;
}
```

#### `ContentExtractor`

Extract content from files.

```rust
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Get supported MIME types.
    fn supported_types(&self) -> &[&str];

    /// Extract content from a file.
    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError>;
}
```

#### `Chunker`

Split content into chunks.

```rust
#[async_trait]
pub trait Chunker: Send + Sync {
    /// Get the chunker name.
    fn name(&self) -> &str;

    /// Get supported content types.
    fn content_types(&self) -> &[&str];

    /// Chunk the content.
    async fn chunk(
        &self,
        content: &ExtractedContent,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError>;
}
```

### Key Types

#### `Chunk`

A chunk of content from a file.

```rust
pub struct Chunk {
    pub id: Uuid,
    pub file_id: Uuid,
    pub file_path: PathBuf,
    pub content: String,
    pub content_type: ContentType,
    pub mime_type: Option<String>,
    pub chunk_index: u32,
    pub byte_range: Range<u64>,
    pub line_range: Option<Range<u32>>,
    pub parent_chunk_id: Option<Uuid>,
    pub depth: u8,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}
```

#### `SearchQuery`

Query for vector search.

```rust
pub struct SearchQuery {
    pub embedding: Vec<f32>,
    pub text: Option<String>,
    pub limit: usize,
    pub filters: Vec<SearchFilter>,
    pub metric: DistanceMetric,
}
```

#### `SearchResult`

Result from a search.

```rust
pub struct SearchResult {
    pub chunk_id: Uuid,
    pub file_path: PathBuf,
    pub content: String,
    pub score: f32,
    pub byte_range: Range<u64>,
    pub line_range: Option<Range<u32>>,
}
```

---

## ragfs-store

LanceDB-based vector storage implementation.

### `LanceStore`

```rust
impl LanceStore {
    /// Create a new LanceStore.
    pub fn new(db_path: PathBuf, embedding_dim: usize) -> Self;

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize;
}
```

Implements `VectorStore` trait. Features:
- HNSW vector index for fast similarity search
- Full-text search (FTS) index for hybrid search
- Automatic index creation on init

### Usage Example

```rust
use ragfs_store::LanceStore;
use ragfs_core::VectorStore;

let store = LanceStore::new("path/to/db.lance".into(), 384);
store.init().await?;

// Store chunks
store.upsert_chunks(&chunks).await?;

// Search
let results = store.search(query).await?;
```

---

## ragfs-embed

Local embedding generation using Candle.

### `CandleEmbedder`

GTE-small model embedder (384 dimensions).

```rust
impl CandleEmbedder {
    /// Create a new embedder with model cache directory.
    pub fn new(cache_dir: PathBuf) -> Self;

    /// Initialize (downloads model if needed).
    pub async fn init(&self) -> Result<(), EmbedError>;
}
```

### `EmbedderPool`

Thread pool for concurrent embedding.

```rust
impl EmbedderPool {
    /// Create a pool with specified concurrency.
    pub fn new(embedder: Arc<dyn Embedder>, pool_size: usize) -> Self;

    /// Embed a batch of texts.
    pub async fn embed_batch(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError>;

    /// Get the document embedder.
    pub fn document_embedder(&self) -> Arc<dyn Embedder>;
}
```

### `EmbeddingCache`

LRU cache for embeddings to avoid redundant computations.

```rust
impl EmbeddingCache {
    /// Create with default size (10,000 entries).
    pub fn new(embedder: Arc<dyn Embedder>) -> Self;

    /// Create with custom capacity.
    pub fn with_capacity(embedder: Arc<dyn Embedder>, max_size: usize) -> Self;

    /// Embed texts with caching.
    pub async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError>;

    /// Get cache statistics.
    pub async fn stats(&self) -> CacheStats;
}
```

---

## ragfs-extract

Content extraction from various file formats.

### `ExtractorRegistry`

Registry of content extractors.

```rust
impl ExtractorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self;

    /// Register an extractor.
    pub fn register<E: ContentExtractor + 'static>(&mut self, name: &str, extractor: E);

    /// Extract content from a file.
    pub async fn extract(
        &self,
        path: &Path,
        mime_type: &str,
    ) -> Result<ExtractedContent, ExtractError>;
}
```

### Built-in Extractors

- `TextExtractor` - Plain text and markdown files
- `PdfExtractor` - PDF documents (text + embedded image extraction)
- `ImageExtractor` - Images (metadata extraction, optional captioning)

### `ImageCaptioner` Trait

Interface for vision-based image captioning.

```rust
#[async_trait]
pub trait ImageCaptioner: Send + Sync {
    /// Initialize the captioner (load model, etc.).
    async fn init(&self) -> Result<(), CaptionError>;

    /// Generate a caption for image bytes.
    async fn caption(&self, image_data: &[u8]) -> Result<Option<String>, CaptionError>;

    /// Check if the captioner is initialized.
    async fn is_initialized(&self) -> bool;

    /// Get the model name.
    fn model_name(&self) -> &str;
}
```

**Built-in Implementations:**
- `PlaceholderCaptioner` - No-op implementation (returns `None`)

### PDF Image Extraction

The `PdfExtractor` extracts embedded images from PDFs:

- **Supported formats**: JPEG (DCTDecode), PNG (FlateDecode), JPEG2000 (JPXDecode)
- **Color spaces**: RGB, Grayscale, CMYK (auto-converted to RGB)
- **Limits**: 100 images max, 50MB total, 50px minimum dimension

---

## ragfs-chunker

Text chunking strategies.

### `ChunkerRegistry`

Registry of chunkers.

```rust
impl ChunkerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self;

    /// Register a chunker.
    pub fn register<C: Chunker + 'static>(&mut self, name: &str, chunker: C);

    /// Set the default chunker.
    pub fn set_default(&mut self, name: &str);

    /// Chunk content.
    pub async fn chunk(
        &self,
        content: &ExtractedContent,
        content_type: &ContentType,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError>;
}
```

### Built-in Chunkers

- `FixedSizeChunker` - Fixed-size chunks with overlap
- `CodeChunker` - AST-aware chunking for code (Tree-sitter)
- `SemanticChunker` - Semantic boundary-aware chunking

### Configuration

```rust
pub struct ChunkConfig {
    pub target_size: usize,    // Target chunk size in tokens (default: 512)
    pub max_size: usize,       // Maximum chunk size (default: 1024)
    pub overlap: usize,        // Overlap between chunks (default: 64)
    pub hierarchical: bool,    // Enable hierarchical chunking (default: true)
    pub max_depth: u8,         // Maximum hierarchy depth (default: 2)
}
```

---

## ragfs-index

Indexing pipeline and file watching.

### `IndexerService`

Main indexing service.

```rust
impl IndexerService {
    /// Create a new indexer service.
    pub fn new(
        root: PathBuf,
        store: Arc<dyn VectorStore>,
        extractors: Arc<ExtractorRegistry>,
        chunkers: Arc<ChunkerRegistry>,
        embedder: Arc<EmbedderPool>,
        config: IndexerConfig,
    ) -> Self;

    /// Subscribe to index updates.
    pub fn subscribe(&self) -> broadcast::Receiver<IndexUpdate>;

    /// Start the indexer (watch + initial scan).
    pub async fn start(&self) -> Result<()>;

    /// Stop the indexer.
    pub async fn stop(&self) -> Result<()>;

    /// Process a single file.
    pub async fn process_single(&self, path: &Path) -> Result<u32>;

    /// Reindex a path (file or directory).
    pub async fn reindex_path(&self, path: &Path) -> Result<()>;
}
```

### Index Updates

```rust
pub enum IndexUpdate {
    FileIndexed { path: PathBuf, chunk_count: u32 },
    FileRemoved { path: PathBuf },
    FileError { path: PathBuf, error: String },
    IndexingStarted { path: PathBuf },
}
```

---

## ragfs-query

Query parsing and execution.

### `QueryExecutor`

```rust
impl QueryExecutor {
    /// Create a new query executor.
    pub fn new(
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        default_limit: usize,
        hybrid: bool,
    ) -> Self;

    /// Execute a query string.
    pub async fn execute(&self, query: &str) -> Result<Vec<SearchResult>, QueryError>;
}
```

---

## ragfs-fuse

FUSE filesystem implementation.

### `RagFs`

```rust
impl RagFs {
    /// Create a basic passthrough filesystem.
    pub fn new(source: PathBuf) -> Self;

    /// Create a filesystem with RAG capabilities.
    pub fn with_rag(
        source: PathBuf,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        runtime: Handle,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
    ) -> Self;

    /// Get the source directory.
    pub fn source(&self) -> &PathBuf;
}
```

### Virtual Directory Structure

```
/mountpoint/
├── .ragfs/                    # Virtual control directory
│   ├── .index                 # Index statistics (JSON)
│   ├── .config                # Configuration (JSON)
│   ├── .reindex               # Write path to trigger reindex
│   ├── .query/                # Query interface
│   │   └── <query>            # Read returns search results (JSON)
│   ├── .search/               # Search interface (symlinks)
│   ├── .similar/              # Similar files
│   └── .help                  # Usage documentation
├── real_dir/                  # Passthrough to source
└── real_file.txt              # Passthrough to source
```

### Usage

```bash
# Query via filesystem
cat "/mnt/ragfs/.ragfs/.query/how to authenticate"

# Check index status
cat /mnt/ragfs/.ragfs/.index

# Trigger reindex
echo "/path/to/file" > /mnt/ragfs/.ragfs/.reindex
```

---

## Error Types

Each crate defines its own error type:

- `ragfs_core::Error` - General errors
- `ragfs_core::StoreError` - Storage errors
- `ragfs_core::ExtractError` - Extraction errors
- `ragfs_core::ChunkError` - Chunking errors
- `ragfs_core::EmbedError` - Embedding errors
- `ragfs_query::QueryError` - Query errors

---

## Cargo Features

Currently no optional features. All functionality is included by default.

---

## Examples

See the [User Guide](USER_GUIDE.md) for complete usage examples.

### Quick Start

```rust
use ragfs_store::LanceStore;
use ragfs_embed::CandleEmbedder;
use ragfs_core::{VectorStore, Embedder, EmbeddingConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize store
    let store = LanceStore::new("data/index.lance".into(), 384);
    store.init().await?;

    // Initialize embedder
    let embedder = CandleEmbedder::new("data/models".into());
    embedder.init().await?;

    // Use the components...
    Ok(())
}
```
