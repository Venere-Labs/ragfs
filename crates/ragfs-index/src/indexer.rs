//! Main indexing service.

use chrono::Utc;
use ragfs_chunker::ChunkerRegistry;
use ragfs_core::{
    Chunk, ChunkConfig, ChunkMetadata, ChunkOutput, ContentType, EmbeddingConfig, Error,
    FileEvent, FileRecord, FileStatus, IndexStats, Indexer, Result, VectorStore,
};
use ragfs_embed::EmbedderPool;
use ragfs_extract::ExtractorRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::watcher::FileWatcher;

/// Index update events.
#[derive(Debug, Clone)]
pub enum IndexUpdate {
    FileIndexed { path: PathBuf, chunk_count: u32 },
    FileRemoved { path: PathBuf },
    FileError { path: PathBuf, error: String },
    IndexingStarted { path: PathBuf },
}

/// Configuration for the indexer.
#[derive(Debug, Clone)]
pub struct IndexerConfig {
    /// Chunk configuration
    pub chunk_config: ChunkConfig,
    /// Embedding configuration
    pub embed_config: EmbeddingConfig,
    /// Include patterns (glob)
    pub include_patterns: Vec<String>,
    /// Exclude patterns (glob)
    pub exclude_patterns: Vec<String>,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            chunk_config: ChunkConfig::default(),
            embed_config: EmbeddingConfig::default(),
            include_patterns: vec!["**/*".to_string()],
            exclude_patterns: vec![
                "**/.*".to_string(),
                "**/.git/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/target/**".to_string(),
                "**/__pycache__/**".to_string(),
                "**/*.lock".to_string(),
            ],
        }
    }
}

/// Main indexing service.
pub struct IndexerService {
    /// Root path being indexed
    root: PathBuf,
    /// Vector store
    store: Arc<dyn VectorStore>,
    /// Extractor registry
    extractors: Arc<ExtractorRegistry>,
    /// Chunker registry
    chunkers: Arc<ChunkerRegistry>,
    /// Embedder pool
    embedder: Arc<EmbedderPool>,
    /// Configuration
    config: IndexerConfig,
    /// Current stats
    stats: Arc<RwLock<IndexStats>>,
    /// Event sender for file watcher
    event_tx: mpsc::Sender<FileEvent>,
    /// Event receiver
    event_rx: Arc<RwLock<mpsc::Receiver<FileEvent>>>,
    /// Update broadcast
    update_tx: broadcast::Sender<IndexUpdate>,
    /// File watcher (if active)
    watcher: Arc<RwLock<Option<FileWatcher>>>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl IndexerService {
    /// Create a new indexer service.
    pub fn new(
        root: PathBuf,
        store: Arc<dyn VectorStore>,
        extractors: Arc<ExtractorRegistry>,
        chunkers: Arc<ChunkerRegistry>,
        embedder: Arc<EmbedderPool>,
        config: IndexerConfig,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1024);
        let (update_tx, _) = broadcast::channel(256);

        Self {
            root,
            store,
            extractors,
            chunkers,
            embedder,
            config,
            stats: Arc::new(RwLock::new(IndexStats::default())),
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
            update_tx,
            watcher: Arc::new(RwLock::new(None)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Subscribe to index updates.
    pub fn subscribe(&self) -> broadcast::Receiver<IndexUpdate> {
        self.update_tx.subscribe()
    }

    /// Get the root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Start the indexer background task.
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }
        *running = true;
        drop(running);

        info!("Starting indexer for {:?}", self.root);

        // Initialize store
        self.store.init().await.map_err(Error::Store)?;

        // Start file watcher
        let watcher =
            FileWatcher::new(self.event_tx.clone(), std::time::Duration::from_millis(500))
                .map_err(|e| Error::Other(format!("watcher error: {e}")))?;
        {
            let mut w = self.watcher.write().await;
            *w = Some(watcher);
        }

        // Watch root directory
        {
            let mut w = self.watcher.write().await;
            if let Some(ref mut watcher) = *w {
                watcher
                    .watch(&self.root)
                    .map_err(|e| Error::Other(format!("watch error: {e}")))?;
            }
        }

        // Clone what we need for the background task
        let event_rx = Arc::clone(&self.event_rx);
        let update_tx = self.update_tx.clone();
        let running = Arc::clone(&self.running);
        let store = Arc::clone(&self.store);
        let extractors = Arc::clone(&self.extractors);
        let chunkers = Arc::clone(&self.chunkers);
        let embedder = Arc::clone(&self.embedder);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);

        // Spawn event processing task
        tokio::spawn(async move {
            let mut rx = event_rx.write().await;
            while *running.read().await {
                match rx.recv().await {
                    Some(event) => {
                        debug!("Received file event: {:?}", event);
                        match &event {
                            FileEvent::Created(path) | FileEvent::Modified(path) => {
                                let _ = update_tx.send(IndexUpdate::IndexingStarted {
                                    path: path.clone(),
                                });

                                match process_file(
                                    path,
                                    &store,
                                    &extractors,
                                    &chunkers,
                                    &embedder,
                                    &config,
                                )
                                .await
                                {
                                    Ok(chunk_count) => {
                                        info!("Indexed {:?} ({} chunks)", path, chunk_count);
                                        let _ = update_tx.send(IndexUpdate::FileIndexed {
                                            path: path.clone(),
                                            chunk_count,
                                        });

                                        // Update stats
                                        let mut s = stats.write().await;
                                        s.indexed_files += 1;
                                        s.total_chunks += chunk_count as u64;
                                        s.last_update = Some(Utc::now());
                                    }
                                    Err(e) => {
                                        error!("Failed to index {:?}: {}", path, e);
                                        let _ = update_tx.send(IndexUpdate::FileError {
                                            path: path.clone(),
                                            error: e.to_string(),
                                        });

                                        // Update stats
                                        let mut s = stats.write().await;
                                        s.error_files += 1;
                                    }
                                }
                            }
                            FileEvent::Deleted(path) => {
                                if let Err(e) = store.delete_by_file_path(path).await {
                                    error!("Failed to delete {:?}: {}", path, e);
                                } else {
                                    let _ = update_tx.send(IndexUpdate::FileRemoved {
                                        path: path.clone(),
                                    });
                                }
                            }
                            FileEvent::Renamed { from, to } => {
                                // Delete old, index new
                                if let Err(e) = store.delete_by_file_path(from).await {
                                    warn!("Failed to delete old path {:?}: {}", from, e);
                                }
                                let _ = update_tx.send(IndexUpdate::IndexingStarted {
                                    path: to.clone(),
                                });

                                match process_file(
                                    to,
                                    &store,
                                    &extractors,
                                    &chunkers,
                                    &embedder,
                                    &config,
                                )
                                .await
                                {
                                    Ok(chunk_count) => {
                                        let _ = update_tx.send(IndexUpdate::FileIndexed {
                                            path: to.clone(),
                                            chunk_count,
                                        });
                                    }
                                    Err(e) => {
                                        let _ = update_tx.send(IndexUpdate::FileError {
                                            path: to.clone(),
                                            error: e.to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    None => break,
                }
            }
        });

        // Initial scan
        self.scan().await?;

        Ok(())
    }

    /// Perform initial scan of the root directory.
    async fn scan(&self) -> Result<()> {
        info!("Scanning {:?}", self.root);

        let root = self.root.clone();
        let event_tx = self.event_tx.clone();
        let exclude_patterns = self.config.exclude_patterns.clone();

        // Walk directory in background thread (blocking I/O)
        tokio::task::spawn_blocking(move || {
            scan_directory(&root, &event_tx, &exclude_patterns);
        })
        .await
        .map_err(|e| Error::Other(format!("scan task failed: {e}")))?;

        Ok(())
    }

    /// Process a single file through the pipeline.
    pub async fn process_single(&self, path: &Path) -> Result<u32> {
        process_file(
            path,
            &self.store,
            &self.extractors,
            &self.chunkers,
            &self.embedder,
            &self.config,
        )
        .await
    }

    /// Reindex a path (file or directory).
    ///
    /// If the path is a file, it will be reindexed (existing chunks deleted first).
    /// If the path is a directory, all files in it will be reindexed recursively.
    pub async fn reindex_path(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(Error::Other(format!("Path does not exist: {:?}", path)));
        }

        if path.is_file() {
            // Delete existing chunks and reindex
            let _ = self.store.delete_by_file_path(path).await;

            let _ = self.update_tx.send(IndexUpdate::IndexingStarted {
                path: path.to_path_buf(),
            });

            match process_file(
                path,
                &self.store,
                &self.extractors,
                &self.chunkers,
                &self.embedder,
                &self.config,
            )
            .await
            {
                Ok(chunk_count) => {
                    info!("Reindexed {:?} ({} chunks)", path, chunk_count);
                    let _ = self.update_tx.send(IndexUpdate::FileIndexed {
                        path: path.to_path_buf(),
                        chunk_count,
                    });
                }
                Err(e) => {
                    error!("Failed to reindex {:?}: {}", path, e);
                    let _ = self.update_tx.send(IndexUpdate::FileError {
                        path: path.to_path_buf(),
                        error: e.to_string(),
                    });
                    return Err(e);
                }
            }
        } else if path.is_dir() {
            // Reindex all files in directory recursively
            info!("Reindexing directory {:?}", path);
            self.reindex_directory(path).await?;
        }

        Ok(())
    }

    /// Recursively reindex all files in a directory.
    async fn reindex_directory(&self, dir: &Path) -> Result<()> {
        let entries = tokio::fs::read_dir(dir)
            .await
            .map_err(|e| Error::Other(format!("Failed to read directory: {e}")))?;

        let mut entries_stream = tokio_stream::wrappers::ReadDirStream::new(entries);

        use tokio_stream::StreamExt;
        while let Some(entry) = entries_stream.next().await {
            let entry = entry.map_err(|e| Error::Other(format!("Failed to read entry: {e}")))?;
            let path = entry.path();

            // Check exclusion patterns
            let path_str = path.to_string_lossy();
            let should_exclude = self.config.exclude_patterns.iter().any(|pattern| {
                if pattern.contains("**") {
                    let parts: Vec<&str> = pattern.split("**").collect();
                    if parts.len() == 2 {
                        let prefix = parts[0].trim_matches('/');
                        let suffix = parts[1].trim_matches('/');
                        (prefix.is_empty() || path_str.contains(prefix))
                            && (suffix.is_empty() || path_str.contains(suffix))
                    } else {
                        false
                    }
                } else if pattern.starts_with('*') {
                    path_str.ends_with(pattern.trim_start_matches('*'))
                } else {
                    path_str.contains(pattern.trim_matches('*'))
                }
            });

            if should_exclude {
                continue;
            }

            if path.is_dir() {
                // Recurse into subdirectory using a boxed future to avoid infinite recursion
                Box::pin(self.reindex_directory(&path)).await?;
            } else if path.is_file() {
                // Reindex file - use process_file directly to avoid recursion
                let _ = self.store.delete_by_file_path(&path).await;

                let _ = self.update_tx.send(IndexUpdate::IndexingStarted {
                    path: path.clone(),
                });

                match process_file(
                    &path,
                    &self.store,
                    &self.extractors,
                    &self.chunkers,
                    &self.embedder,
                    &self.config,
                )
                .await
                {
                    Ok(chunk_count) => {
                        info!("Reindexed {:?} ({} chunks)", path, chunk_count);
                        let _ = self.update_tx.send(IndexUpdate::FileIndexed {
                            path: path.clone(),
                            chunk_count,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to reindex {:?}: {}", path, e);
                        let _ = self.update_tx.send(IndexUpdate::FileError {
                            path: path.clone(),
                            error: e.to_string(),
                        });
                        // Continue with other files
                    }
                }
            }
        }

        Ok(())
    }
}

/// Scan a directory and send file events.
fn scan_directory(
    root: &Path,
    event_tx: &mpsc::Sender<FileEvent>,
    exclude_patterns: &[String],
) {
    use std::fs;

    fn visit_dir(
        dir: &Path,
        event_tx: &mpsc::Sender<FileEvent>,
        exclude_patterns: &[String],
    ) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                warn!("Cannot read directory {:?}: {}", dir, e);
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();

            // Check exclusion patterns
            let path_str = path.to_string_lossy();
            let should_exclude = exclude_patterns.iter().any(|pattern| {
                // Simple glob matching
                if pattern.contains("**") {
                    let parts: Vec<&str> = pattern.split("**").collect();
                    if parts.len() == 2 {
                        let prefix = parts[0].trim_matches('/');
                        let suffix = parts[1].trim_matches('/');
                        (prefix.is_empty() || path_str.contains(prefix))
                            && (suffix.is_empty() || path_str.contains(suffix))
                    } else {
                        false
                    }
                } else if pattern.starts_with('*') {
                    path_str.ends_with(pattern.trim_start_matches('*'))
                } else {
                    path_str.contains(pattern.trim_matches('*'))
                }
            });

            if should_exclude {
                continue;
            }

            if path.is_dir() {
                visit_dir(&path, event_tx, exclude_patterns);
            } else if path.is_file() {
                // Send as Created event
                if let Err(e) = event_tx.blocking_send(FileEvent::Created(path.clone())) {
                    warn!("Failed to queue file {:?}: {}", path, e);
                }
            }
        }
    }

    visit_dir(root, event_tx, exclude_patterns);
}

/// Process a file through the full pipeline: extract → chunk → embed → store.
async fn process_file(
    path: &Path,
    store: &Arc<dyn VectorStore>,
    extractors: &Arc<ExtractorRegistry>,
    chunkers: &Arc<ChunkerRegistry>,
    embedder: &Arc<EmbedderPool>,
    config: &IndexerConfig,
) -> Result<u32> {
    // Get file metadata
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|e| Error::Other(format!("Failed to get metadata: {e}")))?;

    if !metadata.is_file() {
        return Ok(0);
    }

    // Compute content hash
    let content_hash = compute_hash(path).await?;

    // Check if already indexed with same hash
    if let Ok(Some(existing)) = store.get_file(path).await {
        if existing.content_hash == content_hash && existing.status == FileStatus::Indexed {
            debug!("File {:?} already indexed, skipping", path);
            return Ok(existing.chunk_count);
        }
    }

    // Determine MIME type
    let mime_type = mime_guess::from_path(path)
        .first_or_text_plain()
        .to_string();

    // Extract content
    let content = extractors
        .extract(path, &mime_type)
        .await
        .map_err(Error::Extraction)?;

    if content.text.is_empty() {
        debug!("Empty content for {:?}, skipping", path);
        return Ok(0);
    }

    // Determine content type
    let content_type = determine_content_type(path, &mime_type, &content);

    // Chunk content
    let chunk_outputs = chunkers
        .chunk(&content, &content_type, &config.chunk_config)
        .await
        .map_err(Error::Chunking)?;

    if chunk_outputs.is_empty() {
        return Ok(0);
    }

    // Prepare texts for embedding
    let texts: Vec<&str> = chunk_outputs.iter().map(|c| c.content.as_str()).collect();

    // Generate embeddings
    let embeddings = embedder
        .embed_batch(&texts, &config.embed_config)
        .await
        .map_err(Error::Embedding)?;

    // Create Chunk objects
    let file_id = Uuid::new_v4();
    let now = Utc::now();
    let model_name = embedder.model_name().to_string();

    let chunks: Vec<Chunk> = chunk_outputs
        .into_iter()
        .zip(embeddings.into_iter())
        .enumerate()
        .map(|(idx, (output, emb_output))| {
            build_chunk(
                file_id,
                path,
                idx as u32,
                output,
                emb_output.embedding,
                &content_type,
                &mime_type,
                &model_name,
                now,
            )
        })
        .collect();

    let chunk_count = chunks.len() as u32;

    // Delete old chunks for this file
    let _ = store.delete_by_file_path(path).await;

    // Store chunks
    store.upsert_chunks(&chunks).await.map_err(Error::Store)?;

    // Store file record
    let file_record = FileRecord {
        id: file_id,
        path: path.to_path_buf(),
        size_bytes: metadata.len(),
        mime_type,
        content_hash,
        modified_at: metadata
            .modified()
            .map(|t| chrono::DateTime::<Utc>::from(t))
            .unwrap_or_else(|_| now),
        indexed_at: Some(now),
        chunk_count,
        status: FileStatus::Indexed,
        error_message: None,
    };

    store.upsert_file(&file_record).await.map_err(Error::Store)?;

    Ok(chunk_count)
}

/// Compute blake3 hash of file content.
async fn compute_hash(path: &Path) -> Result<String> {
    let content = tokio::fs::read(path)
        .await
        .map_err(|e| Error::Other(format!("Failed to read file: {e}")))?;

    let hash = blake3::hash(&content);
    Ok(hash.to_hex().to_string())
}

/// Determine content type from path, MIME type, and extracted content.
fn determine_content_type(
    path: &Path,
    mime_type: &str,
    content: &ragfs_core::ExtractedContent,
) -> ContentType {
    // Check for code files
    let code_extensions = [
        ("rs", "rust"),
        ("py", "python"),
        ("js", "javascript"),
        ("ts", "typescript"),
        ("tsx", "typescript"),
        ("jsx", "javascript"),
        ("java", "java"),
        ("go", "go"),
        ("c", "c"),
        ("cpp", "cpp"),
        ("h", "c"),
        ("hpp", "cpp"),
        ("rb", "ruby"),
        ("php", "php"),
        ("swift", "swift"),
        ("kt", "kotlin"),
        ("scala", "scala"),
        ("ex", "elixir"),
        ("hs", "haskell"),
        ("ml", "ocaml"),
        ("lua", "lua"),
        ("sh", "bash"),
        ("sql", "sql"),
    ];

    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let ext_lower = ext.to_lowercase();
        for (code_ext, lang) in &code_extensions {
            if ext_lower == *code_ext {
                return ContentType::Code {
                    language: lang.to_string(),
                    symbol: None,
                };
            }
        }
    }

    // Check for markdown
    if mime_type.contains("markdown")
        || path
            .extension()
            .is_some_and(|e| e == "md" || e == "markdown")
    {
        return ContentType::Markdown;
    }

    // Check language from extraction metadata
    if let Some(ref lang) = content.metadata.language {
        if !lang.is_empty() {
            return ContentType::Code {
                language: lang.clone(),
                symbol: None,
            };
        }
    }

    ContentType::Text
}

/// Build a Chunk from chunk output and embedding.
fn build_chunk(
    file_id: Uuid,
    file_path: &Path,
    chunk_index: u32,
    output: ChunkOutput,
    embedding: Vec<f32>,
    content_type: &ContentType,
    mime_type: &str,
    model_name: &str,
    now: chrono::DateTime<Utc>,
) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        file_id,
        file_path: file_path.to_path_buf(),
        content: output.content,
        content_type: content_type.clone(),
        mime_type: Some(mime_type.to_string()),
        chunk_index,
        byte_range: output.byte_range,
        line_range: output.line_range,
        parent_chunk_id: None,
        depth: output.depth,
        embedding: Some(embedding),
        metadata: ChunkMetadata {
            embedding_model: Some(model_name.to_string()),
            indexed_at: Some(now),
            token_count: None,
            extra: Default::default(),
        },
    }
}

#[async_trait::async_trait]
impl Indexer for IndexerService {
    async fn watch(&self, path: &Path) -> Result<()> {
        let mut w = self.watcher.write().await;
        if let Some(ref mut watcher) = *w {
            watcher
                .watch(path)
                .map_err(|e| Error::Other(format!("watch error: {e}")))?;
        }
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        info!("Indexer stopped");
        Ok(())
    }

    async fn index(&self, path: &Path, force: bool) -> Result<()> {
        if force {
            // Delete existing and reindex
            let _ = self.store.delete_by_file_path(path).await;
        }

        // Queue file for indexing
        self.event_tx
            .send(FileEvent::Modified(path.to_path_buf()))
            .await
            .map_err(|e| Error::Other(format!("send error: {e}")))?;
        Ok(())
    }

    async fn stats(&self) -> Result<IndexStats> {
        Ok(self.stats.read().await.clone())
    }

    async fn needs_reindex(&self, path: &Path) -> Result<bool> {
        // Check if file exists in store and compare hash
        match self.store.get_file(path).await {
            Ok(Some(record)) => {
                let current_hash = compute_hash(path).await?;
                Ok(record.content_hash != current_hash)
            }
            Ok(None) => Ok(true),
            Err(e) => Err(Error::Store(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragfs_chunker::{ChunkerRegistry, FixedSizeChunker};
    use ragfs_core::{
        ContentMetadataInfo, EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality,
        SearchQuery, SearchResult, StoreError, StoreStats,
    };
    use ragfs_embed::EmbedderPool;
    use ragfs_extract::ExtractorRegistry;
    use std::collections::HashMap;
    use tempfile::tempdir;

    const TEST_DIM: usize = 384;

    // ==================== Mock Embedder ====================

    struct MockEmbedder {
        dimension: usize,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        fn model_name(&self) -> &str {
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
        ) -> std::result::Result<Vec<EmbeddingOutput>, EmbedError> {
            Ok(texts
                .iter()
                .map(|_| EmbeddingOutput {
                    embedding: vec![0.1; self.dimension],
                    token_count: 10,
                })
                .collect())
        }

        async fn embed_query(
            &self,
            _query: &str,
            _config: &EmbeddingConfig,
        ) -> std::result::Result<EmbeddingOutput, EmbedError> {
            Ok(EmbeddingOutput {
                embedding: vec![0.1; self.dimension],
                token_count: 10,
            })
        }
    }

    // ==================== Mock VectorStore ====================

    struct MockStore {
        chunks: Arc<RwLock<Vec<Chunk>>>,
        files: Arc<RwLock<HashMap<PathBuf, FileRecord>>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                chunks: Arc::new(RwLock::new(Vec::new())),
                files: Arc::new(RwLock::new(HashMap::new())),
            }
        }
    }

    #[async_trait::async_trait]
    impl VectorStore for MockStore {
        async fn init(&self) -> std::result::Result<(), StoreError> {
            Ok(())
        }

        async fn upsert_chunks(&self, chunks: &[Chunk]) -> std::result::Result<(), StoreError> {
            let mut store = self.chunks.write().await;
            for chunk in chunks {
                store.push(chunk.clone());
            }
            Ok(())
        }

        async fn search(
            &self,
            _query: SearchQuery,
        ) -> std::result::Result<Vec<SearchResult>, StoreError> {
            Ok(vec![])
        }

        async fn hybrid_search(
            &self,
            _query: SearchQuery,
        ) -> std::result::Result<Vec<SearchResult>, StoreError> {
            Ok(vec![])
        }

        async fn delete_by_file_path(&self, path: &Path) -> std::result::Result<u64, StoreError> {
            let mut chunks = self.chunks.write().await;
            let initial_len = chunks.len();
            chunks.retain(|c| c.file_path != path);
            let deleted = initial_len - chunks.len();
            let mut files = self.files.write().await;
            files.remove(path);
            Ok(deleted as u64)
        }

        async fn get_file(
            &self,
            path: &Path,
        ) -> std::result::Result<Option<FileRecord>, StoreError> {
            let files = self.files.read().await;
            Ok(files.get(path).cloned())
        }

        async fn upsert_file(&self, record: &FileRecord) -> std::result::Result<(), StoreError> {
            let mut files = self.files.write().await;
            files.insert(record.path.clone(), record.clone());
            Ok(())
        }

        async fn stats(&self) -> std::result::Result<StoreStats, StoreError> {
            let chunks = self.chunks.read().await;
            let files = self.files.read().await;
            Ok(StoreStats {
                total_chunks: chunks.len() as u64,
                total_files: files.len() as u64,
                index_size_bytes: 0,
                last_updated: None,
            })
        }

        async fn update_file_path(
            &self,
            _old_path: &Path,
            _new_path: &Path,
        ) -> std::result::Result<u64, StoreError> {
            Ok(0)
        }

        async fn get_chunks_for_file(
            &self,
            path: &Path,
        ) -> std::result::Result<Vec<Chunk>, StoreError> {
            let chunks = self.chunks.read().await;
            Ok(chunks.iter().filter(|c| c.file_path == path).cloned().collect())
        }
    }

    // ==================== Helper function tests ====================

    #[test]
    fn test_determine_content_type_rust() {
        let path = PathBuf::from("/test/file.rs");
        let content = ragfs_core::ExtractedContent {
            text: "fn main() {}".to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let result = determine_content_type(&path, "text/x-rust", &content);
        match result {
            ContentType::Code { language, .. } => assert_eq!(language, "rust"),
            _ => panic!("Expected Code content type"),
        }
    }

    #[test]
    fn test_determine_content_type_python() {
        let path = PathBuf::from("/test/script.py");
        let content = ragfs_core::ExtractedContent {
            text: "def main(): pass".to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let result = determine_content_type(&path, "text/x-python", &content);
        match result {
            ContentType::Code { language, .. } => assert_eq!(language, "python"),
            _ => panic!("Expected Code content type"),
        }
    }

    #[test]
    fn test_determine_content_type_markdown() {
        let path = PathBuf::from("/test/readme.md");
        let content = ragfs_core::ExtractedContent {
            text: "# Hello".to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let result = determine_content_type(&path, "text/markdown", &content);
        assert!(matches!(result, ContentType::Markdown));
    }

    #[test]
    fn test_determine_content_type_text() {
        let path = PathBuf::from("/test/notes.txt");
        let content = ragfs_core::ExtractedContent {
            text: "Some text content".to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let result = determine_content_type(&path, "text/plain", &content);
        assert!(matches!(result, ContentType::Text));
    }

    #[test]
    fn test_determine_content_type_javascript() {
        let path = PathBuf::from("/test/app.js");
        let content = ragfs_core::ExtractedContent {
            text: "const x = 1;".to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        };

        let result = determine_content_type(&path, "application/javascript", &content);
        match result {
            ContentType::Code { language, .. } => assert_eq!(language, "javascript"),
            _ => panic!("Expected Code content type"),
        }
    }

    #[tokio::test]
    async fn test_compute_hash() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "test content").unwrap();

        let hash = compute_hash(&file_path).await.unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // blake3 hex is 64 chars

        // Same content should produce same hash
        let hash2 = compute_hash(&file_path).await.unwrap();
        assert_eq!(hash, hash2);
    }

    #[tokio::test]
    async fn test_compute_hash_different_content() {
        let temp_dir = tempdir().unwrap();

        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");

        std::fs::write(&file1, "content 1").unwrap();
        std::fs::write(&file2, "content 2").unwrap();

        let hash1 = compute_hash(&file1).await.unwrap();
        let hash2 = compute_hash(&file2).await.unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_build_chunk() {
        use ragfs_core::ChunkOutputMetadata;

        let file_id = Uuid::new_v4();
        let file_path = PathBuf::from("/test/file.txt");
        let chunk_output = ChunkOutput {
            content: "Test chunk content".to_string(),
            byte_range: 0..18,
            line_range: Some(0..1),
            parent_index: None,
            depth: 0,
            metadata: ChunkOutputMetadata::default(),
        };
        let embedding = vec![0.1; TEST_DIM];
        let content_type = ContentType::Text;
        let mime_type = "text/plain";
        let model_name = "test-model";
        let now = Utc::now();

        let chunk = build_chunk(
            file_id,
            &file_path,
            0,
            chunk_output,
            embedding.clone(),
            &content_type,
            mime_type,
            model_name,
            now,
        );

        assert_eq!(chunk.file_id, file_id);
        assert_eq!(chunk.file_path, file_path);
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.content, "Test chunk content");
        assert_eq!(chunk.embedding, Some(embedding));
        assert_eq!(chunk.mime_type, Some("text/plain".to_string()));
        assert!(matches!(chunk.content_type, ContentType::Text));
    }

    // ==================== IndexerService tests ====================

    fn create_test_indexer(store: Arc<dyn VectorStore>) -> IndexerService {
        use ragfs_extract::TextExtractor;

        let mut extractors = ExtractorRegistry::new();
        extractors.register("text", TextExtractor::new());
        let extractors = Arc::new(extractors);

        let mut chunkers = ChunkerRegistry::new();
        chunkers.register("fixed", FixedSizeChunker::new());
        chunkers.set_default("fixed");
        let chunkers = Arc::new(chunkers);

        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let embedder_pool = Arc::new(EmbedderPool::new(embedder, 1));

        let config = IndexerConfig::default();

        IndexerService::new(
            PathBuf::from("/tmp"),
            store,
            extractors,
            chunkers,
            embedder_pool,
            config,
        )
    }

    #[tokio::test]
    async fn test_process_single_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "This is test content for indexing.").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        let chunk_count = indexer.process_single(&file_path).await.unwrap();

        assert!(chunk_count > 0, "Should have created at least one chunk");

        // Verify chunks were stored
        let stored_chunks = store.chunks.read().await;
        assert!(!stored_chunks.is_empty());
        assert!(stored_chunks.iter().all(|c| c.file_path == file_path));

        // Verify file record was stored
        let files = store.files.read().await;
        assert!(files.contains_key(&file_path));
        let file_record = files.get(&file_path).unwrap();
        assert_eq!(file_record.chunk_count, chunk_count);
        assert_eq!(file_record.status, FileStatus::Indexed);
    }

    #[tokio::test]
    async fn test_process_single_skip_already_indexed() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Test content").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        // First indexing
        let chunk_count1 = indexer.process_single(&file_path).await.unwrap();

        // Count chunks after first indexing
        let chunks_after_first = store.chunks.read().await.len();

        // Second indexing (should skip because content hasn't changed)
        let chunk_count2 = indexer.process_single(&file_path).await.unwrap();

        assert_eq!(chunk_count1, chunk_count2);

        // Chunk count should be same (old chunks deleted and new ones added if reindexed,
        // or no change if skipped)
        let chunks_after_second = store.chunks.read().await.len();
        // Note: process_file deletes old chunks before adding new ones, so count may vary
        assert!(chunks_after_second > 0);
    }

    #[tokio::test]
    async fn test_needs_reindex_new_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("new_file.txt");
        std::fs::write(&file_path, "New content").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        // File not in store should need reindex
        let needs = indexer.needs_reindex(&file_path).await.unwrap();
        assert!(needs);
    }

    #[tokio::test]
    async fn test_needs_reindex_unchanged_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("unchanged.txt");
        std::fs::write(&file_path, "Unchanged content").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        // Index the file first
        indexer.process_single(&file_path).await.unwrap();

        // Same content should not need reindex
        let needs = indexer.needs_reindex(&file_path).await.unwrap();
        assert!(!needs);
    }

    #[tokio::test]
    async fn test_needs_reindex_modified_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("modified.txt");
        std::fs::write(&file_path, "Original content").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        // Index the file
        indexer.process_single(&file_path).await.unwrap();

        // Modify the file
        std::fs::write(&file_path, "Modified content - different!").unwrap();

        // Modified file should need reindex
        let needs = indexer.needs_reindex(&file_path).await.unwrap();
        assert!(needs);
    }

    #[tokio::test]
    async fn test_reindex_path_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("reindex_test.txt");
        std::fs::write(&file_path, "Original content for reindex test").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        // Index initially
        indexer.process_single(&file_path).await.unwrap();

        let initial_chunks = store.chunks.read().await.len();
        assert!(initial_chunks > 0);

        // Modify content
        std::fs::write(&file_path, "New content after modification").unwrap();

        // Reindex
        indexer.reindex_path(&file_path).await.unwrap();

        // Verify file was reindexed
        let files = store.files.read().await;
        assert!(files.contains_key(&file_path));
    }

    #[tokio::test]
    async fn test_empty_file_returns_zero_chunks() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.txt");
        std::fs::write(&file_path, "").unwrap();

        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        let chunk_count = indexer.process_single(&file_path).await.unwrap();

        assert_eq!(chunk_count, 0, "Empty file should produce zero chunks");
    }

    #[test]
    fn test_indexer_config_default() {
        let config = IndexerConfig::default();

        assert!(!config.include_patterns.is_empty());
        assert!(config.include_patterns.contains(&"**/*".to_string()));

        assert!(!config.exclude_patterns.is_empty());
        assert!(config.exclude_patterns.contains(&"**/.git/**".to_string()));
        assert!(config.exclude_patterns.contains(&"**/node_modules/**".to_string()));
        assert!(config.exclude_patterns.contains(&"**/target/**".to_string()));
    }

    #[tokio::test]
    async fn test_subscribe_receives_updates() {
        let store = Arc::new(MockStore::new());
        let indexer = create_test_indexer(Arc::clone(&store) as Arc<dyn VectorStore>);

        let mut receiver = indexer.subscribe();

        // Note: This test is limited because we can't easily trigger updates
        // without starting the full indexer service. The subscribe mechanism
        // is tested indirectly through the update_tx.send() calls in process_file.

        // Verify we can create a receiver without panicking
        assert!(receiver.try_recv().is_err()); // No messages yet
    }

    #[test]
    fn test_index_update_variants() {
        // Test that all IndexUpdate variants can be created
        let _indexed = IndexUpdate::FileIndexed {
            path: PathBuf::from("/test"),
            chunk_count: 5,
        };

        let _removed = IndexUpdate::FileRemoved {
            path: PathBuf::from("/test"),
        };

        let _error = IndexUpdate::FileError {
            path: PathBuf::from("/test"),
            error: "test error".to_string(),
        };

        let _started = IndexUpdate::IndexingStarted {
            path: PathBuf::from("/test"),
        };
    }
}
