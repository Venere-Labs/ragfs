//! In-memory store for testing without `LanceDB`.
//!
//! This module provides a [`MemoryStore`] that stores chunks and files in memory.
//! It's useful for:
//! - Testing without the `LanceDB` dependency
//! - Development builds with faster compilation
//! - Unit tests that don't need persistence

use async_trait::async_trait;
use chrono::Utc;
use ragfs_core::{Chunk, FileRecord, SearchQuery, SearchResult, StoreError, StoreStats, VectorStore};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

/// In-memory vector store for testing.
///
/// This store keeps all data in memory and provides basic search functionality
/// using brute-force cosine similarity. It's not suitable for production use
/// but is perfect for testing and development.
///
/// # Example
///
/// ```rust
/// use ragfs_store::MemoryStore;
/// use ragfs_core::VectorStore;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let store = MemoryStore::new(384);
/// store.init().await?;
///
/// // Use like any other VectorStore
/// let stats = store.stats().await?;
/// assert_eq!(stats.total_chunks, 0);
/// # Ok(())
/// # }
/// ```
pub struct MemoryStore {
    dimension: usize,
    chunks: Arc<RwLock<HashMap<Uuid, Chunk>>>,
    files: Arc<RwLock<HashMap<PathBuf, FileRecord>>>,
    initialized: Arc<RwLock<bool>>,
}

impl MemoryStore {
    /// Create a new in-memory store with the given embedding dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            chunks: Arc::new(RwLock::new(HashMap::new())),
            files: Arc::new(RwLock::new(HashMap::new())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new(384)
    }
}

#[async_trait]
impl VectorStore for MemoryStore {
    async fn init(&self) -> Result<(), StoreError> {
        let mut initialized = self.initialized.write().await;
        *initialized = true;
        debug!("MemoryStore initialized (dimension: {})", self.dimension);
        Ok(())
    }

    async fn upsert_chunks(&self, chunks: &[Chunk]) -> Result<(), StoreError> {
        let mut store = self.chunks.write().await;
        for chunk in chunks {
            store.insert(chunk.id, chunk.clone());
        }
        debug!("Upserted {} chunks", chunks.len());
        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
        let chunks = self.chunks.read().await;
        let mut results: Vec<(f32, &Chunk)> = Vec::new();

        // Brute force search with cosine similarity
        for chunk in chunks.values() {
            if let Some(embedding) = &chunk.embedding {
                let score = Self::cosine_similarity(&query.embedding, embedding);
                results.push((score, chunk));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k (limit)
        let top_k = results
            .into_iter()
            .take(query.limit)
            .map(|(score, chunk)| SearchResult {
                chunk_id: chunk.id,
                file_path: chunk.file_path.clone(),
                content: chunk.content.clone(),
                score,
                byte_range: chunk.byte_range.clone(),
                line_range: chunk.line_range.clone(),
                metadata: chunk.metadata.extra.clone(),
            })
            .collect();

        Ok(top_k)
    }

    async fn hybrid_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
        // For memory store, hybrid search is the same as vector search
        // A real implementation would combine FTS and vector scores
        self.search(query).await
    }

    async fn delete_by_file_path(&self, path: &Path) -> Result<u64, StoreError> {
        let mut chunks = self.chunks.write().await;
        let mut files = self.files.write().await;

        let before = chunks.len();
        chunks.retain(|_, chunk| chunk.file_path != path);
        let deleted = (before - chunks.len()) as u64;

        files.remove(path);

        debug!("Deleted {} chunks for {:?}", deleted, path);
        Ok(deleted)
    }

    async fn update_file_path(&self, from: &Path, to: &Path) -> Result<u64, StoreError> {
        let mut chunks = self.chunks.write().await;
        let mut files = self.files.write().await;
        let mut updated = 0u64;

        // Update chunks
        for chunk in chunks.values_mut() {
            if chunk.file_path == from {
                chunk.file_path = to.to_path_buf();
                updated += 1;
            }
        }

        // Update file record
        if let Some(mut record) = files.remove(from) {
            record.path = to.to_path_buf();
            files.insert(to.to_path_buf(), record);
        }

        debug!("Updated {} chunks from {:?} to {:?}", updated, from, to);
        Ok(updated)
    }

    async fn get_chunks_for_file(&self, path: &Path) -> Result<Vec<Chunk>, StoreError> {
        let chunks = self.chunks.read().await;
        let file_chunks: Vec<Chunk> = chunks
            .values()
            .filter(|chunk| chunk.file_path == path)
            .cloned()
            .collect();
        Ok(file_chunks)
    }

    async fn get_file(&self, path: &Path) -> Result<Option<FileRecord>, StoreError> {
        let files = self.files.read().await;
        Ok(files.get(path).cloned())
    }

    async fn upsert_file(&self, record: &FileRecord) -> Result<(), StoreError> {
        let mut files = self.files.write().await;
        files.insert(record.path.clone(), record.clone());
        debug!("Upserted file record for {:?}", record.path);
        Ok(())
    }

    async fn stats(&self) -> Result<StoreStats, StoreError> {
        let chunks = self.chunks.read().await;
        let files = self.files.read().await;

        Ok(StoreStats {
            total_chunks: chunks.len() as u64,
            total_files: files.len() as u64,
            index_size_bytes: 0, // In-memory, no disk usage
            last_updated: Some(Utc::now()),
        })
    }

    async fn get_all_chunks(&self) -> Result<Vec<Chunk>, StoreError> {
        let chunks = self.chunks.read().await;
        Ok(chunks.values().cloned().collect())
    }

    async fn get_all_files(&self) -> Result<Vec<FileRecord>, StoreError> {
        let files = self.files.read().await;
        Ok(files.values().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragfs_core::{ChunkMetadata, ContentType};

    fn create_test_chunk(id: Uuid, file_id: Uuid, path: &str, embedding: Vec<f32>) -> Chunk {
        Chunk {
            id,
            file_id,
            file_path: PathBuf::from(path),
            content: "test content".to_string(),
            content_type: ContentType::Text,
            mime_type: Some("text/plain".to_string()),
            chunk_index: 0,
            byte_range: 0..12,
            line_range: Some(0..1),
            parent_chunk_id: None,
            depth: 0,
            embedding: Some(embedding),
            metadata: ChunkMetadata::default(),
        }
    }

    #[tokio::test]
    async fn test_memory_store_new() {
        let store = MemoryStore::new(384);
        assert_eq!(store.dimension, 384);
    }

    #[tokio::test]
    async fn test_memory_store_init() {
        let store = MemoryStore::new(384);
        let result = store.init().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_memory_store_upsert_and_stats() {
        let store = MemoryStore::new(3);
        store.init().await.unwrap();

        let file_id = Uuid::new_v4();
        let chunks = vec![
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![1.0, 0.0, 0.0]),
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![0.0, 1.0, 0.0]),
        ];

        store.upsert_chunks(&chunks).await.unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 2);
    }

    #[tokio::test]
    async fn test_memory_store_search() {
        let store = MemoryStore::new(3);
        store.init().await.unwrap();

        let file_id = Uuid::new_v4();
        let chunk1_id = Uuid::new_v4();
        let chunks = vec![
            create_test_chunk(chunk1_id, file_id, "/test/file.txt", vec![1.0, 0.0, 0.0]),
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![0.0, 1.0, 0.0]),
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![0.0, 0.0, 1.0]),
        ];

        store.upsert_chunks(&chunks).await.unwrap();

        let query = SearchQuery {
            embedding: vec![1.0, 0.0, 0.0],
            text: None,
            limit: 2,
            filters: vec![],
            metric: Default::default(),
        };

        let results = store.search(query).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, chunk1_id);
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_memory_store_delete_by_file_path() {
        let store = MemoryStore::new(3);
        store.init().await.unwrap();

        let chunks = vec![
            create_test_chunk(Uuid::new_v4(), Uuid::new_v4(), "/test/file1.txt", vec![1.0, 0.0, 0.0]),
            create_test_chunk(Uuid::new_v4(), Uuid::new_v4(), "/test/file2.txt", vec![0.0, 1.0, 0.0]),
        ];

        store.upsert_chunks(&chunks).await.unwrap();

        let deleted = store
            .delete_by_file_path(Path::new("/test/file1.txt"))
            .await
            .unwrap();
        assert_eq!(deleted, 1);

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 1);
    }

    #[tokio::test]
    async fn test_memory_store_get_all_chunks() {
        let store = MemoryStore::new(3);
        store.init().await.unwrap();

        let file_id = Uuid::new_v4();
        let chunks = vec![
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![1.0, 0.0, 0.0]),
            create_test_chunk(Uuid::new_v4(), file_id, "/test/file.txt", vec![0.0, 1.0, 0.0]),
        ];

        store.upsert_chunks(&chunks).await.unwrap();

        let all_chunks = store.get_all_chunks().await.unwrap();
        assert_eq!(all_chunks.len(), 2);
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vector = 1.0
        let sim = MemoryStore::cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors = 0.0
        let sim = MemoryStore::cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);

        // Opposite vectors = -1.0
        let sim = MemoryStore::cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 0.001);
    }
}
