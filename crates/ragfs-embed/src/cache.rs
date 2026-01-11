//! Embedding cache for avoiding redundant computations.
//!
//! This module provides a simple LRU cache for embeddings based on content hashes.

use ragfs_core::{EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// Maximum number of entries in the cache.
const DEFAULT_CACHE_SIZE: usize = 10_000;

/// A cached embedding entry.
#[derive(Clone)]
struct CacheEntry {
    /// The embedding output
    output: EmbeddingOutput,
    /// Access counter for LRU eviction
    access_count: u64,
}

/// Embedding cache with LRU eviction.
pub struct EmbeddingCache {
    /// The underlying embedder
    embedder: Arc<dyn Embedder>,
    /// Cache map: content hash -> embedding
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Maximum cache size
    max_size: usize,
    /// Global access counter
    access_counter: RwLock<u64>,
    /// Cache statistics
    stats: RwLock<CacheStats>,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries evicted
    pub evictions: u64,
}

impl EmbeddingCache {
    /// Create a new embedding cache with default size.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self::with_capacity(embedder, DEFAULT_CACHE_SIZE)
    }

    /// Create a new embedding cache with specified capacity.
    pub fn with_capacity(embedder: Arc<dyn Embedder>, max_size: usize) -> Self {
        Self {
            embedder,
            cache: RwLock::new(HashMap::new()),
            max_size,
            access_counter: RwLock::new(0),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Compute hash for a text.
    fn hash_text(text: &str) -> String {
        let hash = blake3::hash(text.as_bytes());
        hash.to_hex().to_string()
    }

    /// Get the next access count.
    async fn next_access(&self) -> u64 {
        let mut counter = self.access_counter.write().await;
        *counter += 1;
        *counter
    }

    /// Evict oldest entries if cache is full.
    async fn maybe_evict(&self) {
        let mut cache = self.cache.write().await;

        if cache.len() < self.max_size {
            return;
        }

        // Find entries to evict (oldest 10%)
        let evict_count = (self.max_size / 10).max(1);
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.access_count))
            .collect();
        entries.sort_by_key(|(_, count)| *count);

        let mut stats = self.stats.write().await;
        for (key, _) in entries.into_iter().take(evict_count) {
            cache.remove(&key);
            stats.evictions += 1;
        }
    }

    /// Embed texts with caching.
    pub async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each text
        {
            let cache = self.cache.read().await;
            let mut stats = self.stats.write().await;

            for (i, text) in texts.iter().enumerate() {
                let hash = Self::hash_text(text);
                if let Some(entry) = cache.get(&hash) {
                    stats.hits += 1;
                    results.push(Some(entry.output.clone()));
                } else {
                    stats.misses += 1;
                    uncached_texts.push(*text);
                    uncached_indices.push(i);
                    results.push(None);
                }
            }
        }

        // Embed uncached texts
        if !uncached_texts.is_empty() {
            debug!("Cache miss for {} texts, embedding", uncached_texts.len());

            let new_embeddings = self.embedder.embed_text(&uncached_texts, config).await?;

            // Update cache
            self.maybe_evict().await;

            let mut cache = self.cache.write().await;
            for (text, output) in uncached_texts.iter().zip(new_embeddings.iter()) {
                let hash = Self::hash_text(text);
                let access = self.next_access().await;
                cache.insert(
                    hash,
                    CacheEntry {
                        output: output.clone(),
                        access_count: access,
                    },
                );
            }

            // Fill in results
            for (idx, output) in uncached_indices.into_iter().zip(new_embeddings) {
                results[idx] = Some(output);
            }
        }

        // Unwrap all results (all should be Some now)
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Embed a single query (always bypasses cache for queries).
    pub async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        // Queries typically shouldn't be cached as they're one-off
        self.embedder.embed_query(query, config).await
    }

    /// Get the underlying embedder.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.embedder)
    }

    /// Get cache statistics.
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Get cache size.
    pub async fn size(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Clear the cache.
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.embedder.model_name()
    }

    /// Get supported modalities.
    pub fn modalities(&self) -> &[Modality] {
        self.embedder.modalities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    const TEST_DIM: usize = 384;

    struct MockEmbedder {
        dimension: usize,
        call_count: RwLock<usize>,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                call_count: RwLock::new(0),
            }
        }

        async fn get_call_count(&self) -> usize {
            *self.call_count.read().await
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
            let mut count = self.call_count.write().await;
            *count += 1;

            Ok(texts
                .iter()
                .map(|text| {
                    let hash = blake3::hash(text.as_bytes());
                    let bytes = hash.as_bytes();
                    let embedding: Vec<f32> = (0..self.dimension)
                        .map(|i| f32::from(bytes[i % 32]) / 255.0)
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

    #[tokio::test]
    async fn test_cache_hit() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::new(Arc::clone(&embedder) as Arc<dyn Embedder>);
        let config = EmbeddingConfig::default();

        // First call - cache miss
        let result1 = cache.embed_text(&["hello world"], &config).await.unwrap();
        assert_eq!(embedder.get_call_count().await, 1);

        // Second call - cache hit
        let result2 = cache.embed_text(&["hello world"], &config).await.unwrap();
        assert_eq!(embedder.get_call_count().await, 1); // No additional call

        // Results should be identical
        assert_eq!(result1[0].embedding, result2[0].embedding);

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::new(Arc::clone(&embedder) as Arc<dyn Embedder>);
        let config = EmbeddingConfig::default();

        // Different texts - all misses
        cache.embed_text(&["text one"], &config).await.unwrap();
        cache.embed_text(&["text two"], &config).await.unwrap();
        cache.embed_text(&["text three"], &config).await.unwrap();

        assert_eq!(embedder.get_call_count().await, 3);

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 3);
    }

    #[tokio::test]
    async fn test_batch_with_mixed_cache() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::new(Arc::clone(&embedder) as Arc<dyn Embedder>);
        let config = EmbeddingConfig::default();

        // Prime cache with one text
        cache.embed_text(&["cached text"], &config).await.unwrap();
        assert_eq!(embedder.get_call_count().await, 1);

        // Batch with mix of cached and uncached
        let results = cache
            .embed_text(&["cached text", "new text", "cached text"], &config)
            .await
            .unwrap();

        // Only one new embedding call for "new text"
        assert_eq!(embedder.get_call_count().await, 2);
        assert_eq!(results.len(), 3);

        // First and third results should be identical
        assert_eq!(results[0].embedding, results[2].embedding);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::new(Arc::clone(&embedder) as Arc<dyn Embedder>);
        let config = EmbeddingConfig::default();

        cache.embed_text(&["test"], &config).await.unwrap();
        assert_eq!(cache.size().await, 1);

        cache.clear().await;
        assert_eq!(cache.size().await, 0);

        // Should be a miss now
        cache.embed_text(&["test"], &config).await.unwrap();
        assert_eq!(embedder.get_call_count().await, 2);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::with_capacity(Arc::clone(&embedder) as Arc<dyn Embedder>, 10);
        let config = EmbeddingConfig::default();

        // Fill cache beyond capacity
        for i in 0..15 {
            let text = format!("text number {i}");
            cache.embed_text(&[&text], &config).await.unwrap();
        }

        // Cache should have evicted some entries
        assert!(cache.size().await < 15);

        let stats = cache.stats().await;
        assert!(stats.evictions > 0);
    }

    #[tokio::test]
    async fn test_embedder_properties() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let cache = EmbeddingCache::new(Arc::clone(&embedder) as Arc<dyn Embedder>);

        assert_eq!(cache.dimension(), TEST_DIM);
        assert_eq!(cache.model_name(), "mock-embedder");
        assert_eq!(cache.modalities(), &[Modality::Text]);
    }
}
