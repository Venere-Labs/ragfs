//! Embedder pool for concurrent embedding operations.

use ragfs_core::{EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Pool of embedders with concurrency control.
pub struct EmbedderPool {
    /// Primary embedder for documents
    document_embedder: Arc<dyn Embedder>,
    /// Semaphore to limit concurrent inference
    semaphore: Semaphore,
    /// Maximum concurrent operations
    max_concurrent: usize,
}

impl EmbedderPool {
    /// Create a new embedder pool.
    pub fn new(embedder: Arc<dyn Embedder>, max_concurrent: usize) -> Self {
        Self {
            document_embedder: embedder,
            semaphore: Semaphore::new(max_concurrent),
            max_concurrent,
        }
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.document_embedder.dimension()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.document_embedder.model_name()
    }

    /// Get supported modalities.
    pub fn modalities(&self) -> &[Modality] {
        self.document_embedder.modalities()
    }

    /// Get the underlying embedder.
    pub fn document_embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.document_embedder)
    }

    /// Embed a batch of texts.
    pub async fn embed_batch(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| EmbedError::Inference(format!("semaphore error: {e}")))?;

        self.document_embedder.embed_text(texts, config).await
    }

    /// Embed a single query.
    pub async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| EmbedError::Inference(format!("semaphore error: {e}")))?;

        self.document_embedder.embed_query(query, config).await
    }

    /// Get pool statistics.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Get max concurrent operations.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    const TEST_DIM: usize = 384;

    /// Mock embedder for testing.
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
            // Return deterministic embeddings based on text length
            Ok(texts
                .iter()
                .map(|text| {
                    let embedding: Vec<f32> = (0..self.dimension)
                        .map(|i| ((i + text.len()) as f32 * 0.001).sin())
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
    async fn test_pool_creation() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);

        assert_eq!(pool.dimension(), TEST_DIM);
        assert_eq!(pool.model_name(), "mock-embedder");
        assert_eq!(pool.max_concurrent(), 4);
        assert_eq!(pool.available_permits(), 4);
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);
        let config = EmbeddingConfig::default();

        let texts = vec!["hello world", "test embedding"];
        let results = pool.embed_batch(&texts, &config).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].embedding.len(), TEST_DIM);
        assert_eq!(results[1].embedding.len(), TEST_DIM);
    }

    #[tokio::test]
    async fn test_embed_query() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);
        let config = EmbeddingConfig::default();

        let result = pool.embed_query("search query", &config).await.unwrap();

        assert_eq!(result.embedding.len(), TEST_DIM);
        assert_eq!(result.token_count, 2); // "search" and "query"
    }

    #[tokio::test]
    async fn test_semaphore_limits_concurrency() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = Arc::new(EmbedderPool::new(embedder, 2));

        // Initially all permits available
        assert_eq!(pool.available_permits(), 2);

        // Spawn multiple concurrent tasks
        let pool1 = Arc::clone(&pool);
        let pool2 = Arc::clone(&pool);

        let handle1 = tokio::spawn(async move {
            let _ = pool1
                .embed_query("query1", &EmbeddingConfig::default())
                .await;
        });

        let handle2 = tokio::spawn(async move {
            let _ = pool2
                .embed_query("query2", &EmbeddingConfig::default())
                .await;
        });

        // Wait for both to complete
        let _ = handle1.await;
        let _ = handle2.await;

        // All permits should be returned
        assert_eq!(pool.available_permits(), 2);
    }

    #[tokio::test]
    async fn test_document_embedder_access() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);

        let doc_embedder = pool.document_embedder();
        assert_eq!(doc_embedder.dimension(), TEST_DIM);
        assert_eq!(doc_embedder.model_name(), "mock-embedder");
    }

    #[tokio::test]
    async fn test_modalities() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);

        let modalities = pool.modalities();
        assert_eq!(modalities.len(), 1);
        assert!(matches!(modalities[0], Modality::Text));
    }

    #[tokio::test]
    async fn test_empty_batch() {
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));
        let pool = EmbedderPool::new(embedder, 4);
        let config = EmbeddingConfig::default();

        let texts: Vec<&str> = vec![];
        let results = pool.embed_batch(&texts, &config).await.unwrap();

        assert!(results.is_empty());
    }
}
