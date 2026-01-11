//! Query execution.

use ragfs_core::{
    DistanceMetric, Embedder, EmbeddingConfig, SearchQuery, SearchResult, VectorStore,
};
use std::sync::Arc;
use tracing::debug;

use crate::parser::{ParsedQuery, QueryParser};

/// Query executor.
pub struct QueryExecutor {
    /// Vector store
    store: Arc<dyn VectorStore>,
    /// Embedder for query embedding
    embedder: Arc<dyn Embedder>,
    /// Query parser
    parser: QueryParser,
    /// Whether to use hybrid search
    hybrid: bool,
}

impl QueryExecutor {
    /// Create a new query executor.
    pub fn new(
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        default_limit: usize,
        hybrid: bool,
    ) -> Self {
        Self {
            store,
            embedder,
            parser: QueryParser::new(default_limit),
            hybrid,
        }
    }

    /// Execute a query string.
    pub async fn execute(&self, query_str: &str) -> Result<Vec<SearchResult>, ragfs_core::Error> {
        debug!("Executing query: {}", query_str);

        // Parse query
        let parsed = self.parser.parse(query_str);

        // Embed query text
        let config = EmbeddingConfig::default();
        let embedding = self
            .embedder
            .embed_query(&parsed.text, &config)
            .await
            .map_err(ragfs_core::Error::Embedding)?;

        // Build search query
        let search_query = SearchQuery {
            embedding: embedding.embedding,
            text: if self.hybrid {
                Some(parsed.text.clone())
            } else {
                None
            },
            limit: parsed.limit,
            filters: parsed.filters,
            metric: DistanceMetric::Cosine,
        };

        // Execute search
        let results = if self.hybrid {
            self.store.hybrid_search(search_query).await
        } else {
            self.store.search(search_query).await
        }
        .map_err(ragfs_core::Error::Store)?;

        debug!("Found {} results", results.len());
        Ok(results)
    }

    /// Execute with a pre-parsed query.
    pub async fn execute_parsed(
        &self,
        parsed: ParsedQuery,
    ) -> Result<Vec<SearchResult>, ragfs_core::Error> {
        let config = EmbeddingConfig::default();
        let embedding = self
            .embedder
            .embed_query(&parsed.text, &config)
            .await
            .map_err(ragfs_core::Error::Embedding)?;

        let search_query = SearchQuery {
            embedding: embedding.embedding,
            text: if self.hybrid {
                Some(parsed.text.clone())
            } else {
                None
            },
            limit: parsed.limit,
            filters: parsed.filters,
            metric: DistanceMetric::Cosine,
        };

        let results = if self.hybrid {
            self.store.hybrid_search(search_query).await
        } else {
            self.store.search(search_query).await
        }
        .map_err(ragfs_core::Error::Store)?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ragfs_core::{
        Chunk, EmbedError, EmbeddingOutput, FileRecord, Modality, StoreError, StoreStats,
    };
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use tokio::sync::RwLock;
    use uuid::Uuid;

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
        ) -> Result<EmbeddingOutput, EmbedError> {
            Ok(EmbeddingOutput {
                embedding: vec![0.1; self.dimension],
                token_count: 10,
            })
        }
    }

    // ==================== Mock VectorStore ====================

    struct MockStore {
        results: Arc<RwLock<Vec<SearchResult>>>,
        hybrid_results: Arc<RwLock<Vec<SearchResult>>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                results: Arc::new(RwLock::new(Vec::new())),
                hybrid_results: Arc::new(RwLock::new(Vec::new())),
            }
        }

        fn with_results(results: Vec<SearchResult>) -> Self {
            Self {
                results: Arc::new(RwLock::new(results)),
                hybrid_results: Arc::new(RwLock::new(Vec::new())),
            }
        }

        fn with_hybrid_results(results: Vec<SearchResult>, hybrid: Vec<SearchResult>) -> Self {
            Self {
                results: Arc::new(RwLock::new(results)),
                hybrid_results: Arc::new(RwLock::new(hybrid)),
            }
        }
    }

    #[async_trait]
    impl VectorStore for MockStore {
        async fn init(&self) -> Result<(), StoreError> {
            Ok(())
        }

        async fn upsert_chunks(&self, _chunks: &[Chunk]) -> Result<(), StoreError> {
            Ok(())
        }

        async fn search(&self, _query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
            let results = self.results.read().await;
            Ok(results.clone())
        }

        async fn hybrid_search(
            &self,
            _query: SearchQuery,
        ) -> Result<Vec<SearchResult>, StoreError> {
            let results = self.hybrid_results.read().await;
            Ok(results.clone())
        }

        async fn delete_by_file_path(&self, _path: &Path) -> Result<u64, StoreError> {
            Ok(0)
        }

        async fn get_file(&self, _path: &Path) -> Result<Option<FileRecord>, StoreError> {
            Ok(None)
        }

        async fn upsert_file(&self, _record: &FileRecord) -> Result<(), StoreError> {
            Ok(())
        }

        async fn stats(&self) -> Result<StoreStats, StoreError> {
            Ok(StoreStats {
                total_chunks: 0,
                total_files: 0,
                index_size_bytes: 0,
                last_updated: None,
            })
        }

        async fn update_file_path(&self, _from: &Path, _to: &Path) -> Result<u64, StoreError> {
            Ok(0)
        }

        async fn get_chunks_for_file(&self, _path: &Path) -> Result<Vec<Chunk>, StoreError> {
            Ok(vec![])
        }
    }

    // ==================== Helper functions ====================

    fn create_test_result(path: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::new_v4(),
            file_path: PathBuf::from(path),
            content: content.to_string(),
            score,
            byte_range: 0..content.len() as u64,
            line_range: Some(0..1),
            metadata: HashMap::new(),
        }
    }

    // ==================== Tests ====================

    #[tokio::test]
    async fn test_execute_simple_query() {
        let results = vec![
            create_test_result("/test/file1.txt", "Authentication module", 0.9),
            create_test_result("/test/file2.txt", "Auth config", 0.8),
        ];

        let store = Arc::new(MockStore::with_results(results.clone()));
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        let executor = QueryExecutor::new(store, embedder, 10, false);

        let query_results = executor.execute("authentication").await.unwrap();

        assert_eq!(query_results.len(), 2);
        assert_eq!(query_results[0].content, "Authentication module");
        assert_eq!(query_results[1].content, "Auth config");
    }

    #[tokio::test]
    async fn test_execute_with_hybrid_search() {
        let vector_results = vec![create_test_result("/test/vector.txt", "Vector result", 0.8)];
        let hybrid_results = vec![
            create_test_result("/test/hybrid1.txt", "Hybrid result 1", 0.95),
            create_test_result("/test/hybrid2.txt", "Hybrid result 2", 0.85),
        ];

        let store = Arc::new(MockStore::with_hybrid_results(
            vector_results,
            hybrid_results.clone(),
        ));
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        // hybrid=true should use hybrid_search
        let executor = QueryExecutor::new(store, embedder, 10, true);

        let query_results = executor.execute("search query").await.unwrap();

        assert_eq!(query_results.len(), 2);
        assert_eq!(query_results[0].content, "Hybrid result 1");
    }

    #[tokio::test]
    async fn test_execute_vector_only() {
        let vector_results = vec![create_test_result(
            "/test/vector.txt",
            "Vector only result",
            0.9,
        )];
        let hybrid_results = vec![create_test_result(
            "/test/hybrid.txt",
            "Hybrid result",
            0.95,
        )];

        let store = Arc::new(MockStore::with_hybrid_results(
            vector_results.clone(),
            hybrid_results,
        ));
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        // hybrid=false should use regular search
        let executor = QueryExecutor::new(store, embedder, 10, false);

        let query_results = executor.execute("search query").await.unwrap();

        assert_eq!(query_results.len(), 1);
        assert_eq!(query_results[0].content, "Vector only result");
    }

    #[tokio::test]
    async fn test_execute_empty_results() {
        let store = Arc::new(MockStore::new());
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        let executor = QueryExecutor::new(store, embedder, 10, false);

        let query_results = executor.execute("no results query").await.unwrap();

        assert!(query_results.is_empty());
    }

    #[tokio::test]
    async fn test_execute_with_limit_in_query() {
        let results = vec![
            create_test_result("/test/file1.txt", "Result 1", 0.9),
            create_test_result("/test/file2.txt", "Result 2", 0.8),
            create_test_result("/test/file3.txt", "Result 3", 0.7),
        ];

        let store = Arc::new(MockStore::with_results(results));
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        let executor = QueryExecutor::new(store, embedder, 10, false);

        // Query with limit:2 filter
        let query_results = executor.execute("search query limit:2").await.unwrap();

        // Note: The mock store returns all results regardless of limit.
        // In a real test with actual store, we'd verify the limit is applied.
        assert!(!query_results.is_empty());
    }

    #[tokio::test]
    async fn test_execute_parsed_query() {
        use crate::parser::ParsedQuery;

        let results = vec![create_test_result("/test/file.txt", "Parsed result", 0.9)];

        let store = Arc::new(MockStore::with_results(results));
        let embedder = Arc::new(MockEmbedder::new(TEST_DIM));

        let executor = QueryExecutor::new(store, embedder, 10, false);

        let parsed = ParsedQuery {
            text: "pre-parsed query".to_string(),
            limit: 5,
            filters: vec![],
        };

        let query_results = executor.execute_parsed(parsed).await.unwrap();

        assert_eq!(query_results.len(), 1);
        assert_eq!(query_results[0].content, "Parsed result");
    }

    #[test]
    fn test_query_executor_creation() {
        let store: Arc<dyn VectorStore> = Arc::new(MockStore::new());
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(TEST_DIM));

        // Test with hybrid=false
        let executor = QueryExecutor::new(Arc::clone(&store), Arc::clone(&embedder), 10, false);
        assert!(!executor.hybrid);

        // Test with hybrid=true
        let executor2 = QueryExecutor::new(store, embedder, 20, true);
        assert!(executor2.hybrid);
    }
}
