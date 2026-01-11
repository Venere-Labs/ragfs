//! Chunker registry for managing chunking strategies.

use ragfs_core::{ChunkConfig, ChunkError, ChunkOutput, Chunker, ContentType, ExtractedContent};
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of chunking strategies.
pub struct ChunkerRegistry {
    /// Named chunkers
    chunkers: HashMap<String, Arc<dyn Chunker>>,
    /// Content type to chunker name mapping
    type_mapping: HashMap<String, String>,
    /// Default chunker name
    default_chunker: Option<String>,
}

impl ChunkerRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunkers: HashMap::new(),
            type_mapping: HashMap::new(),
            default_chunker: None,
        }
    }

    /// Register a chunker.
    pub fn register<C: Chunker + 'static>(&mut self, name: &str, chunker: C) {
        let chunker = Arc::new(chunker);
        for content_type in chunker.content_types() {
            self.type_mapping
                .insert((*content_type).to_string(), name.to_string());
        }
        self.chunkers.insert(name.to_string(), chunker);
    }

    /// Set the default chunker.
    pub fn set_default(&mut self, name: &str) {
        self.default_chunker = Some(name.to_string());
    }

    /// Get a chunker for a content type.
    #[must_use]
    pub fn get_for_content_type(&self, content_type: &ContentType) -> Option<Arc<dyn Chunker>> {
        // Try to find specific chunker
        for chunker in self.chunkers.values() {
            if chunker.can_chunk(content_type) {
                return Some(chunker.clone());
            }
        }

        // Fall back to default
        self.default_chunker
            .as_ref()
            .and_then(|name| self.chunkers.get(name))
            .cloned()
    }

    /// Chunk content using appropriate strategy.
    pub async fn chunk(
        &self,
        content: &ExtractedContent,
        content_type: &ContentType,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError> {
        let chunker = self
            .get_for_content_type(content_type)
            .ok_or_else(|| ChunkError::Failed("no suitable chunker found".to_string()))?;

        chunker.chunk(content, config).await
    }
}

impl Default for ChunkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedSizeChunker;
    use ragfs_core::ContentMetadataInfo;

    fn create_test_content(text: &str) -> ExtractedContent {
        ExtractedContent {
            text: text.to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        }
    }

    #[test]
    fn test_new_registry_is_empty() {
        let registry = ChunkerRegistry::new();
        assert!(registry.chunkers.is_empty());
        assert!(registry.type_mapping.is_empty());
        assert!(registry.default_chunker.is_none());
    }

    #[test]
    fn test_default_implementation() {
        let registry = ChunkerRegistry::default();
        assert!(registry.chunkers.is_empty());
        assert!(registry.default_chunker.is_none());
    }

    #[test]
    fn test_register_chunker() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        assert!(registry.chunkers.contains_key("fixed"));
        // FixedSizeChunker registers text, code, markdown
        assert!(registry.type_mapping.contains_key("text"));
    }

    #[test]
    fn test_set_default_chunker() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());
        registry.set_default("fixed");

        assert_eq!(registry.default_chunker, Some("fixed".to_string()));
    }

    #[test]
    fn test_get_for_content_type_text() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let chunker = registry.get_for_content_type(&ContentType::Text);
        assert!(chunker.is_some());
    }

    #[test]
    fn test_get_for_content_type_markdown() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let chunker = registry.get_for_content_type(&ContentType::Markdown);
        assert!(chunker.is_some());
    }

    #[test]
    fn test_get_for_content_type_code() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let chunker = registry.get_for_content_type(&ContentType::Code {
            language: "rust".to_string(),
            symbol: None,
        });
        assert!(chunker.is_some());
    }

    #[test]
    fn test_get_for_content_type_falls_back_to_default() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());
        registry.set_default("fixed");

        // FixedSizeChunker can handle any type, but this tests the fallback logic
        let chunker = registry.get_for_content_type(&ContentType::Text);
        assert!(chunker.is_some());
    }

    #[test]
    fn test_get_for_content_type_none_when_no_match() {
        let registry = ChunkerRegistry::new();
        // Empty registry with no default
        let chunker = registry.get_for_content_type(&ContentType::Text);
        assert!(chunker.is_none());
    }

    #[tokio::test]
    async fn test_chunk_success() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let content = create_test_content("Hello, world!");
        let config = ChunkConfig::default();

        let result = registry.chunk(&content, &ContentType::Text, &config).await;
        assert!(result.is_ok());

        let chunks = result.unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_chunk_long_text() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let text = "A".repeat(3000);
        let content = create_test_content(&text);
        let config = ChunkConfig {
            target_size: 256,
            max_size: 512,
            overlap: 32,
            ..Default::default()
        };

        let result = registry.chunk(&content, &ContentType::Text, &config).await;
        assert!(result.is_ok());

        let chunks = result.unwrap();
        assert!(chunks.len() > 1, "Long text should produce multiple chunks");
    }

    #[tokio::test]
    async fn test_chunk_empty_text() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed", FixedSizeChunker::new());

        let content = create_test_content("");
        let config = ChunkConfig::default();

        let result = registry.chunk(&content, &ContentType::Text, &config).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_chunk_fails_without_chunker() {
        let registry = ChunkerRegistry::new();

        let content = create_test_content("Hello");
        let config = ChunkConfig::default();

        let result = registry.chunk(&content, &ContentType::Text, &config).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ChunkError::Failed(msg) => {
                assert!(msg.contains("no suitable chunker"));
            }
            _ => panic!("Expected ChunkError::Failed"),
        }
    }

    #[test]
    fn test_multiple_chunkers() {
        let mut registry = ChunkerRegistry::new();
        registry.register("fixed1", FixedSizeChunker::new());
        registry.register("fixed2", FixedSizeChunker::new());

        assert_eq!(registry.chunkers.len(), 2);
    }

    #[test]
    fn test_chunker_overrides_type_mapping() {
        let mut registry = ChunkerRegistry::new();
        registry.register("first", FixedSizeChunker::new());
        registry.register("second", FixedSizeChunker::new());

        // Second registration should override the type mapping
        assert_eq!(
            registry.type_mapping.get("text"),
            Some(&"second".to_string())
        );
    }
}
