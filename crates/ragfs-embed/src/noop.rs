//! No-op embedder for testing without Candle.
//!
//! This module provides a [`NoopEmbedder`] that returns zero-vectors for all embeddings.
//! It's useful for:
//! - Testing without the Candle ML stack
//! - Development builds with faster compilation
//! - Stubbing embeddings in unit tests

use async_trait::async_trait;
use ragfs_core::{EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality};

/// No-op embedder that returns zero-vectors.
///
/// This embedder is always available, even without the `candle` feature.
/// It returns 384-dimensional zero-vectors for all inputs, making it useful
/// for testing and development.
///
/// # Example
///
/// ```rust
/// use ragfs_embed::NoopEmbedder;
/// use ragfs_core::{Embedder, EmbeddingConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let embedder = NoopEmbedder::new();
/// let config = EmbeddingConfig::default();
/// let texts: Vec<&str> = vec!["Hello", "World"];
/// let outputs = embedder.embed_text(&texts, &config).await?;
///
/// assert_eq!(outputs.len(), 2);
/// assert_eq!(outputs[0].embedding.len(), 384);
/// assert!(outputs[0].embedding.iter().all(|&v| v == 0.0));
/// # Ok(())
/// # }
/// ```
pub struct NoopEmbedder {
    dimension: usize,
    modalities: Vec<Modality>,
}

impl NoopEmbedder {
    /// Create a new no-op embedder with default dimension (384).
    #[must_use]
    pub fn new() -> Self {
        Self {
            dimension: 384,
            modalities: vec![Modality::Text],
        }
    }

    /// Create a new no-op embedder with custom dimension.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            modalities: vec![Modality::Text],
        }
    }
}

impl Default for NoopEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for NoopEmbedder {
    fn model_name(&self) -> &str {
        "noop"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        512
    }

    fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    async fn embed_text(
        &self,
        texts: &[&str],
        _config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        Ok(texts
            .iter()
            .map(|_| EmbeddingOutput {
                embedding: vec![0.0; self.dimension],
                token_count: 0,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_new() {
        let embedder = NoopEmbedder::new();
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_name(), "noop");
    }

    #[test]
    fn test_noop_default() {
        let embedder = NoopEmbedder::default();
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn test_noop_with_dimension() {
        let embedder = NoopEmbedder::with_dimension(768);
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn test_noop_modalities() {
        let embedder = NoopEmbedder::new();
        let modalities = embedder.modalities();
        assert_eq!(modalities.len(), 1);
        assert!(matches!(modalities[0], Modality::Text));
    }

    #[test]
    fn test_noop_max_tokens() {
        let embedder = NoopEmbedder::new();
        assert_eq!(embedder.max_tokens(), 512);
    }

    #[tokio::test]
    async fn test_noop_embed_text() {
        let embedder = NoopEmbedder::new();
        let texts: Vec<&str> = vec!["Hello", "World"];
        let config = EmbeddingConfig::default();

        let outputs = embedder.embed_text(&texts, &config).await.unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].embedding.len(), 384);
        assert_eq!(outputs[1].embedding.len(), 384);
        assert!(outputs[0].embedding.iter().all(|&v| v == 0.0));
        assert_eq!(outputs[0].token_count, 0);
    }

    #[tokio::test]
    async fn test_noop_embed_empty() {
        let embedder = NoopEmbedder::new();
        let texts: Vec<&str> = vec![];
        let config = EmbeddingConfig::default();

        let outputs = embedder.embed_text(&texts, &config).await.unwrap();

        assert!(outputs.is_empty());
    }

    #[tokio::test]
    async fn test_noop_embed_custom_dimension() {
        let embedder = NoopEmbedder::with_dimension(768);
        let texts: Vec<&str> = vec!["Test"];
        let config = EmbeddingConfig::default();

        let outputs = embedder.embed_text(&texts, &config).await.unwrap();

        assert_eq!(outputs[0].embedding.len(), 768);
    }
}
