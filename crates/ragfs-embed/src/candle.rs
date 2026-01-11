//! GTE-small embedder using Candle.
//!
//! Uses thenlper/gte-small model for text embeddings:
//! - 384 dimensions
//! - 512 max tokens
//! - BERT architecture

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use ragfs_core::{EmbedError, Embedder, EmbeddingConfig, EmbeddingOutput, Modality};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Model identifier on HuggingFace Hub.
const MODEL_ID: &str = "thenlper/gte-small";

/// Embedding dimension for gte-small.
const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length.
const MAX_TOKENS: usize = 512;

/// GTE-small embedder using Candle.
pub struct CandleEmbedder {
    /// Device to run inference on (CPU or CUDA)
    device: Device,
    /// Loaded model
    model: Arc<RwLock<Option<BertModel>>>,
    /// Tokenizer
    tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    /// Model configuration
    config: Arc<RwLock<Option<Config>>>,
    /// Cache directory for models
    #[allow(dead_code)]
    cache_dir: PathBuf,
    /// Whether model is initialized
    initialized: Arc<RwLock<bool>>,
}

impl CandleEmbedder {
    /// Create a new CandleEmbedder.
    pub fn new(cache_dir: PathBuf) -> Self {
        // Try to use CUDA if available, fallback to CPU
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("CandleEmbedder using device: {:?}", device);

        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Create with specific device.
    pub fn with_device(cache_dir: PathBuf, device: Device) -> Self {
        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize the model (download if needed, load into memory).
    pub async fn init(&self) -> Result<(), EmbedError> {
        {
            let initialized = self.initialized.read().await;
            if *initialized {
                return Ok(());
            }
        }

        info!("Initializing CandleEmbedder with model: {}", MODEL_ID);

        // Download model files from HuggingFace Hub
        let api = Api::new()
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to create HF API: {e}")))?;

        let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

        // Download tokenizer
        debug!("Downloading tokenizer...");
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to download tokenizer: {e}")))?;

        // Download model config
        debug!("Downloading config...");
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to download config: {e}")))?;

        // Download model weights
        debug!("Downloading model weights...");
        let weights_path = repo
            .get("model.safetensors")
            .await
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to download weights: {e}")))?;

        // Load tokenizer
        debug!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;

        // Load config
        debug!("Loading config...");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to read config: {e}")))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to parse config: {e}")))?;

        // Load model weights
        debug!("Loading model weights...");
        // SAFETY: The safetensors file is downloaded from HuggingFace Hub and is trusted.
        // Memory mapping is safe for read-only access to model weights.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &self.device)
                .map_err(|e| EmbedError::ModelLoad(format!("Failed to load weights: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbedError::ModelLoad(format!("Failed to create BERT model: {e}")))?;

        // Store in instance
        {
            let mut tok = self.tokenizer.write().await;
            *tok = Some(tokenizer);
        }
        {
            let mut cfg = self.config.write().await;
            *cfg = Some(config);
        }
        {
            let mut mdl = self.model.write().await;
            *mdl = Some(model);
        }
        {
            let mut init = self.initialized.write().await;
            *init = true;
        }

        info!("CandleEmbedder initialized successfully");
        Ok(())
    }

    /// Mean pooling with attention mask.
    fn mean_pooling(
        &self,
        token_embeddings: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, EmbedError> {
        // Expand attention mask to match embedding dimensions
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| EmbedError::Inference(format!("unsqueeze failed: {e}")))?
            .broadcast_as(token_embeddings.shape())
            .map_err(|e| EmbedError::Inference(format!("broadcast failed: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| EmbedError::Inference(format!("dtype conversion failed: {e}")))?;

        // Masked sum
        let masked = token_embeddings
            .mul(&mask)
            .map_err(|e| EmbedError::Inference(format!("mul failed: {e}")))?;

        let sum = masked
            .sum(1)
            .map_err(|e| EmbedError::Inference(format!("sum failed: {e}")))?;

        // Count non-masked tokens
        let mask_sum = mask
            .sum(1)
            .map_err(|e| EmbedError::Inference(format!("mask sum failed: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| EmbedError::Inference(format!("clamp failed: {e}")))?;

        // Mean
        let mean = sum
            .div(&mask_sum)
            .map_err(|e| EmbedError::Inference(format!("div failed: {e}")))?;

        Ok(mean)
    }

    /// L2 normalize embeddings.
    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor, EmbedError> {
        let norm = embeddings
            .sqr()
            .map_err(|e| EmbedError::Inference(format!("sqr failed: {e}")))?
            .sum_keepdim(1)
            .map_err(|e| EmbedError::Inference(format!("sum_keepdim failed: {e}")))?
            .sqrt()
            .map_err(|e| EmbedError::Inference(format!("sqrt failed: {e}")))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| EmbedError::Inference(format!("clamp failed: {e}")))?;

        let normalized = embeddings
            .broadcast_div(&norm)
            .map_err(|e| EmbedError::Inference(format!("div failed: {e}")))?;

        Ok(normalized)
    }

    /// Encode a batch of texts.
    async fn encode_batch(
        &self,
        texts: &[&str],
        normalize: bool,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        // Ensure initialized
        self.init().await?;

        let tokenizer = self.tokenizer.read().await;
        let tokenizer = tokenizer
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Tokenizer not loaded".to_string()))?;

        let model = self.model.read().await;
        let model = model
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Model not loaded".to_string()))?;

        // Tokenize all texts
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Inference(format!("Tokenization failed: {e}")))?;

        // Find max length for padding
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let max_len = max_len.min(MAX_TOKENS);

        // Prepare input tensors
        let mut input_ids_vec: Vec<u32> = Vec::new();
        let mut attention_mask_vec: Vec<u32> = Vec::new();
        let mut token_type_ids_vec: Vec<u32> = Vec::new();
        let mut token_counts = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len);
            token_counts.push(len);

            // Add IDs with padding
            for i in 0..max_len {
                if i < len {
                    input_ids_vec.push(ids[i]);
                    attention_mask_vec.push(1);
                    token_type_ids_vec.push(0);
                } else {
                    input_ids_vec.push(0); // PAD token
                    attention_mask_vec.push(0);
                    token_type_ids_vec.push(0);
                }
            }
        }

        let batch_size = texts.len();

        // Create tensors
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)
            .map_err(|e| EmbedError::Inference(format!("Failed to create input_ids tensor: {e}")))?;

        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device).map_err(
                |e| EmbedError::Inference(format!("Failed to create attention_mask tensor: {e}")),
            )?;

        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device).map_err(
                |e| EmbedError::Inference(format!("Failed to create token_type_ids tensor: {e}")),
            )?;

        // Run model
        let output = model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| EmbedError::Inference(format!("Model forward failed: {e}")))?;

        // Mean pooling
        let pooled = self.mean_pooling(&output, &attention_mask)?;

        // Normalize if requested
        let final_embeddings = if normalize {
            self.normalize(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<EmbeddingOutput>
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let embedding = final_embeddings
                .get(i)
                .map_err(|e| EmbedError::Inference(format!("Failed to get embedding {i}: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| EmbedError::Inference(format!("Failed to convert to vec: {e}")))?;

            results.push(EmbeddingOutput {
                embedding,
                token_count: token_counts[i],
            });
        }

        Ok(results)
    }
}

#[async_trait]
impl Embedder for CandleEmbedder {
    fn model_name(&self) -> &str {
        MODEL_ID
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }

    fn max_tokens(&self) -> usize {
        MAX_TOKENS
    }

    fn modalities(&self) -> &[Modality] {
        &[Modality::Text]
    }

    async fn embed_text(
        &self,
        texts: &[&str],
        config: &EmbeddingConfig,
    ) -> Result<Vec<EmbeddingOutput>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding {} texts with batch_size {}", texts.len(), config.batch_size);

        // Process in batches
        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(config.batch_size) {
            let batch_results = self.encode_batch(chunk, config.normalize).await?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    async fn embed_query(
        &self,
        query: &str,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingOutput, EmbedError> {
        // For GTE models, queries and documents use the same embedding process
        // Some models use different prefixes, but GTE doesn't need that
        let results = self.embed_text(&[query], config).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Inference("Empty embedding result".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_candle_embedder() {
        let cache_dir = tempdir().unwrap();
        let embedder = CandleEmbedder::new(cache_dir.path().to_path_buf());

        embedder.init().await.unwrap();

        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_name(), "thenlper/gte-small");

        let config = EmbeddingConfig::default();
        let texts = &["Hello world", "This is a test"];

        let results = embedder.embed_text(texts, &config).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].embedding.len(), 384);
        assert_eq!(results[1].embedding.len(), 384);

        // Check normalization (should have unit length)
        let norm: f32 = results[0].embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
