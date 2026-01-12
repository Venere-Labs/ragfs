//! Vision model captioning for images.
//!
//! This module provides infrastructure for generating captions from images
//! using vision models. With the `vision` feature enabled, a BLIP-based
//! captioner is available. Otherwise, only a placeholder implementation exists.

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::debug;

#[cfg(feature = "vision")]
use candle_core::{DType, Device, IndexOp, Module, Tensor};
#[cfg(feature = "vision")]
use candle_nn::VarBuilder;
#[cfg(feature = "vision")]
use candle_transformers::models::blip;
#[cfg(feature = "vision")]
use hf_hub::{Repo, RepoType, api::tokio::Api};
#[cfg(feature = "vision")]
use tokenizers::Tokenizer;
#[cfg(feature = "vision")]
use tracing::info;

/// Error type for vision captioning operations.
#[derive(Debug, Error)]
pub enum CaptionError {
    /// Model loading failed.
    #[error("model loading failed: {0}")]
    ModelLoad(String),

    /// Image preprocessing failed.
    #[error("image preprocessing failed: {0}")]
    ImagePreprocess(String),

    /// Caption generation failed.
    #[error("caption generation failed: {0}")]
    Generation(String),

    /// Model not initialized.
    #[error("model not initialized")]
    NotInitialized,

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Trait for vision-based image captioning.
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

/// Placeholder vision captioner that returns no captions.
///
/// This is a no-op implementation that can be used when vision captioning
/// is not available or not desired. It always returns `None` for captions.
pub struct PlaceholderCaptioner {
    initialized: Arc<RwLock<bool>>,
}

impl PlaceholderCaptioner {
    /// Create a new placeholder captioner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: Arc::new(RwLock::new(false)),
        }
    }
}

impl Default for PlaceholderCaptioner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageCaptioner for PlaceholderCaptioner {
    async fn init(&self) -> Result<(), CaptionError> {
        let mut initialized = self.initialized.write().await;
        *initialized = true;
        debug!("Placeholder captioner initialized (no-op)");
        Ok(())
    }

    async fn caption(&self, _image_data: &[u8]) -> Result<Option<String>, CaptionError> {
        // Placeholder returns no caption
        Ok(None)
    }

    async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    fn model_name(&self) -> &str {
        "placeholder"
    }
}

/// Configuration for vision captioning.
#[derive(Debug, Clone)]
pub struct CaptionConfig {
    /// Enable captioning (default: false until model is implemented).
    pub enabled: bool,
    /// Use quantized model for lower memory usage.
    pub quantized: bool,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Cache directory for model files.
    pub cache_dir: PathBuf,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            quantized: false,
            max_tokens: 100,
            cache_dir: PathBuf::from("~/.local/share/ragfs/models"),
        }
    }
}

// ============================================================================
// BLIP Captioner (requires "vision" feature)
// ============================================================================

/// BLIP model identifier on HuggingFace Hub.
#[cfg(feature = "vision")]
#[allow(clippy::doc_markdown)]
const BLIP_MODEL_ID: &str = "Salesforce/blip-image-captioning-base";

/// Image size for BLIP preprocessing.
#[cfg(feature = "vision")]
const BLIP_IMAGE_SIZE: u32 = 384;

/// BLIP-based image captioner using Candle.
///
/// Uses the `Salesforce/blip-image-captioning-base` model from HuggingFace Hub.
/// Requires the `vision` feature to be enabled.
#[cfg(feature = "vision")]
#[allow(clippy::doc_markdown)]
pub struct BlipCaptioner {
    /// Device to run inference on (CPU or CUDA)
    device: Device,
    /// BLIP model
    model: Arc<RwLock<Option<blip::BlipForConditionalGeneration>>>,
    /// Text tokenizer
    tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    /// Cache directory for model files
    #[allow(dead_code)]
    cache_dir: PathBuf,
    /// Whether model is initialized
    initialized: Arc<RwLock<bool>>,
    /// Configuration
    config: CaptionConfig,
}

#[cfg(feature = "vision")]
impl BlipCaptioner {
    /// Create a new BLIP captioner.
    #[must_use]
    pub fn new(cache_dir: PathBuf) -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("BlipCaptioner using device: {:?}", device);

        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
            config: CaptionConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(cache_dir: PathBuf, config: CaptionConfig) -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("BlipCaptioner using device: {:?}", device);

        Self {
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            cache_dir,
            initialized: Arc::new(RwLock::new(false)),
            config,
        }
    }

    /// Preprocess image data into a tensor suitable for BLIP.
    fn preprocess_image(&self, image_data: &[u8]) -> Result<Tensor, CaptionError> {
        // Decode image
        let img = image::load_from_memory(image_data)
            .map_err(|e| CaptionError::ImagePreprocess(format!("Failed to decode image: {e}")))?;

        // Resize to 384x384
        let img = img.resize_exact(
            BLIP_IMAGE_SIZE,
            BLIP_IMAGE_SIZE,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB and normalize
        let img = img.to_rgb8();
        let (width, height) = (img.width() as usize, img.height() as usize);

        // CLIP normalization values (standard ImageNet values, kept as-is for reference)
        #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
        let std = [0.26862954f32, 0.26130258, 0.27577711];

        // Convert to tensor [C, H, W] format with normalization
        let mut data = vec![0f32; 3 * width * height];
        for (x, y, pixel) in img.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            for c in 0..3 {
                let val = f32::from(pixel[c]) / 255.0;
                let normalized = (val - mean[c]) / std[c];
                data[c * height * width + y * width + x] = normalized;
            }
        }

        let tensor = Tensor::from_vec(data, (3, height, width), &self.device)
            .map_err(|e| CaptionError::ImagePreprocess(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0) // Add batch dimension [1, C, H, W]
            .map_err(|e| CaptionError::ImagePreprocess(format!("Unsqueeze failed: {e}")))?;

        Ok(tensor)
    }

    /// Generate caption from image embedding.
    async fn generate_caption(&self, image_tensor: &Tensor) -> Result<String, CaptionError> {
        // Use write lock since text_decoder may need mutable access
        let mut model_guard = self.model.write().await;
        let model = model_guard.as_mut().ok_or(CaptionError::NotInitialized)?;

        let tokenizer_guard = self.tokenizer.read().await;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or(CaptionError::NotInitialized)?;

        // Get image embeddings from vision encoder
        let image_embeds = model
            .vision_model()
            .forward(image_tensor)
            .map_err(|e| CaptionError::Generation(format!("Vision forward failed: {e}")))?;

        // Initialize with BOS token
        let mut token_ids = vec![tokenizer.token_to_id("[CLS]").unwrap_or(101)]; // Default BERT [CLS] id

        let eos_token_id = tokenizer.token_to_id("[SEP]").unwrap_or(102);
        let max_tokens = self.config.max_tokens;

        // Autoregressive generation
        for _ in 0..max_tokens {
            let input_ids = Tensor::new(&token_ids[..], &self.device)
                .map_err(|e| CaptionError::Generation(format!("Token tensor failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| CaptionError::Generation(format!("Unsqueeze failed: {e}")))?;

            let logits = model
                .text_decoder()
                .forward(&input_ids, &image_embeds)
                .map_err(|e| CaptionError::Generation(format!("Text decoder failed: {e}")))?;

            // Get next token (greedy decoding)
            let seq_len = logits
                .dim(1)
                .map_err(|e| CaptionError::Generation(format!("Dim failed: {e}")))?;
            let next_token_logits = logits
                .i((.., seq_len - 1, ..))
                .map_err(|e| CaptionError::Generation(format!("Index failed: {e}")))?;

            let next_token = next_token_logits
                .argmax(candle_core::D::Minus1)
                .map_err(|e| CaptionError::Generation(format!("Argmax failed: {e}")))?
                .to_scalar::<u32>()
                .map_err(|e| CaptionError::Generation(format!("Scalar failed: {e}")))?;

            if next_token == eos_token_id {
                break;
            }

            token_ids.push(next_token);
        }

        // Decode tokens to string
        let caption = tokenizer
            .decode(&token_ids, true)
            .map_err(|e| CaptionError::Generation(format!("Decode failed: {e}")))?;

        Ok(caption.trim().to_string())
    }
}

#[cfg(feature = "vision")]
#[async_trait]
impl ImageCaptioner for BlipCaptioner {
    async fn init(&self) -> Result<(), CaptionError> {
        {
            let initialized = self.initialized.read().await;
            if *initialized {
                return Ok(());
            }
        }

        info!("Initializing BlipCaptioner with model: {}", BLIP_MODEL_ID);

        // Download model files from HuggingFace Hub
        let api = Api::new()
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to create HF API: {e}")))?;

        let repo = api.repo(Repo::new(BLIP_MODEL_ID.to_string(), RepoType::Model));

        // Download tokenizer
        debug!("Downloading tokenizer...");
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to download tokenizer: {e}")))?;

        // Download model config
        debug!("Downloading config...");
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to download config: {e}")))?;

        // Download model weights
        debug!("Downloading model weights...");
        let weights_path = repo
            .get("model.safetensors")
            .await
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to download weights: {e}")))?;

        // Load tokenizer
        debug!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;

        // Load config
        debug!("Loading config...");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to read config: {e}")))?;
        let config: blip::Config = serde_json::from_str(&config_str)
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to parse config: {e}")))?;

        // Load model weights
        debug!("Loading model weights...");
        let dtype = if self.config.quantized {
            DType::BF16
        } else {
            DType::F32
        };

        // SAFETY: The safetensors file is downloaded from HuggingFace Hub and is trusted.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &self.device)
                .map_err(|e| CaptionError::ModelLoad(format!("Failed to load weights: {e}")))?
        };

        let model = blip::BlipForConditionalGeneration::new(&config, vb)
            .map_err(|e| CaptionError::ModelLoad(format!("Failed to create BLIP model: {e}")))?;

        // Store in instance
        {
            let mut tok = self.tokenizer.write().await;
            *tok = Some(tokenizer);
        }
        {
            let mut mdl = self.model.write().await;
            *mdl = Some(model);
        }
        {
            let mut initialized = self.initialized.write().await;
            *initialized = true;
        }

        info!("BlipCaptioner initialized successfully");
        Ok(())
    }

    async fn caption(&self, image_data: &[u8]) -> Result<Option<String>, CaptionError> {
        if !self.is_initialized().await {
            return Err(CaptionError::NotInitialized);
        }

        if !self.config.enabled {
            return Ok(None);
        }

        debug!("Generating caption for image ({} bytes)", image_data.len());

        // Preprocess image
        let image_tensor = self.preprocess_image(image_data)?;

        // Generate caption
        let caption = self.generate_caption(&image_tensor).await?;

        if caption.is_empty() {
            Ok(None)
        } else {
            Ok(Some(caption))
        }
    }

    async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    fn model_name(&self) -> &str {
        BLIP_MODEL_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_placeholder_captioner_new() {
        let captioner = PlaceholderCaptioner::new();
        assert!(!captioner.is_initialized().await);
    }

    #[tokio::test]
    async fn test_placeholder_captioner_init() {
        let captioner = PlaceholderCaptioner::new();
        let result = captioner.init().await;
        assert!(result.is_ok());
        assert!(captioner.is_initialized().await);
    }

    #[tokio::test]
    async fn test_placeholder_captioner_returns_none() {
        let captioner = PlaceholderCaptioner::new();
        captioner.init().await.unwrap();

        let result = captioner.caption(b"fake image data").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_placeholder_captioner_model_name() {
        let captioner = PlaceholderCaptioner::new();
        assert_eq!(captioner.model_name(), "placeholder");
    }

    #[test]
    fn test_caption_config_default() {
        let config = CaptionConfig::default();
        assert!(!config.enabled);
        assert!(!config.quantized);
        assert_eq!(config.max_tokens, 100);
    }

    #[test]
    fn test_caption_error_display() {
        let err = CaptionError::NotInitialized;
        assert_eq!(err.to_string(), "model not initialized");

        let err = CaptionError::ModelLoad("test error".to_string());
        assert!(err.to_string().contains("model loading failed"));
    }
}
