//! Error types for RAGFS.

use thiserror::Error;

/// Main error type for RAGFS operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Content extraction failed
    #[error("extraction error: {0}")]
    Extraction(#[from] ExtractError),

    /// Chunking failed
    #[error("chunking error: {0}")]
    Chunking(#[from] ChunkError),

    /// Embedding generation failed
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbedError),

    /// Vector store operation failed
    #[error("store error: {0}")]
    Store(#[from] StoreError),

    /// I/O error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error
    #[error("config error: {0}")]
    Config(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Content extraction errors.
#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("unsupported file type: {0}")]
    UnsupportedType(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("extraction failed: {0}")]
    Failed(String),
}

/// Chunking errors.
#[derive(Error, Debug)]
pub enum ChunkError {
    #[error("chunking failed: {0}")]
    Failed(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Embedding errors.
#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("model loading failed: {0}")]
    ModelLoad(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("modality not supported: {0:?}")]
    ModalityNotSupported(crate::types::Modality),

    #[error("input too long: {tokens} tokens, max {max}")]
    InputTooLong { tokens: usize, max: usize },
}

/// Vector store errors.
#[derive(Error, Debug)]
pub enum StoreError {
    #[error("store initialization failed: {0}")]
    Init(String),

    #[error("insert failed: {0}")]
    Insert(String),

    #[error("query failed: {0}")]
    Query(String),

    #[error("delete failed: {0}")]
    Delete(String),

    #[error("schema error: {0}")]
    Schema(String),
}

/// Result type alias for RAGFS operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Modality;

    // ========== ExtractError Tests ==========

    #[test]
    fn test_extract_error_unsupported_type_display() {
        let err = ExtractError::UnsupportedType("application/octet-stream".to_string());
        assert_eq!(
            err.to_string(),
            "unsupported file type: application/octet-stream"
        );
    }

    #[test]
    fn test_extract_error_parse_display() {
        let err = ExtractError::Parse("invalid UTF-8".to_string());
        assert_eq!(err.to_string(), "parse error: invalid UTF-8");
    }

    #[test]
    fn test_extract_error_io_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = ExtractError::Io(io_err);
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_extract_error_failed_display() {
        let err = ExtractError::Failed("PDF parsing crashed".to_string());
        assert_eq!(err.to_string(), "extraction failed: PDF parsing crashed");
    }

    #[test]
    fn test_extract_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err: ExtractError = io_err.into();
        assert!(matches!(err, ExtractError::Io(_)));
    }

    // ========== ChunkError Tests ==========

    #[test]
    fn test_chunk_error_failed_display() {
        let err = ChunkError::Failed("empty content".to_string());
        assert_eq!(err.to_string(), "chunking failed: empty content");
    }

    #[test]
    fn test_chunk_error_invalid_config_display() {
        let err = ChunkError::InvalidConfig("target_size must be > 0".to_string());
        assert_eq!(
            err.to_string(),
            "invalid configuration: target_size must be > 0"
        );
    }

    // ========== EmbedError Tests ==========

    #[test]
    fn test_embed_error_model_load_display() {
        let err = EmbedError::ModelLoad("weights file not found".to_string());
        assert_eq!(
            err.to_string(),
            "model loading failed: weights file not found"
        );
    }

    #[test]
    fn test_embed_error_inference_display() {
        let err = EmbedError::Inference("CUDA out of memory".to_string());
        assert_eq!(err.to_string(), "inference failed: CUDA out of memory");
    }

    #[test]
    fn test_embed_error_modality_not_supported_display() {
        let err = EmbedError::ModalityNotSupported(Modality::Audio);
        assert_eq!(err.to_string(), "modality not supported: Audio");
    }

    #[test]
    fn test_embed_error_input_too_long_display() {
        let err = EmbedError::InputTooLong {
            tokens: 10000,
            max: 8192,
        };
        assert_eq!(err.to_string(), "input too long: 10000 tokens, max 8192");
    }

    // ========== StoreError Tests ==========

    #[test]
    fn test_store_error_init_display() {
        let err = StoreError::Init("database locked".to_string());
        assert_eq!(err.to_string(), "store initialization failed: database locked");
    }

    #[test]
    fn test_store_error_insert_display() {
        let err = StoreError::Insert("duplicate key".to_string());
        assert_eq!(err.to_string(), "insert failed: duplicate key");
    }

    #[test]
    fn test_store_error_query_display() {
        let err = StoreError::Query("invalid vector dimension".to_string());
        assert_eq!(err.to_string(), "query failed: invalid vector dimension");
    }

    #[test]
    fn test_store_error_delete_display() {
        let err = StoreError::Delete("file not indexed".to_string());
        assert_eq!(err.to_string(), "delete failed: file not indexed");
    }

    #[test]
    fn test_store_error_schema_display() {
        let err = StoreError::Schema("missing embedding column".to_string());
        assert_eq!(err.to_string(), "schema error: missing embedding column");
    }

    // ========== Main Error Tests ==========

    #[test]
    fn test_error_from_extract_error() {
        let extract_err = ExtractError::UnsupportedType("video/mp4".to_string());
        let err: Error = extract_err.into();
        assert!(matches!(err, Error::Extraction(_)));
        assert!(err.to_string().contains("video/mp4"));
    }

    #[test]
    fn test_error_from_chunk_error() {
        let chunk_err = ChunkError::Failed("too short".to_string());
        let err: Error = chunk_err.into();
        assert!(matches!(err, Error::Chunking(_)));
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn test_error_from_embed_error() {
        let embed_err = EmbedError::ModelLoad("missing model".to_string());
        let err: Error = embed_err.into();
        assert!(matches!(err, Error::Embedding(_)));
        assert!(err.to_string().contains("missing model"));
    }

    #[test]
    fn test_error_from_store_error() {
        let store_err = StoreError::Query("timeout".to_string());
        let err: Error = store_err.into();
        assert!(matches!(err, Error::Store(_)));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_error_config_display() {
        let err = Error::Config("invalid path".to_string());
        assert_eq!(err.to_string(), "config error: invalid path");
    }

    #[test]
    fn test_error_other_display() {
        let err = Error::Other("unexpected condition".to_string());
        assert_eq!(err.to_string(), "unexpected condition");
    }

    // ========== Error Debug Tests ==========

    #[test]
    fn test_extract_error_debug() {
        let err = ExtractError::Parse("bad format".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Parse"));
        assert!(debug_str.contains("bad format"));
    }

    #[test]
    fn test_chunk_error_debug() {
        let err = ChunkError::InvalidConfig("negative size".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidConfig"));
    }

    #[test]
    fn test_embed_error_debug() {
        let err = EmbedError::InputTooLong {
            tokens: 5000,
            max: 4096,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InputTooLong"));
        assert!(debug_str.contains("5000"));
    }

    #[test]
    fn test_store_error_debug() {
        let err = StoreError::Schema("wrong type".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Schema"));
    }

    #[test]
    fn test_main_error_debug() {
        let err = Error::Config("missing key".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Config"));
    }

    // ========== Error Chaining Tests ==========

    #[test]
    fn test_error_chain_io_to_extract_to_main() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file.txt not found");
        let extract_err: ExtractError = io_err.into();
        let main_err: Error = extract_err.into();

        assert!(matches!(main_err, Error::Extraction(ExtractError::Io(_))));
        assert!(main_err.to_string().contains("extraction error"));
    }

    #[test]
    fn test_result_type_alias() {
        fn example_function() -> Result<i32> {
            Ok(42)
        }

        fn failing_function() -> Result<i32> {
            Err(Error::Other("test failure".to_string()))
        }

        assert!(example_function().is_ok());
        assert!(failing_function().is_err());
    }

    // ========== All Modality Variants ==========

    #[test]
    fn test_embed_error_all_modalities() {
        let text_err = EmbedError::ModalityNotSupported(Modality::Text);
        assert!(text_err.to_string().contains("Text"));

        let image_err = EmbedError::ModalityNotSupported(Modality::Image);
        assert!(image_err.to_string().contains("Image"));

        let audio_err = EmbedError::ModalityNotSupported(Modality::Audio);
        assert!(audio_err.to_string().contains("Audio"));
    }
}
