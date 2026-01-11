//! Extractor registry for managing content extractors.

use ragfs_core::{ContentExtractor, ExtractError, ExtractedContent};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Registry of content extractors.
pub struct ExtractorRegistry {
    /// Named extractors
    extractors: HashMap<String, Arc<dyn ContentExtractor>>,
    /// MIME type to extractor name mapping
    mime_mapping: HashMap<String, String>,
}

impl ExtractorRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extractors: HashMap::new(),
            mime_mapping: HashMap::new(),
        }
    }

    /// Register an extractor.
    pub fn register<E: ContentExtractor + 'static>(&mut self, name: &str, extractor: E) {
        let extractor = Arc::new(extractor);
        for mime in extractor.supported_types() {
            self.mime_mapping
                .insert((*mime).to_string(), name.to_string());
        }
        self.extractors.insert(name.to_string(), extractor);
    }

    /// Get an extractor for a MIME type.
    #[must_use]
    pub fn get_for_mime(&self, mime_type: &str) -> Option<Arc<dyn ContentExtractor>> {
        self.mime_mapping
            .get(mime_type)
            .and_then(|name| self.extractors.get(name))
            .cloned()
    }

    /// Get an extractor that can handle a file.
    #[must_use]
    pub fn get_for_file(&self, path: &Path, mime_type: &str) -> Option<Arc<dyn ContentExtractor>> {
        // First try by MIME type
        if let Some(extractor) = self.get_for_mime(mime_type) {
            return Some(extractor);
        }

        // Then try by extension
        for extractor in self.extractors.values() {
            if extractor.can_extract(path, mime_type) {
                return Some(extractor.clone());
            }
        }

        None
    }

    /// Extract content from a file.
    pub async fn extract(
        &self,
        path: &Path,
        mime_type: &str,
    ) -> Result<ExtractedContent, ExtractError> {
        let extractor = self
            .get_for_file(path, mime_type)
            .ok_or_else(|| ExtractError::UnsupportedType(mime_type.to_string()))?;

        extractor.extract(path).await
    }
}

impl Default for ExtractorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextExtractor;
    use tempfile::tempdir;

    #[test]
    fn test_new_registry_is_empty() {
        let registry = ExtractorRegistry::new();
        assert!(registry.extractors.is_empty());
        assert!(registry.mime_mapping.is_empty());
    }

    #[test]
    fn test_register_extractor() {
        let mut registry = ExtractorRegistry::new();
        registry.register("text", TextExtractor::new());

        assert!(registry.extractors.contains_key("text"));
        // TextExtractor supports text/plain and text/markdown
        assert!(registry.mime_mapping.contains_key("text/plain"));
    }

    #[test]
    fn test_get_for_mime_existing() {
        let mut registry = ExtractorRegistry::new();
        registry.register("text", TextExtractor::new());

        let extractor = registry.get_for_mime("text/plain");
        assert!(extractor.is_some());
    }

    #[test]
    fn test_get_for_mime_nonexistent() {
        let registry = ExtractorRegistry::new();
        let extractor = registry.get_for_mime("video/mp4");
        assert!(extractor.is_none());
    }

    #[test]
    fn test_get_for_file_by_mime() {
        let mut registry = ExtractorRegistry::new();
        registry.register("text", TextExtractor::new());

        let path = std::path::PathBuf::from("/test/file.txt");
        let extractor = registry.get_for_file(&path, "text/plain");
        assert!(extractor.is_some());
    }

    #[test]
    fn test_get_for_file_unknown_type() {
        let registry = ExtractorRegistry::new();
        let path = std::path::PathBuf::from("/test/file.xyz");
        let extractor = registry.get_for_file(&path, "application/unknown");
        assert!(extractor.is_none());
    }

    #[tokio::test]
    async fn test_extract_success() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Hello, world!").unwrap();

        let mut registry = ExtractorRegistry::new();
        registry.register("text", TextExtractor::new());

        let result = registry.extract(&file_path, "text/plain").await;
        assert!(result.is_ok());

        let content = result.unwrap();
        assert_eq!(content.text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_extract_unsupported_type() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");
        std::fs::write(&file_path, [0u8; 10]).unwrap();

        let registry = ExtractorRegistry::new();

        let result = registry
            .extract(&file_path, "application/octet-stream")
            .await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ExtractError::UnsupportedType(mime) => {
                assert_eq!(mime, "application/octet-stream");
            }
            _ => panic!("Expected UnsupportedType error"),
        }
    }

    #[test]
    fn test_multiple_extractors() {
        let mut registry = ExtractorRegistry::new();
        registry.register("text", TextExtractor::new());
        // Could register more extractors here

        assert_eq!(registry.extractors.len(), 1);
        // TextExtractor registers multiple MIME types
        assert!(!registry.mime_mapping.is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let registry = ExtractorRegistry::default();
        assert!(registry.extractors.is_empty());
    }
}
