//! Image content extractor.
//!
//! Extracts image files and prepares them for embedding.

use async_trait::async_trait;
use image::GenericImageView;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
    ExtractedImage,
};
use std::path::Path;
use tracing::debug;

/// Extractor for image files.
pub struct ImageExtractor;

impl ImageExtractor {
    /// Create a new image extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for ImageExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for ImageExtractor {
    fn supported_types(&self) -> &[&str] {
        &[
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
        ]
    }

    fn can_extract_by_extension(&self, path: &Path) -> bool {
        let extensions = ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"];

        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| extensions.contains(&ext.to_lowercase().as_str()))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        debug!("Extracting image: {:?}", path);

        // Read image file
        let bytes = tokio::fs::read(path).await?;

        // Decode image to get metadata (blocking operation)
        let (width, height, format) =
            tokio::task::spawn_blocking(move || decode_image_metadata(&bytes))
                .await
                .map_err(|e| ExtractError::Failed(format!("Task join error: {e}")))?
                .map_err(|e| ExtractError::Failed(format!("Image decode failed: {e}")))?;

        // Get MIME type
        let mime_type = mime_type_from_extension(path);

        // Read bytes again for storage (we consumed them in decoding)
        let data = tokio::fs::read(path).await?;

        // Create text representation with metadata
        let text = format!(
            "Image: {} ({}x{}, {})",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
            width,
            height,
            format
        );

        // Create the extracted image
        let extracted_image = ExtractedImage {
            data,
            mime_type: mime_type.clone(),
            caption: None, // Will be filled by vision model in future
            page: None,
        };

        Ok(ExtractedContent {
            text,
            elements: vec![ContentElement::Paragraph {
                text: format!("{width}x{height} {format} image"),
                byte_offset: 0,
            }],
            images: vec![extracted_image],
            metadata: ContentMetadataInfo {
                title: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(std::string::ToString::to_string),
                ..Default::default()
            },
        })
    }
}

/// Decode image to get dimensions and format.
fn decode_image_metadata(bytes: &[u8]) -> Result<(u32, u32, String), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("Failed to load image: {e}"))?;

    let (width, height) = img.dimensions();

    // Try to detect format
    let format = image::guess_format(bytes).map_or_else(
        |_| "unknown".to_string(),
        |f| format!("{f:?}").to_lowercase(),
    );

    Ok((width, height, format))
}

/// Get MIME type from file extension.
fn mime_type_from_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map_or("application/octet-stream", |ext| {
            match ext.to_lowercase().as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "bmp" => "image/bmp",
                "tiff" | "tif" => "image/tiff",
                _ => "application/octet-stream",
            }
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Create a simple 2x2 PNG image for testing
    fn create_test_png() -> Vec<u8> {
        use image::{ImageBuffer, Rgba};

        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(2, 2, |x, y| {
            if (x + y) % 2 == 0 {
                Rgba([255, 0, 0, 255]) // Red
            } else {
                Rgba([0, 255, 0, 255]) // Green
            }
        });

        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
        bytes
    }

    /// Create a simple 2x2 JPEG image for testing
    fn create_test_jpeg() -> Vec<u8> {
        use image::{ImageBuffer, Rgb};

        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(2, 2, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([255, 0, 0]) // Red
            } else {
                Rgb([0, 255, 0]) // Green
            }
        });

        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
        bytes
    }

    #[test]
    fn test_new_extractor() {
        let extractor = ImageExtractor::new();
        assert!(!extractor.supported_types().is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let extractor = ImageExtractor;
        assert!(!extractor.supported_types().is_empty());
    }

    #[test]
    fn test_supported_types_includes_common_formats() {
        let extractor = ImageExtractor::new();
        let types = extractor.supported_types();

        assert!(types.contains(&"image/jpeg"));
        assert!(types.contains(&"image/png"));
        assert!(types.contains(&"image/gif"));
        assert!(types.contains(&"image/webp"));
    }

    #[test]
    fn test_can_extract_by_extension() {
        let extractor = ImageExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("photo.jpg")));
        assert!(extractor.can_extract_by_extension(Path::new("image.PNG")));
        assert!(extractor.can_extract_by_extension(Path::new("animation.gif")));
        assert!(extractor.can_extract_by_extension(Path::new("photo.webp")));
        assert!(!extractor.can_extract_by_extension(Path::new("document.txt")));
        assert!(!extractor.can_extract_by_extension(Path::new("code.rs")));
    }

    #[test]
    fn test_can_extract_jpeg_variants() {
        let extractor = ImageExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("photo.jpg")));
        assert!(extractor.can_extract_by_extension(Path::new("photo.jpeg")));
        assert!(extractor.can_extract_by_extension(Path::new("photo.JPG")));
        assert!(extractor.can_extract_by_extension(Path::new("photo.JPEG")));
    }

    #[test]
    fn test_can_extract_tiff_variants() {
        let extractor = ImageExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("image.tiff")));
        assert!(extractor.can_extract_by_extension(Path::new("image.tif")));
    }

    #[test]
    fn test_cannot_extract_non_image() {
        let extractor = ImageExtractor::new();

        assert!(!extractor.can_extract_by_extension(Path::new("file.pdf")));
        assert!(!extractor.can_extract_by_extension(Path::new("file.mp4")));
        assert!(!extractor.can_extract_by_extension(Path::new("file.zip")));
    }

    #[test]
    fn test_mime_type_from_extension() {
        assert_eq!(
            mime_type_from_extension(Path::new("test.jpg")),
            "image/jpeg"
        );
        assert_eq!(mime_type_from_extension(Path::new("test.png")), "image/png");
        assert_eq!(mime_type_from_extension(Path::new("test.gif")), "image/gif");
    }

    #[test]
    fn test_mime_type_from_extension_case_insensitive() {
        assert_eq!(
            mime_type_from_extension(Path::new("test.JPG")),
            "image/jpeg"
        );
        assert_eq!(mime_type_from_extension(Path::new("test.PNG")), "image/png");
    }

    #[test]
    fn test_mime_type_unknown_extension() {
        assert_eq!(
            mime_type_from_extension(Path::new("test.xyz")),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn test_extract_png_image() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.png");
        std::fs::write(&file_path, create_test_png()).unwrap();

        let extractor = ImageExtractor::new();
        let result = extractor.extract(&file_path).await;

        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(content.text.contains("test.png"));
        assert!(content.text.contains("2x2")); // Dimensions
        assert_eq!(content.images.len(), 1);
        assert_eq!(content.images[0].mime_type, "image/png");
    }

    #[tokio::test]
    async fn test_extract_jpeg_image() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.jpg");
        std::fs::write(&file_path, create_test_jpeg()).unwrap();

        let extractor = ImageExtractor::new();
        let result = extractor.extract(&file_path).await;

        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(content.text.contains("test.jpg"));
        assert_eq!(content.images.len(), 1);
        assert_eq!(content.images[0].mime_type, "image/jpeg");
    }

    #[tokio::test]
    async fn test_extract_creates_paragraph_element() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("photo.png");
        std::fs::write(&file_path, create_test_png()).unwrap();

        let extractor = ImageExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.elements.len(), 1);
        match &content.elements[0] {
            ragfs_core::ContentElement::Paragraph { text, .. } => {
                assert!(text.contains("2x2"));
                assert!(text.contains("image"));
            }
            _ => panic!("Expected Paragraph element"),
        }
    }

    #[tokio::test]
    async fn test_extract_stores_image_data() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("image.png");
        let png_data = create_test_png();
        std::fs::write(&file_path, &png_data).unwrap();

        let extractor = ImageExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.images.len(), 1);
        assert_eq!(content.images[0].data, png_data);
    }

    #[tokio::test]
    async fn test_extract_sets_title_metadata() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("my_photo.png");
        std::fs::write(&file_path, create_test_png()).unwrap();

        let extractor = ImageExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.metadata.title, Some("my_photo.png".to_string()));
    }

    #[tokio::test]
    async fn test_extract_nonexistent_file_fails() {
        let extractor = ImageExtractor::new();
        let result = extractor.extract(Path::new("/nonexistent/image.png")).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_invalid_image_fails() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("fake.png");
        // Write non-image data
        std::fs::write(&file_path, b"This is not an image").unwrap();

        let extractor = ImageExtractor::new();
        let result = extractor.extract(&file_path).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_decode_image_metadata() {
        let png_data = create_test_png();
        let result = decode_image_metadata(&png_data);

        assert!(result.is_ok());
        let (width, height, format) = result.unwrap();
        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert!(format.contains("png"));
    }

    #[test]
    fn test_decode_image_metadata_invalid() {
        let result = decode_image_metadata(b"not an image");
        assert!(result.is_err());
    }
}
