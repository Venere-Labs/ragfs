//! PDF content extractor.
//!
//! Uses pdf-extract to extract text content and lopdf for embedded images.

use async_trait::async_trait;
use flate2::read::ZlibDecoder;
use lopdf::Document;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
    ExtractedImage,
};
use std::io::Read;
use std::path::Path;
use tracing::{debug, warn};

/// Extractor for PDF files.
pub struct PdfExtractor;

impl PdfExtractor {
    /// Create a new PDF extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for PdfExtractor {
    fn supported_types(&self) -> &[&str] {
        &["application/pdf"]
    }

    fn can_extract_by_extension(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        debug!("Extracting PDF: {:?}", path);

        // Read PDF file
        let bytes = tokio::fs::read(path).await?;

        // Extract text using pdf-extract (blocking operation)
        let text = tokio::task::spawn_blocking({
            let bytes = bytes.clone();
            move || extract_pdf_text(&bytes)
        })
        .await
        .map_err(|e| ExtractError::Failed(format!("Task join error: {e}")))?
        .map_err(|e| ExtractError::Failed(format!("PDF extraction failed: {e}")))?;

        // Extract images using lopdf (blocking operation)
        let images = tokio::task::spawn_blocking(move || extract_pdf_images(&bytes))
            .await
            .map_err(|e| ExtractError::Failed(format!("Image extraction task error: {e}")))?;

        // Split into pages/paragraphs for elements
        let elements = build_elements(&text);

        // Estimate page count from page breaks or text length
        let page_count = estimate_page_count(&text);

        Ok(ExtractedContent {
            text,
            elements,
            images,
            metadata: ContentMetadataInfo {
                page_count: Some(page_count),
                ..Default::default()
            },
        })
    }
}

/// Extract text from PDF bytes using pdf-extract.
fn extract_pdf_text(bytes: &[u8]) -> Result<String, String> {
    pdf_extract::extract_text_from_mem(bytes).map_err(|e| e.to_string())
}

/// Configuration for image extraction limits.
const MAX_IMAGES: usize = 100;
const MAX_TOTAL_BYTES: usize = 50 * 1024 * 1024; // 50MB
const MIN_DIMENSION: u32 = 50; // Skip tiny images (icons, etc.)

/// Extract images from PDF document using lopdf.
fn extract_pdf_images(bytes: &[u8]) -> Vec<ExtractedImage> {
    let doc = match Document::load_mem(bytes) {
        Ok(d) => d,
        Err(e) => {
            warn!("Failed to load PDF for image extraction: {}", e);
            return vec![];
        }
    };

    let mut images = Vec::new();
    let mut total_bytes = 0usize;

    let pages = doc.get_pages();
    for (page_num, page_id) in pages {
        if images.len() >= MAX_IMAGES {
            debug!(
                "Reached maximum image count ({}), stopping extraction",
                MAX_IMAGES
            );
            break;
        }

        match doc.get_page_images(page_id) {
            Ok(page_images) => {
                for pdf_image in page_images {
                    if images.len() >= MAX_IMAGES || total_bytes >= MAX_TOTAL_BYTES {
                        break;
                    }

                    // Skip tiny images
                    if pdf_image.width < i64::from(MIN_DIMENSION)
                        || pdf_image.height < i64::from(MIN_DIMENSION)
                    {
                        debug!(
                            "Skipping small image: {}x{}",
                            pdf_image.width, pdf_image.height
                        );
                        continue;
                    }

                    if let Some(extracted) = decode_pdf_image(&pdf_image, page_num) {
                        total_bytes += extracted.data.len();
                        images.push(extracted);
                    }
                }
            }
            Err(e) => {
                debug!("Failed to get images from page {}: {}", page_num, e);
            }
        }
    }

    debug!(
        "Extracted {} images from PDF ({} bytes total)",
        images.len(),
        total_bytes
    );
    images
}

/// Decode a PDF image into `ExtractedImage` format.
fn decode_pdf_image(pdf_image: &lopdf::xobject::PdfImage, page_num: u32) -> Option<ExtractedImage> {
    let filters = pdf_image.filters.as_ref()?;

    // Determine MIME type and decode based on filter
    let (data, mime_type) = if filters.iter().any(|f| f == "DCTDecode") {
        // JPEG - can use raw content directly
        (pdf_image.content.to_vec(), "image/jpeg".to_string())
    } else if filters.iter().any(|f| f == "FlateDecode") {
        // Compressed raw image data - decompress and convert to PNG
        match decode_flate_image(pdf_image) {
            Ok((data, mime)) => (data, mime),
            Err(e) => {
                debug!("Failed to decode FlateDecode image: {}", e);
                return None;
            }
        }
    } else if filters.iter().any(|f| f == "JPXDecode") {
        // JPEG 2000 - use raw content
        (pdf_image.content.to_vec(), "image/jp2".to_string())
    } else {
        // Unsupported filter
        debug!("Unsupported image filter: {:?}", filters);
        return None;
    };

    Some(ExtractedImage {
        data,
        mime_type,
        caption: None, // Will be filled by vision model in future
        page: Some(page_num),
    })
}

/// Decode `FlateDecode` compressed image to PNG.
fn decode_flate_image(pdf_image: &lopdf::xobject::PdfImage) -> Result<(Vec<u8>, String), String> {
    // Decompress the data
    let mut decoder = ZlibDecoder::new(pdf_image.content);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| format!("Decompression failed: {e}"))?;

    // Determine color space and create image
    let color_space = pdf_image.color_space.as_deref().unwrap_or("DeviceRGB");
    let width = pdf_image.width as u32;
    let height = pdf_image.height as u32;

    let img = match color_space {
        "DeviceRGB" | "RGB" => image::RgbImage::from_raw(width, height, decompressed)
            .map(image::DynamicImage::ImageRgb8),
        "DeviceGray" | "Gray" => image::GrayImage::from_raw(width, height, decompressed)
            .map(image::DynamicImage::ImageLuma8),
        "DeviceCMYK" | "CMYK" => {
            // Convert CMYK to RGB
            let rgb_data = cmyk_to_rgb(&decompressed);
            image::RgbImage::from_raw(width, height, rgb_data).map(image::DynamicImage::ImageRgb8)
        }
        _ => {
            // Attempt RGB as fallback
            debug!("Unknown color space '{}', attempting RGB", color_space);
            image::RgbImage::from_raw(width, height, decompressed)
                .map(image::DynamicImage::ImageRgb8)
        }
    };

    let img = img.ok_or_else(|| "Failed to create image from raw data".to_string())?;

    // Encode to PNG
    let mut png_data = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_data),
        image::ImageFormat::Png,
    )
    .map_err(|e| format!("PNG encoding failed: {e}"))?;

    Ok((png_data, "image/png".to_string()))
}

/// Convert CMYK bytes to RGB.
#[allow(clippy::many_single_char_names)]
fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((cmyk.len() / 4) * 3);
    for chunk in cmyk.chunks_exact(4) {
        let c = f32::from(chunk[0]) / 255.0;
        let m = f32::from(chunk[1]) / 255.0;
        let y = f32::from(chunk[2]) / 255.0;
        let k = f32::from(chunk[3]) / 255.0;

        let r = 255.0 * (1.0 - c) * (1.0 - k);
        let g = 255.0 * (1.0 - m) * (1.0 - k);
        let b = 255.0 * (1.0 - y) * (1.0 - k);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            rgb.push(r as u8);
            rgb.push(g as u8);
            rgb.push(b as u8);
        }
    }
    rgb
}

/// Build `ContentElements` from extracted text.
fn build_elements(text: &str) -> Vec<ContentElement> {
    let mut elements = Vec::new();
    let mut current_offset = 0u64;

    // Split by double newlines to get paragraphs
    for paragraph in text.split("\n\n") {
        let trimmed = paragraph.trim();
        if trimmed.is_empty() {
            current_offset += paragraph.len() as u64 + 2; // +2 for \n\n
            continue;
        }

        // Check if it looks like a heading (short, possibly capitalized)
        if looks_like_heading(trimmed) {
            elements.push(ContentElement::Heading {
                level: 1,
                text: trimmed.to_string(),
                byte_offset: current_offset,
            });
        } else {
            elements.push(ContentElement::Paragraph {
                text: trimmed.to_string(),
                byte_offset: current_offset,
            });
        }

        current_offset += paragraph.len() as u64 + 2;
    }

    elements
}

/// Heuristic to detect if text looks like a heading.
fn looks_like_heading(text: &str) -> bool {
    // Short text (likely a title/heading)
    if text.len() > 100 {
        return false;
    }

    // No period at end (headings typically don't end with periods)
    if text.ends_with('.') {
        return false;
    }

    // Single line
    if text.contains('\n') {
        return false;
    }

    // All caps or title case with few words
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 8 {
        // Check if mostly capitalized
        let caps_count = words
            .iter()
            .filter(|w| w.chars().next().is_some_and(char::is_uppercase))
            .count();
        return caps_count >= words.len() / 2;
    }

    false
}

/// Estimate page count from text.
fn estimate_page_count(text: &str) -> u32 {
    // Look for form feed characters (page breaks)
    let form_feeds = text.matches('\x0C').count();
    if form_feeds > 0 {
        return (form_feeds + 1) as u32;
    }

    // Estimate based on character count (~3000 chars per page average)
    let chars = text.len();
    std::cmp::max(1, (chars / 3000) as u32)
}

// ============================================================================
// Alternative PDF extractor using pdf_oxide (optional feature)
// ============================================================================

/// Alternative PDF extractor using the pdf_oxide library.
///
/// This extractor provides potentially better performance and a cleaner API
/// compared to the default pdf-extract + lopdf combination.
///
/// Enable with: `cargo build --features pdf_oxide`
#[cfg(feature = "pdf_oxide")]
pub struct PdfOxideExtractor;

#[cfg(feature = "pdf_oxide")]
impl PdfOxideExtractor {
    /// Create a new PDF oxide extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "pdf_oxide")]
impl Default for PdfOxideExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "pdf_oxide")]
#[async_trait]
impl ContentExtractor for PdfOxideExtractor {
    fn supported_types(&self) -> &[&str] {
        &["application/pdf"]
    }

    fn can_extract_by_extension(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        debug!("Extracting PDF with pdf_oxide: {:?}", path);

        let path_owned = path.to_path_buf();

        // pdf_oxide operations are blocking
        let (text, page_count, images) =
            tokio::task::spawn_blocking(move || extract_with_pdf_oxide(&path_owned))
                .await
                .map_err(|e| ExtractError::Failed(format!("Task join error: {e}")))?
                .map_err(|e| ExtractError::Failed(format!("PDF extraction failed: {e}")))?;

        let elements = build_elements(&text);

        Ok(ExtractedContent {
            text,
            elements,
            images,
            metadata: ContentMetadataInfo {
                page_count: Some(page_count),
                ..Default::default()
            },
        })
    }
}

#[cfg(feature = "pdf_oxide")]
fn extract_with_pdf_oxide(
    path: &std::path::Path,
) -> Result<(String, u32, Vec<ExtractedImage>), String> {
    use pdf_oxide::PdfDocument;

    let mut doc = PdfDocument::open(path).map_err(|e| format!("Failed to open PDF: {e}"))?;

    let page_count = doc
        .page_count()
        .map_err(|e| format!("Failed to get page count: {e}"))?;

    // Extract text from all pages
    let mut text = String::new();
    for page_idx in 0..page_count {
        match doc.extract_text(page_idx) {
            Ok(page_text) => {
                text.push_str(&page_text);
                text.push_str("\n\n");
            }
            Err(e) => {
                debug!("Failed to extract text from page {}: {}", page_idx + 1, e);
            }
        }
    }

    // Extract images from all pages
    let mut images = Vec::new();
    let mut total_bytes = 0usize;

    for page_idx in 0..page_count {
        if images.len() >= MAX_IMAGES || total_bytes >= MAX_TOTAL_BYTES {
            break;
        }

        match doc.extract_images(page_idx) {
            Ok(page_images) => {
                for img in page_images {
                    if images.len() >= MAX_IMAGES || total_bytes >= MAX_TOTAL_BYTES {
                        break;
                    }

                    // Get image data and determine MIME type
                    let data = img.data;
                    let mime_type = match img.format.as_deref() {
                        Some("jpeg") | Some("jpg") => "image/jpeg",
                        Some("png") => "image/png",
                        Some("jp2") | Some("jpeg2000") => "image/jp2",
                        _ => "image/png", // default assumption
                    };

                    total_bytes += data.len();
                    images.push(ExtractedImage {
                        data,
                        mime_type: mime_type.to_string(),
                        caption: None,
                        page: Some(page_idx as u32 + 1),
                    });
                }
            }
            Err(e) => {
                debug!("Failed to extract images from page {}: {}", page_idx + 1, e);
            }
        }
    }

    debug!(
        "pdf_oxide: Extracted {} pages, {} chars, {} images",
        page_count,
        text.len(),
        images.len()
    );

    Ok((text, page_count as u32, images))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_looks_like_heading() {
        assert!(looks_like_heading("Chapter 1"));
        assert!(looks_like_heading("INTRODUCTION"));
        assert!(looks_like_heading("The Quick Brown Fox"));
        assert!(!looks_like_heading("This is a normal sentence."));
        assert!(!looks_like_heading(
            "This is a very long paragraph that goes on and on and definitely is not a heading"
        ));
    }

    #[test]
    fn test_estimate_page_count() {
        assert_eq!(estimate_page_count("short"), 1);
        assert_eq!(estimate_page_count(&"x".repeat(6000)), 2);
        assert_eq!(estimate_page_count("page1\x0Cpage2\x0Cpage3"), 3);
    }

    #[test]
    fn test_build_elements() {
        let text = "Title\n\nFirst paragraph here.\n\nSecond paragraph.";
        let elements = build_elements(text);
        assert_eq!(elements.len(), 3);
    }
}
