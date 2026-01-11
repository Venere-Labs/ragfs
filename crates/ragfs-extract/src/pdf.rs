//! PDF content extractor.
//!
//! Uses pdf-extract to extract text content from PDF files.

use async_trait::async_trait;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
};
use std::path::Path;
use tracing::debug;

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
        let text = tokio::task::spawn_blocking(move || extract_pdf_text(&bytes))
            .await
            .map_err(|e| ExtractError::Failed(format!("Task join error: {e}")))?
            .map_err(|e| ExtractError::Failed(format!("PDF extraction failed: {e}")))?;

        // Split into pages/paragraphs for elements
        let elements = build_elements(&text);

        // Estimate page count from page breaks or text length
        let page_count = estimate_page_count(&text);

        Ok(ExtractedContent {
            text,
            elements,
            images: vec![], // TODO: Extract embedded images in future
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
