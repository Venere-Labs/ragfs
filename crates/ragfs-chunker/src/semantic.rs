//! Semantic chunking strategy.
//!
//! Chunks content based on semantic structure (headings, paragraphs, sections).
//! Best for markdown, documentation, and prose text.

use async_trait::async_trait;
use ragfs_core::{
    ChunkConfig, ChunkError, ChunkOutput, ChunkOutputMetadata, Chunker, ContentElement,
    ContentType, ExtractedContent,
};
use tracing::debug;

/// Semantic chunker that splits at document structure boundaries.
pub struct SemanticChunker;

impl SemanticChunker {
    /// Create a new semantic chunker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Chunker for SemanticChunker {
    fn name(&self) -> &str {
        "semantic"
    }

    fn content_types(&self) -> &[&str] {
        &["text", "markdown"]
    }

    fn can_chunk(&self, content_type: &ContentType) -> bool {
        matches!(content_type, ContentType::Text | ContentType::Markdown)
    }

    async fn chunk(
        &self,
        content: &ExtractedContent,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError> {
        let text = &content.text;
        if text.is_empty() {
            return Ok(vec![]);
        }

        debug!("Semantic chunking {} bytes", text.len());

        // If we have structured elements, use them
        if !content.elements.is_empty() {
            return chunk_from_elements(text, &content.elements, config);
        }

        // Otherwise, parse structure from text
        let sections = parse_sections(text);
        chunk_sections(text, &sections, config)
    }
}

/// A parsed section of the document.
#[derive(Debug)]
struct Section {
    heading: Option<String>,
    heading_level: u8,
    start_byte: usize,
    end_byte: usize,
    content: String,
}

/// Parse document into sections based on headings.
fn parse_sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut current_section = Section {
        heading: None,
        heading_level: 0,
        start_byte: 0,
        end_byte: 0,
        content: String::new(),
    };

    let lines: Vec<&str> = text.lines().collect();
    let mut byte_offset = 0;

    for (i, line) in lines.iter().enumerate() {
        // Check for markdown-style headings
        if let Some((level, heading_text)) = parse_markdown_heading(line) {
            // Save current section if it has content
            if !current_section.content.trim().is_empty() || current_section.heading.is_some() {
                current_section.end_byte = byte_offset;
                sections.push(current_section);
            }

            // Start new section
            current_section = Section {
                heading: Some(heading_text),
                heading_level: level,
                start_byte: byte_offset,
                end_byte: 0,
                content: String::new(),
            };
        }
        // Check for underline-style headings (=== or ---)
        else if i > 0 && is_underline_heading(line, &lines[..i]) {
            let prev_line = lines[i - 1];
            let level = if line.starts_with('=') { 1 } else { 2 };

            // Update the previous section to end before the heading
            if !sections.is_empty() {
                let last = sections.last_mut().unwrap();
                // Remove the heading text from previous section
                if last.content.ends_with(prev_line) {
                    last.content = last.content[..last.content.len() - prev_line.len()]
                        .trim_end()
                        .to_string();
                    last.end_byte = byte_offset - prev_line.len() - 1;
                }
            } else if current_section.content.ends_with(prev_line) {
                current_section.content = current_section.content
                    [..current_section.content.len() - prev_line.len()]
                    .trim_end()
                    .to_string();
            }

            // Save current section
            if !current_section.content.trim().is_empty() || current_section.heading.is_some() {
                current_section.end_byte = byte_offset - prev_line.len() - 1;
                sections.push(current_section);
            }

            // Start new section
            current_section = Section {
                heading: Some(prev_line.to_string()),
                heading_level: level,
                start_byte: byte_offset - prev_line.len() - 1,
                end_byte: 0,
                content: String::new(),
            };
        } else {
            // Add to current section
            if !current_section.content.is_empty() {
                current_section.content.push('\n');
            }
            current_section.content.push_str(line);
        }

        byte_offset += line.len() + 1; // +1 for newline
    }

    // Save final section
    current_section.end_byte = text.len();
    if !current_section.content.trim().is_empty() || current_section.heading.is_some() {
        sections.push(current_section);
    }

    sections
}

/// Parse a markdown-style heading (# Heading).
fn parse_markdown_heading(line: &str) -> Option<(u8, String)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }

    let hash_count = trimmed.chars().take_while(|c| *c == '#').count();
    if hash_count > 6 {
        return None; // Max 6 levels in markdown
    }

    let rest = &trimmed[hash_count..];
    if rest.is_empty() || !rest.starts_with(char::is_whitespace) {
        return None;
    }

    Some((hash_count as u8, rest.trim().to_string()))
}

/// Check if a line is an underline-style heading indicator.
fn is_underline_heading(line: &str, previous_lines: &[&str]) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Must be at least 3 characters
    if trimmed.len() < 3 {
        return false;
    }

    // Must be all = or all -
    let is_equals = trimmed.chars().all(|c| c == '=');
    let is_dashes = trimmed.chars().all(|c| c == '-');

    if !is_equals && !is_dashes {
        return false;
    }

    // Previous line must exist and not be empty
    if previous_lines.is_empty() {
        return false;
    }

    let prev = previous_lines.last().unwrap().trim();
    !prev.is_empty() && !prev.starts_with('#')
}

/// Chunk sections into appropriately-sized chunks.
fn chunk_sections(
    text: &str,
    sections: &[Section],
    config: &ChunkConfig,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let target_chars = config.target_size * chars_per_token;
    let max_chars = config.max_size * chars_per_token;

    let mut current_chunk = String::new();
    let mut chunk_start = 0;
    let mut current_heading: Option<String> = None;

    for section in sections {
        let section_text = if let Some(ref heading) = section.heading {
            format!(
                "{} {}\n\n{}",
                "#".repeat(section.heading_level as usize),
                heading,
                section.content.trim()
            )
        } else {
            section.content.trim().to_string()
        };

        // If adding this section would exceed max, flush current chunk
        if !current_chunk.is_empty() && current_chunk.len() + section_text.len() > max_chars {
            chunks.push(create_chunk(
                text,
                &current_chunk,
                chunk_start,
                current_heading.take(),
            ));
            current_chunk = String::new();
            chunk_start = section.start_byte;
        }

        // If section itself is too large, split it
        if section_text.len() > max_chars {
            // Flush current chunk first
            if !current_chunk.is_empty() {
                chunks.push(create_chunk(
                    text,
                    &current_chunk,
                    chunk_start,
                    current_heading.take(),
                ));
                current_chunk = String::new();
            }

            // Split large section by paragraphs
            let sub_chunks = split_large_section(&section_text, section.start_byte, config)?;
            for mut sub_chunk in sub_chunks {
                // Add section heading to first sub-chunk's metadata
                if chunks.len() == 0 || sub_chunk.metadata.symbol_name.is_none() {
                    sub_chunk.metadata.symbol_name = section.heading.clone();
                }
                chunks.push(sub_chunk);
            }
            chunk_start = section.end_byte;
        } else {
            // Add section to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(&section_text);

            // Update heading if this section has one
            if section.heading.is_some() {
                current_heading = section.heading.clone();
            }

            // If we've reached target size, consider flushing
            if current_chunk.len() >= target_chars {
                chunks.push(create_chunk(
                    text,
                    &current_chunk,
                    chunk_start,
                    current_heading.take(),
                ));
                current_chunk = String::new();
                chunk_start = section.end_byte;
            }
        }
    }

    // Flush remaining content
    if !current_chunk.is_empty() {
        chunks.push(create_chunk(
            text,
            &current_chunk,
            chunk_start,
            current_heading,
        ));
    }

    Ok(chunks)
}

/// Create a chunk output.
fn create_chunk(
    _text: &str,
    content: &str,
    start_byte: usize,
    heading: Option<String>,
) -> ChunkOutput {
    let line_count = content.matches('\n').count() as u32;

    ChunkOutput {
        content: content.to_string(),
        byte_range: start_byte as u64..(start_byte + content.len()) as u64,
        line_range: Some(0..line_count),
        parent_index: None,
        depth: 0,
        metadata: ChunkOutputMetadata {
            symbol_type: heading.as_ref().map(|_| "section".to_string()),
            symbol_name: heading,
            language: None,
        },
    }
}

/// Split a large section into smaller chunks.
fn split_large_section(
    text: &str,
    base_offset: usize,
    config: &ChunkConfig,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let target_chars = config.target_size * chars_per_token;
    let overlap_chars = config.overlap * chars_per_token;

    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut current = String::new();
    let mut current_offset = base_offset;

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        // If adding this paragraph exceeds target, flush
        if !current.is_empty() && current.len() + para.len() > target_chars {
            let line_count = current.matches('\n').count() as u32;
            chunks.push(ChunkOutput {
                content: current.clone(),
                byte_range: current_offset as u64..(current_offset + current.len()) as u64,
                line_range: Some(0..line_count),
                parent_index: None,
                depth: 0,
                metadata: ChunkOutputMetadata::default(),
            });

            // Keep overlap
            let overlap_start = current.len().saturating_sub(overlap_chars);
            let overlap = &current[overlap_start..];
            current = format!("{}\n\n{}", overlap, para);
            current_offset += overlap_start;
        } else {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }

    // Flush remaining
    if !current.is_empty() {
        let line_count = current.matches('\n').count() as u32;
        chunks.push(ChunkOutput {
            content: current.clone(),
            byte_range: current_offset as u64..(current_offset + current.len()) as u64,
            line_range: Some(0..line_count),
            parent_index: None,
            depth: 0,
            metadata: ChunkOutputMetadata::default(),
        });
    }

    Ok(chunks)
}

/// Chunk from structured elements.
fn chunk_from_elements(
    text: &str,
    elements: &[ContentElement],
    config: &ChunkConfig,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let target_chars = config.target_size * chars_per_token;
    let max_chars = config.max_size * chars_per_token;

    let mut current_chunk = String::new();
    let mut chunk_start = 0u64;
    let mut current_heading: Option<String> = None;

    for element in elements {
        let (elem_text, elem_offset, is_heading) = match element {
            ContentElement::Heading {
                level,
                text,
                byte_offset,
            } => {
                let heading_text = format!("{} {}", "#".repeat(*level as usize), text);
                (heading_text, *byte_offset, true)
            }
            ContentElement::Paragraph { text, byte_offset } => (text.clone(), *byte_offset, false),
            ContentElement::CodeBlock {
                language,
                code,
                byte_offset,
            } => {
                let lang = language.as_deref().unwrap_or("");
                let block = format!("```{}\n{}\n```", lang, code);
                (block, *byte_offset, false)
            }
            ContentElement::List {
                items,
                ordered,
                byte_offset,
            } => {
                let list_text = items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        if *ordered {
                            format!("{}. {}", i + 1, item)
                        } else {
                            format!("- {}", item)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                (list_text, *byte_offset, false)
            }
            ContentElement::Table {
                headers,
                rows,
                byte_offset,
            } => {
                let mut table = String::new();
                table.push_str(&format!("| {} |\n", headers.join(" | ")));
                table.push_str(&format!(
                    "| {} |\n",
                    headers.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")
                ));
                for row in rows {
                    table.push_str(&format!("| {} |\n", row.join(" | ")));
                }
                (table, *byte_offset, false)
            }
        };

        // If adding this element would exceed max, flush current chunk
        if !current_chunk.is_empty() && current_chunk.len() + elem_text.len() + 2 > max_chars {
            chunks.push(create_chunk(
                text,
                &current_chunk,
                chunk_start as usize,
                current_heading.take(),
            ));
            current_chunk = String::new();
            chunk_start = elem_offset;
        }

        // Add element to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(&elem_text);

        if is_heading {
            current_heading = Some(elem_text.trim_start_matches('#').trim().to_string());
        }

        // If we've reached target size, flush
        if current_chunk.len() >= target_chars {
            chunks.push(create_chunk(
                text,
                &current_chunk,
                chunk_start as usize,
                current_heading.take(),
            ));
            current_chunk = String::new();
            chunk_start = elem_offset + elem_text.len() as u64;
        }
    }

    // Flush remaining
    if !current_chunk.is_empty() {
        chunks.push(create_chunk(
            text,
            &current_chunk,
            chunk_start as usize,
            current_heading,
        ));
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragfs_core::ContentMetadataInfo;

    fn create_test_content(text: &str) -> ExtractedContent {
        ExtractedContent {
            text: text.to_string(),
            elements: vec![],
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        }
    }

    fn create_content_with_elements(text: &str, elements: Vec<ContentElement>) -> ExtractedContent {
        ExtractedContent {
            text: text.to_string(),
            elements,
            images: vec![],
            metadata: ContentMetadataInfo::default(),
        }
    }

    #[test]
    fn test_parse_markdown_heading() {
        assert_eq!(
            parse_markdown_heading("# Title"),
            Some((1, "Title".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("## Subtitle"),
            Some((2, "Subtitle".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("### Section"),
            Some((3, "Section".to_string()))
        );
        assert_eq!(parse_markdown_heading("Not a heading"), None);
        assert_eq!(parse_markdown_heading("#NoSpace"), None);
    }

    #[test]
    fn test_is_underline_heading() {
        assert!(is_underline_heading("===", &["Title"]));
        assert!(is_underline_heading("---", &["Subtitle"]));
        assert!(!is_underline_heading("===", &[""]));
        assert!(!is_underline_heading("---", &[]));
    }

    #[test]
    fn test_parse_sections() {
        let text = "# Introduction\n\nThis is the intro.\n\n## Details\n\nMore details here.";
        let sections = parse_sections(text);

        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, Some("Introduction".to_string()));
        assert_eq!(sections[1].heading, Some("Details".to_string()));
    }

    #[test]
    fn test_semantic_chunker_can_chunk() {
        let chunker = SemanticChunker::new();
        assert!(chunker.can_chunk(&ContentType::Text));
        assert!(chunker.can_chunk(&ContentType::Markdown));
        assert!(!chunker.can_chunk(&ContentType::Code {
            language: "rust".to_string(),
            symbol: None
        }));
    }

    #[test]
    fn test_chunker_name() {
        let chunker = SemanticChunker::new();
        assert_eq!(chunker.name(), "semantic");
    }

    #[test]
    fn test_chunker_content_types() {
        let chunker = SemanticChunker::new();
        let types = chunker.content_types();
        assert!(types.contains(&"text"));
        assert!(types.contains(&"markdown"));
    }

    #[test]
    fn test_default_implementation() {
        let chunker = SemanticChunker::default();
        assert_eq!(chunker.name(), "semantic");
    }

    #[tokio::test]
    async fn test_chunk_empty_text() {
        let chunker = SemanticChunker::new();
        let content = create_test_content("");
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn test_chunk_simple_text() {
        let chunker = SemanticChunker::new();
        let content = create_test_content("This is simple text without headings.");
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("simple text"));
    }

    #[tokio::test]
    async fn test_chunk_markdown_with_headings() {
        let chunker = SemanticChunker::new();
        let text = "# Introduction\n\nThis is the intro.\n\n## Details\n\nMore details here.";
        let content = create_test_content(text);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
        // Check that headings are preserved in output
        let all_content: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert!(all_content.contains("Introduction"));
        assert!(all_content.contains("Details"));
    }

    #[tokio::test]
    async fn test_chunk_with_structured_elements() {
        let chunker = SemanticChunker::new();
        let text = "# Title\n\nParagraph content.";
        let elements = vec![
            ContentElement::Heading {
                level: 1,
                text: "Title".to_string(),
                byte_offset: 0,
            },
            ContentElement::Paragraph {
                text: "Paragraph content.".to_string(),
                byte_offset: 9,
            },
        ];
        let content = create_content_with_elements(text, elements);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
    }

    #[tokio::test]
    async fn test_chunk_large_text_splits() {
        let chunker = SemanticChunker::new();
        // Create text with multiple sections that exceed target size
        let section = "# Section\n\nThis is a paragraph with some content.\n\n";
        let text = section.repeat(50);
        let content = create_test_content(&text);
        let config = ChunkConfig {
            target_size: 100,
            max_size: 200,
            overlap: 20,
            ..Default::default()
        };

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(chunks.len() > 1, "Large text should produce multiple chunks");
    }

    #[tokio::test]
    async fn test_chunk_preserves_heading_metadata() {
        let chunker = SemanticChunker::new();
        let text = "# Important Section\n\nContent under the section.";
        let content = create_test_content(text);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
        // First chunk should have section metadata
        if let Some(symbol_name) = &chunks[0].metadata.symbol_name {
            assert!(symbol_name.contains("Important Section"));
        }
    }

    #[tokio::test]
    async fn test_chunk_with_code_block_element() {
        let chunker = SemanticChunker::new();
        let text = "# Code Example\n\n```rust\nfn main() {}\n```";
        let elements = vec![
            ContentElement::Heading {
                level: 1,
                text: "Code Example".to_string(),
                byte_offset: 0,
            },
            ContentElement::CodeBlock {
                language: Some("rust".to_string()),
                code: "fn main() {}".to_string(),
                byte_offset: 16,
            },
        ];
        let content = create_content_with_elements(text, elements);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert!(all_content.contains("fn main()"));
    }

    #[tokio::test]
    async fn test_chunk_with_list_element() {
        let chunker = SemanticChunker::new();
        let text = "# Items\n\n- Item 1\n- Item 2\n- Item 3";
        let elements = vec![
            ContentElement::Heading {
                level: 1,
                text: "Items".to_string(),
                byte_offset: 0,
            },
            ContentElement::List {
                items: vec![
                    "Item 1".to_string(),
                    "Item 2".to_string(),
                    "Item 3".to_string(),
                ],
                ordered: false,
                byte_offset: 9,
            },
        ];
        let content = create_content_with_elements(text, elements);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert!(all_content.contains("Item 1"));
        assert!(all_content.contains("Item 2"));
    }

    #[tokio::test]
    async fn test_chunk_with_table_element() {
        let chunker = SemanticChunker::new();
        let text = "# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |";
        let elements = vec![
            ContentElement::Heading {
                level: 1,
                text: "Data".to_string(),
                byte_offset: 0,
            },
            ContentElement::Table {
                headers: vec!["A".to_string(), "B".to_string()],
                rows: vec![vec!["1".to_string(), "2".to_string()]],
                byte_offset: 8,
            },
        ];
        let content = create_content_with_elements(text, elements);
        let config = ChunkConfig::default();

        let chunks = chunker.chunk(&content, &config).await.unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_parse_sections_with_underline_headings() {
        let text = "Title\n=====\n\nContent under title.\n\nSubtitle\n--------\n\nMore content.";
        let sections = parse_sections(text);

        assert!(sections.len() >= 2);
    }

    #[test]
    fn test_parse_sections_plain_text() {
        let text = "Just some plain text\n\nwith multiple paragraphs\n\nbut no headings.";
        let sections = parse_sections(text);

        // Should still create at least one section
        assert!(!sections.is_empty());
    }

    #[test]
    fn test_parse_markdown_heading_levels() {
        assert_eq!(parse_markdown_heading("# H1"), Some((1, "H1".to_string())));
        assert_eq!(
            parse_markdown_heading("## H2"),
            Some((2, "H2".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("### H3"),
            Some((3, "H3".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("#### H4"),
            Some((4, "H4".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("##### H5"),
            Some((5, "H5".to_string()))
        );
        assert_eq!(
            parse_markdown_heading("###### H6"),
            Some((6, "H6".to_string()))
        );
        // 7 hashes is not valid
        assert_eq!(parse_markdown_heading("####### H7"), None);
    }
}
