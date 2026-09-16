//! Office document extractor (OOXML and ODT).
//!
//! Reads ZIP containers and pulls visible text from XML parts.
//! Supported: `.docx`, `.xlsx`, `.pptx`, `.odt`.
//! Not supported: legacy binary `.doc` / `.xls` / `.ppt`, RTF, EPUB.

use async_trait::async_trait;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
};
use std::io::{Cursor, Read};
use std::path::Path;
use tracing::debug;
use zip::ZipArchive;

const DOCX_MIME: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
const PPTX_MIME: &str = "application/vnd.openxmlformats-officedocument.presentationml.presentation";
const ODT_MIME: &str = "application/vnd.oasis.opendocument.text";

/// Office/OpenDocument text extractor.
pub struct OfficeExtractor;

impl OfficeExtractor {
    /// Create a new office extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for OfficeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OfficeKind {
    Docx,
    Xlsx,
    Pptx,
    Odt,
}

impl OfficeKind {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "docx" => Some(Self::Docx),
            "xlsx" => Some(Self::Xlsx),
            "pptx" => Some(Self::Pptx),
            "odt" => Some(Self::Odt),
            _ => None,
        }
    }

    fn from_mime(mime: &str) -> Option<Self> {
        match mime {
            DOCX_MIME => Some(Self::Docx),
            XLSX_MIME => Some(Self::Xlsx),
            PPTX_MIME => Some(Self::Pptx),
            ODT_MIME => Some(Self::Odt),
            _ => None,
        }
    }

    fn mime(self) -> &'static str {
        match self {
            Self::Docx => DOCX_MIME,
            Self::Xlsx => XLSX_MIME,
            Self::Pptx => PPTX_MIME,
            Self::Odt => ODT_MIME,
        }
    }
}

#[async_trait]
impl ContentExtractor for OfficeExtractor {
    fn supported_types(&self) -> &[&str] {
        &[DOCX_MIME, XLSX_MIME, PPTX_MIME, ODT_MIME]
    }

    fn can_extract_by_extension(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(OfficeKind::from_ext)
            .is_some()
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        debug!("Extracting office document: {:?}", path);
        let bytes = tokio::fs::read(path).await?;
        let kind = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(OfficeKind::from_ext)
            .ok_or_else(|| ExtractError::UnsupportedType(path.display().to_string()))?;
        extract_kind(&bytes, kind)
    }

    async fn extract_bytes(
        &self,
        data: &[u8],
        mime_type: &str,
    ) -> Result<ExtractedContent, ExtractError> {
        let kind = OfficeKind::from_mime(mime_type)
            .ok_or_else(|| ExtractError::UnsupportedType(mime_type.to_string()))?;
        extract_kind(data, kind)
    }
}

fn extract_kind(bytes: &[u8], kind: OfficeKind) -> Result<ExtractedContent, ExtractError> {
    let text = match kind {
        OfficeKind::Docx => extract_named_parts(bytes, |name| {
            name == "word/document.xml"
                || name.starts_with("word/header")
                || name.starts_with("word/footer")
        })?,
        OfficeKind::Xlsx => extract_named_parts(bytes, |name| {
            name == "xl/sharedStrings.xml" || name.starts_with("xl/worksheets/")
        })?,
        OfficeKind::Pptx => extract_named_parts(bytes, |name| {
            name.starts_with("ppt/slides/slide")
                && Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("xml"))
        })?,
        OfficeKind::Odt => extract_named_parts(bytes, |name| name == "content.xml")?,
    };

    if text.trim().is_empty() {
        return Err(ExtractError::Failed(format!(
            "no text extracted from {}",
            kind.mime()
        )));
    }

    let elements = text
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .scan(0u64, |offset, line| {
            let element = ContentElement::Paragraph {
                text: line.to_string(),
                byte_offset: *offset,
            };
            *offset += line.len() as u64 + 1;
            Some(element)
        })
        .collect();

    Ok(ExtractedContent {
        text,
        elements,
        images: vec![],
        metadata: ContentMetadataInfo::default(),
    })
}

fn extract_named_parts(
    bytes: &[u8],
    include: impl Fn(&str) -> bool,
) -> Result<String, ExtractError> {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor)
        .map_err(|e| ExtractError::Parse(format!("not a ZIP office document: {e}")))?;

    let mut names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            let file = archive.by_index(i).ok()?;
            let name = file.name().to_string();
            include(&name).then_some(name)
        })
        .collect();
    names.sort();

    let mut parts = Vec::new();
    for name in names {
        let mut file = archive
            .by_name(&name)
            .map_err(|e| ExtractError::Parse(format!("missing {name}: {e}")))?;
        let mut xml = String::new();
        file.read_to_string(&mut xml)
            .map_err(|e| ExtractError::Parse(format!("failed to read {name}: {e}")))?;
        let part = xml_to_text(&xml);
        if !part.is_empty() {
            parts.push(part);
        }
    }

    Ok(parts.join("\n"))
}

fn xml_to_text(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    while let Some(start) = rest.find('<') {
        if start > 0 {
            push_decoded(&mut out, &rest[..start]);
        }
        let after = &rest[start + 1..];
        let Some(end_rel) = after.find('>') else {
            break;
        };
        let tag = &after[..end_rel];
        let name = tag_name(tag);
        if tag.starts_with('/') && is_block_tag(name) {
            out.push('\n');
        } else if matches!(name, "w:tab" | "w:br" | "br") {
            out.push(' ');
        }
        rest = &after[end_rel + 1..];
    }
    if !rest.is_empty() {
        push_decoded(&mut out, rest);
    }
    normalize_ws(&out)
}

fn tag_name(tag: &str) -> &str {
    let trimmed = tag.trim_start_matches('/').trim_start_matches('?');
    trimmed
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("")
}

fn is_block_tag(name: &str) -> bool {
    matches!(
        name,
        "w:p" | "a:p" | "text:p" | "text:h" | "p" | "tr" | "si" | "c"
    )
}

fn push_decoded(out: &mut String, raw: &str) {
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '&' {
            let mut entity = String::new();
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == ';' {
                    break;
                }
                entity.push(next);
                if entity.len() > 10 {
                    break;
                }
            }
            match entity.as_str() {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                "nbsp" => out.push(' '),
                other if other.starts_with('#') => {
                    let code = if let Some(hex) = other.strip_prefix("#x") {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        other.strip_prefix('#').and_then(|n| n.parse().ok())
                    };
                    if let Some(ch) = code.and_then(char::from_u32) {
                        out.push(ch);
                    }
                }
                _ => {
                    out.push('&');
                    out.push_str(&entity);
                }
            }
        } else {
            out.push(c);
        }
    }
}

fn normalize_ws(text: &str) -> String {
    text.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    fn zip_with(files: &[(&str, &str)]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut zip = ZipWriter::new(&mut cursor);
            let opts = SimpleFileOptions::default();
            for (name, body) in files {
                zip.start_file(*name, opts).unwrap();
                zip.write_all(body.as_bytes()).unwrap();
            }
            zip.finish().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn xml_to_text_strips_tags_and_entities() {
        let xml = r"<w:p><w:r><w:t>Hello &amp; world</w:t></w:r></w:p>";
        assert_eq!(xml_to_text(xml), "Hello & world");
    }

    #[test]
    fn rejects_legacy_doc_extension() {
        let extractor = OfficeExtractor::new();
        assert!(!extractor.can_extract_by_extension(Path::new("report.doc")));
        assert!(extractor.can_extract_by_extension(Path::new("report.docx")));
    }

    #[tokio::test]
    async fn extracts_docx_paragraph() {
        let xml = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Indexed from DOCX</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let bytes = zip_with(&[("word/document.xml", xml)]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, DOCX_MIME).await.unwrap();
        assert!(content.text.contains("Indexed from DOCX"));
    }

    #[tokio::test]
    async fn extracts_xlsx_shared_strings() {
        let xml = r#"<?xml version="1.0"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>Revenue</t></si>
  <si><t>Q1 actuals</t></si>
</sst>"#;
        let bytes = zip_with(&[("xl/sharedStrings.xml", xml)]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, XLSX_MIME).await.unwrap();
        assert!(content.text.contains("Revenue"));
        assert!(content.text.contains("Q1 actuals"));
    }

    #[tokio::test]
    async fn extracts_pptx_slide() {
        let xml = r#"<?xml version="1.0"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:p><a:r><a:t>Slide title</a:t></a:r></a:p>
</p:sld>"#;
        let bytes = zip_with(&[("ppt/slides/slide1.xml", xml)]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, PPTX_MIME).await.unwrap();
        assert!(content.text.contains("Slide title"));
    }

    #[tokio::test]
    async fn extracts_odt_content() {
        let xml = r#"<?xml version="1.0"?>
<office:document-content xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <text:p>OpenDocument text</text:p>
</office:document-content>"#;
        let bytes = zip_with(&[("content.xml", xml)]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, ODT_MIME).await.unwrap();
        assert!(content.text.contains("OpenDocument text"));
    }

    #[tokio::test]
    async fn extract_bytes_rejects_unknown_mime() {
        let extractor = OfficeExtractor::new();
        let err = extractor
            .extract_bytes(b"not zip", "application/msword")
            .await
            .unwrap_err();
        assert!(matches!(err, ExtractError::UnsupportedType(_)));
    }
}
