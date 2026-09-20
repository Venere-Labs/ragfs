//! Office document extractor (OOXML and ODT).
//!
//! Reads ZIP containers and pulls visible text from XML parts.
//! Supported: `.docx`, `.xlsx`, `.pptx`, `.odt`.
//! Not supported: legacy binary `.doc` / `.xls` / `.ppt`, RTF, EPUB.

use async_trait::async_trait;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
};
use std::cmp::Ordering;
use std::io::{Cursor, Read};
use std::path::Path;
use tracing::debug;
use zip::ZipArchive;

const DOCX_MIME: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
const PPTX_MIME: &str = "application/vnd.openxmlformats-officedocument.presentationml.presentation";
const ODT_MIME: &str = "application/vnd.oasis.opendocument.text";

/// Per-entry uncompressed cap (zip-bomb guard).
const MAX_ENTRY_UNCOMPRESSED: u64 = 8 * 1024 * 1024;
/// Aggregate uncompressed cap across extracted XML parts.
const MAX_TOTAL_UNCOMPRESSED: u64 = 32 * 1024 * 1024;
/// Upper bound on spaces expanded from one ODT `text:c` count.
const MAX_ODT_SPACES: usize = 255;

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
        OfficeKind::Xlsx => extract_xlsx(bytes)?,
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

fn open_zip(bytes: &[u8]) -> Result<ZipArchive<Cursor<&[u8]>>, ExtractError> {
    ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| ExtractError::Parse(format!("not a ZIP office document: {e}")))
}

fn collect_part_names(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    include: impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            let file = archive.by_index(i).ok()?;
            let name = file.name().to_string();
            include(&name).then_some(name)
        })
        .collect();
    names.sort_by(|a, b| natural_cmp(a, b));
    names
}

fn extract_named_parts(
    bytes: &[u8],
    include: impl Fn(&str) -> bool,
) -> Result<String, ExtractError> {
    let mut archive = open_zip(bytes)?;
    let names = collect_part_names(&mut archive, include);
    let mut remaining = MAX_TOTAL_UNCOMPRESSED;
    let mut parts = Vec::new();
    for name in names {
        let mut file = archive
            .by_name(&name)
            .map_err(|e| ExtractError::Parse(format!("missing {name}: {e}")))?;
        let xml = read_xml_part(&mut file, &mut remaining, MAX_ENTRY_UNCOMPRESSED)?;
        let part = xml_to_text(&xml);
        if !part.is_empty() {
            parts.push(part);
        }
    }
    Ok(parts.join("\n"))
}

fn extract_xlsx(bytes: &[u8]) -> Result<String, ExtractError> {
    let mut archive = open_zip(bytes)?;
    let mut remaining = MAX_TOTAL_UNCOMPRESSED;

    let sst = if archive.by_name("xl/sharedStrings.xml").is_ok() {
        let mut file = archive
            .by_name("xl/sharedStrings.xml")
            .map_err(|e| ExtractError::Parse(format!("missing sharedStrings: {e}")))?;
        let xml = read_xml_part(&mut file, &mut remaining, MAX_ENTRY_UNCOMPRESSED)?;
        drop(file);
        parse_shared_strings(&xml)
    } else {
        Vec::new()
    };

    let sheets = collect_part_names(&mut archive, |name| {
        name.starts_with("xl/worksheets/")
            && Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("xml"))
    });

    let mut parts = Vec::new();
    for name in sheets {
        let mut file = archive
            .by_name(&name)
            .map_err(|e| ExtractError::Parse(format!("missing {name}: {e}")))?;
        let xml = read_xml_part(&mut file, &mut remaining, MAX_ENTRY_UNCOMPRESSED)?;
        drop(file);
        let part = xlsx_sheet_to_text(&xml, &sst);
        if !part.is_empty() {
            parts.push(part);
        }
    }
    Ok(parts.join("\n"))
}

fn read_xml_part(
    file: &mut zip::read::ZipFile<'_>,
    remaining: &mut u64,
    max_entry: u64,
) -> Result<String, ExtractError> {
    let bytes = read_entry_bytes(file, remaining, max_entry)?;
    decode_xml_part(&bytes)
}

fn read_entry_bytes(
    file: &mut zip::read::ZipFile<'_>,
    remaining: &mut u64,
    max_entry: u64,
) -> Result<Vec<u8>, ExtractError> {
    let name = file.name().to_string();
    let declared = file.size();
    if declared > max_entry {
        return Err(ExtractError::Failed(format!(
            "office ZIP entry {name} exceeds {max_entry} uncompressed bytes"
        )));
    }
    if declared > *remaining {
        return Err(ExtractError::Failed(format!(
            "office ZIP aggregate uncompressed limit exceeded at {name}"
        )));
    }

    let mut buf = Vec::new();
    let mut limited = file.take(max_entry.saturating_add(1));
    limited
        .read_to_end(&mut buf)
        .map_err(|e| ExtractError::Parse(format!("failed to read {name}: {e}")))?;
    let len = buf.len() as u64;
    if len > max_entry {
        return Err(ExtractError::Failed(format!(
            "office ZIP entry {name} exceeds {max_entry} uncompressed bytes"
        )));
    }
    if len > *remaining {
        return Err(ExtractError::Failed(format!(
            "office ZIP aggregate uncompressed limit exceeded at {name}"
        )));
    }
    *remaining -= len;
    Ok(buf)
}

fn decode_xml_part(bytes: &[u8]) -> Result<String, ExtractError> {
    if bytes.starts_with(&[0xFF, 0xFE]) {
        return decode_utf16(&bytes[2..], true);
    }
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return decode_utf16(&bytes[2..], false);
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => decode_utf16(bytes, true).or_else(|_| decode_utf16(bytes, false)),
    }
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> Result<String, ExtractError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(ExtractError::Parse(
            "UTF-16 XML part has an odd number of bytes".into(),
        ));
    }
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| {
            if little_endian {
                u16::from_le_bytes(c)
            } else {
                u16::from_be_bytes(c)
            }
        })
        .collect();
    String::from_utf16(&units)
        .map_err(|e| ExtractError::Parse(format!("invalid UTF-16 XML part: {e}")))
}

fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();
    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ac), Some(bc)) if ac.is_ascii_digit() && bc.is_ascii_digit() => {
                let mut an = 0u64;
                while matches!(ai.peek(), Some(c) if c.is_ascii_digit()) {
                    let d = ai.next().expect("digit") as u32 - u32::from(b'0');
                    an = an.saturating_mul(10).saturating_add(u64::from(d));
                }
                let mut bn = 0u64;
                while matches!(bi.peek(), Some(c) if c.is_ascii_digit()) {
                    let d = bi.next().expect("digit") as u32 - u32::from(b'0');
                    bn = bn.saturating_mul(10).saturating_add(u64::from(d));
                }
                match an.cmp(&bn) {
                    Ordering::Equal => {}
                    other => return other,
                }
            }
            (Some(_), Some(_)) => {
                let ac = ai.next().expect("char");
                let bc = bi.next().expect("char");
                match ac.cmp(&bc) {
                    Ordering::Equal => {}
                    other => return other,
                }
            }
        }
    }
}

fn parse_shared_strings(xml: &str) -> Vec<String> {
    inner_elements(xml, "si")
        .into_iter()
        .map(xml_to_text)
        .collect()
}

fn xlsx_sheet_to_text(xml: &str, sst: &[String]) -> String {
    let mut out = String::new();
    let mut rest = xml;
    let mut shared = false;
    let mut in_v = false;
    let mut in_f = false;
    let mut index_buf = String::new();

    while let Some(start) = rest.find('<') {
        if start > 0 {
            if shared && in_v {
                index_buf.push_str(&rest[..start]);
            } else if !shared && !in_f {
                push_decoded(&mut out, &rest[..start]);
            }
        }
        let after = &rest[start + 1..];
        let Some(end_rel) = after.find('>') else {
            break;
        };
        let tag = &after[..end_rel];
        let local = local_name(tag_name(tag));
        let is_end = tag.starts_with('/');

        if !is_end && local == "c" {
            shared = cell_is_shared_string(tag);
            in_f = false;
            index_buf.clear();
        } else if is_end && local == "c" {
            shared = false;
            in_v = false;
            in_f = false;
            index_buf.clear();
            out.push('\n');
        } else if !is_end && local == "f" {
            in_f = true;
        } else if is_end && local == "f" {
            in_f = false;
        } else if !is_end && local == "v" {
            in_v = true;
            index_buf.clear();
        } else if is_end && local == "v" {
            if shared {
                if let Ok(i) = index_buf.trim().parse::<usize>()
                    && let Some(s) = sst.get(i)
                {
                    if !out.is_empty() && !out.ends_with(['\n', ' ']) {
                        out.push(' ');
                    }
                    out.push_str(s);
                }
                index_buf.clear();
            }
            in_v = false;
        } else if is_end && is_block_local(local) {
            out.push('\n');
        }
        rest = &after[end_rel + 1..];
    }
    if !rest.is_empty() && !shared && !in_f {
        push_decoded(&mut out, rest);
    }
    normalize_ws(&out)
}

fn cell_is_shared_string(tag: &str) -> bool {
    tag.contains("t=\"s\"") || tag.contains("t='s'")
}

fn inner_elements<'a>(xml: &'a str, local: &str) -> Vec<&'a str> {
    let mut rest = xml;
    let mut out = Vec::new();
    while let Some(i) = rest.find('<') {
        let after = &rest[i + 1..];
        let Some(gt) = after.find('>') else {
            break;
        };
        let tag = &after[..gt];
        let is_end = tag.starts_with('/');
        let is_empty = tag.ends_with('/');
        if !is_end && !is_empty && local_name(tag_name(tag)) == local {
            let inner = &after[gt + 1..];
            if let Some(end) = find_close_local(inner, local) {
                out.push(&inner[..end]);
                rest = &inner[end..];
                continue;
            }
        }
        rest = &after[gt + 1..];
    }
    out
}

fn find_close_local(inner: &str, local: &str) -> Option<usize> {
    let mut offset = 0;
    let mut rest = inner;
    while let Some(i) = rest.find("</") {
        let after = &rest[i + 2..];
        let gt = after.find('>')?;
        if local_name(tag_name(&after[..gt])) == local {
            return Some(offset + i);
        }
        offset += i + 2 + gt + 1;
        rest = &inner[offset..];
    }
    None
}

fn xml_to_text(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    let mut skip_depth = 0u32;
    let mut vanish_run = false;
    while let Some(start) = rest.find('<') {
        if start > 0 && skip_depth == 0 && !vanish_run {
            push_decoded(&mut out, &rest[..start]);
        }
        let after = &rest[start + 1..];
        let Some(end_rel) = after.find('>') else {
            break;
        };
        let tag = &after[..end_rel];
        let local = local_name(tag_name(tag));
        let is_end = tag.starts_with('/');
        let is_empty = tag.ends_with('/');

        if !is_end && matches!(local, "delText" | "del") {
            if !is_empty {
                skip_depth = skip_depth.saturating_add(1);
            }
        } else if is_end && matches!(local, "delText" | "del") {
            skip_depth = skip_depth.saturating_sub(1);
        }

        if !is_end && local == "vanish" && !vanish_disabled(tag) {
            vanish_run = true;
        }
        if is_end && local == "r" {
            vanish_run = false;
        }

        if skip_depth == 0 && !vanish_run {
            if is_end && is_block_local(local) {
                out.push('\n');
            } else if !is_end && local == "s" {
                for _ in 0..odt_space_count(tag) {
                    out.push(' ');
                }
            } else if matches!(local, "tab" | "br") {
                out.push(' ');
            } else if local == "line-break" {
                out.push('\n');
            }
        }
        rest = &after[end_rel + 1..];
    }
    if !rest.is_empty() && skip_depth == 0 && !vanish_run {
        push_decoded(&mut out, rest);
    }
    normalize_ws(&out)
}

fn vanish_disabled(tag: &str) -> bool {
    tag.contains("val=\"0\"")
        || tag.contains("val='0'")
        || tag.contains("val=\"false\"")
        || tag.contains("val='false'")
}

fn tag_name(tag: &str) -> &str {
    let trimmed = tag.trim_start_matches('/').trim_start_matches('?');
    trimmed
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("")
}

fn local_name(qname: &str) -> &str {
    qname.rsplit_once(':').map_or(qname, |(_, local)| local)
}

fn is_block_local(local: &str) -> bool {
    matches!(local, "p" | "h" | "tr" | "si" | "c")
}

fn odt_space_count(tag: &str) -> usize {
    for key in ["text:c=", "c="] {
        for quote in ['"', '\''] {
            let pat = format!("{key}{quote}");
            if let Some(i) = tag.find(&pat) {
                let rest = &tag[i + pat.len()..];
                if let Some(end) = rest.find(quote)
                    && let Ok(n) = rest[..end].parse::<usize>()
                {
                    return n.clamp(1, MAX_ODT_SPACES);
                }
            }
        }
    }
    1
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
        let owned: Vec<(&str, Vec<u8>)> = files
            .iter()
            .map(|(n, b)| (*n, b.as_bytes().to_vec()))
            .collect();
        let refs: Vec<(&str, &[u8])> = owned.iter().map(|(n, b)| (*n, b.as_slice())).collect();
        zip_with_bytes(&refs)
    }

    fn zip_with_bytes(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut zip = ZipWriter::new(&mut cursor);
            let opts = SimpleFileOptions::default();
            for (name, body) in files {
                zip.start_file(*name, opts).unwrap();
                zip.write_all(body).unwrap();
            }
            zip.finish().unwrap();
        }
        cursor.into_inner()
    }

    fn utf16_le_bom(s: &str) -> Vec<u8> {
        let mut out = vec![0xFF, 0xFE];
        for unit in s.encode_utf16() {
            out.extend_from_slice(&unit.to_le_bytes());
        }
        out
    }

    #[test]
    fn xml_to_text_strips_tags_and_entities() {
        let xml = r"<w:p><w:r><w:t>Hello &amp; world</w:t></w:r></w:p>";
        assert_eq!(xml_to_text(xml), "Hello & world");
    }

    #[test]
    fn xml_to_text_emits_odt_whitespace() {
        assert_eq!(xml_to_text("A<text:s/>B"), "A B");
        assert_eq!(xml_to_text(r#"A<text:s text:c="3"/>B"#), "A B");
        assert_eq!(xml_to_text("A<text:tab/>B"), "A B");
        assert_eq!(xml_to_text("A<text:line-break/>B"), "A\nB");
    }

    #[test]
    fn odt_space_count_honors_text_c_but_clamps() {
        assert_eq!(odt_space_count(r#"text:s text:c="3""#), 3);
        assert_eq!(odt_space_count(r#"text:s text:c="0""#), 1);
        assert_eq!(
            odt_space_count(r#"text:s text:c="18446744073709551615""#),
            255
        );
        assert_eq!(
            xml_to_text(r#"A<text:s text:c="18446744073709551615"/>B"#),
            "A B"
        );
    }

    #[test]
    fn xml_to_text_skips_deleted_and_vanished() {
        let xml = r"<w:p><w:r><w:t>Keep</w:t></w:r><w:del><w:r><w:delText>Gone</w:delText></w:r></w:del><w:r><w:rPr><w:vanish/></w:rPr><w:t>Hidden</w:t></w:r><w:r><w:t>Visible</w:t></w:r></w:p>";
        let text = xml_to_text(xml);
        assert!(text.contains("Keep"));
        assert!(text.contains("Visible"));
        assert!(!text.contains("Gone"));
        assert!(!text.contains("Hidden"));
    }

    #[test]
    fn natural_cmp_orders_slide10_after_slide2() {
        assert_eq!(
            natural_cmp("ppt/slides/slide2.xml", "ppt/slides/slide10.xml"),
            Ordering::Less
        );
        let mut names = vec![
            "ppt/slides/slide10.xml".to_string(),
            "ppt/slides/slide2.xml".to_string(),
        ];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            vec![
                "ppt/slides/slide2.xml".to_string(),
                "ppt/slides/slide10.xml".to_string()
            ]
        );
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
    async fn extracts_xlsx_skips_formula_keeps_cached_value() {
        let sheet = r#"<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row><c><f>SUM(A1:A2)</f><v>3</v></c></row>
  </sheetData>
</worksheet>"#;
        let bytes = zip_with(&[("xl/worksheets/sheet1.xml", sheet)]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, XLSX_MIME).await.unwrap();
        assert!(content.text.contains('3'));
        assert!(!content.text.contains("SUM"));
    }

    #[tokio::test]
    async fn extracts_xlsx_prefixed_shared_string_cells() {
        let sst = r#"<?xml version="1.0"?>
<x:sst xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <x:si><x:t>Prefixed</x:t></x:si>
</x:sst>"#;
        let sheet = r#"<?xml version="1.0"?>
<x:worksheet xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <x:sheetData>
    <x:row><x:c t="s"><x:v>0</x:v></x:c></x:row>
  </x:sheetData>
</x:worksheet>"#;
        let bytes = zip_with(&[
            ("xl/sharedStrings.xml", sst),
            ("xl/worksheets/sheet1.xml", sheet),
        ]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, XLSX_MIME).await.unwrap();
        assert!(content.text.contains("Prefixed"));
        assert!(!content.text.split_whitespace().any(|w| w == "0"));
    }

    #[test]
    fn xml_to_text_skips_prefixed_deleted_text() {
        let xml = r"<ns:p><ns:r><ns:t>Keep</ns:t></ns:r><ns:del><ns:r><ns:delText>Gone</ns:delText></ns:r></ns:del></ns:p>";
        let text = xml_to_text(xml);
        assert!(text.contains("Keep"));
        assert!(!text.contains("Gone"));
    }

    #[tokio::test]
    async fn extracts_xlsx_shared_strings_in_sheet_order() {
        let sst = r#"<?xml version="1.0"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>Revenue</t></si>
  <si><t>Q1 actuals</t></si>
</sst>"#;
        let sheet = r#"<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row>
      <c t="s"><v>1</v></c>
      <c t="s"><v>0</v></c>
    </row>
  </sheetData>
</worksheet>"#;
        let bytes = zip_with(&[
            ("xl/sharedStrings.xml", sst),
            ("xl/worksheets/sheet1.xml", sheet),
        ]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, XLSX_MIME).await.unwrap();
        assert!(content.text.contains("Q1 actuals"));
        assert!(content.text.contains("Revenue"));
        let q1 = content.text.find("Q1 actuals").unwrap();
        let rev = content.text.find("Revenue").unwrap();
        assert!(q1 < rev, "worksheet order should resolve index 1 then 0");
        assert!(
            !content
                .text
                .split_whitespace()
                .any(|w| w == "0" || w == "1")
        );
    }

    #[tokio::test]
    async fn extracts_pptx_slides_in_numeric_order() {
        let slide = |title: &str| {
            format!(
                r#"<?xml version="1.0"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:p><a:r><a:t>{title}</a:t></a:r></a:p>
</p:sld>"#
            )
        };
        let s10 = slide("Slide ten");
        let s2 = slide("Slide two");
        let bytes = zip_with(&[
            ("ppt/slides/slide10.xml", s10.as_str()),
            ("ppt/slides/slide2.xml", s2.as_str()),
        ]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, PPTX_MIME).await.unwrap();
        let two = content.text.find("Slide two").unwrap();
        let ten = content.text.find("Slide ten").unwrap();
        assert!(two < ten);
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
    async fn extracts_utf16_docx_part() {
        let xml = r#"<?xml version="1.0" encoding="UTF-16"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>UTF16 body</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let encoded = utf16_le_bom(xml);
        let bytes = zip_with_bytes(&[("word/document.xml", encoded.as_slice())]);
        let extractor = OfficeExtractor::new();
        let content = extractor.extract_bytes(&bytes, DOCX_MIME).await.unwrap();
        assert!(content.text.contains("UTF16 body"));
    }

    #[test]
    fn rejects_oversized_zip_entry() {
        let bytes = zip_with(&[("word/document.xml", "hello")]);
        let mut archive = ZipArchive::new(Cursor::new(bytes.as_slice())).unwrap();
        let mut file = archive.by_name("word/document.xml").unwrap();
        let mut remaining = 1024u64;
        let err = read_entry_bytes(&mut file, &mut remaining, 2).unwrap_err();
        assert!(matches!(err, ExtractError::Failed(_)));
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
