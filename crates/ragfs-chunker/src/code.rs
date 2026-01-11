//! Code-aware chunking strategy.
//!
//! Chunks code at function/class/method boundaries using pattern matching.
//! Supports Rust, Python, JavaScript, TypeScript, Go, Java, and C/C++.

use async_trait::async_trait;
use ragfs_core::{
    ChunkConfig, ChunkError, ChunkOutput, ChunkOutputMetadata, Chunker, ContentType,
    ExtractedContent,
};
use tracing::debug;

/// Code-aware chunker that splits at function/class boundaries.
pub struct CodeChunker;

impl CodeChunker {
    /// Create a new code chunker.
    pub fn new() -> Self {
        Self
    }

    /// Detect language from metadata or infer from content.
    fn detect_language(content: &ExtractedContent) -> Option<Language> {
        content
            .metadata
            .language
            .as_ref()
            .and_then(|l| Language::from_extension(l))
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Chunker for CodeChunker {
    fn name(&self) -> &str {
        "code"
    }

    fn content_types(&self) -> &[&str] {
        &["code"]
    }

    fn can_chunk(&self, content_type: &ContentType) -> bool {
        matches!(content_type, ContentType::Code { .. })
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

        let language = Self::detect_language(content);
        debug!("Code chunking with language: {:?}", language);

        let lines: Vec<&str> = text.lines().collect();
        let boundaries = find_code_boundaries(&lines, language.as_ref());

        // If no boundaries found, fall back to line-based chunking
        if boundaries.is_empty() {
            return chunk_by_lines(text, &lines, config, &content.metadata.language);
        }

        create_chunks_from_boundaries(text, &lines, &boundaries, config, &content.metadata.language)
    }
}

/// Supported programming languages.
#[derive(Debug, Clone, Copy)]
enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
}

impl Language {
    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "js" | "jsx" | "mjs" => Some(Language::JavaScript),
            "ts" | "tsx" => Some(Language::TypeScript),
            "go" => Some(Language::Go),
            "java" => Some(Language::Java),
            "c" | "h" => Some(Language::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Language::Cpp),
            _ => None,
        }
    }
}

/// A code boundary (start of a function/class/method).
#[derive(Debug)]
struct CodeBoundary {
    line: usize,
    kind: BoundaryKind,
    name: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum BoundaryKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Impl,
    Module,
}

impl BoundaryKind {
    fn as_str(&self) -> &'static str {
        match self {
            BoundaryKind::Function => "function",
            BoundaryKind::Method => "method",
            BoundaryKind::Class => "class",
            BoundaryKind::Struct => "struct",
            BoundaryKind::Enum => "enum",
            BoundaryKind::Impl => "impl",
            BoundaryKind::Module => "module",
        }
    }
}

/// Find code boundaries (functions, classes, etc.) in source code.
fn find_code_boundaries(lines: &[&str], language: Option<&Language>) -> Vec<CodeBoundary> {
    let mut boundaries = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        if let Some(boundary) = detect_boundary(trimmed, language) {
            boundaries.push(CodeBoundary {
                line: i,
                kind: boundary.0,
                name: boundary.1,
            });
        }
    }

    boundaries
}

/// Detect if a line starts a code boundary.
fn detect_boundary(line: &str, language: Option<&Language>) -> Option<(BoundaryKind, Option<String>)> {
    // Language-specific patterns
    match language {
        Some(Language::Rust) => detect_rust_boundary(line),
        Some(Language::Python) => detect_python_boundary(line),
        Some(Language::JavaScript | Language::TypeScript) => detect_js_boundary(line),
        Some(Language::Go) => detect_go_boundary(line),
        Some(Language::Java) => detect_java_boundary(line),
        Some(Language::C | Language::Cpp) => detect_c_boundary(line),
        None => {
            // Try all patterns
            detect_rust_boundary(line)
                .or_else(|| detect_python_boundary(line))
                .or_else(|| detect_js_boundary(line))
                .or_else(|| detect_go_boundary(line))
                .or_else(|| detect_java_boundary(line))
                .or_else(|| detect_c_boundary(line))
        }
    }
}

/// Detect Rust code boundaries.
fn detect_rust_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // fn name(
    if line.starts_with("pub fn ")
        || line.starts_with("fn ")
        || line.starts_with("pub async fn ")
        || line.starts_with("async fn ")
        || line.starts_with("pub(crate) fn ")
        || line.starts_with("pub(super) fn ")
    {
        let name = extract_rust_fn_name(line);
        return Some((BoundaryKind::Function, name));
    }

    // impl block
    if line.starts_with("impl ") || line.starts_with("impl<") {
        let name = extract_after_keyword(line, "impl");
        return Some((BoundaryKind::Impl, name));
    }

    // struct
    if line.starts_with("pub struct ")
        || line.starts_with("struct ")
        || line.starts_with("pub(crate) struct ")
    {
        let name = extract_after_keyword(line, "struct");
        return Some((BoundaryKind::Struct, name));
    }

    // enum
    if line.starts_with("pub enum ")
        || line.starts_with("enum ")
        || line.starts_with("pub(crate) enum ")
    {
        let name = extract_after_keyword(line, "enum");
        return Some((BoundaryKind::Enum, name));
    }

    // mod
    if line.starts_with("pub mod ") || line.starts_with("mod ") {
        let name = extract_after_keyword(line, "mod");
        return Some((BoundaryKind::Module, name));
    }

    None
}

/// Detect Python code boundaries.
fn detect_python_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // def name(
    if line.starts_with("def ") || line.starts_with("async def ") {
        let name = extract_python_fn_name(line);
        return Some((BoundaryKind::Function, name));
    }

    // class Name
    if line.starts_with("class ") {
        let name = extract_after_keyword(line, "class");
        return Some((BoundaryKind::Class, name));
    }

    None
}

/// Detect JavaScript/TypeScript code boundaries.
fn detect_js_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // function name(
    if line.starts_with("function ")
        || line.starts_with("async function ")
        || line.starts_with("export function ")
        || line.starts_with("export async function ")
    {
        let name = extract_js_fn_name(line);
        return Some((BoundaryKind::Function, name));
    }

    // const name = (
    if (line.starts_with("const ") || line.starts_with("export const "))
        && (line.contains(" = (") || line.contains(" = async ("))
    {
        let name = extract_const_fn_name(line);
        return Some((BoundaryKind::Function, name));
    }

    // class Name
    if line.starts_with("class ")
        || line.starts_with("export class ")
        || line.starts_with("export default class ")
    {
        let name = extract_after_keyword(line, "class");
        return Some((BoundaryKind::Class, name));
    }

    // interface Name (TypeScript)
    if line.starts_with("interface ") || line.starts_with("export interface ") {
        let name = extract_after_keyword(line, "interface");
        return Some((BoundaryKind::Struct, name));
    }

    // type Name (TypeScript)
    if line.starts_with("type ") || line.starts_with("export type ") {
        let name = extract_after_keyword(line, "type");
        return Some((BoundaryKind::Struct, name));
    }

    None
}

/// Detect Go code boundaries.
fn detect_go_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // func name(
    if line.starts_with("func ") {
        // Check if it's a method (has receiver)
        if line.contains(") ") && line.find('(') < line.find(')') {
            let name = extract_go_method_name(line);
            return Some((BoundaryKind::Method, name));
        }
        let name = extract_go_fn_name(line);
        return Some((BoundaryKind::Function, name));
    }

    // type Name struct
    if line.starts_with("type ") && line.contains(" struct") {
        let name = extract_after_keyword(line, "type");
        return Some((BoundaryKind::Struct, name));
    }

    // type Name interface
    if line.starts_with("type ") && line.contains(" interface") {
        let name = extract_after_keyword(line, "type");
        return Some((BoundaryKind::Struct, name));
    }

    None
}

/// Detect Java code boundaries.
fn detect_java_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // public/private/protected class Name
    if line.contains(" class ") || line.starts_with("class ") {
        let name = extract_after_keyword(line, "class");
        return Some((BoundaryKind::Class, name));
    }

    // interface
    if line.contains(" interface ") || line.starts_with("interface ") {
        let name = extract_after_keyword(line, "interface");
        return Some((BoundaryKind::Struct, name));
    }

    // enum
    if line.contains(" enum ") || line.starts_with("enum ") {
        let name = extract_after_keyword(line, "enum");
        return Some((BoundaryKind::Enum, name));
    }

    // Method detection (simplified - after class/interface)
    if (line.contains("public ") || line.contains("private ") || line.contains("protected "))
        && line.contains('(')
        && !line.contains(" class ")
        && !line.contains(" interface ")
    {
        let name = extract_java_method_name(line);
        return Some((BoundaryKind::Method, name));
    }

    None
}

/// Detect C/C++ code boundaries.
fn detect_c_boundary(line: &str) -> Option<(BoundaryKind, Option<String>)> {
    // class Name (C++)
    if line.starts_with("class ") || line.contains(" class ") {
        let name = extract_after_keyword(line, "class");
        return Some((BoundaryKind::Class, name));
    }

    // struct Name
    if line.starts_with("struct ") || line.contains(" struct ") {
        let name = extract_after_keyword(line, "struct");
        return Some((BoundaryKind::Struct, name));
    }

    // enum
    if line.starts_with("enum ") || line.contains(" enum ") {
        let name = extract_after_keyword(line, "enum");
        return Some((BoundaryKind::Enum, name));
    }

    // Function (simplified - type name( pattern)
    if line.contains('(') && !line.starts_with('#') && !line.starts_with("//") {
        // Very basic function detection
        let name = extract_c_fn_name(line);
        if name.is_some() {
            return Some((BoundaryKind::Function, name));
        }
    }

    None
}

// Helper functions to extract names

fn extract_rust_fn_name(line: &str) -> Option<String> {
    let line = line
        .trim_start_matches("pub ")
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub(super) ")
        .trim_start_matches("async ")
        .trim_start_matches("fn ");
    line.split('(')
        .next()
        .map(|s| s.split('<').next().unwrap_or(s).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_python_fn_name(line: &str) -> Option<String> {
    let line = line
        .trim_start_matches("async ")
        .trim_start_matches("def ");
    line.split('(')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_js_fn_name(line: &str) -> Option<String> {
    let line = line
        .trim_start_matches("export ")
        .trim_start_matches("async ")
        .trim_start_matches("function ");
    line.split('(')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_const_fn_name(line: &str) -> Option<String> {
    let line = line.trim_start_matches("export ").trim_start_matches("const ");
    line.split(" =")
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_go_fn_name(line: &str) -> Option<String> {
    let line = line.trim_start_matches("func ");
    line.split('(')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_go_method_name(line: &str) -> Option<String> {
    // func (r *Receiver) MethodName(
    let line = line.trim_start_matches("func ");
    if let Some(idx) = line.find(") ") {
        let after_receiver = &line[idx + 2..];
        return after_receiver
            .split('(')
            .next()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
    }
    None
}

fn extract_java_method_name(line: &str) -> Option<String> {
    // Find the part before (
    if let Some(paren_idx) = line.find('(') {
        let before_paren = &line[..paren_idx];
        // Last word before ( is the method name
        before_paren
            .split_whitespace()
            .last()
            .map(|s| s.to_string())
    } else {
        None
    }
}

fn extract_c_fn_name(line: &str) -> Option<String> {
    // Very simplified: look for word before (
    if let Some(paren_idx) = line.find('(') {
        let before_paren = line[..paren_idx].trim();
        // Skip control flow keywords
        let last_word = before_paren.split_whitespace().last()?;
        if ["if", "while", "for", "switch", "return", "sizeof", "typeof"]
            .contains(&last_word.to_lowercase().as_str())
        {
            return None;
        }
        // Skip pointer/ref symbols
        let name = last_word.trim_start_matches('*').trim_start_matches('&');
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(name.to_string());
        }
    }
    None
}

fn extract_after_keyword(line: &str, keyword: &str) -> Option<String> {
    if let Some(idx) = line.find(keyword) {
        let after = &line[idx + keyword.len()..];
        let after = after.trim_start();
        // Handle generics like impl<T>
        let name = if after.starts_with('<') {
            after.split('>').nth(1).unwrap_or(after).trim_start()
        } else {
            after
        };
        // Extract until space, {, (, or <
        let name = name
            .split(|c: char| c.is_whitespace() || c == '{' || c == '(' || c == '<' || c == ':')
            .next()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());
        return name;
    }
    None
}

/// Create chunks from detected boundaries.
fn create_chunks_from_boundaries(
    text: &str,
    lines: &[&str],
    boundaries: &[CodeBoundary],
    config: &ChunkConfig,
    language: &Option<String>,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let max_chars = config.max_size * chars_per_token;

    for (i, boundary) in boundaries.iter().enumerate() {
        let start_line = boundary.line;
        let end_line = if i + 1 < boundaries.len() {
            boundaries[i + 1].line
        } else {
            lines.len()
        };

        // Skip if this section is too small
        if end_line <= start_line {
            continue;
        }

        // Get content for this boundary
        let chunk_lines = &lines[start_line..end_line];
        let content: String = chunk_lines.join("\n");

        // If content is too large, split it
        if content.len() > max_chars {
            let sub_chunks = split_large_chunk(&content, start_line, config, language, &boundary.kind, &boundary.name)?;
            chunks.extend(sub_chunks);
        } else {
            let (byte_start, byte_end) = calculate_byte_range(text, lines, start_line, end_line);

            chunks.push(ChunkOutput {
                content,
                byte_range: byte_start..byte_end,
                line_range: Some(start_line as u32..end_line as u32),
                parent_index: None,
                depth: 0,
                metadata: ChunkOutputMetadata {
                    symbol_type: Some(boundary.kind.as_str().to_string()),
                    symbol_name: boundary.name.clone(),
                    language: language.clone(),
                },
            });
        }
    }

    // If no chunks created, fall back to line-based chunking
    if chunks.is_empty() {
        return chunk_by_lines(text, lines, config, language);
    }

    Ok(chunks)
}

/// Split a chunk that's too large.
fn split_large_chunk(
    content: &str,
    base_line: usize,
    config: &ChunkConfig,
    language: &Option<String>,
    kind: &BoundaryKind,
    name: &Option<String>,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let target_chars = config.target_size * chars_per_token;
    let overlap_chars = config.overlap * chars_per_token;
    let step = target_chars.saturating_sub(overlap_chars).max(1);

    let lines: Vec<&str> = content.lines().collect();
    let mut start = 0;

    while start < lines.len() {
        let mut char_count = 0;
        let mut end = start;

        // Find end based on character count
        while end < lines.len() && char_count < target_chars {
            char_count += lines[end].len() + 1; // +1 for newline
            end += 1;
        }

        let chunk_content: String = lines[start..end].join("\n");
        let chunk_lines = end - start;

        // Calculate byte range within content
        let byte_start = lines[..start].iter().map(|l| l.len() + 1).sum::<usize>() as u64;
        let byte_end = byte_start + chunk_content.len() as u64;

        chunks.push(ChunkOutput {
            content: chunk_content,
            byte_range: byte_start..byte_end,
            line_range: Some((base_line + start) as u32..(base_line + end) as u32),
            parent_index: None,
            depth: 0,
            metadata: ChunkOutputMetadata {
                symbol_type: Some(kind.as_str().to_string()),
                symbol_name: name.clone(),
                language: language.clone(),
            },
        });

        // Move start forward
        let line_step = (step / (char_count / chunk_lines.max(1))).max(1);
        start += line_step;

        if end >= lines.len() {
            break;
        }
    }

    Ok(chunks)
}

/// Chunk by lines when no code boundaries are found.
fn chunk_by_lines(
    text: &str,
    lines: &[&str],
    config: &ChunkConfig,
    language: &Option<String>,
) -> Result<Vec<ChunkOutput>, ChunkError> {
    let mut chunks = Vec::new();
    let chars_per_token = 4;
    let target_chars = config.target_size * chars_per_token;
    let overlap_lines = (config.overlap * chars_per_token) / 80; // Assuming ~80 chars per line

    let mut start = 0;
    while start < lines.len() {
        let mut char_count = 0;
        let mut end = start;

        while end < lines.len() && char_count < target_chars {
            char_count += lines[end].len() + 1;
            end += 1;
        }

        let chunk_content: String = lines[start..end].join("\n");
        let (byte_start, byte_end) = calculate_byte_range(text, lines, start, end);

        chunks.push(ChunkOutput {
            content: chunk_content,
            byte_range: byte_start..byte_end,
            line_range: Some(start as u32..end as u32),
            parent_index: None,
            depth: 0,
            metadata: ChunkOutputMetadata {
                language: language.clone(),
                ..Default::default()
            },
        });

        start = (end).saturating_sub(overlap_lines).max(start + 1);
    }

    Ok(chunks)
}

/// Calculate byte range for a line range.
fn calculate_byte_range(text: &str, lines: &[&str], start_line: usize, end_line: usize) -> (u64, u64) {
    let byte_start: usize = lines[..start_line].iter().map(|l| l.len() + 1).sum();
    let byte_end: usize = lines[..end_line].iter().map(|l| l.len() + 1).sum();
    (byte_start as u64, byte_end.min(text.len()) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_rust_boundary() {
        assert!(detect_rust_boundary("fn main() {").is_some());
        assert!(detect_rust_boundary("pub fn new() -> Self {").is_some());
        assert!(detect_rust_boundary("pub async fn process() {").is_some());
        assert!(detect_rust_boundary("impl Foo {").is_some());
        assert!(detect_rust_boundary("pub struct Bar {").is_some());
        assert!(detect_rust_boundary("enum Baz {").is_some());
        assert!(detect_rust_boundary("let x = 5;").is_none());
    }

    #[test]
    fn test_detect_python_boundary() {
        assert!(detect_python_boundary("def hello():").is_some());
        assert!(detect_python_boundary("async def world():").is_some());
        assert!(detect_python_boundary("class MyClass:").is_some());
        assert!(detect_python_boundary("    x = 5").is_none());
    }

    #[test]
    fn test_detect_js_boundary() {
        assert!(detect_js_boundary("function foo() {").is_some());
        assert!(detect_js_boundary("async function bar() {").is_some());
        assert!(detect_js_boundary("export function baz() {").is_some());
        assert!(detect_js_boundary("const fn = () => {").is_some());
        assert!(detect_js_boundary("class Component {").is_some());
        assert!(detect_js_boundary("export interface Props {").is_some());
    }

    #[test]
    fn test_extract_rust_fn_name() {
        assert_eq!(
            extract_rust_fn_name("fn main() {"),
            Some("main".to_string())
        );
        assert_eq!(
            extract_rust_fn_name("pub fn new() -> Self {"),
            Some("new".to_string())
        );
        assert_eq!(
            extract_rust_fn_name("pub async fn process<T>() {"),
            Some("process".to_string())
        );
    }

    #[test]
    fn test_language_detection() {
        assert!(matches!(
            Language::from_extension("rs"),
            Some(Language::Rust)
        ));
        assert!(matches!(
            Language::from_extension("py"),
            Some(Language::Python)
        ));
        assert!(matches!(
            Language::from_extension("js"),
            Some(Language::JavaScript)
        ));
        assert!(Language::from_extension("unknown").is_none());
    }
}
