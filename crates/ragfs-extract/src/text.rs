//! Text content extractor.

use async_trait::async_trait;
use ragfs_core::{
    ContentElement, ContentExtractor, ContentMetadataInfo, ExtractError, ExtractedContent,
};
use std::path::Path;
use tokio::fs;

/// Extractor for plain text files.
pub struct TextExtractor;

impl TextExtractor {
    /// Create a new text extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for TextExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContentExtractor for TextExtractor {
    fn supported_types(&self) -> &[&str] {
        &[
            "text/plain",
            "text/markdown",
            "text/x-markdown",
            "application/x-sh",
            "text/x-rust",
            "text/x-python",
            "text/x-java",
            "text/javascript",
            "text/typescript",
            "text/x-go",
            "text/x-c",
            "text/x-c++",
            "application/json",
            "application/xml",
            "text/xml",
            "text/html",
            "text/css",
            "application/toml",
            "text/x-toml",
            "application/yaml",
            "text/x-yaml",
        ]
    }

    fn can_extract_by_extension(&self, path: &Path) -> bool {
        let extensions = [
            "txt",
            "md",
            "markdown",
            "rs",
            "py",
            "java",
            "js",
            "ts",
            "tsx",
            "jsx",
            "go",
            "c",
            "cpp",
            "cc",
            "h",
            "hpp",
            "json",
            "xml",
            "html",
            "htm",
            "css",
            "scss",
            "sass",
            "toml",
            "yaml",
            "yml",
            "sh",
            "bash",
            "zsh",
            "fish",
            "sql",
            "rb",
            "php",
            "swift",
            "kt",
            "kts",
            "scala",
            "clj",
            "ex",
            "exs",
            "erl",
            "hs",
            "ml",
            "mli",
            "fs",
            "fsx",
            "lua",
            "vim",
            "el",
            "lisp",
            "scm",
            "rkt",
            "asm",
            "s",
            "dockerfile",
            "makefile",
            "cmake",
            "gradle",
            "sbt",
            "cabal",
            "nix",
            "tf",
            "hcl",
        ];

        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| extensions.contains(&ext.to_lowercase().as_str()))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        let content = fs::read_to_string(path).await?;

        // Detect language from extension
        let language = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_lowercase);

        // Build elements (simple paragraph-based for now)
        let elements = content
            .split("\n\n")
            .enumerate()
            .filter(|(_, para)| !para.trim().is_empty())
            .map(|(_idx, para)| ContentElement::Paragraph {
                text: para.to_string(),
                byte_offset: content[..content.find(para).unwrap_or(0)].len() as u64,
            })
            .collect();

        Ok(ExtractedContent {
            text: content,
            elements,
            images: vec![],
            metadata: ContentMetadataInfo {
                language,
                ..Default::default()
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_new_extractor() {
        let extractor = TextExtractor::new();
        assert!(!extractor.supported_types().is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let extractor = TextExtractor;
        assert!(!extractor.supported_types().is_empty());
    }

    #[test]
    fn test_supported_types_includes_common_types() {
        let extractor = TextExtractor::new();
        let types = extractor.supported_types();

        assert!(types.contains(&"text/plain"));
        assert!(types.contains(&"text/markdown"));
        assert!(types.contains(&"text/x-rust"));
        assert!(types.contains(&"text/x-python"));
        assert!(types.contains(&"application/json"));
        assert!(types.contains(&"text/javascript"));
    }

    #[test]
    fn test_can_extract_by_extension_txt() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/file.txt");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_by_extension_rust() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/main.rs");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_by_extension_python() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/script.py");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_by_extension_markdown() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/README.md");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_by_extension_json() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/config.json");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_by_extension_typescript() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/app.tsx");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_cannot_extract_binary() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/image.png");
        assert!(!extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_cannot_extract_executable() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/program.exe");
        assert!(!extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_cannot_extract_no_extension() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/file_without_extension");
        assert!(!extractor.can_extract_by_extension(&path));
    }

    #[test]
    fn test_can_extract_case_insensitive() {
        let extractor = TextExtractor::new();
        let path = std::path::PathBuf::from("/test/FILE.TXT");
        assert!(extractor.can_extract_by_extension(&path));
    }

    #[tokio::test]
    async fn test_extract_simple_text() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Hello, world!").unwrap();

        let extractor = TextExtractor::new();
        let result = extractor.extract(&file_path).await;

        assert!(result.is_ok());
        let content = result.unwrap();
        assert_eq!(content.text, "Hello, world!");
        assert!(content.images.is_empty());
    }

    #[tokio::test]
    async fn test_extract_detects_language() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("main.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.metadata.language, Some("rs".to_string()));
    }

    #[tokio::test]
    async fn test_extract_python_language() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("script.py");
        std::fs::write(&file_path, "print('hello')").unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.metadata.language, Some("py".to_string()));
    }

    #[tokio::test]
    async fn test_extract_creates_paragraph_elements() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        std::fs::write(&file_path, text).unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.elements.len(), 3);
    }

    #[tokio::test]
    async fn test_extract_handles_empty_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.txt");
        std::fs::write(&file_path, "").unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.text, "");
        assert!(content.elements.is_empty());
    }

    #[tokio::test]
    async fn test_extract_handles_unicode() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("unicode.txt");
        let text = "Hello 世界! 🌍 Привет мир!";
        std::fs::write(&file_path, text).unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.text, text);
    }

    #[tokio::test]
    async fn test_extract_nonexistent_file_fails() {
        let extractor = TextExtractor::new();
        let result = extractor.extract(Path::new("/nonexistent/file.txt")).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_multiline_content() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("multi.txt");
        let text = "Line 1\nLine 2\nLine 3";
        std::fs::write(&file_path, text).unwrap();

        let extractor = TextExtractor::new();
        let content = extractor.extract(&file_path).await.unwrap();

        assert_eq!(content.text, text);
        // Single paragraph (no double newlines)
        assert_eq!(content.elements.len(), 1);
    }

    #[test]
    fn test_can_extract_config_files() {
        let extractor = TextExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("config.toml")));
        assert!(extractor.can_extract_by_extension(Path::new("config.yaml")));
        assert!(extractor.can_extract_by_extension(Path::new("config.yml")));
    }

    #[test]
    fn test_can_extract_shell_scripts() {
        let extractor = TextExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("script.sh")));
        assert!(extractor.can_extract_by_extension(Path::new("script.bash")));
        assert!(extractor.can_extract_by_extension(Path::new("script.zsh")));
    }

    #[test]
    fn test_can_extract_web_files() {
        let extractor = TextExtractor::new();

        assert!(extractor.can_extract_by_extension(Path::new("index.html")));
        assert!(extractor.can_extract_by_extension(Path::new("styles.css")));
        assert!(extractor.can_extract_by_extension(Path::new("app.js")));
        assert!(extractor.can_extract_by_extension(Path::new("app.ts")));
    }
}
