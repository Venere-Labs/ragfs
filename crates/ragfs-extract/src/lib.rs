//! # ragfs-extract
//!
//! Content extraction from various file formats for the RAGFS indexing pipeline.
//!
//! This crate provides the extraction layer that reads files and produces
//! [`ExtractedContent`](ragfs_core::ExtractedContent) for downstream chunking and embedding.
//!
//! ## Supported Formats
//!
//! | Extractor | Formats | Features |
//! |-----------|---------|----------|
//! | [`TextExtractor`] | `.txt`, `.md`, `.rs`, `.py`, `.js`, `.ts`, `.go`, `.java`, `.json`, `.yaml`, `.toml`, `.xml`, `.html`, `.css`, and 30+ more | UTF-8 text extraction |
//! | [`PdfExtractor`] | `.pdf` | Text extraction + embedded images (JPEG, PNG, JPEG2000) |
//! | [`ImageExtractor`] | `.png`, `.jpg`, `.gif`, `.webp`, `.bmp` | Metadata extraction, optional vision captioning |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ragfs_extract::{ExtractorRegistry, TextExtractor, PdfExtractor, ImageExtractor};
//! use std::path::Path;
//!
//! // Create a registry with all extractors
//! let mut registry = ExtractorRegistry::new();
//! registry.register("text", TextExtractor);
//! registry.register("pdf", PdfExtractor::new());
//! registry.register("image", ImageExtractor::new(None));
//!
//! // Extract content from a file
//! let content = registry.extract(Path::new("document.pdf"), "application/pdf").await?;
//! println!("Extracted {} bytes", content.text.len());
//! ```
//!
//! ## PDF Image Extraction
//!
//! The [`PdfExtractor`] can extract embedded images from PDF documents:
//!
//! - **Supported formats**: JPEG (`DCTDecode`), PNG (`FlateDecode`), JPEG2000 (`JPXDecode`)
//! - **Color spaces**: RGB, Grayscale, CMYK (auto-converted to RGB)
//! - **Limits**: 100 images max, 50MB total, 50px minimum dimension
//!
//! ## Vision Captioning
//!
//! The [`ImageExtractor`] supports optional vision-based captioning via the
//! [`ImageCaptioner`] trait. A [`PlaceholderCaptioner`] is provided as a no-op default.
//!
//! ## Components
//!
//! | Type | Description |
//! |------|-------------|
//! | [`ExtractorRegistry`] | Routes files to appropriate extractors by MIME type |
//! | [`TextExtractor`] | Handles text-based files (40+ types) |
//! | [`PdfExtractor`] | PDF text and image extraction |
//! | [`ImageExtractor`] | Image metadata and optional captioning |
//! | [`ImageCaptioner`] | Trait for vision model integration |
//! | [`PlaceholderCaptioner`] | No-op captioner implementation |

pub mod image;
pub mod pdf;
pub mod registry;
pub mod text;
pub mod vision;

pub use image::ImageExtractor;
pub use pdf::PdfExtractor;
#[cfg(feature = "pdf_oxide")]
pub use pdf::PdfOxideExtractor;
pub use registry::ExtractorRegistry;
pub use text::TextExtractor;
#[cfg(feature = "vision")]
pub use vision::BlipCaptioner;
pub use vision::{CaptionConfig, CaptionError, ImageCaptioner, PlaceholderCaptioner};
