//! Content extraction for RAGFS.

pub mod image;
pub mod pdf;
pub mod registry;
pub mod text;
pub mod vision;

pub use image::ImageExtractor;
pub use pdf::PdfExtractor;
pub use registry::ExtractorRegistry;
pub use text::TextExtractor;
pub use vision::{CaptionConfig, CaptionError, ImageCaptioner, PlaceholderCaptioner};
