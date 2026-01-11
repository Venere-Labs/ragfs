//! Arrow schema definitions for `LanceDB` tables.

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;

/// Schema for the chunks table.
#[must_use]
pub fn chunks_schema(embedding_dim: usize) -> Schema {
    Schema::new(vec![
        // Identity
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("file_id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        // Content
        Field::new("content", DataType::Utf8, false),
        Field::new("content_type", DataType::Utf8, false),
        // Position
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("start_byte", DataType::UInt64, false),
        Field::new("end_byte", DataType::UInt64, false),
        Field::new("start_line", DataType::UInt32, true),
        Field::new("end_line", DataType::UInt32, true),
        // Hierarchy
        Field::new("parent_chunk_id", DataType::Utf8, true),
        Field::new("depth", DataType::UInt8, false),
        // Embedding
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
        // Metadata
        Field::new("embedding_model", DataType::Utf8, true),
        Field::new(
            "indexed_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        // File metadata (denormalized)
        Field::new("file_mime_type", DataType::Utf8, true),
        Field::new("file_size_bytes", DataType::UInt64, true),
        // Code-specific
        Field::new("language", DataType::Utf8, true),
        Field::new("symbol_type", DataType::Utf8, true),
        Field::new("symbol_name", DataType::Utf8, true),
    ])
}

/// Schema for the files metadata table.
#[must_use]
pub fn files_schema() -> Schema {
    Schema::new(vec![
        Field::new("file_id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("mime_type", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new(
            "modified_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new(
            "indexed_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new("chunk_count", DataType::UInt32, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("error_message", DataType::Utf8, true),
    ])
}
