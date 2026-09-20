//! Arrow schema definitions for `LanceDB` tables.

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Current chunks table schema version.
///
/// Version 1: pre-scope columns (no `dir_path` / `dir_depth` / `path_components`).
/// Version 2: directory-scope columns for `ragfs query --scope`.
pub const CHUNKS_SCHEMA_VERSION: u32 = 2;

/// Sidecar written beside the Lance database (`…/indices/<hash>/ragfs-schema.json`).
pub const SCHEMA_SIDECAR_FILENAME: &str = "ragfs-schema.json";

/// Persisted schema version for an existing Lance index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchemaSidecar {
    pub chunks_schema_version: u32,
}

/// Scope columns missing from a pre-v2 chunks table (nullable so `add_columns` can fill nulls).
#[must_use]
pub fn missing_scope_columns(schema: &Schema) -> Vec<Field> {
    let mut missing = Vec::new();
    if schema.field_with_name("dir_path").is_err() {
        missing.push(Field::new("dir_path", DataType::Utf8, true));
    }
    if schema.field_with_name("dir_depth").is_err() {
        missing.push(Field::new("dir_depth", DataType::UInt16, true));
    }
    if schema.field_with_name("path_components").is_err() {
        missing.push(Field::new("path_components", DataType::Utf8, true));
    }
    missing
}

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
        Field::new("dir_path", DataType::Utf8, false),
        Field::new("dir_depth", DataType::UInt16, false),
        Field::new("path_components", DataType::Utf8, false),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_scope_columns_on_v1_schema() {
        let v1 = Schema::new(vec![Field::new("file_path", DataType::Utf8, false)]);
        let missing = missing_scope_columns(&v1);
        assert_eq!(missing.len(), 3);
        assert_eq!(missing[0].name(), "dir_path");
        assert_eq!(missing[1].name(), "dir_depth");
        assert_eq!(missing[2].name(), "path_components");
        assert!(missing.iter().all(Field::is_nullable));
    }

    #[test]
    fn test_missing_scope_columns_on_v2_schema() {
        let v2 = chunks_schema(384);
        assert!(missing_scope_columns(&v2).is_empty());
    }
}
