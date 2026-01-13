//! `LanceDB` implementation of `VectorStore`.

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt8Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions};
use lancedb::{Connection, Table, connect};
use ragfs_core::{
    Chunk, ChunkMetadata, ContentType, FileRecord, FileStatus, SearchQuery, SearchResult,
    StoreError, StoreStats, VectorStore,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

const CHUNKS_TABLE: &str = "chunks";
const FILES_TABLE: &str = "files";

/// LanceDB-based vector store.
pub struct LanceStore {
    /// Path to the `LanceDB` database
    db_path: PathBuf,
    /// Embedding dimension
    embedding_dim: usize,
    /// Database connection (lazy initialized)
    connection: RwLock<Option<Connection>>,
    /// Chunks table handle
    chunks_table: RwLock<Option<Table>>,
    /// Files table handle
    files_table: RwLock<Option<Table>>,
}

impl LanceStore {
    /// Create a new `LanceStore`.
    #[must_use]
    pub fn new(db_path: PathBuf, embedding_dim: usize) -> Self {
        Self {
            db_path,
            embedding_dim,
            connection: RwLock::new(None),
            chunks_table: RwLock::new(None),
            files_table: RwLock::new(None),
        }
    }

    /// Get the database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get or create connection.
    async fn get_connection(&self) -> Result<Connection, StoreError> {
        {
            let conn = self.connection.read().await;
            if let Some(ref c) = *conn {
                return Ok(c.clone());
            }
        }

        let mut conn = self.connection.write().await;
        if conn.is_none() {
            let db_path_str = self.db_path.to_string_lossy().to_string();
            let new_conn = connect(&db_path_str)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to connect to LanceDB: {e}")))?;
            *conn = Some(new_conn);
        }
        Ok(conn.as_ref().unwrap().clone())
    }

    /// Build chunks table schema.
    fn chunks_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("file_id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("content_type", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("start_byte", DataType::UInt64, false),
            Field::new("end_byte", DataType::UInt64, false),
            Field::new("start_line", DataType::UInt32, true),
            Field::new("end_line", DataType::UInt32, true),
            Field::new("parent_chunk_id", DataType::Utf8, true),
            Field::new("depth", DataType::UInt8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim as i32,
                ),
                false,
            ),
            Field::new("embedding_model", DataType::Utf8, true),
            Field::new("indexed_at", DataType::Utf8, false),
            Field::new("file_mime_type", DataType::Utf8, true),
            Field::new("language", DataType::Utf8, true),
            Field::new("symbol_type", DataType::Utf8, true),
            Field::new("symbol_name", DataType::Utf8, true),
        ])
    }

    /// Build files table schema.
    fn files_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("file_id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("size_bytes", DataType::UInt64, false),
            Field::new("mime_type", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, false),
            Field::new("modified_at", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Utf8, true),
            Field::new("chunk_count", DataType::UInt32, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("error_message", DataType::Utf8, true),
        ])
    }

    /// Get or open chunks table.
    async fn get_chunks_table(&self) -> Result<Table, StoreError> {
        {
            let table = self.chunks_table.read().await;
            if let Some(ref t) = *table {
                return Ok(t.clone());
            }
        }

        let conn = self.get_connection().await?;
        let mut table_lock = self.chunks_table.write().await;

        if table_lock.is_none() {
            let t = conn
                .open_table(CHUNKS_TABLE)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to open chunks table: {e}")))?;
            *table_lock = Some(t);
        }

        Ok(table_lock.as_ref().unwrap().clone())
    }

    /// Get or open files table.
    async fn get_files_table(&self) -> Result<Table, StoreError> {
        {
            let table = self.files_table.read().await;
            if let Some(ref t) = *table {
                return Ok(t.clone());
            }
        }

        let conn = self.get_connection().await?;
        let mut table_lock = self.files_table.write().await;

        if table_lock.is_none() {
            let t = conn
                .open_table(FILES_TABLE)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to open files table: {e}")))?;
            *table_lock = Some(t);
        }

        Ok(table_lock.as_ref().unwrap().clone())
    }

    /// Convert chunks to Arrow `RecordBatch`.
    fn chunks_to_batch(&self, chunks: &[Chunk]) -> Result<RecordBatch, StoreError> {
        let chunk_ids: Vec<_> = chunks.iter().map(|c| c.id.to_string()).collect();
        let file_ids: Vec<_> = chunks.iter().map(|c| c.file_id.to_string()).collect();
        let file_paths: Vec<_> = chunks
            .iter()
            .map(|c| c.file_path.to_string_lossy().to_string())
            .collect();
        let contents: Vec<_> = chunks.iter().map(|c| c.content.clone()).collect();
        let content_types: Vec<_> = chunks
            .iter()
            .map(|c| content_type_to_string(&c.content_type))
            .collect();
        let chunk_indices: Vec<_> = chunks.iter().map(|c| c.chunk_index).collect();
        let start_bytes: Vec<_> = chunks.iter().map(|c| c.byte_range.start).collect();
        let end_bytes: Vec<_> = chunks.iter().map(|c| c.byte_range.end).collect();
        let start_lines: Vec<_> = chunks
            .iter()
            .map(|c| c.line_range.as_ref().map(|r| r.start))
            .collect();
        let end_lines: Vec<_> = chunks
            .iter()
            .map(|c| c.line_range.as_ref().map(|r| r.end))
            .collect();
        let parent_ids: Vec<_> = chunks
            .iter()
            .map(|c| c.parent_chunk_id.map(|id| id.to_string()))
            .collect();
        let depths: Vec<_> = chunks.iter().map(|c| c.depth).collect();

        // Build embeddings as FixedSizeList
        let embeddings: Vec<Option<Vec<Option<f32>>>> = chunks
            .iter()
            .map(|c| {
                c.embedding
                    .as_ref()
                    .map(|e| e.iter().map(|&v| Some(v)).collect())
            })
            .collect();

        let embedding_models: Vec<_> = chunks
            .iter()
            .map(|c| c.metadata.embedding_model.clone())
            .collect();
        let indexed_ats: Vec<_> = chunks
            .iter()
            .map(|c| {
                c.metadata
                    .indexed_at
                    .map_or_else(|| Utc::now().to_rfc3339(), |t| t.to_rfc3339())
            })
            .collect();

        // Extract language/symbol info from content_type
        let languages: Vec<_> = chunks
            .iter()
            .map(|c| match &c.content_type {
                ContentType::Code { language, .. } => Some(language.clone()),
                _ => None,
            })
            .collect();
        let symbol_types: Vec<_> = chunks
            .iter()
            .map(|c| match &c.content_type {
                ContentType::Code { symbol, .. } => {
                    symbol.as_ref().map(|s| format!("{:?}", s.kind))
                }
                _ => None,
            })
            .collect();
        let symbol_names: Vec<_> = chunks
            .iter()
            .map(|c| match &c.content_type {
                ContentType::Code { symbol, .. } => symbol.as_ref().map(|s| s.name.clone()),
                _ => None,
            })
            .collect();

        let mime_types: Vec<Option<String>> = chunks.iter().map(|c| c.mime_type.clone()).collect();

        // Build arrays
        let schema = Arc::new(self.chunks_schema());

        let vector_array = build_vector_array(&embeddings, self.embedding_dim)?;

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(chunk_ids)),
                Arc::new(StringArray::from(file_ids)),
                Arc::new(StringArray::from(file_paths)),
                Arc::new(StringArray::from(contents)),
                Arc::new(StringArray::from(content_types)),
                Arc::new(UInt32Array::from(chunk_indices)),
                Arc::new(UInt64Array::from(start_bytes)),
                Arc::new(UInt64Array::from(end_bytes)),
                Arc::new(UInt32Array::from(start_lines)),
                Arc::new(UInt32Array::from(end_lines)),
                Arc::new(StringArray::from(parent_ids)),
                Arc::new(UInt8Array::from(depths)),
                vector_array,
                Arc::new(StringArray::from(embedding_models)),
                Arc::new(StringArray::from(indexed_ats)),
                Arc::new(StringArray::from(mime_types.clone())),
                Arc::new(StringArray::from(languages)),
                Arc::new(StringArray::from(symbol_types)),
                Arc::new(StringArray::from(symbol_names)),
            ],
        )
        .map_err(|e| StoreError::Insert(format!("Failed to create RecordBatch: {e}")))?;

        Ok(batch)
    }

    /// Convert file record to Arrow `RecordBatch`.
    fn file_to_batch(&self, record: &FileRecord) -> Result<RecordBatch, StoreError> {
        let schema = Arc::new(self.files_schema());

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![record.id.to_string()])),
                Arc::new(StringArray::from(vec![
                    record.path.to_string_lossy().to_string(),
                ])),
                Arc::new(UInt64Array::from(vec![record.size_bytes])),
                Arc::new(StringArray::from(vec![record.mime_type.clone()])),
                Arc::new(StringArray::from(vec![record.content_hash.clone()])),
                Arc::new(StringArray::from(vec![record.modified_at.to_rfc3339()])),
                Arc::new(StringArray::from(vec![
                    record.indexed_at.map(|t| t.to_rfc3339()),
                ])),
                Arc::new(UInt32Array::from(vec![record.chunk_count])),
                Arc::new(StringArray::from(vec![status_to_string(&record.status)])),
                Arc::new(StringArray::from(vec![record.error_message.clone()])),
            ],
        )
        .map_err(|e| StoreError::Insert(format!("Failed to create file RecordBatch: {e}")))?;

        Ok(batch)
    }
}

#[async_trait]
impl VectorStore for LanceStore {
    async fn init(&self) -> Result<(), StoreError> {
        info!("Initializing LanceDB at {:?}", self.db_path);

        // Ensure directory exists
        if let Some(parent) = self.db_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| StoreError::Init(format!("Failed to create db directory: {e}")))?;
        }

        let conn = self.get_connection().await?;

        // Check existing tables
        let tables = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| StoreError::Init(format!("Failed to list tables: {e}")))?;

        // Create chunks table if not exists
        if !tables.contains(&CHUNKS_TABLE.to_string()) {
            info!("Creating chunks table");
            let schema = Arc::new(self.chunks_schema());
            conn.create_empty_table(CHUNKS_TABLE, schema)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to create chunks table: {e}")))?;

            // Create FTS index on content column for hybrid search
            info!("Creating FTS index on content column");
            let table = conn
                .open_table(CHUNKS_TABLE)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to open chunks table: {e}")))?;

            if let Err(e) = table
                .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await
            {
                warn!("Failed to create FTS index (may already exist): {e}");
            }
        }

        // Create files table if not exists
        if !tables.contains(&FILES_TABLE.to_string()) {
            info!("Creating files table");
            let schema = Arc::new(self.files_schema());
            conn.create_empty_table(FILES_TABLE, schema)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to create files table: {e}")))?;
        }

        info!("LanceDB initialized successfully");
        Ok(())
    }

    async fn upsert_chunks(&self, chunks: &[Chunk]) -> Result<(), StoreError> {
        if chunks.is_empty() {
            return Ok(());
        }

        debug!("Upserting {} chunks", chunks.len());

        let table = self.get_chunks_table().await?;
        let batch = self.chunks_to_batch(chunks)?;
        let schema = batch.schema();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| StoreError::Insert(format!("Failed to insert chunks: {e}")))?;

        debug!("Successfully upserted {} chunks", chunks.len());
        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
        debug!("Searching with limit {}", query.limit);

        let table = self.get_chunks_table().await?;

        let mut results = table
            .vector_search(query.embedding.clone())
            .map_err(|e| StoreError::Query(format!("Failed to create search query: {e}")))?
            .limit(query.limit)
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to execute search: {e}")))?;

        let mut search_results = Vec::new();

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch results: {e}")))?
        {
            search_results.extend(batch_to_search_results(&batch)?);
        }

        debug!("Found {} results", search_results.len());
        Ok(search_results)
    }

    async fn hybrid_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
        // If no text query provided, fall back to vector-only search
        let query_text = match &query.text {
            Some(text) if !text.is_empty() => text.clone(),
            _ => return self.search(query).await,
        };

        debug!(
            "Performing hybrid search with text: '{}' and limit {}",
            query_text, query.limit
        );

        let table = self.get_chunks_table().await?;

        // Build hybrid query combining FTS and vector search
        let fts_query = FullTextSearchQuery::new(query_text);

        let mut results = table
            .query()
            .full_text_search(fts_query)
            .nearest_to(query.embedding.clone())
            .map_err(|e| StoreError::Query(format!("Failed to create hybrid query: {e}")))?
            .limit(query.limit)
            .execute_hybrid(QueryExecutionOptions::default())
            .await
            .map_err(|e| StoreError::Query(format!("Failed to execute hybrid search: {e}")))?;

        let mut search_results = Vec::new();

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch hybrid results: {e}")))?
        {
            search_results.extend(batch_to_search_results(&batch)?);
        }

        debug!("Hybrid search found {} results", search_results.len());
        Ok(search_results)
    }

    async fn delete_by_file_path(&self, path: &Path) -> Result<u64, StoreError> {
        let path_str = path.to_string_lossy().to_string();
        debug!("Deleting chunks for file: {}", path_str);

        let table = self.get_chunks_table().await?;

        table
            .delete(&format!("file_path = '{}'", path_str.replace('\'', "''")))
            .await
            .map_err(|e| StoreError::Delete(format!("Failed to delete chunks: {e}")))?;

        // Also delete from files table
        let files_table = self.get_files_table().await?;
        files_table
            .delete(&format!("path = '{}'", path_str.replace('\'', "''")))
            .await
            .map_err(|e| StoreError::Delete(format!("Failed to delete file record: {e}")))?;

        Ok(1) // LanceDB doesn't return count, we assume success
    }

    async fn update_file_path(&self, from: &Path, to: &Path) -> Result<u64, StoreError> {
        // LanceDB doesn't support UPDATE directly, so we read-delete-insert
        debug!("Updating file path from {:?} to {:?}", from, to);

        // 1. Get all chunks for the old path
        let mut chunks = self.get_chunks_for_file(from).await?;
        if chunks.is_empty() {
            debug!("No chunks found for path {:?}", from);
            return Ok(0);
        }

        let chunk_count = chunks.len() as u64;

        // 2. Update the file_path in each chunk
        for chunk in &mut chunks {
            chunk.file_path = to.to_path_buf();
        }

        // 3. Delete old chunks
        self.delete_by_file_path(from).await?;

        // 4. Insert updated chunks
        self.upsert_chunks(&chunks).await?;

        // 5. Also update file record if exists
        if let Ok(Some(mut file_record)) = self.get_file(from).await {
            file_record.path = to.to_path_buf();
            self.upsert_file(&file_record).await?;
        }

        info!("Updated {} chunks from {:?} to {:?}", chunk_count, from, to);
        Ok(chunk_count)
    }

    async fn get_chunks_for_file(&self, path: &Path) -> Result<Vec<Chunk>, StoreError> {
        let path_str = path.to_string_lossy().to_string();
        debug!("Getting chunks for file: {}", path_str);

        let table = self.get_chunks_table().await?;

        let mut results = table
            .query()
            .only_if(format!("file_path = '{}'", path_str.replace('\'', "''")))
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query chunks: {e}")))?;

        let mut chunks = Vec::new();

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch chunks: {e}")))?
        {
            chunks.extend(batch_to_chunks(&batch)?);
        }

        Ok(chunks)
    }

    async fn get_file(&self, path: &Path) -> Result<Option<FileRecord>, StoreError> {
        let path_str = path.to_string_lossy().to_string();
        debug!("Getting file record: {}", path_str);

        let table = self.get_files_table().await?;

        let mut results = table
            .query()
            .only_if(format!("path = '{}'", path_str.replace('\'', "''")))
            .limit(1)
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query file: {e}")))?;

        if let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch file: {e}")))?
        {
            let records = batch_to_file_records(&batch)?;
            return Ok(records.into_iter().next());
        }

        Ok(None)
    }

    async fn upsert_file(&self, record: &FileRecord) -> Result<(), StoreError> {
        debug!("Upserting file record: {:?}", record.path);

        let path_str = record.path.to_string_lossy().to_string();

        // Delete existing file record only (not chunks!)
        let files_table = self.get_files_table().await?;
        let _ = files_table
            .delete(&format!("path = '{}'", path_str.replace('\'', "''")))
            .await;

        // Insert new record
        let batch = self.file_to_batch(record)?;
        let schema = batch.schema();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        files_table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| StoreError::Insert(format!("Failed to insert file record: {e}")))?;

        Ok(())
    }

    async fn stats(&self) -> Result<StoreStats, StoreError> {
        let chunks_table = self.get_chunks_table().await?;
        let files_table = self.get_files_table().await?;

        // Count chunks - use exact same pattern as get_chunks_for_file
        let mut chunk_count = 0u64;
        let mut results = chunks_table
            .query()
            .only_if("file_path LIKE '%'")
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query chunks: {e}")))?;

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to count chunks: {e}")))?
        {
            chunk_count += batch.num_rows() as u64;
        }

        // Count files - use filter
        let mut file_count = 0u64;
        let mut results = files_table
            .query()
            .only_if("size_bytes >= 0")
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query files: {e}")))?;

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to count files: {e}")))?
        {
            file_count += batch.num_rows() as u64;
        }

        // Calculate actual index size from disk
        let index_size_bytes = calculate_dir_size(&self.db_path);

        Ok(StoreStats {
            total_chunks: chunk_count,
            total_files: file_count,
            index_size_bytes,
            last_updated: Some(Utc::now()),
        })
    }

    async fn get_all_chunks(&self) -> Result<Vec<Chunk>, StoreError> {
        debug!("Getting all chunks");

        let table = self.get_chunks_table().await?;

        let mut results = table
            .query()
            .only_if("file_path LIKE '%'")
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query all chunks: {e}")))?;

        let mut chunks = Vec::new();

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch chunks: {e}")))?
        {
            chunks.extend(batch_to_chunks(&batch)?);
        }

        debug!("Retrieved {} chunks", chunks.len());
        Ok(chunks)
    }

    async fn get_all_files(&self) -> Result<Vec<FileRecord>, StoreError> {
        debug!("Getting all file records");

        let table = self.get_files_table().await?;

        let mut results = table
            .query()
            .only_if("size_bytes >= 0")
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to query all files: {e}")))?;

        let mut records = Vec::new();

        while let Some(batch) = results
            .try_next()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to fetch files: {e}")))?
        {
            records.extend(batch_to_file_records(&batch)?);
        }

        debug!("Retrieved {} file records", records.len());
        Ok(records)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Calculate the total size of a directory recursively.
fn calculate_dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }

    let mut total_size = 0u64;

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            } else if entry_path.is_dir() {
                total_size += calculate_dir_size(&entry_path);
            }
        }
    }

    total_size
}

fn content_type_to_string(ct: &ContentType) -> String {
    match ct {
        ContentType::Text => "text".to_string(),
        ContentType::Code { language, .. } => format!("code:{language}"),
        ContentType::ImageCaption => "image_caption".to_string(),
        ContentType::PdfPage { page_num } => format!("pdf:{page_num}"),
        ContentType::Markdown => "markdown".to_string(),
    }
}

fn string_to_content_type(s: &str) -> ContentType {
    if s == "text" {
        ContentType::Text
    } else if s == "markdown" {
        ContentType::Markdown
    } else if s == "image_caption" {
        ContentType::ImageCaption
    } else if let Some(lang) = s.strip_prefix("code:") {
        ContentType::Code {
            language: lang.to_string(),
            symbol: None,
        }
    } else if let Some(page) = s.strip_prefix("pdf:") {
        ContentType::PdfPage {
            page_num: page.parse().unwrap_or(1),
        }
    } else {
        ContentType::Text
    }
}

fn status_to_string(status: &FileStatus) -> String {
    match status {
        FileStatus::Pending => "pending".to_string(),
        FileStatus::Indexing => "indexing".to_string(),
        FileStatus::Indexed => "indexed".to_string(),
        FileStatus::Error => "error".to_string(),
        FileStatus::Deleted => "deleted".to_string(),
    }
}

fn string_to_status(s: &str) -> FileStatus {
    match s {
        "pending" => FileStatus::Pending,
        "indexing" => FileStatus::Indexing,
        "indexed" => FileStatus::Indexed,
        "error" => FileStatus::Error,
        "deleted" => FileStatus::Deleted,
        _ => FileStatus::Pending,
    }
}

fn build_vector_array(
    embeddings: &[Option<Vec<Option<f32>>>],
    dim: usize,
) -> Result<ArrayRef, StoreError> {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);

    for emb in embeddings {
        if let Some(values) = emb {
            let values_builder = builder.values();
            for &v in values {
                values_builder.append_option(v);
            }
            builder.append(true);
        } else {
            // Append zeros for missing embeddings
            let values_builder = builder.values();
            for _ in 0..dim {
                values_builder.append_value(0.0);
            }
            builder.append(true);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn batch_to_search_results(batch: &RecordBatch) -> Result<Vec<SearchResult>, StoreError> {
    let mut results = Vec::new();

    let chunk_ids = batch
        .column_by_name("chunk_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let file_paths = batch
        .column_by_name("file_path")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let contents = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let start_bytes = batch
        .column_by_name("start_byte")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
    let end_bytes = batch
        .column_by_name("end_byte")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
    let start_lines = batch
        .column_by_name("start_line")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let end_lines = batch
        .column_by_name("end_line")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let distances = batch
        .column_by_name("_distance")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

    let (Some(chunk_ids), Some(file_paths), Some(contents), Some(start_bytes), Some(end_bytes)) =
        (chunk_ids, file_paths, contents, start_bytes, end_bytes)
    else {
        return Err(StoreError::Query("Missing required columns".to_string()));
    };

    for i in 0..batch.num_rows() {
        let chunk_id = chunk_ids.value(i);
        let file_path = file_paths.value(i);
        let content = contents.value(i);
        let start = start_bytes.value(i);
        let end = end_bytes.value(i);

        let line_range = match (start_lines, end_lines) {
            (Some(sl), Some(el)) if !sl.is_null(i) && !el.is_null(i) => {
                Some(sl.value(i)..el.value(i))
            }
            _ => None,
        };

        let score = distances.map_or(0.0, |d| 1.0 - d.value(i));

        results.push(SearchResult {
            chunk_id: Uuid::parse_str(chunk_id).unwrap_or_default(),
            file_path: PathBuf::from(file_path),
            content: content.to_string(),
            score,
            byte_range: start..end,
            line_range,
            metadata: HashMap::new(),
        });
    }

    Ok(results)
}

fn batch_to_chunks(batch: &RecordBatch) -> Result<Vec<Chunk>, StoreError> {
    let mut chunks = Vec::new();

    // Similar to batch_to_search_results but returns full Chunk structs
    // Simplified for now - full implementation would parse all fields

    let chunk_ids = batch
        .column_by_name("chunk_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let file_ids = batch
        .column_by_name("file_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let file_paths = batch
        .column_by_name("file_path")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let contents = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let content_types = batch
        .column_by_name("content_type")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let chunk_indices = batch
        .column_by_name("chunk_index")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let start_bytes = batch
        .column_by_name("start_byte")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
    let end_bytes = batch
        .column_by_name("end_byte")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
    let depths = batch
        .column_by_name("depth")
        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>());
    let mime_types = batch
        .column_by_name("mime_type")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let start_lines = batch
        .column_by_name("start_line")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let end_lines = batch
        .column_by_name("end_line")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let embeddings = batch
        .column_by_name("embedding")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

    let (
        Some(chunk_ids),
        Some(file_ids),
        Some(file_paths),
        Some(contents),
        Some(content_types),
        Some(chunk_indices),
        Some(start_bytes),
        Some(end_bytes),
        Some(depths),
    ) = (
        chunk_ids,
        file_ids,
        file_paths,
        contents,
        content_types,
        chunk_indices,
        start_bytes,
        end_bytes,
        depths,
    )
    else {
        return Err(StoreError::Query(
            "Missing required columns in chunks".to_string(),
        ));
    };

    for i in 0..batch.num_rows() {
        let mime_type = mime_types.and_then(|m| {
            if m.is_null(i) {
                None
            } else {
                Some(m.value(i).to_string())
            }
        });

        // Parse line range from start_line and end_line columns
        let line_range = match (start_lines, end_lines) {
            (Some(starts), Some(ends)) if !starts.is_null(i) && !ends.is_null(i) => {
                Some(starts.value(i)..ends.value(i))
            }
            _ => None,
        };

        // Parse embedding vector
        let embedding = embeddings.and_then(|emb_array| {
            if emb_array.is_null(i) {
                None
            } else {
                let values = emb_array.value(i);
                values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|arr| arr.values().to_vec())
            }
        });

        chunks.push(Chunk {
            id: Uuid::parse_str(chunk_ids.value(i)).unwrap_or_default(),
            file_id: Uuid::parse_str(file_ids.value(i)).unwrap_or_default(),
            file_path: PathBuf::from(file_paths.value(i)),
            content: contents.value(i).to_string(),
            content_type: string_to_content_type(content_types.value(i)),
            mime_type,
            chunk_index: chunk_indices.value(i),
            byte_range: start_bytes.value(i)..end_bytes.value(i),
            line_range,
            parent_chunk_id: None,
            depth: depths.value(i),
            embedding,
            metadata: ChunkMetadata::default(),
        });
    }

    Ok(chunks)
}

fn batch_to_file_records(batch: &RecordBatch) -> Result<Vec<FileRecord>, StoreError> {
    let mut records = Vec::new();

    let file_ids = batch
        .column_by_name("file_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let paths = batch
        .column_by_name("path")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let sizes = batch
        .column_by_name("size_bytes")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
    let mime_types = batch
        .column_by_name("mime_type")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let hashes = batch
        .column_by_name("content_hash")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let modified_ats = batch
        .column_by_name("modified_at")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let chunk_counts = batch
        .column_by_name("chunk_count")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let statuses = batch
        .column_by_name("status")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let indexed_ats = batch
        .column_by_name("indexed_at")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let (
        Some(file_ids),
        Some(paths),
        Some(sizes),
        Some(mime_types),
        Some(hashes),
        Some(modified_ats),
        Some(chunk_counts),
        Some(statuses),
    ) = (
        file_ids,
        paths,
        sizes,
        mime_types,
        hashes,
        modified_ats,
        chunk_counts,
        statuses,
    )
    else {
        return Err(StoreError::Query(
            "Missing required columns in files".to_string(),
        ));
    };

    for i in 0..batch.num_rows() {
        let modified_at = chrono::DateTime::parse_from_rfc3339(modified_ats.value(i))
            .map_or_else(|_| Utc::now(), |dt| dt.with_timezone(&Utc));

        // Parse indexed_at timestamp
        let indexed_at = indexed_ats.and_then(|arr| {
            if arr.is_null(i) {
                None
            } else {
                chrono::DateTime::parse_from_rfc3339(arr.value(i))
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }
        });

        records.push(FileRecord {
            id: Uuid::parse_str(file_ids.value(i)).unwrap_or_default(),
            path: PathBuf::from(paths.value(i)),
            size_bytes: sizes.value(i),
            mime_type: mime_types.value(i).to_string(),
            content_hash: hashes.value(i).to_string(),
            modified_at,
            indexed_at,
            chunk_count: chunk_counts.value(i),
            status: string_to_status(statuses.value(i)),
            error_message: None,
        });
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragfs_core::DistanceMetric;
    use std::collections::HashMap;
    use tempfile::tempdir;

    const TEST_DIM: usize = 384;

    fn create_test_chunk(
        file_path: &Path,
        content: &str,
        embedding: Vec<f32>,
        chunk_index: u32,
    ) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            file_id: Uuid::new_v4(),
            file_path: file_path.to_path_buf(),
            content: content.to_string(),
            content_type: ContentType::Text,
            mime_type: Some("text/plain".to_string()),
            chunk_index,
            byte_range: 0..content.len() as u64,
            line_range: Some(0..1),
            parent_chunk_id: None,
            depth: 0,
            embedding: Some(embedding),
            metadata: ChunkMetadata {
                indexed_at: Some(Utc::now()),
                embedding_model: Some("test-model".to_string()),
                token_count: None,
                extra: HashMap::new(),
            },
        }
    }

    fn create_random_embedding(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.001).sin()).collect()
    }

    fn create_test_file_record(path: &Path) -> FileRecord {
        FileRecord {
            id: Uuid::new_v4(),
            path: path.to_path_buf(),
            size_bytes: 1024,
            mime_type: "text/plain".to_string(),
            content_hash: "abc123".to_string(),
            modified_at: Utc::now(),
            indexed_at: Some(Utc::now()),
            chunk_count: 1,
            status: FileStatus::Indexed,
            error_message: None,
        }
    }

    #[tokio::test]
    async fn test_init_creates_tables() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);

        let result = store.init().await;
        assert!(result.is_ok(), "Init failed: {:?}", result.err());

        // Verify tables exist
        let conn = store.get_connection().await.unwrap();
        let tables = conn.table_names().execute().await.unwrap();
        assert!(tables.contains(&"chunks".to_string()));
        assert!(tables.contains(&"files".to_string()));
    }

    #[tokio::test]
    async fn test_init_idempotent() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);

        // Init twice should succeed
        store.init().await.unwrap();
        let result = store.init().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_upsert_and_get_chunks() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/file.txt");
        let embedding = create_random_embedding(TEST_DIM);
        let chunk = create_test_chunk(&file_path, "Hello world", embedding, 0);

        // Upsert
        let result = store.upsert_chunks(&[chunk]).await;
        assert!(result.is_ok(), "Upsert failed: {:?}", result.err());

        // Get chunks back
        let chunks = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello world");
    }

    #[tokio::test]
    async fn test_upsert_multiple_chunks() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/multi.txt");
        let chunks: Vec<Chunk> = (0..5)
            .map(|i| {
                create_test_chunk(
                    &file_path,
                    &format!("Chunk content {i}"),
                    create_random_embedding(TEST_DIM),
                    i,
                )
            })
            .collect();

        store.upsert_chunks(&chunks).await.unwrap();

        let retrieved = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(retrieved.len(), 5);
    }

    #[tokio::test]
    async fn test_upsert_empty_chunks() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        // Empty vec should succeed
        let result = store.upsert_chunks(&[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_returns_results() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        // Insert some chunks
        let file_path = PathBuf::from("/test/search.txt");
        let embedding = create_random_embedding(TEST_DIM);
        let chunk = create_test_chunk(&file_path, "Authentication logic", embedding.clone(), 0);
        store.upsert_chunks(&[chunk]).await.unwrap();

        // Search with same embedding should find it
        let query = SearchQuery {
            text: Some("auth".to_string()),
            embedding: embedding.clone(),
            limit: 10,
            filters: vec![],
            metric: DistanceMetric::Cosine,
        };

        let results = store.search(query).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "Authentication logic");
    }

    #[tokio::test]
    async fn test_search_respects_limit() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        // Insert 10 chunks
        let file_path = PathBuf::from("/test/limit.txt");
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| {
                create_test_chunk(
                    &file_path,
                    &format!("Content {i}"),
                    create_random_embedding(TEST_DIM),
                    i,
                )
            })
            .collect();
        store.upsert_chunks(&chunks).await.unwrap();

        // Search with limit 3
        let query = SearchQuery {
            text: Some("test".to_string()),
            embedding: create_random_embedding(TEST_DIM),
            limit: 3,
            filters: vec![],
            metric: DistanceMetric::Cosine,
        };

        let results = store.search(query).await.unwrap();
        assert!(results.len() <= 3);
    }

    #[tokio::test]
    async fn test_delete_by_file_path() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/delete.txt");
        let chunk = create_test_chunk(
            &file_path,
            "To be deleted",
            create_random_embedding(TEST_DIM),
            0,
        );
        store.upsert_chunks(&[chunk]).await.unwrap();

        // Verify it exists
        let chunks = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(chunks.len(), 1);

        // Delete
        store.delete_by_file_path(&file_path).await.unwrap();

        // Verify it's gone
        let chunks = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[tokio::test]
    async fn test_upsert_and_get_file_record() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/record.txt");
        let record = create_test_file_record(&file_path);

        // Upsert
        store.upsert_file(&record).await.unwrap();

        // Get
        let retrieved = store.get_file(&file_path).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.path, file_path);
        assert_eq!(retrieved.mime_type, "text/plain");
    }

    #[tokio::test]
    async fn test_get_nonexistent_file() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let result = store
            .get_file(&PathBuf::from("/nonexistent"))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_stats() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        // Initially empty
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_files, 0);

        // Add chunk
        let file_path = PathBuf::from("/test/stats.txt");
        let chunk = create_test_chunk(
            &file_path,
            "Stats test",
            create_random_embedding(TEST_DIM),
            0,
        );
        store.upsert_chunks(&[chunk]).await.unwrap();

        // Add file record (should NOT delete chunks)
        let record = create_test_file_record(&file_path);
        store.upsert_file(&record).await.unwrap();

        // Verify chunks still exist after upsert_file
        let chunks = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(
            chunks.len(),
            1,
            "Chunks should still exist after upsert_file"
        );

        // Verify file exists
        let file = store.get_file(&file_path).await.unwrap();
        assert!(file.is_some(), "File should be retrievable");

        // Check stats
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 1);
        assert_eq!(stats.total_files, 1);
        assert!(
            stats.index_size_bytes > 0,
            "index_size_bytes should be > 0, got {}",
            stats.index_size_bytes
        );
    }

    #[tokio::test]
    async fn test_content_type_conversion() {
        assert_eq!(content_type_to_string(&ContentType::Text), "text");
        assert_eq!(content_type_to_string(&ContentType::Markdown), "markdown");
        assert_eq!(
            content_type_to_string(&ContentType::Code {
                language: "rust".to_string(),
                symbol: None
            }),
            "code:rust"
        );
        assert_eq!(
            content_type_to_string(&ContentType::PdfPage { page_num: 5 }),
            "pdf:5"
        );

        assert!(matches!(string_to_content_type("text"), ContentType::Text));
        assert!(matches!(
            string_to_content_type("markdown"),
            ContentType::Markdown
        ));
        assert!(matches!(
            string_to_content_type("code:python"),
            ContentType::Code { language, .. } if language == "python"
        ));
    }

    #[tokio::test]
    async fn test_file_status_conversion() {
        assert_eq!(status_to_string(&FileStatus::Pending), "pending");
        assert_eq!(status_to_string(&FileStatus::Indexed), "indexed");
        assert_eq!(status_to_string(&FileStatus::Error), "error");

        assert!(matches!(string_to_status("pending"), FileStatus::Pending));
        assert!(matches!(string_to_status("indexed"), FileStatus::Indexed));
        assert!(matches!(string_to_status("error"), FileStatus::Error));
        assert!(matches!(string_to_status("unknown"), FileStatus::Pending));
    }

    #[tokio::test]
    async fn test_chunks_with_code_content_type() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/code.rs");
        let chunk = Chunk {
            id: Uuid::new_v4(),
            file_id: Uuid::new_v4(),
            file_path: file_path.clone(),
            content: "fn main() {}".to_string(),
            content_type: ContentType::Code {
                language: "rust".to_string(),
                symbol: None,
            },
            mime_type: Some("text/x-rust".to_string()),
            chunk_index: 0,
            byte_range: 0..12,
            line_range: Some(0..1),
            parent_chunk_id: None,
            depth: 0,
            embedding: Some(create_random_embedding(TEST_DIM)),
            metadata: ChunkMetadata::default(),
        };

        store.upsert_chunks(&[chunk]).await.unwrap();

        let chunks = store.get_chunks_for_file(&file_path).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(
            &chunks[0].content_type,
            ContentType::Code { language, .. } if language == "rust"
        ));
    }
}
