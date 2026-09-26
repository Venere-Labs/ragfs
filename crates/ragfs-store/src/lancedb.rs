//! `LanceDB` implementation of `VectorStore`.

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions};
use lancedb::table::{NewColumnTransform, OptimizeAction, OptimizeOptions};
use lancedb::{Connection, DistanceType, Table, connect};
use ragfs_core::{
    Chunk, ChunkMetadata, ContentType, DirectoryScope, DistanceMetric, FileRecord, FileStatus,
    SearchFilter, SearchQuery, SearchResult, StoreError, StoreStats, VectorStore,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::schema::{
    CHUNKS_SCHEMA_VERSION, SCHEMA_SIDECAR_FILENAME, SchemaSidecar, missing_scope_columns,
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

const CHUNKS_TABLE: &str = "chunks";
const FILES_TABLE: &str = "files";
/// Lance IVF training uses a default sample rate of 256. Below this, stay on exact scan.
const MIN_ANN_ROWS: usize = 256;
/// Product default search metric (`gte-small`). Train IVF-PQ with this so ANN is valid.
const ANN_INDEX_METRIC: DistanceMetric = DistanceMetric::Cosine;

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

    /// Whether the table is large enough for Lance IVF-PQ training.
    pub(crate) fn should_build_ann_index(row_count: usize) -> bool {
        row_count >= MIN_ANN_ROWS
    }

    /// Refresh IVF-PQ once the unindexed tail is large enough to train another sample.
    pub(crate) fn should_refresh_ann_index(unindexed_rows: usize) -> bool {
        unindexed_rows >= MIN_ANN_ROWS
    }

    /// ANN is trained as cosine; other metrics must not use the index.
    pub(crate) fn ann_index_covers_metric(metric: DistanceMetric) -> bool {
        metric == ANN_INDEX_METRIC
    }

    /// Reuse/refresh only a cosine-trained vector index.
    pub(crate) fn ann_index_is_cosine(distance_type: Option<DistanceType>) -> bool {
        distance_type == Some(DistanceType::Cosine)
    }

    async fn vector_index_name(table: &Table) -> Option<String> {
        match table.list_indices().await {
            Ok(indices) => indices
                .into_iter()
                .find(|idx| idx.columns.iter().any(|col| col == "vector"))
                .map(|idx| idx.name),
            Err(_) => None,
        }
    }

    async fn refresh_vector_index(table: &Table, index_name: &str) {
        let unindexed = match table.index_stats(index_name).await {
            Ok(Some(stats)) => stats.num_unindexed_rows,
            Ok(None) => return,
            Err(e) => {
                debug!("Skipping ANN refresh; could not read index stats: {e}");
                return;
            }
        };

        if !Self::should_refresh_ann_index(unindexed) {
            debug!("Skipping ANN refresh: {unindexed} unindexed rows < {MIN_ANN_ROWS}");
            return;
        }

        info!("Refreshing IVF-PQ ANN index ({unindexed} unindexed rows)");
        match table
            .optimize(OptimizeAction::Index(
                OptimizeOptions::append().index_names(vec![index_name.to_string()]),
            ))
            .await
        {
            Ok(_) => info!("IVF-PQ ANN index refreshed"),
            Err(e) => warn!("IVF-PQ ANN index refresh failed (unindexed tail stays exact): {e}"),
        }
    }

    /// Best-effort IVF-PQ on `vector`. Small tables keep exact scan.
    /// Existing cosine indexes are incrementally refreshed after large appends.
    /// A non-cosine vector index is dropped and replaced when the table is large enough.
    async fn ensure_vector_index(&self) -> Result<(), StoreError> {
        let table = self.get_chunks_table().await?;
        if let Some(name) = Self::vector_index_name(&table).await {
            match table.index_stats(&name).await {
                Ok(Some(stats)) if Self::ann_index_is_cosine(stats.distance_type) => {
                    Self::refresh_vector_index(&table, &name).await;
                    return Ok(());
                }
                Ok(Some(stats)) => {
                    info!(
                        "Replacing vector index trained as {:?} with cosine IVF-PQ",
                        stats.distance_type
                    );
                    if let Err(e) = table.drop_index(&name).await {
                        warn!("Could not drop mismatched vector index (search may bypass): {e}");
                        return Ok(());
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    debug!("Skipping ANN reuse; could not read index stats: {e}");
                    return Ok(());
                }
            }
        }

        let count = match table.count_rows(None).await {
            Ok(n) => n,
            Err(e) => {
                debug!("Skipping ANN index; could not count rows: {e}");
                return Ok(());
            }
        };

        if !Self::should_build_ann_index(count) {
            debug!("Skipping IVF-PQ ANN index: {count} rows < {MIN_ANN_ROWS} (exact scan)");
            return Ok(());
        }

        info!("Creating IVF-PQ ANN index on vector ({count} rows, cosine)");
        match table
            .create_index(
                &["vector"],
                Index::IvfPq(IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine)),
            )
            .execute()
            .await
        {
            Ok(()) => info!("IVF-PQ ANN index ready"),
            Err(e) => warn!("IVF-PQ ANN index not created (search stays exact scan): {e}"),
        }
        Ok(())
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
            Field::new("dir_path", DataType::Utf8, false),
            Field::new("dir_depth", DataType::UInt16, false),
            Field::new("path_components", DataType::Utf8, false),
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

    fn schema_sidecar_path(&self) -> PathBuf {
        self.db_path
            .parent()
            .unwrap_or(self.db_path.as_path())
            .join(SCHEMA_SIDECAR_FILENAME)
    }

    async fn read_schema_sidecar(&self) -> Option<u32> {
        let bytes = tokio::fs::read(self.schema_sidecar_path()).await.ok()?;
        serde_json::from_slice::<SchemaSidecar>(&bytes)
            .ok()
            .map(|s| s.chunks_schema_version)
    }

    async fn write_schema_sidecar(&self, version: u32) -> Result<(), StoreError> {
        let path = self.schema_sidecar_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                StoreError::Schema(format!(
                    "Failed to create schema sidecar directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let json = serde_json::to_string_pretty(&SchemaSidecar {
            chunks_schema_version: version,
        })
        .map_err(|e| StoreError::Schema(format!("Failed to serialize schema sidecar: {e}")))?;
        tokio::fs::write(&path, json).await.map_err(|e| {
            StoreError::Schema(format!(
                "Failed to write schema sidecar {}: {e}",
                path.display()
            ))
        })?;
        Ok(())
    }

    fn schema_migration_failed_message(db_path: &Path, error: &impl std::fmt::Display) -> String {
        let index_dir = db_path.parent().unwrap_or(db_path);
        format!(
            "Existing Lance index is missing directory-scope columns \
             (dir_path, dir_depth, path_components) and automatic schema \
             migration to v{CHUNKS_SCHEMA_VERSION} failed ({error}). \
             The index was not deleted. Remove `{}` and run \
             `ragfs index <dir> --force` to rebuild.",
            index_dir.display()
        )
    }

    /// Detect a pre-v2 chunks table, add scope columns, and backfill from `file_path`.
    ///
    /// `--force` rewrites index-root-relative `dir_path` on new upserts; it cannot
    /// add Lance columns. This path is what existing indexes need on upgrade.
    async fn ensure_chunks_schema_v2(&self) -> Result<(), StoreError> {
        let table = {
            let conn = self.get_connection().await?;
            conn.open_table(CHUNKS_TABLE)
                .execute()
                .await
                .map_err(|e| StoreError::Init(format!("Failed to open chunks table: {e}")))?
        };

        let schema = table
            .schema()
            .await
            .map_err(|e| StoreError::Schema(format!("Failed to read chunks schema: {e}")))?;
        let missing = missing_scope_columns(&schema);
        let sidecar_version = self.read_schema_sidecar().await.unwrap_or(0);

        if missing.is_empty() && sidecar_version >= CHUNKS_SCHEMA_VERSION {
            return Ok(());
        }

        if !missing.is_empty() {
            info!(
                "Migrating chunks table to schema v{CHUNKS_SCHEMA_VERSION} \
                 (adding directory-scope columns)"
            );
            table
                .add_columns()
                .transform(NewColumnTransform::AllNulls(Arc::new(Schema::new(missing))))
                .execute()
                .await
                .map_err(|e| {
                    StoreError::Schema(Self::schema_migration_failed_message(&self.db_path, &e))
                })?;

            // Drop the cached handle so later ops see the evolved schema.
            *self.chunks_table.write().await = None;
        }

        self.backfill_scope_columns().await?;
        self.write_schema_sidecar(CHUNKS_SCHEMA_VERSION).await?;
        info!("Chunks table schema is v{CHUNKS_SCHEMA_VERSION}");
        Ok(())
    }

    async fn backfill_scope_columns(&self) -> Result<(), StoreError> {
        let table = self.get_chunks_table().await?;
        let mut results = table
            .query()
            .only_if("dir_path IS NULL OR dir_path = ''")
            .execute()
            .await
            .map_err(|e| {
                StoreError::Schema(format!(
                    "Failed to scan chunks for directory-scope backfill: {e}"
                ))
            })?;

        let mut paths = HashSet::new();
        while let Some(batch) = results.try_next().await.map_err(|e| {
            StoreError::Schema(format!(
                "Failed to read chunks for directory-scope backfill: {e}"
            ))
        })? {
            let Some(file_paths) = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                continue;
            };
            for i in 0..batch.num_rows() {
                if !file_paths.is_null(i) {
                    paths.insert(file_paths.value(i).to_string());
                }
            }
        }

        if paths.is_empty() {
            return Ok(());
        }

        info!(
            "Backfilling directory-scope columns for {} file path(s)",
            paths.len()
        );
        for path in paths {
            let scope = DirectoryScope::from_file_path(Path::new(&path));
            table
                .update()
                .only_if(format!("file_path = '{}'", escape_sql_literal(&path)))
                .column(
                    "dir_path",
                    format!("'{}'", escape_sql_literal(&scope.dir_path)),
                )
                .column("dir_depth", scope.dir_depth.to_string())
                .column(
                    "path_components",
                    format!("'{}'", escape_sql_literal(&scope.path_components)),
                )
                .execute()
                .await
                .map_err(|e| {
                    StoreError::Schema(format!(
                        "Failed to backfill directory-scope columns for '{path}': {e}"
                    ))
                })?;
        }
        Ok(())
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

        let dir_paths: Vec<_> = chunks.iter().map(|c| c.dir_path.clone()).collect();
        let dir_depths: Vec<_> = chunks.iter().map(|c| c.dir_depth).collect();
        let path_components: Vec<_> = chunks.iter().map(|c| c.path_components.clone()).collect();

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
                Arc::new(StringArray::from(dir_paths)),
                Arc::new(UInt16Array::from(dir_depths)),
                Arc::new(StringArray::from(path_components)),
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

    /// Combine chunk-column filters with source-file `modified_at` constraints.
    async fn chunk_filter_sql(
        &self,
        filters: &[SearchFilter],
    ) -> Result<Option<String>, StoreError> {
        Ok(combine_predicates(
            filters_to_sql(filters),
            self.modified_at_path_predicate(filters).await?,
        ))
    }

    /// Resolve `ModifiedAfter` / `ModifiedBefore` against the files table.
    async fn modified_at_path_predicate(
        &self,
        filters: &[SearchFilter],
    ) -> Result<Option<String>, StoreError> {
        let Some(date_sql) = file_date_filters_to_sql(filters) else {
            return Ok(None);
        };

        let table = self.get_files_table().await?;
        let mut results = table
            .query()
            .only_if(date_sql)
            .execute()
            .await
            .map_err(|e| StoreError::Query(format!("Failed to apply modified_at filter: {e}")))?;

        let mut paths = Vec::new();
        while let Some(batch) = results.try_next().await.map_err(|e| {
            StoreError::Query(format!("Failed to fetch files for modified_at filter: {e}"))
        })? {
            for record in batch_to_file_records(&batch)? {
                paths.push(record.path.to_string_lossy().to_string());
            }
        }

        if paths.is_empty() {
            return Ok(Some("1 = 0".to_string()));
        }

        let in_list = paths
            .iter()
            .map(|p| format!("'{}'", escape_sql_literal(p)))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(Some(format!("file_path IN ({in_list})")))
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

        if tables.contains(&CHUNKS_TABLE.to_string()) {
            self.ensure_chunks_schema_v2().await?;
        } else {
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

            self.write_schema_sidecar(CHUNKS_SCHEMA_VERSION).await?;
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

        // Existing tables may already have enough rows for ANN.
        if let Err(e) = self.ensure_vector_index().await {
            warn!("ANN index check on init failed: {e}");
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
            .add(Box::new(batches) as Box<dyn RecordBatchReader + Send>)
            .execute()
            .await
            .map_err(|e| StoreError::Insert(format!("Failed to insert chunks: {e}")))?;

        debug!("Successfully upserted {} chunks", chunks.len());
        if let Err(e) = self.ensure_vector_index().await {
            warn!("ANN index check after upsert failed: {e}");
        }
        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, StoreError> {
        debug!(
            "Searching with limit {} metric {:?} filters {}",
            query.limit,
            query.metric,
            query.filters.len()
        );

        let table = self.get_chunks_table().await?;
        let filter_sql = combine_predicates(
            self.chunk_filter_sql(&query.filters).await?,
            query.scope_prefix.as_deref().and_then(scope_prefix_sql),
        );

        let mut search_q = table
            .vector_search(query.embedding.clone())
            .map_err(|e| StoreError::Query(format!("Failed to create search query: {e}")))?
            .distance_type(distance_type_from_metric(query.metric))
            .limit(query.limit);

        if !Self::ann_index_covers_metric(query.metric) {
            search_q = search_q.bypass_vector_index();
        }

        if let Some(ref filter) = filter_sql {
            debug!("Applying search filter: {filter}");
            search_q = search_q.only_if(filter);
        }

        let mut results = search_q
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
            "Performing hybrid search with text: '{}' limit {} metric {:?} filters {}",
            query_text,
            query.limit,
            query.metric,
            query.filters.len()
        );

        let table = self.get_chunks_table().await?;
        let filter_sql = combine_predicates(
            self.chunk_filter_sql(&query.filters).await?,
            query.scope_prefix.as_deref().and_then(scope_prefix_sql),
        );

        // Build hybrid query combining FTS and vector search
        let fts_query = FullTextSearchQuery::new(query_text);

        let mut search_q = table
            .query()
            .full_text_search(fts_query)
            .nearest_to(query.embedding.clone())
            .map_err(|e| StoreError::Query(format!("Failed to create hybrid query: {e}")))?
            .distance_type(distance_type_from_metric(query.metric))
            .limit(query.limit);

        if !Self::ann_index_covers_metric(query.metric) {
            search_q = search_q.bypass_vector_index();
        }

        if let Some(ref filter) = filter_sql {
            debug!("Applying hybrid search filter: {filter}");
            search_q = search_q.only_if(filter);
        }

        let mut results = search_q
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

        // 2. Update the file_path and directory-scope fields in each chunk
        for chunk in &mut chunks {
            let root = DirectoryScope::infer_root(&chunk.file_path, &chunk.dir_path);
            let scope = DirectoryScope::from_paths(to, root.as_deref());
            chunk.file_path = to.to_path_buf();
            chunk.dir_path = scope.dir_path;
            chunk.dir_depth = scope.dir_depth;
            chunk.path_components = scope.path_components;
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
            .add(Box::new(batches) as Box<dyn RecordBatchReader + Send>)
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

/// Map [`DistanceMetric`] to the `LanceDB` distance type used at query time.
fn distance_type_from_metric(metric: DistanceMetric) -> DistanceType {
    match metric {
        DistanceMetric::Cosine => DistanceType::Cosine,
        DistanceMetric::L2 => DistanceType::L2,
        DistanceMetric::Dot => DistanceType::Dot,
    }
}

/// Escape a string for use inside a single-quoted SQL literal.
fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

/// Escape `%` and `_` so they are treated as literals in `LIKE` patterns.
fn escape_like_literal(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Convert a glob (`*`, `**`, `?`) into an anchored regular expression.
///
/// `*` and `?` do not cross `/`. `**` matches across directories; `**/` also
/// matches zero intervening segments without collapsing the following name.
fn glob_to_regex(glob: &str) -> String {
    let mut regex = String::from("^");
    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    i += 2;
                    if i < chars.len() && chars[i] == '/' {
                        i += 1;
                        regex.push_str("(?:.*/)?");
                    } else {
                        regex.push_str(".*");
                    }
                } else {
                    regex.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                regex.push_str("[^/]");
                i += 1;
            }
            c => {
                regex.push_str(&regex_escape_char(c));
                i += 1;
            }
        }
    }

    regex.push('$');
    regex
}

fn regex_escape_char(c: char) -> String {
    if matches!(
        c,
        '.' | '+' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '\\'
    ) {
        format!("\\{c}")
    } else {
        c.to_string()
    }
}

/// Convert one [`SearchFilter`] into a chunks-table SQL predicate.
///
/// Date filters are omitted here; they are resolved against `files.modified_at`.
fn filter_to_sql(filter: &SearchFilter) -> Option<String> {
    Some(match filter {
        SearchFilter::PathPrefix(prefix) => {
            let pattern = escape_sql_literal(&format!("{}%", escape_like_literal(prefix)));
            format!("file_path LIKE '{pattern}' ESCAPE '\\'")
        }
        SearchFilter::PathGlob(glob) => {
            let pattern = escape_sql_literal(&glob_to_regex(glob));
            format!("regexp_like(file_path, '{pattern}')")
        }
        SearchFilter::MimeType(value) => type_or_mime_sql(value),
        SearchFilter::Language(lang) => {
            let escaped = escape_sql_literal(&lang.to_lowercase());
            format!("(LOWER(language) = '{escaped}' OR LOWER(content_type) = 'code:{escaped}')")
        }
        SearchFilter::ModifiedAfter(_) | SearchFilter::ModifiedBefore(_) => return None,
        SearchFilter::MinDepth(depth) => format!("depth >= {depth}"),
        SearchFilter::MaxDepth(depth) => format!("depth <= {depth}"),
    })
}

/// Files-table predicates for source modification time (inclusive).
fn file_date_filters_to_sql(filters: &[SearchFilter]) -> Option<String> {
    let clauses: Vec<String> = filters
        .iter()
        .filter_map(|filter| match filter {
            SearchFilter::ModifiedAfter(ts) => {
                let escaped = escape_sql_literal(&ts.to_rfc3339());
                Some(format!("modified_at >= '{escaped}'"))
            }
            SearchFilter::ModifiedBefore(ts) => {
                let escaped = escape_sql_literal(&ts.to_rfc3339());
                Some(format!("modified_at <= '{escaped}'"))
            }
            _ => None,
        })
        .collect();

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

fn combine_predicates(left: Option<String>, right: Option<String>) -> Option<String> {
    match (left, right) {
        (Some(a), Some(b)) => Some(format!("{a} AND {b}")),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// `type:` / `mime:` values may name a content-type alias or a MIME type.
fn type_or_mime_sql(value: &str) -> String {
    let lowered = value.to_lowercase();
    let escaped = escape_sql_literal(&lowered);

    if lowered.contains('/') {
        return format!("LOWER(file_mime_type) = '{escaped}'");
    }

    match lowered.as_str() {
        "code" => "(LOWER(content_type) LIKE 'code:%' OR LOWER(content_type) = 'code')".to_string(),
        "text" => "LOWER(content_type) = 'text'".to_string(),
        "markdown" | "md" => "LOWER(content_type) = 'markdown'".to_string(),
        "pdf" => "(LOWER(content_type) LIKE 'pdf:%' OR LOWER(content_type) = 'pdf')".to_string(),
        "image" | "image_caption" => "LOWER(content_type) = 'image_caption'".to_string(),
        _ => format!(
            "(LOWER(content_type) = '{escaped}' OR LOWER(content_type) LIKE '{escaped}:%' OR LOWER(file_mime_type) = '{escaped}')"
        ),
    }
}

/// Pre-filter for directory-scoped search (exact dir, subdirectory, or suffix).
///
/// Used with IVF-PQ ANN via `only_if`; this is not a replacement for ANN.
/// Suffix `LIKE` clauses cover schema-v1 rows backfilled from absolute `file_path`.
fn scope_prefix_sql(scope: &str) -> Option<String> {
    let scope = DirectoryScope::normalize_prefix(scope);
    if scope.is_empty() || scope == "." {
        return None;
    }
    let exact = escape_sql_literal(&scope);
    let prefix = escape_sql_literal(&format!("{}/%", escape_like_literal(&scope)));
    let suffix = escape_sql_literal(&format!("%/{}", escape_like_literal(&scope)));
    let mid = escape_sql_literal(&format!("%/{}/%", escape_like_literal(&scope)));
    Some(format!(
        "dir_path = '{exact}' OR dir_path LIKE '{prefix}' ESCAPE '\\' \
         OR dir_path LIKE '{suffix}' ESCAPE '\\' OR dir_path LIKE '{mid}' ESCAPE '\\'"
    ))
}

/// Combine [`SearchQuery`] filters into a single SQL predicate, if any.
fn filters_to_sql(filters: &[SearchFilter]) -> Option<String> {
    if filters.is_empty() {
        return None;
    }

    let clauses: Vec<String> = filters.iter().filter_map(filter_to_sql).collect();
    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
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
    let dir_paths = batch
        .column_by_name("dir_path")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let dir_depths = batch
        .column_by_name("dir_depth")
        .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
    let path_components_arr = batch
        .column_by_name("path_components")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

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
            dir_path: dir_paths.map_or_else(
                || DirectoryScope::from_file_path(&PathBuf::from(file_paths.value(i))).dir_path,
                |arr| {
                    if arr.is_null(i) {
                        String::new()
                    } else {
                        arr.value(i).to_string()
                    }
                },
            ),
            dir_depth: dir_depths.map_or(0, |arr| if arr.is_null(i) { 0 } else { arr.value(i) }),
            path_components: path_components_arr.map_or_else(String::new, |arr| {
                if arr.is_null(i) {
                    String::new()
                } else {
                    arr.value(i).to_string()
                }
            }),
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
    use crate::schema::{CHUNKS_SCHEMA_VERSION, SCHEMA_SIDECAR_FILENAME, SchemaSidecar};
    use ragfs_core::{DistanceMetric, SearchFilter};
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
            dir_path: DirectoryScope::from_file_path(file_path).dir_path,
            dir_depth: DirectoryScope::from_file_path(file_path).dir_depth,
            path_components: DirectoryScope::from_file_path(file_path).path_components,
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
            scope_prefix: None,
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
            scope_prefix: None,
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
            dir_path: DirectoryScope::from_file_path(&file_path).dir_path,
            dir_depth: DirectoryScope::from_file_path(&file_path).dir_depth,
            path_components: DirectoryScope::from_file_path(&file_path).path_components,
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

    #[tokio::test]
    async fn test_get_all_files_empty() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let files = store.get_all_files().await.unwrap();
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_file() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let path = PathBuf::from("/nonexistent/file.txt");
        // Deleting a nonexistent file should not error
        let result = store.delete_by_file_path(&path).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_ann_threshold() {
        assert!(!LanceStore::should_build_ann_index(0));
        assert!(!LanceStore::should_build_ann_index(255));
        assert!(LanceStore::should_build_ann_index(256));
        assert!(LanceStore::should_build_ann_index(10_000));
    }

    #[test]
    fn test_should_refresh_ann_index_after_post_threshold_appends() {
        // Repeated-batch ingest: first batch crosses 256, later batches are unindexed.
        assert!(LanceStore::should_build_ann_index(300));
        assert!(!LanceStore::should_refresh_ann_index(0));
        assert!(!LanceStore::should_refresh_ann_index(255));
        assert!(LanceStore::should_refresh_ann_index(256));
        assert!(LanceStore::should_refresh_ann_index(700));
        assert!(LanceStore::should_refresh_ann_index(9_700));
    }

    #[test]
    fn test_ann_index_covers_cosine_only() {
        assert!(LanceStore::ann_index_covers_metric(DistanceMetric::Cosine));
        assert!(!LanceStore::ann_index_covers_metric(DistanceMetric::L2));
        assert!(!LanceStore::ann_index_covers_metric(DistanceMetric::Dot));
        assert!(LanceStore::ann_index_is_cosine(Some(DistanceType::Cosine)));
        assert!(!LanceStore::ann_index_is_cosine(Some(DistanceType::L2)));
        assert!(!LanceStore::ann_index_is_cosine(Some(DistanceType::Dot)));
        assert!(!LanceStore::ann_index_is_cosine(None));
    }

    #[tokio::test]
    async fn test_small_table_skips_ann_and_still_searches() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/small.txt");
        let chunk = create_test_chunk(
            &file_path,
            "hello ann",
            create_random_embedding(TEST_DIM),
            0,
        );
        store.upsert_chunks(&[chunk]).await.unwrap();
        store.ensure_vector_index().await.unwrap();

        let results = store
            .search(SearchQuery {
                text: None,
                embedding: create_random_embedding(TEST_DIM),
                limit: 5,
                filters: vec![],
                metric: DistanceMetric::Cosine,
                scope_prefix: None,
            })
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_replaces_preexisting_l2_vector_index() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let file_path = PathBuf::from("/test/l2.txt");
        let chunks: Vec<Chunk> = (0..16)
            .map(|i| {
                create_test_chunk(
                    &file_path,
                    &format!("legacy l2 chunk {i}"),
                    create_random_embedding(TEST_DIM),
                    i,
                )
            })
            .collect();
        store.upsert_chunks(&chunks).await.unwrap();

        let table = store.get_chunks_table().await.unwrap();
        let created = table
            .create_index(
                &["vector"],
                Index::IvfPq(
                    IvfPqIndexBuilder::default()
                        .distance_type(DistanceType::L2)
                        .num_partitions(1)
                        .sample_rate(4),
                ),
            )
            .execute()
            .await;
        if created.is_err() {
            // Fixture only: Lance may refuse a tiny L2 index. Cosine search must still work.
            let results = store
                .search(SearchQuery {
                    text: None,
                    embedding: create_random_embedding(TEST_DIM),
                    limit: 5,
                    filters: vec![],
                    metric: DistanceMetric::Cosine,
                    scope_prefix: None,
                })
                .await
                .unwrap();
            assert!(!results.is_empty());
            return;
        }

        let name = LanceStore::vector_index_name(&table)
            .await
            .expect("L2 fixture index");
        let before = table.index_stats(&name).await.unwrap().unwrap();
        assert_eq!(before.distance_type, Some(DistanceType::L2));

        store.ensure_vector_index().await.unwrap();

        if let Some(after_name) = LanceStore::vector_index_name(&table).await {
            let after = table.index_stats(&after_name).await.unwrap().unwrap();
            assert!(
                LanceStore::ann_index_is_cosine(after.distance_type),
                "mismatched L2 index must be replaced with cosine"
            );
        } else {
            // Below MIN_ANN_ROWS: drop L2 and stay on exact scan.
        }

        let results = store
            .search(SearchQuery {
                text: None,
                embedding: create_random_embedding(TEST_DIM),
                limit: 5,
                filters: vec![],
                metric: DistanceMetric::Cosine,
                scope_prefix: None,
            })
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_get_file_not_found() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path.clone(), TEST_DIM);
        store.init().await.unwrap();

        let path = PathBuf::from("/nonexistent/file.txt");
        let result = store.get_file(&path).await.unwrap();
        assert!(result.is_none());
    }

    fn create_chunk_with_meta(
        file_path: &Path,
        content: &str,
        embedding: Vec<f32>,
        content_type: ContentType,
        mime_type: Option<String>,
        depth: u8,
    ) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            file_id: Uuid::new_v4(),
            file_path: file_path.to_path_buf(),
            content: content.to_string(),
            content_type,
            mime_type,
            chunk_index: 0,
            byte_range: 0..content.len() as u64,
            line_range: Some(0..1),
            parent_chunk_id: None,
            depth,
            embedding: Some(embedding),
            dir_path: DirectoryScope::from_file_path(file_path).dir_path,
            dir_depth: DirectoryScope::from_file_path(file_path).dir_depth,
            path_components: DirectoryScope::from_file_path(file_path).path_components,
            metadata: ChunkMetadata {
                indexed_at: Some(Utc::now()),
                embedding_model: Some("test-model".to_string()),
                token_count: None,
                extra: HashMap::new(),
            },
        }
    }

    async fn seeded_filter_store() -> (tempfile::TempDir, LanceStore, Vec<f32>) {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let embedding = create_random_embedding(TEST_DIM);
        let chunks = vec![
            create_chunk_with_meta(
                Path::new("src/lib.rs"),
                "fn rust_auth() {}",
                embedding.clone(),
                ContentType::Code {
                    language: "rust".to_string(),
                    symbol: None,
                },
                Some("text/x-rust".to_string()),
                0,
            ),
            create_chunk_with_meta(
                Path::new("src/app.py"),
                "def python_auth(): pass",
                embedding.clone(),
                ContentType::Code {
                    language: "python".to_string(),
                    symbol: None,
                },
                Some("text/x-python".to_string()),
                1,
            ),
            create_chunk_with_meta(
                Path::new("docs/readme.md"),
                "authentication notes",
                embedding.clone(),
                ContentType::Markdown,
                Some("text/markdown".to_string()),
                3,
            ),
            create_chunk_with_meta(
                Path::new("notes/plain.txt"),
                "plain authentication text",
                embedding.clone(),
                ContentType::Text,
                Some("text/plain".to_string()),
                2,
            ),
        ];
        store.upsert_chunks(&chunks).await.unwrap();
        (temp, store, embedding)
    }

    fn search_query(
        embedding: Vec<f32>,
        filters: Vec<SearchFilter>,
        metric: DistanceMetric,
    ) -> SearchQuery {
        SearchQuery {
            text: Some("authentication".to_string()),
            embedding,
            limit: 10,
            filters,
            metric,
            scope_prefix: None,
        }
    }

    #[test]
    fn test_filters_to_sql_language_path_type_depth() {
        let sql = filters_to_sql(&[
            SearchFilter::Language("Rust".to_string()),
            SearchFilter::PathPrefix("src/".to_string()),
            SearchFilter::MimeType("code".to_string()),
            SearchFilter::MaxDepth(2),
        ])
        .unwrap();

        assert!(sql.contains("LOWER(language) = 'rust'"));
        assert!(sql.contains("LOWER(content_type) = 'code:rust'"));
        assert!(sql.contains("file_path LIKE 'src/%' ESCAPE '\\'"));
        assert!(sql.contains("LOWER(content_type) LIKE 'code:%'"));
        assert!(sql.contains("depth <= 2"));
        assert!(sql.contains(" AND "));
    }

    #[test]
    fn test_filters_to_sql_path_glob_and_mime() {
        let sql = filters_to_sql(&[
            SearchFilter::PathGlob("src/**/*.rs".to_string()),
            SearchFilter::MimeType("text/x-rust".to_string()),
            SearchFilter::MinDepth(1),
        ])
        .unwrap();

        assert!(sql.contains("regexp_like(file_path, '^src/(?:.*/)?[^/]*\\.rs$')"));
        assert!(sql.contains("LOWER(file_mime_type) = 'text/x-rust'"));
        assert!(sql.contains("depth >= 1"));
    }

    #[test]
    fn test_filters_to_sql_escapes_quotes() {
        let sql = filters_to_sql(&[SearchFilter::PathPrefix("o'brien".to_string())]).unwrap();
        assert!(sql.contains("o''brien"));
    }

    #[test]
    fn test_filters_to_sql_empty() {
        assert!(filters_to_sql(&[]).is_none());
    }

    #[test]
    fn test_glob_to_regex_preserves_path_segments() {
        assert_eq!(glob_to_regex("src/*.rs"), r"^src/[^/]*\.rs$");
        assert_eq!(glob_to_regex("src/**/mod.rs"), r"^src/(?:.*/)?mod\.rs$");
        assert_eq!(glob_to_regex("src/**"), r"^src/.*$");
        assert_eq!(glob_to_regex("file?.txt"), r"^file[^/]\.txt$");
        assert!(!glob_to_regex("src/*.rs").contains(".*"));
        assert!(!glob_to_regex("src/**/mod.rs").contains("%mod"));
    }

    #[test]
    fn test_file_date_filters_use_modified_at() {
        let after = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let sql = file_date_filters_to_sql(&[SearchFilter::ModifiedAfter(after)]).unwrap();
        assert!(sql.contains("modified_at >= "));
        assert!(!sql.contains("indexed_at"));
        assert!(filters_to_sql(&[SearchFilter::ModifiedAfter(after)]).is_none());
    }

    #[test]
    fn test_distance_type_from_metric() {
        assert_eq!(
            distance_type_from_metric(DistanceMetric::Cosine),
            DistanceType::Cosine
        );
        assert_eq!(
            distance_type_from_metric(DistanceMetric::L2),
            DistanceType::L2
        );
        assert_eq!(
            distance_type_from_metric(DistanceMetric::Dot),
            DistanceType::Dot
        );
    }

    #[tokio::test]
    async fn test_search_filters_by_language() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::Language("rust".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, PathBuf::from("src/lib.rs"));
    }

    #[tokio::test]
    async fn test_search_filters_by_path_prefix() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::PathPrefix("src/".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert_eq!(results.len(), 2);
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
        assert!(paths.contains(&PathBuf::from("src/app.py")));
        assert!(!paths.contains(&PathBuf::from("docs/readme.md")));
    }

    #[tokio::test]
    async fn test_search_filters_by_path_glob() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::PathGlob("src/**".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert_eq!(results.len(), 2);
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
        assert!(paths.contains(&PathBuf::from("src/app.py")));
    }

    #[tokio::test]
    async fn test_search_filters_by_type_code() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::MimeType("code".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert_eq!(results.len(), 2);
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
        assert!(paths.contains(&PathBuf::from("src/app.py")));
        assert!(!paths.contains(&PathBuf::from("notes/plain.txt")));
    }

    #[tokio::test]
    async fn test_search_filters_by_mime_type() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::MimeType("text/plain".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, PathBuf::from("notes/plain.txt"));
    }

    #[tokio::test]
    async fn test_search_filters_by_max_depth() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::MaxDepth(1)],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert_eq!(results.len(), 2);
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
        assert!(paths.contains(&PathBuf::from("src/app.py")));
        assert!(!paths.contains(&PathBuf::from("docs/readme.md")));
        assert!(!paths.contains(&PathBuf::from("notes/plain.txt")));
    }

    #[tokio::test]
    async fn test_search_combines_filters() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![
                    SearchFilter::PathPrefix("src/".to_string()),
                    SearchFilter::Language("python".to_string()),
                    SearchFilter::MaxDepth(2),
                ],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, PathBuf::from("src/app.py"));
    }

    #[tokio::test]
    async fn test_search_applies_l2_and_dot_metrics() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let filters = vec![SearchFilter::Language("rust".to_string())];

        for metric in [
            DistanceMetric::L2,
            DistanceMetric::Dot,
            DistanceMetric::Cosine,
        ] {
            let results = store
                .search(search_query(embedding.clone(), filters.clone(), metric))
                .await
                .unwrap();
            assert_eq!(
                results.len(),
                1,
                "metric {metric:?} should still honor language filter"
            );
            assert_eq!(results[0].file_path, PathBuf::from("src/lib.rs"));
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_applies_filters_and_metric() {
        let (_temp, store, embedding) = seeded_filter_store().await;
        let results = store
            .hybrid_search(search_query(
                embedding,
                vec![
                    SearchFilter::Language("rust".to_string()),
                    SearchFilter::MimeType("code".to_string()),
                    SearchFilter::MaxDepth(1),
                ],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, PathBuf::from("src/lib.rs"));
        assert!(results[0].content.contains("rust_auth"));
    }

    async fn seeded_glob_boundary_store() -> (tempfile::TempDir, LanceStore, Vec<f32>) {
        let temp = tempdir().unwrap();
        let store = LanceStore::new(temp.path().join("test.lance"), TEST_DIM);
        store.init().await.unwrap();
        let embedding = create_random_embedding(TEST_DIM);
        store
            .upsert_chunks(&[
                create_test_chunk(Path::new("src/lib.rs"), "lib", embedding.clone(), 0),
                create_test_chunk(
                    Path::new("src/nested/file.rs"),
                    "nested",
                    embedding.clone(),
                    0,
                ),
                create_test_chunk(Path::new("src/mod.rs"), "mod file", embedding.clone(), 0),
                create_test_chunk(Path::new("src/notmod.rs"), "not mod", embedding.clone(), 0),
            ])
            .await
            .unwrap();
        (temp, store, embedding)
    }

    #[tokio::test]
    async fn test_search_path_glob_star_does_not_cross_slash() {
        let (_temp, store, embedding) = seeded_glob_boundary_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::PathGlob("src/*.rs".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
        assert!(paths.contains(&PathBuf::from("src/mod.rs")));
        assert!(paths.contains(&PathBuf::from("src/notmod.rs")));
        assert!(!paths.contains(&PathBuf::from("src/nested/file.rs")));
    }

    #[tokio::test]
    async fn test_search_path_glob_doublestar_keeps_separator() {
        let (_temp, store, embedding) = seeded_glob_boundary_store().await;
        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::PathGlob("src/**/mod.rs".to_string())],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        let paths: Vec<_> = results.iter().map(|r| r.file_path.clone()).collect();
        assert_eq!(results.len(), 1);
        assert!(paths.contains(&PathBuf::from("src/mod.rs")));
        assert!(!paths.contains(&PathBuf::from("src/notmod.rs")));
        assert!(!paths.contains(&PathBuf::from("src/lib.rs")));
    }

    #[tokio::test]
    async fn test_search_filters_by_source_modified_at() {
        let temp = tempdir().unwrap();
        let store = LanceStore::new(temp.path().join("test.lance"), TEST_DIM);
        store.init().await.unwrap();
        let embedding = create_random_embedding(TEST_DIM);

        let old_path = PathBuf::from("old.txt");
        let new_path = PathBuf::from("new.txt");
        store
            .upsert_chunks(&[
                create_test_chunk(&old_path, "old file", embedding.clone(), 0),
                create_test_chunk(&new_path, "new file", embedding.clone(), 0),
            ])
            .await
            .unwrap();

        let old_modified = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let new_modified = chrono::DateTime::parse_from_rfc3339("2024-06-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2022-01-01T00:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);

        let mut old_record = create_test_file_record(&old_path);
        old_record.modified_at = old_modified;
        let mut new_record = create_test_file_record(&new_path);
        new_record.modified_at = new_modified;
        store.upsert_file(&old_record).await.unwrap();
        store.upsert_file(&new_record).await.unwrap();

        let results = store
            .search(search_query(
                embedding,
                vec![SearchFilter::ModifiedAfter(cutoff)],
                DistanceMetric::Cosine,
            ))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, new_path);
    }

    fn create_chunk_with_scope(
        file_path: &Path,
        root: Option<&Path>,
        content: &str,
        embedding: Vec<f32>,
    ) -> Chunk {
        let scope = DirectoryScope::from_paths(file_path, root);
        let mut chunk = create_test_chunk(file_path, content, embedding, 0);
        chunk.dir_path = scope.dir_path;
        chunk.dir_depth = scope.dir_depth;
        chunk.path_components = scope.path_components;
        chunk
    }

    async fn seeded_scope_store() -> (tempfile::TempDir, LanceStore, Vec<f32>) {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let root = Path::new("/project");
        let embedding = create_random_embedding(TEST_DIM);
        let chunks = vec![
            create_chunk_with_scope(
                Path::new("/project/src/auth/login.rs"),
                Some(root),
                "login handler",
                embedding.clone(),
            ),
            create_chunk_with_scope(
                Path::new("/project/src/auth/oauth/token.rs"),
                Some(root),
                "oauth token",
                embedding.clone(),
            ),
            create_chunk_with_scope(
                Path::new("/project/src/db.rs"),
                Some(root),
                "database pool",
                embedding.clone(),
            ),
            create_chunk_with_scope(
                Path::new("/project/docs/readme.md"),
                Some(root),
                "project readme",
                embedding.clone(),
            ),
        ];
        store.upsert_chunks(&chunks).await.unwrap();
        (temp, store, embedding)
    }

    fn scoped_query(embedding: Vec<f32>, scope: &str) -> SearchQuery {
        SearchQuery {
            text: None,
            embedding,
            limit: 10,
            filters: vec![],
            metric: DistanceMetric::Cosine,
            scope_prefix: Some(scope.to_string()),
        }
    }

    #[test]
    fn test_scope_prefix_sql_exact_or_subdirectory() {
        let sql = scope_prefix_sql("src/auth/").unwrap();
        assert!(sql.contains("dir_path = 'src/auth'"));
        assert!(sql.contains("dir_path LIKE 'src/auth/%'"));
        assert!(sql.contains("dir_path LIKE '%/src/auth'"));
        assert!(sql.contains("dir_path LIKE '%/src/auth/%'"));
        assert!(scope_prefix_sql("").is_none());
        assert!(scope_prefix_sql(".").is_none());
    }

    #[test]
    fn test_scope_prefix_sql_escapes_quotes() {
        let sql = scope_prefix_sql("o'brien").unwrap();
        assert!(sql.contains("o''brien"));
    }

    #[tokio::test]
    async fn test_scoped_search_stores_relative_dir_path() {
        let (_temp, store, _) = seeded_scope_store().await;
        let chunks = store
            .get_chunks_for_file(Path::new("/project/src/auth/login.rs"))
            .await
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].dir_path, "src/auth");
        assert_eq!(chunks[0].dir_depth, 2);
        assert_eq!(chunks[0].path_components, "src,auth,login.rs");
        assert!(!chunks[0].dir_path.starts_with('/'));
    }

    #[tokio::test]
    async fn test_scoped_search_exact_directory() {
        let (_temp, store, embedding) = seeded_scope_store().await;
        let results = store
            .search(scoped_query(embedding, "src/auth"))
            .await
            .unwrap();
        let paths: Vec<_> = results
            .iter()
            .map(|r| r.file_path.to_string_lossy().to_string())
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("src/auth/login.rs")));
        assert!(paths.iter().any(|p| p.ends_with("src/auth/oauth/token.rs")));
        assert!(!paths.iter().any(|p| p.ends_with("src/db.rs")));
        assert!(!paths.iter().any(|p| p.ends_with("docs/readme.md")));
    }

    #[tokio::test]
    async fn test_scoped_search_includes_subdirectory() {
        let (_temp, store, embedding) = seeded_scope_store().await;
        let results = store.search(scoped_query(embedding, "src")).await.unwrap();
        let paths: Vec<_> = results
            .iter()
            .map(|r| r.file_path.to_string_lossy().to_string())
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("src/auth/login.rs")));
        assert!(paths.iter().any(|p| p.ends_with("src/auth/oauth/token.rs")));
        assert!(paths.iter().any(|p| p.ends_with("src/db.rs")));
        assert!(!paths.iter().any(|p| p.ends_with("docs/readme.md")));
    }

    #[tokio::test]
    async fn test_scoped_search_excludes_outside_directory() {
        let (_temp, store, embedding) = seeded_scope_store().await;
        let results = store.search(scoped_query(embedding, "docs")).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .file_path
                .to_string_lossy()
                .ends_with("docs/readme.md")
        );
    }

    fn v1_chunks_schema(dim: usize) -> Schema {
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
                    dim as i32,
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

    async fn seed_v1_chunks_table(
        db_path: &Path,
        file_path: &str,
        content: &str,
        embedding: &[f32],
    ) {
        let conn = connect(db_path.to_str().unwrap()).execute().await.unwrap();
        let schema = Arc::new(v1_chunks_schema(TEST_DIM));
        conn.create_empty_table(CHUNKS_TABLE, schema.clone())
            .execute()
            .await
            .unwrap();

        let embeddings = vec![Some(embedding.iter().copied().map(Some).collect())];
        let vector = build_vector_array(&embeddings, TEST_DIM).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Uuid::new_v4().to_string()])),
                Arc::new(StringArray::from(vec![Uuid::new_v4().to_string()])),
                Arc::new(StringArray::from(vec![file_path.to_string()])),
                Arc::new(StringArray::from(vec![content.to_string()])),
                Arc::new(StringArray::from(vec!["text".to_string()])),
                Arc::new(UInt32Array::from(vec![0u32])),
                Arc::new(UInt64Array::from(vec![0u64])),
                Arc::new(UInt64Array::from(vec![content.len() as u64])),
                Arc::new(UInt32Array::from(vec![Some(0u32)])),
                Arc::new(UInt32Array::from(vec![Some(1u32)])),
                Arc::new(StringArray::from(vec![None::<String>])),
                Arc::new(UInt8Array::from(vec![0u8])),
                vector,
                Arc::new(StringArray::from(vec![Some("test-model".to_string())])),
                Arc::new(StringArray::from(vec![Utc::now().to_rfc3339()])),
                Arc::new(StringArray::from(vec![Some("text/plain".to_string())])),
                Arc::new(StringArray::from(vec![None::<String>])),
                Arc::new(StringArray::from(vec![None::<String>])),
                Arc::new(StringArray::from(vec![None::<String>])),
            ],
        )
        .unwrap();

        let table = conn.open_table(CHUNKS_TABLE).execute().await.unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table
            .add(Box::new(batches) as Box<dyn RecordBatchReader + Send>)
            .execute()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_schema_v2_migrates_existing_index() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let file_path = "/project/src/auth/login.rs";
        let embedding = create_random_embedding(TEST_DIM);
        seed_v1_chunks_table(&db_path, file_path, "login handler", &embedding).await;

        assert!(!temp.path().join(SCHEMA_SIDECAR_FILENAME).exists());

        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let sidecar: SchemaSidecar = serde_json::from_slice(
            &std::fs::read(temp.path().join(SCHEMA_SIDECAR_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar.chunks_schema_version, CHUNKS_SCHEMA_VERSION);

        let chunks = store
            .get_chunks_for_file(Path::new(file_path))
            .await
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].dir_path, "project/src/auth");
        assert_eq!(chunks[0].dir_depth, 3);
        assert_eq!(chunks[0].path_components, "project,src,auth,login.rs");

        let results = store
            .search(scoped_query(embedding, "src/auth"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .file_path
                .to_string_lossy()
                .ends_with("src/auth/login.rs")
        );

        // Second init is idempotent (sidecar already v2, columns present).
        store.init().await.unwrap();
        let again = store
            .get_chunks_for_file(Path::new(file_path))
            .await
            .unwrap();
        assert_eq!(again[0].dir_path, "project/src/auth");
    }

    #[tokio::test]
    async fn test_new_index_writes_schema_v2_sidecar() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.lance");
        let store = LanceStore::new(db_path, TEST_DIM);
        store.init().await.unwrap();

        let sidecar: SchemaSidecar = serde_json::from_slice(
            &std::fs::read(temp.path().join(SCHEMA_SIDECAR_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar.chunks_schema_version, CHUNKS_SCHEMA_VERSION);
    }
}
