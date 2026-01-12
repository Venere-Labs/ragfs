//! FUSE filesystem implementation.

use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
};
use libc::{EEXIST, EINVAL, EIO, EISDIR, ENOENT, ENOTEMPTY, ENOSYS, ENOTDIR, EPERM};
use ragfs_core::{Embedder, VectorStore};
use ragfs_query::QueryExecutor;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::runtime::Handle;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn};

use crate::inode::{
    APPROVE_FILE_INO, CLEANUP_FILE_INO, CONFIG_FILE_INO, DEDUPE_FILE_INO, FIRST_REAL_INO,
    HELP_FILE_INO, HISTORY_FILE_INO, INDEX_FILE_INO, InodeKind, InodeTable, OPS_BATCH_INO,
    OPS_CREATE_INO, OPS_DELETE_INO, OPS_DIR_INO, OPS_MOVE_INO, OPS_RESULT_INO, ORGANIZE_FILE_INO,
    PENDING_DIR_INO, QUERY_DIR_INO, RAGFS_DIR_INO, REINDEX_FILE_INO, REJECT_FILE_INO, ROOT_INO,
    SAFETY_DIR_INO, SEARCH_DIR_INO, SEMANTIC_DIR_INO, SIMILAR_DIR_INO, SIMILAR_OPS_FILE_INO,
    TRASH_DIR_INO, UNDO_FILE_INO,
};
use crate::ops::OpsManager;
use crate::safety::SafetyManager;
use crate::semantic::SemanticManager;

const TTL: Duration = Duration::from_secs(1);
const BLOCK_SIZE: u64 = 512;

/// RAGFS FUSE filesystem.
pub struct RagFs {
    /// Source directory being indexed
    source: PathBuf,
    /// Inode table
    inodes: Arc<RwLock<InodeTable>>,
    /// Vector store for queries and stats
    store: Option<Arc<dyn VectorStore>>,
    /// Query executor
    query_executor: Option<Arc<QueryExecutor>>,
    /// Tokio runtime handle for async operations
    runtime: Handle,
    /// Cache for virtual file contents (query results, index status)
    content_cache: Arc<RwLock<HashMap<u64, Vec<u8>>>>,
    /// Channel sender for reindex requests
    reindex_sender: Option<mpsc::Sender<PathBuf>>,
    /// Operations manager for agent file management
    ops_manager: Arc<OpsManager>,
    /// Safety manager for trash/history/undo
    safety_manager: Arc<SafetyManager>,
    /// Semantic manager for intelligent operations
    semantic_manager: Arc<SemanticManager>,
}

impl RagFs {
    /// Create a new RAGFS filesystem (basic, for passthrough only).
    #[must_use]
    pub fn new(source: PathBuf) -> Self {
        let safety_manager = Arc::new(SafetyManager::new(&source, None));
        let ops_manager = Arc::new(OpsManager::with_safety(
            source.clone(),
            None,
            None,
            safety_manager.clone(),
        ));
        let semantic_manager = Arc::new(SemanticManager::new(source.clone(), None, None, None));
        Self {
            source,
            inodes: Arc::new(RwLock::new(InodeTable::new())),
            store: None,
            query_executor: None,
            runtime: Handle::current(),
            content_cache: Arc::new(RwLock::new(HashMap::new())),
            reindex_sender: None,
            ops_manager,
            safety_manager,
            semantic_manager,
        }
    }

    /// Create a new RAGFS filesystem with full RAG capabilities.
    pub fn with_rag(
        source: PathBuf,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        runtime: Handle,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
    ) -> Self {
        let query_executor = Arc::new(QueryExecutor::new(
            store.clone(),
            embedder.clone(),
            10,    // default limit
            false, // hybrid search
        ));

        let safety_manager = Arc::new(SafetyManager::new(&source, None));
        let ops_manager = Arc::new(OpsManager::with_safety(
            source.clone(),
            Some(store.clone()),
            reindex_sender.clone(),
            safety_manager.clone(),
        ));
        let semantic_manager = Arc::new(SemanticManager::new(
            source.clone(),
            Some(store.clone()),
            Some(embedder.clone()),
            None,
        ));

        Self {
            source,
            inodes: Arc::new(RwLock::new(InodeTable::new())),
            store: Some(store),
            query_executor: Some(query_executor),
            runtime,
            content_cache: Arc::new(RwLock::new(HashMap::new())),
            reindex_sender,
            ops_manager,
            safety_manager,
            semantic_manager,
        }
    }

    /// Get the source directory.
    #[must_use]
    pub fn source(&self) -> &PathBuf {
        &self.source
    }

    /// Convert a real path to a FUSE inode.
    fn real_path_to_attr(&self, path: &PathBuf, ino: u64) -> Option<FileAttr> {
        let metadata = fs::metadata(path).ok()?;
        let kind = if metadata.is_dir() {
            FileType::Directory
        } else if metadata.is_file() {
            FileType::RegularFile
        } else if metadata.file_type().is_symlink() {
            FileType::Symlink
        } else {
            return None;
        };

        let atime = metadata.accessed().unwrap_or(SystemTime::UNIX_EPOCH);
        let mtime = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let ctime = UNIX_EPOCH + Duration::from_secs(metadata.ctime() as u64);

        Some(FileAttr {
            ino,
            size: metadata.len(),
            blocks: metadata.len().div_ceil(BLOCK_SIZE),
            atime,
            mtime,
            ctime,
            crtime: ctime,
            kind,
            perm: (metadata.mode() & 0o7777) as u16,
            nlink: metadata.nlink() as u32,
            uid: metadata.uid(),
            gid: metadata.gid(),
            rdev: metadata.rdev() as u32,
            blksize: BLOCK_SIZE as u32,
            flags: 0,
        })
    }

    #[allow(unsafe_code)]
    fn make_attr(&self, ino: u64, kind: FileType, size: u64) -> FileAttr {
        let now = SystemTime::now();
        // SAFETY: getuid() and getgid() are always safe to call
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        FileAttr {
            ino,
            size,
            blocks: size.div_ceil(BLOCK_SIZE),
            atime: now,
            mtime: now,
            ctime: now,
            crtime: now,
            kind,
            perm: if kind == FileType::Directory {
                0o755
            } else {
                0o644
            },
            nlink: if kind == FileType::Directory { 2 } else { 1 },
            uid,
            gid,
            rdev: 0,
            blksize: BLOCK_SIZE as u32,
            flags: 0,
        }
    }

    /// Get index status as JSON.
    fn get_index_status(&self) -> Vec<u8> {
        if let Some(ref store) = self.store {
            let store = store.clone();
            let result = self.runtime.block_on(async { store.stats().await });

            match result {
                Ok(stats) => {
                    let json = serde_json::json!({
                        "status": "indexed",
                        "total_files": stats.total_files,
                        "total_chunks": stats.total_chunks,
                        "index_size_bytes": stats.index_size_bytes,
                        "last_updated": stats.last_updated.map(|t| t.to_rfc3339()),
                    });
                    serde_json::to_string_pretty(&json)
                        .unwrap_or_default()
                        .into_bytes()
                }
                Err(e) => {
                    let json = serde_json::json!({
                        "status": "error",
                        "error": e.to_string(),
                    });
                    serde_json::to_string_pretty(&json)
                        .unwrap_or_default()
                        .into_bytes()
                }
            }
        } else {
            let json = serde_json::json!({
                "status": "not_initialized",
                "message": "No store configured",
            });
            serde_json::to_string_pretty(&json)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Execute a query and return results as JSON.
    fn execute_query(&self, query: &str) -> Vec<u8> {
        if let Some(ref executor) = self.query_executor {
            let executor = executor.clone();
            let query_str = query.to_string();
            let query_for_result = query_str.clone();
            let result = self
                .runtime
                .block_on(async move { executor.execute(&query_str).await });

            match result {
                Ok(results) => {
                    let json_results: Vec<_> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "file": r.file_path.to_string_lossy(),
                                "score": r.score,
                                "content": truncate(&r.content, 500),
                                "byte_range": [r.byte_range.start, r.byte_range.end],
                                "line_range": r.line_range.as_ref().map(|lr| [lr.start, lr.end]),
                            })
                        })
                        .collect();

                    let json = serde_json::json!({
                        "query": query_for_result,
                        "results": json_results,
                    });
                    serde_json::to_string_pretty(&json)
                        .unwrap_or_default()
                        .into_bytes()
                }
                Err(e) => {
                    let json = serde_json::json!({
                        "query": query_for_result,
                        "error": e.to_string(),
                    });
                    serde_json::to_string_pretty(&json)
                        .unwrap_or_default()
                        .into_bytes()
                }
            }
        } else {
            let json = serde_json::json!({
                "error": "Query executor not configured",
            });
            serde_json::to_string_pretty(&json)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Get configuration as JSON.
    fn get_config(&self) -> Vec<u8> {
        let json = serde_json::json!({
            "source": self.source.to_string_lossy(),
            "store_configured": self.store.is_some(),
            "query_executor_configured": self.query_executor.is_some(),
        });
        serde_json::to_string_pretty(&json)
            .unwrap_or_default()
            .into_bytes()
    }

    /// Resolve a parent inode to a real path.
    /// Returns None if the parent doesn't exist or is not a directory.
    fn resolve_parent_path(&self, parent: u64) -> Option<PathBuf> {
        if parent == ROOT_INO {
            return Some(self.source.clone());
        }

        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(parent)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            Some(path.clone())
        } else {
            None
        }
    }

    /// Get help content for the virtual control directory.
    fn get_help_content(&self) -> Vec<u8> {
        r#"RAGFS Virtual Control Directory
================================

The .ragfs directory provides a virtual interface to RAGFS functionality.

Available paths:

  .index          Read to get index statistics (JSON)
                  Shows file count, chunk count, and last update time.

  .config         Read to get current configuration (JSON)
                  Shows source directory and component status.

  .reindex        Write a file path to trigger reindexing.
                  Example: echo "src/main.rs" > .ragfs/.reindex

  .query/<q>      Read to execute a semantic search query.
                  The filename is the query string.
                  Returns JSON with matching results.
                  Example: cat .ragfs/.query/authentication

  .search/        Directory for search results (symlinks).
                  Access .search/<query>/ to get symlinks to matching files.

  .similar/       Directory for finding similar files.
                  Access .similar/<path>/ to get symlinks to similar files.

  .help           This help file.

Examples:

  # Check index status
  cat .ragfs/.index

  # Search for files about authentication
  cat ".ragfs/.query/how to authenticate users"

  # Trigger reindex of a specific file
  echo "src/lib.rs" > .ragfs/.reindex

  # View configuration
  cat .ragfs/.config
"#
        .as_bytes()
        .to_vec()
    }
}

impl Filesystem for RagFs {
    fn init(
        &mut self,
        _req: &Request<'_>,
        _config: &mut fuser::KernelConfig,
    ) -> Result<(), libc::c_int> {
        debug!("FUSE init for source: {:?}", self.source);
        Ok(())
    }

    fn destroy(&mut self) {
        debug!("FUSE destroy");
    }

    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let name_str = name.to_string_lossy();
        debug!("lookup: parent={}, name={}", parent, name_str);

        // Handle root directory lookups
        if parent == ROOT_INO {
            if name_str == ".ragfs" {
                let attr = self.make_attr(RAGFS_DIR_INO, FileType::Directory, 0);
                reply.entry(&TTL, &attr, 0);
                return;
            }

            // Try to find real file/directory in source
            let real_path = self.source.join(&*name_str);
            if real_path.exists() {
                let metadata = if let Ok(m) = fs::metadata(&real_path) {
                    m
                } else {
                    reply.error(ENOENT);
                    return;
                };

                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_real(real_path.clone(), metadata.ino());
                drop(inodes);

                if let Some(attr) = self.real_path_to_attr(&real_path, ino) {
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
            }
        }

        // Handle .ragfs directory lookups
        if parent == RAGFS_DIR_INO {
            match name_str.as_ref() {
                ".query" => {
                    let attr = self.make_attr(QUERY_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".search" => {
                    let attr = self.make_attr(SEARCH_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".index" => {
                    let content = self.get_index_status();
                    let attr =
                        self.make_attr(INDEX_FILE_INO, FileType::RegularFile, content.len() as u64);
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(INDEX_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".config" => {
                    let content = self.get_config();
                    let attr = self.make_attr(
                        CONFIG_FILE_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(CONFIG_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".reindex" => {
                    let attr = self.make_attr(REINDEX_FILE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".similar" => {
                    let attr = self.make_attr(SIMILAR_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".help" => {
                    let content = self.get_help_content();
                    let attr =
                        self.make_attr(HELP_FILE_INO, FileType::RegularFile, content.len() as u64);
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(HELP_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".ops" => {
                    let attr = self.make_attr(OPS_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".safety" => {
                    let attr = self.make_attr(SAFETY_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".semantic" => {
                    let attr = self.make_attr(SEMANTIC_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                _ => {}
            }
        }

        // Handle .semantic directory lookups
        if parent == SEMANTIC_DIR_INO {
            match name_str.as_ref() {
                ".organize" => {
                    let attr = self.make_attr(ORGANIZE_FILE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".similar" => {
                    let content = self.runtime.block_on(self.semantic_manager.get_similar_json());
                    let attr = self.make_attr(
                        SIMILAR_OPS_FILE_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(SIMILAR_OPS_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".cleanup" => {
                    let content = self.runtime.block_on(self.semantic_manager.get_cleanup_json());
                    let attr = self.make_attr(
                        CLEANUP_FILE_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(CLEANUP_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".dedupe" => {
                    let content = self.runtime.block_on(self.semantic_manager.get_dedupe_json());
                    let attr = self.make_attr(
                        DEDUPE_FILE_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(DEDUPE_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".pending" => {
                    let attr = self.make_attr(PENDING_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".approve" => {
                    let attr = self.make_attr(APPROVE_FILE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".reject" => {
                    let attr = self.make_attr(REJECT_FILE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                _ => {}
            }
        }

        // Handle .pending directory lookups (semantic plans)
        if parent == PENDING_DIR_INO {
            // Lookup a specific pending plan by ID
            let plan_ids = self
                .runtime
                .block_on(self.semantic_manager.get_pending_plan_ids());
            if plan_ids.contains(&name_str.to_string()) {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_plan_json(&name_str));
                // Use dynamic inode for pending plans
                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_query_result(PENDING_DIR_INO, name_str.to_string());
                let attr = self.make_attr(ino, FileType::RegularFile, content.len() as u64);
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(ino, content);
                reply.entry(&TTL, &attr, 0);
                return;
            }
        }

        // Handle .safety directory lookups
        if parent == SAFETY_DIR_INO {
            match name_str.as_ref() {
                ".trash" => {
                    let attr = self.make_attr(TRASH_DIR_INO, FileType::Directory, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".history" => {
                    let content = self.safety_manager.get_history_json(Some(100));
                    let attr = self.make_attr(
                        HISTORY_FILE_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(HISTORY_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".undo" => {
                    let attr = self.make_attr(UNDO_FILE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                _ => {}
            }
        }

        // Handle .ops directory lookups
        if parent == OPS_DIR_INO {
            match name_str.as_ref() {
                ".create" => {
                    let attr = self.make_attr(OPS_CREATE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".delete" => {
                    let attr = self.make_attr(OPS_DELETE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".move" => {
                    let attr = self.make_attr(OPS_MOVE_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".batch" => {
                    let attr = self.make_attr(OPS_BATCH_INO, FileType::RegularFile, 0);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".result" => {
                    let content = self.runtime.block_on(self.ops_manager.get_last_result());
                    let attr = self.make_attr(
                        OPS_RESULT_INO,
                        FileType::RegularFile,
                        content.len() as u64,
                    );
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(OPS_RESULT_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                _ => {}
            }
        }

        // Handle .query directory lookups (dynamic query files)
        if parent == QUERY_DIR_INO {
            let query = name_str.to_string();
            let content = self.execute_query(&query);

            let mut inodes = self.runtime.block_on(self.inodes.write());
            let ino = inodes.get_or_create_query_result(QUERY_DIR_INO, query);
            drop(inodes);

            let attr = self.make_attr(ino, FileType::RegularFile, content.len() as u64);
            let mut cache = self.runtime.block_on(self.content_cache.write());
            cache.insert(ino, content);

            reply.entry(&TTL, &attr, 0);
            return;
        }

        // Handle lookups in real directories
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(parent)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            let real_path = path.join(&*name_str);
            if real_path.exists() {
                drop(inodes);
                let metadata = if let Ok(m) = fs::metadata(&real_path) {
                    m
                } else {
                    reply.error(ENOENT);
                    return;
                };

                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_real(real_path.clone(), metadata.ino());
                drop(inodes);

                if let Some(attr) = self.real_path_to_attr(&real_path, ino) {
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
            }
        }

        reply.error(ENOENT);
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        debug!("getattr: ino={}", ino);

        match ino {
            ROOT_INO => {
                let attr = self.make_attr(ROOT_INO, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            RAGFS_DIR_INO => {
                let attr = self.make_attr(RAGFS_DIR_INO, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            QUERY_DIR_INO | SEARCH_DIR_INO | SIMILAR_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            INDEX_FILE_INO => {
                let content = self.get_index_status();
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(INDEX_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            CONFIG_FILE_INO => {
                let content = self.get_config();
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(CONFIG_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            REINDEX_FILE_INO => {
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            HELP_FILE_INO => {
                let content = self.get_help_content();
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(HELP_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            // .ops/ directory and files
            OPS_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
                // Write-only files have size 0
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            OPS_RESULT_INO => {
                let content = self.runtime.block_on(self.ops_manager.get_last_result());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(OPS_RESULT_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            // .safety/ directory and files
            SAFETY_DIR_INO | TRASH_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            HISTORY_FILE_INO => {
                let content = self.safety_manager.get_history_json(Some(100));
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(HISTORY_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            UNDO_FILE_INO => {
                // Write-only file
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            // .semantic/ directory and files
            SEMANTIC_DIR_INO | PENDING_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            SIMILAR_OPS_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_similar_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(SIMILAR_OPS_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            CLEANUP_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_cleanup_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(CLEANUP_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            DEDUPE_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_dedupe_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(DEDUPE_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            ORGANIZE_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                // Write-only files
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            _ => {
                // Check if it's a cached query result
                let cache = self.runtime.block_on(self.content_cache.read());
                if let Some(content) = cache.get(&ino) {
                    let attr = self.make_attr(ino, FileType::RegularFile, content.len() as u64);
                    reply.attr(&TTL, &attr);
                    return;
                }
                drop(cache);

                // Check if it's a real file
                let inodes = self.runtime.block_on(self.inodes.read());
                if let Some(entry) = inodes.get(ino)
                    && let InodeKind::Real { path, .. } = &entry.kind
                    && let Some(attr) = self.real_path_to_attr(path, ino)
                {
                    reply.attr(&TTL, &attr);
                    return;
                }
                reply.error(ENOENT);
            }
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        debug!("read: ino={}, offset={}, size={}", ino, offset, size);

        // Check content cache first (for virtual files)
        let cache = self.runtime.block_on(self.content_cache.read());
        if let Some(content) = cache.get(&ino) {
            let offset = offset as usize;
            let size = size as usize;
            if offset >= content.len() {
                reply.data(&[]);
            } else {
                let end = (offset + size).min(content.len());
                reply.data(&content[offset..end]);
            }
            return;
        }
        drop(cache);

        // Handle virtual files by inode
        match ino {
            INDEX_FILE_INO => {
                let content = self.get_index_status();
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            CONFIG_FILE_INO => {
                let content = self.get_config();
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            REINDEX_FILE_INO => {
                reply.data(&[]);
                return;
            }
            HELP_FILE_INO => {
                let content = self.get_help_content();
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            OPS_RESULT_INO => {
                let content = self.runtime.block_on(self.ops_manager.get_last_result());
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            // Write-only .ops/ files return empty
            OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
                reply.data(&[]);
                return;
            }
            HISTORY_FILE_INO => {
                let content = self.safety_manager.get_history_json(Some(100));
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            // Write-only .safety/ files return empty
            UNDO_FILE_INO => {
                reply.data(&[]);
                return;
            }
            // .semantic/ read-only files
            SIMILAR_OPS_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_similar_json());
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            CLEANUP_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_cleanup_json());
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            DEDUPE_FILE_INO => {
                let content = self.runtime.block_on(self.semantic_manager.get_dedupe_json());
                let offset = offset as usize;
                let size = size as usize;
                if offset >= content.len() {
                    reply.data(&[]);
                } else {
                    let end = (offset + size).min(content.len());
                    reply.data(&content[offset..end]);
                }
                return;
            }
            // Write-only .semantic/ files return empty
            ORGANIZE_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                reply.data(&[]);
                return;
            }
            _ => {}
        }

        // Try to read real file
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(ino)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            let path = path.clone();
            drop(inodes);

            match fs::read(&path) {
                Ok(content) => {
                    let offset = offset as usize;
                    let size = size as usize;
                    if offset >= content.len() {
                        reply.data(&[]);
                    } else {
                        let end = (offset + size).min(content.len());
                        reply.data(&content[offset..end]);
                    }
                    return;
                }
                Err(e) => {
                    warn!("Failed to read file {:?}: {}", path, e);
                    reply.error(EIO);
                    return;
                }
            }
        }

        reply.error(ENOENT);
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        debug!("readdir: ino={}, offset={}", ino, offset);

        match ino {
            ROOT_INO => {
                let mut entries = vec![
                    (ROOT_INO, FileType::Directory, ".".to_string()),
                    (ROOT_INO, FileType::Directory, "..".to_string()),
                    (RAGFS_DIR_INO, FileType::Directory, ".ragfs".to_string()),
                ];

                // Add real files/directories from source
                if let Ok(read_dir) = fs::read_dir(&self.source) {
                    for entry in read_dir.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with('.') {
                            continue; // Skip hidden files
                        }
                        let file_type = if entry.path().is_dir() {
                            FileType::Directory
                        } else {
                            FileType::RegularFile
                        };

                        let metadata = match entry.metadata() {
                            Ok(m) => m,
                            Err(_) => continue,
                        };

                        let mut inodes = self.runtime.block_on(self.inodes.write());
                        let entry_ino = inodes.get_or_create_real(entry.path(), metadata.ino());
                        entries.push((entry_ino, file_type, name));
                    }
                }

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            RAGFS_DIR_INO => {
                let entries = [
                    (RAGFS_DIR_INO, FileType::Directory, "."),
                    (ROOT_INO, FileType::Directory, ".."),
                    (QUERY_DIR_INO, FileType::Directory, ".query"),
                    (SEARCH_DIR_INO, FileType::Directory, ".search"),
                    (INDEX_FILE_INO, FileType::RegularFile, ".index"),
                    (CONFIG_FILE_INO, FileType::RegularFile, ".config"),
                    (REINDEX_FILE_INO, FileType::RegularFile, ".reindex"),
                    (HELP_FILE_INO, FileType::RegularFile, ".help"),
                    (SIMILAR_DIR_INO, FileType::Directory, ".similar"),
                    (OPS_DIR_INO, FileType::Directory, ".ops"),
                    (SAFETY_DIR_INO, FileType::Directory, ".safety"),
                    (SEMANTIC_DIR_INO, FileType::Directory, ".semantic"),
                ];

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            SAFETY_DIR_INO => {
                let entries = [
                    (SAFETY_DIR_INO, FileType::Directory, "."),
                    (RAGFS_DIR_INO, FileType::Directory, ".."),
                    (TRASH_DIR_INO, FileType::Directory, ".trash"),
                    (HISTORY_FILE_INO, FileType::RegularFile, ".history"),
                    (UNDO_FILE_INO, FileType::RegularFile, ".undo"),
                ];

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            TRASH_DIR_INO => {
                // List trash entries dynamically
                let mut entries: Vec<(u64, FileType, String)> = vec![
                    (TRASH_DIR_INO, FileType::Directory, ".".to_string()),
                    (SAFETY_DIR_INO, FileType::Directory, "..".to_string()),
                ];

                // Add trash entries from SafetyManager
                let trash = self.runtime.block_on(self.safety_manager.list_trash());
                for entry in trash {
                    // Use a dynamic inode (starting from a high number to avoid conflicts)
                    let entry_ino = FIRST_REAL_INO + 500_000 + (entry.id.as_u128() % 100_000) as u64;
                    entries.push((entry_ino, FileType::RegularFile, entry.id.to_string()));
                }

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            OPS_DIR_INO => {
                let entries = [
                    (OPS_DIR_INO, FileType::Directory, "."),
                    (RAGFS_DIR_INO, FileType::Directory, ".."),
                    (OPS_CREATE_INO, FileType::RegularFile, ".create"),
                    (OPS_DELETE_INO, FileType::RegularFile, ".delete"),
                    (OPS_MOVE_INO, FileType::RegularFile, ".move"),
                    (OPS_BATCH_INO, FileType::RegularFile, ".batch"),
                    (OPS_RESULT_INO, FileType::RegularFile, ".result"),
                ];

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            SEMANTIC_DIR_INO => {
                let entries = [
                    (SEMANTIC_DIR_INO, FileType::Directory, "."),
                    (RAGFS_DIR_INO, FileType::Directory, ".."),
                    (ORGANIZE_FILE_INO, FileType::RegularFile, ".organize"),
                    (SIMILAR_OPS_FILE_INO, FileType::RegularFile, ".similar"),
                    (CLEANUP_FILE_INO, FileType::RegularFile, ".cleanup"),
                    (DEDUPE_FILE_INO, FileType::RegularFile, ".dedupe"),
                    (PENDING_DIR_INO, FileType::Directory, ".pending"),
                    (APPROVE_FILE_INO, FileType::RegularFile, ".approve"),
                    (REJECT_FILE_INO, FileType::RegularFile, ".reject"),
                ];

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            PENDING_DIR_INO => {
                // List pending plans dynamically
                let mut entries: Vec<(u64, FileType, String)> = vec![
                    (PENDING_DIR_INO, FileType::Directory, ".".to_string()),
                    (SEMANTIC_DIR_INO, FileType::Directory, "..".to_string()),
                ];

                // Add pending plan entries from SemanticManager
                let plan_ids = self
                    .runtime
                    .block_on(self.semantic_manager.get_pending_plan_ids());
                for plan_id in plan_ids {
                    // Use a dynamic inode
                    let mut inodes = self.runtime.block_on(self.inodes.write());
                    let entry_ino =
                        inodes.get_or_create_query_result(PENDING_DIR_INO, plan_id.clone());
                    entries.push((entry_ino, FileType::RegularFile, plan_id));
                }

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            QUERY_DIR_INO | SEARCH_DIR_INO | SIMILAR_DIR_INO => {
                // These directories are empty - files are created dynamically on lookup
                let entries = [
                    (ino, FileType::Directory, "."),
                    (RAGFS_DIR_INO, FileType::Directory, ".."),
                ];

                for (i, (entry_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize)
                {
                    if reply.add(*entry_ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            _ => {
                // Try to read real directory
                let inodes = self.runtime.block_on(self.inodes.read());
                if let Some(entry) = inodes.get(ino)
                    && let InodeKind::Real { path, .. } = &entry.kind
                {
                    let path = path.clone();
                    let parent_ino = entry.parent;
                    drop(inodes);

                    if path.is_dir() {
                        let mut entries = vec![
                            (ino, FileType::Directory, ".".to_string()),
                            (parent_ino, FileType::Directory, "..".to_string()),
                        ];

                        if let Ok(read_dir) = fs::read_dir(&path) {
                            for dir_entry in read_dir.flatten() {
                                let name = dir_entry.file_name().to_string_lossy().to_string();
                                if name.starts_with('.') {
                                    continue;
                                }
                                let file_type = if dir_entry.path().is_dir() {
                                    FileType::Directory
                                } else {
                                    FileType::RegularFile
                                };

                                let metadata = match dir_entry.metadata() {
                                    Ok(m) => m,
                                    Err(_) => continue,
                                };

                                let mut inodes = self.runtime.block_on(self.inodes.write());
                                let entry_ino =
                                    inodes.get_or_create_real(dir_entry.path(), metadata.ino());
                                entries.push((entry_ino, file_type, name));
                            }
                        }

                        for (i, (entry_ino, kind, name)) in
                            entries.iter().enumerate().skip(offset as usize)
                        {
                            if reply.add(*entry_ino, (i + 1) as i64, *kind, name) {
                                break;
                            }
                        }
                        reply.ok();
                        return;
                    }
                }
                reply.error(ENOENT);
            }
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        debug!("open: ino={}, flags={}", ino, flags);
        // Allow opening any file for now
        reply.opened(0, 0);
    }

    fn opendir(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        debug!("opendir: ino={}, flags={}", ino, flags);
        reply.opened(0, 0);
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        debug!("write: ino={}, len={}", ino, data.len());

        // Handle .reindex writes
        if ino == REINDEX_FILE_INO {
            let path_str = String::from_utf8_lossy(data).trim().to_string();

            if path_str.is_empty() {
                debug!("Empty reindex request, ignoring");
                reply.written(data.len() as u32);
                return;
            }

            let path = PathBuf::from(&path_str);

            // Convert relative paths to absolute paths relative to source
            let absolute_path = if path.is_absolute() {
                path
            } else {
                self.source.join(&path)
            };

            info!("Reindex requested for: {:?}", absolute_path);

            // Send reindex request if sender is configured
            if let Some(ref sender) = self.reindex_sender {
                let sender = sender.clone();
                let path_to_send = absolute_path.clone();

                // Use runtime to send asynchronously
                self.runtime.spawn(async move {
                    if let Err(e) = sender.send(path_to_send).await {
                        warn!("Failed to send reindex request: {}", e);
                    }
                });

                debug!("Reindex request sent for: {:?}", absolute_path);
            } else {
                warn!("Reindex requested but no sender configured");
            }

            reply.written(data.len() as u32);
            return;
        }

        // Handle .ops/ virtual file writes
        if ino == OPS_CREATE_INO {
            let data_str = String::from_utf8_lossy(data).to_string();
            let ops_manager = self.ops_manager.clone();
            self.runtime.block_on(async move {
                ops_manager.parse_and_create(&data_str).await;
            });
            reply.written(data.len() as u32);
            return;
        }

        if ino == OPS_DELETE_INO {
            let data_str = String::from_utf8_lossy(data).to_string();
            let ops_manager = self.ops_manager.clone();
            self.runtime.block_on(async move {
                ops_manager.parse_and_delete(&data_str).await;
            });
            reply.written(data.len() as u32);
            return;
        }

        if ino == OPS_MOVE_INO {
            let data_str = String::from_utf8_lossy(data).to_string();
            let ops_manager = self.ops_manager.clone();
            self.runtime.block_on(async move {
                ops_manager.parse_and_move(&data_str).await;
            });
            reply.written(data.len() as u32);
            return;
        }

        if ino == OPS_BATCH_INO {
            let data_str = String::from_utf8_lossy(data).to_string();
            let ops_manager = self.ops_manager.clone();
            self.runtime.block_on(async move {
                ops_manager.parse_and_batch(&data_str).await;
            });
            reply.written(data.len() as u32);
            return;
        }

        // Handle .safety/.undo writes
        if ino == UNDO_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            if let Ok(operation_id) = uuid::Uuid::parse_str(&data_str) {
                let safety_manager = self.safety_manager.clone();
                let result = self.runtime.block_on(async move {
                    safety_manager.undo(operation_id).await
                });
                match result {
                    Ok(msg) => info!("Undo successful: {}", msg),
                    Err(e) => warn!("Undo failed: {}", e),
                }
            } else {
                warn!("Invalid operation ID for undo: {}", data_str);
            }
            reply.written(data.len() as u32);
            return;
        }

        // Handle .semantic/.organize writes
        if ino == ORGANIZE_FILE_INO {
            let data_str = String::from_utf8_lossy(data).to_string();
            let semantic_manager = self.semantic_manager.clone();
            let result = self.runtime.block_on(async move {
                match serde_json::from_str::<crate::semantic::OrganizeRequest>(&data_str) {
                    Ok(request) => semantic_manager.create_organize_plan(request).await,
                    Err(e) => Err(format!("Invalid OrganizeRequest JSON: {e}")),
                }
            });
            match result {
                Ok(plan) => info!("Created organization plan: {}", plan.id),
                Err(e) => warn!("Failed to create organization plan: {}", e),
            }
            reply.written(data.len() as u32);
            return;
        }

        // Handle .semantic/.similar writes
        if ino == SIMILAR_OPS_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            let path = PathBuf::from(&data_str);
            let semantic_manager = self.semantic_manager.clone();
            let result = self.runtime.block_on(async move {
                semantic_manager.find_similar(&path).await
            });
            match result {
                Ok(r) => info!("Found {} similar files to {}", r.similar.len(), data_str),
                Err(e) => warn!("Failed to find similar files: {}", e),
            }
            reply.written(data.len() as u32);
            return;
        }

        // Handle .semantic/.approve writes
        if ino == APPROVE_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            if let Ok(plan_id) = uuid::Uuid::parse_str(&data_str) {
                let semantic_manager = self.semantic_manager.clone();
                let result = self.runtime.block_on(async move {
                    semantic_manager.approve_plan(plan_id).await
                });
                match result {
                    Ok(plan) => info!("Approved plan: {}", plan.id),
                    Err(e) => warn!("Failed to approve plan: {}", e),
                }
            } else {
                warn!("Invalid plan ID for approve: {}", data_str);
            }
            reply.written(data.len() as u32);
            return;
        }

        // Handle .semantic/.reject writes
        if ino == REJECT_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            if let Ok(plan_id) = uuid::Uuid::parse_str(&data_str) {
                let semantic_manager = self.semantic_manager.clone();
                let result = self.runtime.block_on(async move {
                    semantic_manager.reject_plan(plan_id).await
                });
                match result {
                    Ok(plan) => info!("Rejected plan: {}", plan.id),
                    Err(e) => warn!("Failed to reject plan: {}", e),
                }
            } else {
                warn!("Invalid plan ID for reject: {}", data_str);
            }
            reply.written(data.len() as u32);
            return;
        }

        // Real file writes (passthrough)
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(ino)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            let path = path.clone();
            drop(inodes);

            match fs::write(&path, data) {
                Ok(()) => {
                    reply.written(data.len() as u32);
                    return;
                }
                Err(e) => {
                    warn!("Failed to write file {:?}: {}", path, e);
                    reply.error(EIO);
                    return;
                }
            }
        }

        reply.error(ENOSYS);
    }

    fn forget(&mut self, _req: &Request<'_>, ino: u64, nlookup: u64) {
        debug!("forget: ino={}, nlookup={}", ino, nlookup);
        let mut inodes = self.runtime.block_on(self.inodes.write());
        inodes.forget(ino, nlookup);
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let name_str = name.to_string_lossy();
        debug!("create: parent={}, name={}, mode={:o}", parent, name_str, mode);

        // Prevent creating files in virtual directories
        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        // Resolve parent to path
        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let new_path = parent_path.join(&*name_str);

        // Check if file already exists
        if new_path.exists() {
            reply.error(EEXIST);
            return;
        }

        // Create the file
        match fs::File::create(&new_path) {
            Ok(_) => {
                // Get metadata for the new file
                let metadata = match fs::metadata(&new_path) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for new file {:?}: {}", new_path, e);
                        reply.error(EIO);
                        return;
                    }
                };

                // Create inode entry
                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_real(new_path.clone(), metadata.ino());
                drop(inodes);

                // Trigger reindex for the new file
                if let Some(ref sender) = self.reindex_sender {
                    let sender = sender.clone();
                    let path_to_send = new_path.clone();
                    self.runtime.spawn(async move {
                        if let Err(e) = sender.send(path_to_send).await {
                            warn!("Failed to send reindex request: {}", e);
                        }
                    });
                }

                if let Some(attr) = self.real_path_to_attr(&new_path, ino) {
                    reply.created(&TTL, &attr, 0, 0, flags as u32);
                } else {
                    reply.error(EIO);
                }
            }
            Err(e) => {
                warn!("Failed to create file {:?}: {}", new_path, e);
                reply.error(EIO);
            }
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy();
        debug!("unlink: parent={}, name={}", parent, name_str);

        // Prevent unlinking from virtual directories
        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        // Resolve parent to path
        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let file_path = parent_path.join(&*name_str);

        // Check if path exists and is a file
        if !file_path.exists() {
            reply.error(ENOENT);
            return;
        }

        if file_path.is_dir() {
            reply.error(EISDIR);
            return;
        }

        // Delete from vector store first
        if let Some(ref store) = self.store {
            let store = store.clone();
            let path_for_delete = file_path.clone();
            let result = self.runtime.block_on(async move {
                store.delete_by_file_path(&path_for_delete).await
            });
            if let Err(e) = result {
                warn!("Failed to delete from store {:?}: {}", file_path, e);
                // Continue with file deletion anyway
            }
        }

        // Delete the file
        match fs::remove_file(&file_path) {
            Ok(()) => {
                // Remove inode entry
                let mut inodes = self.runtime.block_on(self.inodes.write());
                if let Some(ino) = inodes.get_by_path(&file_path) {
                    inodes.remove(ino);
                }
                info!("Deleted file: {:?}", file_path);
                reply.ok();
            }
            Err(e) => {
                warn!("Failed to delete file {:?}: {}", file_path, e);
                reply.error(EIO);
            }
        }
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let name_str = name.to_string_lossy();
        debug!("mkdir: parent={}, name={}, mode={:o}", parent, name_str, mode);

        // Prevent creating directories in virtual directories
        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        // Resolve parent to path
        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let new_path = parent_path.join(&*name_str);

        // Check if path already exists
        if new_path.exists() {
            reply.error(EEXIST);
            return;
        }

        // Create the directory
        match fs::create_dir(&new_path) {
            Ok(()) => {
                // Get metadata
                let metadata = match fs::metadata(&new_path) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for new dir {:?}: {}", new_path, e);
                        reply.error(EIO);
                        return;
                    }
                };

                // Create inode entry
                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_real(new_path.clone(), metadata.ino());
                drop(inodes);

                if let Some(attr) = self.real_path_to_attr(&new_path, ino) {
                    info!("Created directory: {:?}", new_path);
                    reply.entry(&TTL, &attr, 0);
                } else {
                    reply.error(EIO);
                }
            }
            Err(e) => {
                warn!("Failed to create directory {:?}: {}", new_path, e);
                reply.error(EIO);
            }
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy();
        debug!("rmdir: parent={}, name={}", parent, name_str);

        // Prevent removing directories from virtual areas
        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        // Resolve parent to path
        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let dir_path = parent_path.join(&*name_str);

        // Check if path exists and is a directory
        if !dir_path.exists() {
            reply.error(ENOENT);
            return;
        }

        if !dir_path.is_dir() {
            reply.error(ENOTDIR);
            return;
        }

        // Remove the directory (will fail if not empty)
        match fs::remove_dir(&dir_path) {
            Ok(()) => {
                // Remove inode entry
                let mut inodes = self.runtime.block_on(self.inodes.write());
                if let Some(ino) = inodes.get_by_path(&dir_path) {
                    inodes.remove(ino);
                }
                info!("Removed directory: {:?}", dir_path);
                reply.ok();
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::DirectoryNotEmpty
                    || e.raw_os_error() == Some(libc::ENOTEMPTY)
                {
                    reply.error(ENOTEMPTY);
                } else {
                    warn!("Failed to remove directory {:?}: {}", dir_path, e);
                    reply.error(EIO);
                }
            }
        }
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let name_str = name.to_string_lossy();
        let newname_str = newname.to_string_lossy();
        debug!(
            "rename: parent={}, name={}, newparent={}, newname={}",
            parent, name_str, newparent, newname_str
        );

        // Prevent renaming from/to virtual directories
        if (parent < FIRST_REAL_INO && parent != ROOT_INO)
            || (newparent < FIRST_REAL_INO && newparent != ROOT_INO)
        {
            reply.error(EPERM);
            return;
        }

        // Resolve source path
        let Some(src_parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        // Resolve destination path
        let Some(dst_parent_path) = self.resolve_parent_path(newparent) else {
            reply.error(ENOENT);
            return;
        };

        let src_path = src_parent_path.join(&*name_str);
        let dst_path = dst_parent_path.join(&*newname_str);

        // Check source exists
        if !src_path.exists() {
            reply.error(ENOENT);
            return;
        }

        // Perform rename
        match fs::rename(&src_path, &dst_path) {
            Ok(()) => {
                // Update vector store path
                if let Some(ref store) = self.store {
                    let store = store.clone();
                    let src = src_path.clone();
                    let dst = dst_path.clone();
                    let result = self.runtime.block_on(async move {
                        store.update_file_path(&src, &dst).await
                    });
                    if let Err(e) = result {
                        warn!("Failed to update store path {:?} -> {:?}: {}", src_path, dst_path, e);
                    }
                }

                // Update inode table
                let mut inodes = self.runtime.block_on(self.inodes.write());
                if let Some(ino) = inodes.get_by_path(&src_path) {
                    inodes.update_path(ino, dst_path.clone());
                }
                drop(inodes);

                info!("Renamed: {:?} -> {:?}", src_path, dst_path);
                reply.ok();
            }
            Err(e) => {
                warn!("Failed to rename {:?} -> {:?}: {}", src_path, dst_path, e);
                reply.error(EIO);
            }
        }
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        debug!("setattr: ino={}, size={:?}", ino, size);

        // Don't allow setattr on virtual files
        if ino < FIRST_REAL_INO {
            reply.error(EPERM);
            return;
        }

        // Get the file path
        let inodes = self.runtime.block_on(self.inodes.read());
        let Some(entry) = inodes.get(ino) else {
            drop(inodes);
            reply.error(ENOENT);
            return;
        };
        let InodeKind::Real { path, .. } = &entry.kind else {
            drop(inodes);
            reply.error(EINVAL);
            return;
        };
        let path = path.clone();
        drop(inodes);

        // Handle truncate
        if let Some(new_size) = size {
            match fs::OpenOptions::new().write(true).open(&path) {
                Ok(file) => {
                    if let Err(e) = file.set_len(new_size) {
                        warn!("Failed to truncate {:?}: {}", path, e);
                        reply.error(EIO);
                        return;
                    }

                    // Trigger reindex after truncate
                    if let Some(ref sender) = self.reindex_sender {
                        let sender = sender.clone();
                        let path_to_send = path.clone();
                        self.runtime.spawn(async move {
                            if let Err(e) = sender.send(path_to_send).await {
                                warn!("Failed to send reindex request: {}", e);
                            }
                        });
                    }
                }
                Err(e) => {
                    warn!("Failed to open {:?} for truncate: {}", path, e);
                    reply.error(EIO);
                    return;
                }
            }
        }

        // Return updated attributes
        if let Some(attr) = self.real_path_to_attr(&path, ino) {
            reply.attr(&TTL, &attr);
        } else {
            reply.error(EIO);
        }
    }
}

/// Truncate a string to max length, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== truncate() Helper Function Tests ==========

    #[test]
    fn test_truncate_short_string() {
        let result = truncate("Hello", 10);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        let result = truncate("Hello", 5);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("Hello, World!", 8);
        assert_eq!(result, "Hello...");
    }

    #[test]
    fn test_truncate_removes_newlines() {
        let result = truncate("Hello\nWorld\nTest", 100);
        assert_eq!(result, "Hello World Test");
    }

    #[test]
    fn test_truncate_removes_carriage_returns() {
        // \n is replaced with space, \r is deleted
        let result = truncate("Hello\r\nWorld", 100);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_truncate_empty_string() {
        let result = truncate("", 10);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_very_short_max() {
        let result = truncate("Hello", 3);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_max_zero() {
        let result = truncate("Hello", 0);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_unicode() {
        let result = truncate("こんにちは世界", 100);
        assert_eq!(result, "こんにちは世界");
    }

    #[test]
    fn test_truncate_with_mixed_whitespace() {
        // \n\n -> "  ", \r\n\r\n -> "  " (two \n->space, two \r->deleted)
        let result = truncate("Line1\n\nLine2\r\n\r\nLine3", 100);
        assert_eq!(result, "Line1  Line2  Line3");
    }

    // ========== RagFs Construction Tests ==========

    #[tokio::test]
    async fn test_ragfs_new() {
        let source = PathBuf::from("/tmp/test");
        let fs = RagFs::new(source.clone());

        assert_eq!(fs.source(), &source);
        assert!(fs.store.is_none());
        assert!(fs.query_executor.is_none());
    }

    #[tokio::test]
    async fn test_ragfs_source_getter() {
        let source = PathBuf::from("/my/test/directory");
        let fs = RagFs::new(source.clone());

        assert_eq!(fs.source(), &source);
    }

    #[tokio::test]
    async fn test_ragfs_inode_table_initialized() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));

        let inodes = fs.inodes.read().await;
        // Virtual inodes should be initialized
        assert!(inodes.get(ROOT_INO).is_some());
        assert!(inodes.get(RAGFS_DIR_INO).is_some());
        assert!(inodes.get(QUERY_DIR_INO).is_some());
    }

    #[tokio::test]
    async fn test_ragfs_content_cache_empty() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));

        let cache = fs.content_cache.read().await;
        assert!(cache.is_empty());
    }

    // ========== get_config() Tests ==========

    #[tokio::test]
    async fn test_get_config_without_rag() {
        let fs = RagFs::new(PathBuf::from("/tmp/test-config"));
        let config = fs.get_config();

        let json: serde_json::Value = serde_json::from_slice(&config).expect("Valid JSON");

        assert_eq!(json["source"], "/tmp/test-config");
        assert_eq!(json["store_configured"], false);
        assert_eq!(json["query_executor_configured"], false);
    }

    #[tokio::test]
    async fn test_get_config_returns_valid_json() {
        let fs = RagFs::new(PathBuf::from("/test/path"));
        let config = fs.get_config();

        // Should be valid UTF-8
        let config_str = String::from_utf8(config).expect("Valid UTF-8");

        // Should be parseable JSON
        let json: serde_json::Value = serde_json::from_str(&config_str).expect("Valid JSON");

        // Should have expected fields
        assert!(json.get("source").is_some());
        assert!(json.get("store_configured").is_some());
        assert!(json.get("query_executor_configured").is_some());
    }

    // ========== get_index_status() Tests ==========

    #[tokio::test]
    async fn test_get_index_status_without_store() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));
        let status = fs.get_index_status();

        let json: serde_json::Value = serde_json::from_slice(&status).expect("Valid JSON");

        assert_eq!(json["status"], "not_initialized");
        assert_eq!(json["message"], "No store configured");
    }

    #[tokio::test]
    async fn test_get_index_status_returns_valid_json() {
        let fs = RagFs::new(PathBuf::from("/test/path"));
        let status = fs.get_index_status();

        // Should be valid UTF-8
        let status_str = String::from_utf8(status).expect("Valid UTF-8");

        // Should be parseable JSON
        let _json: serde_json::Value = serde_json::from_str(&status_str).expect("Valid JSON");
    }

    // ========== execute_query() Tests ==========

    #[tokio::test]
    async fn test_execute_query_without_executor() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));
        let result = fs.execute_query("test query");

        let json: serde_json::Value = serde_json::from_slice(&result).expect("Valid JSON");

        assert_eq!(json["error"], "Query executor not configured");
    }

    #[tokio::test]
    async fn test_execute_query_returns_valid_json() {
        let fs = RagFs::new(PathBuf::from("/test/path"));
        let result = fs.execute_query("any query");

        // Should be valid UTF-8
        let result_str = String::from_utf8(result).expect("Valid UTF-8");

        // Should be parseable JSON
        let _json: serde_json::Value = serde_json::from_str(&result_str).expect("Valid JSON");
    }

    // ========== Constants Tests ==========

    #[test]
    fn test_ttl_is_reasonable() {
        assert_eq!(TTL, Duration::from_secs(1));
    }

    #[test]
    fn test_block_size_is_standard() {
        assert_eq!(BLOCK_SIZE, 512);
    }

    // ========== make_attr() Tests ==========

    #[tokio::test]
    async fn test_make_attr_directory() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));
        let attr = fs.make_attr(100, fuser::FileType::Directory, 0);

        assert_eq!(attr.ino, 100);
        assert_eq!(attr.size, 0);
        assert_eq!(attr.kind, fuser::FileType::Directory);
        assert_eq!(attr.perm, 0o755);
        assert_eq!(attr.nlink, 2);
    }

    #[tokio::test]
    async fn test_make_attr_regular_file() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));
        let attr = fs.make_attr(200, fuser::FileType::RegularFile, 1024);

        assert_eq!(attr.ino, 200);
        assert_eq!(attr.size, 1024);
        assert_eq!(attr.kind, fuser::FileType::RegularFile);
        assert_eq!(attr.perm, 0o644);
        assert_eq!(attr.nlink, 1);
    }

    #[tokio::test]
    async fn test_make_attr_blocks_calculation() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));

        // Test exact block boundary
        let attr = fs.make_attr(1, fuser::FileType::RegularFile, 512);
        assert_eq!(attr.blocks, 1);

        // Test one byte over
        let attr = fs.make_attr(1, fuser::FileType::RegularFile, 513);
        assert_eq!(attr.blocks, 2);

        // Test empty file
        let attr = fs.make_attr(1, fuser::FileType::RegularFile, 0);
        assert_eq!(attr.blocks, 0);
    }

    #[tokio::test]
    async fn test_make_attr_has_current_uid_gid() {
        let fs = RagFs::new(PathBuf::from("/tmp/test"));
        let attr = fs.make_attr(1, fuser::FileType::RegularFile, 0);

        // Should have current user's uid/gid
        #[allow(unsafe_code)]
        let expected_uid = unsafe { libc::getuid() };
        #[allow(unsafe_code)]
        let expected_gid = unsafe { libc::getgid() };

        assert_eq!(attr.uid, expected_uid);
        assert_eq!(attr.gid, expected_gid);
    }
}
