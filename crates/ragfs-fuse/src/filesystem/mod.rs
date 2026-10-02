//! FUSE filesystem implementation.

use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
};
use ragfs_core::{Embedder, VectorStore};
use ragfs_query::QueryExecutor;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::runtime::Handle;
use tokio::sync::{RwLock, mpsc};
use tracing::debug;

use crate::inode::{
    APPROVE_FILE_INO, CLEANUP_FILE_INO, CONFIG_FILE_INO, DEDUPE_FILE_INO, HELP_FILE_INO,
    HISTORY_FILE_INO, INDEX_FILE_INO, InodeKind, InodeTable, OPS_BATCH_INO, OPS_CREATE_INO,
    OPS_DELETE_INO, OPS_DIR_INO, OPS_MOVE_INO, OPS_RESULT_INO, ORGANIZE_FILE_INO, PENDING_DIR_INO,
    QUERY_DIR_INO, RAGFS_DIR_INO, REINDEX_FILE_INO, REJECT_FILE_INO, ROOT_INO, SAFETY_DIR_INO,
    SEARCH_DIR_INO, SEMANTIC_DIR_INO, SIMILAR_DIR_INO, SIMILAR_OPS_FILE_INO, TRASH_DIR_INO,
    UNDO_FILE_INO,
};
use crate::ops::OpsManager;
use crate::safety::SafetyManager;
use crate::semantic::SemanticManager;

mod ops;
mod passthrough;
mod query;
mod safety;
mod semantic;

pub(crate) const TTL: Duration = Duration::from_secs(1);
pub(crate) const BLOCK_SIZE: u64 = 512;

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
        let semantic_manager = Arc::new(SemanticManager::with_ops(
            source.clone(),
            None,
            None,
            None,
            ops_manager.clone(),
        ));
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
    ///
    /// Hybrid search defaults to on (same as `[query].hybrid` in config.toml).
    pub fn with_rag(
        source: PathBuf,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        runtime: Handle,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
    ) -> Self {
        Self::with_rag_query(
            source,
            store,
            embedder,
            runtime,
            reindex_sender,
            10,
            true,
            100,
        )
    }

    /// Create a RAG-enabled filesystem with query settings from config / CLI.
    pub fn with_rag_query(
        source: PathBuf,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        runtime: Handle,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
        default_limit: usize,
        hybrid: bool,
        max_limit: usize,
    ) -> Self {
        let query_executor = Arc::new(
            QueryExecutor::new(store.clone(), embedder.clone(), default_limit, hybrid)
                .with_max_limit(max_limit),
        );

        let safety_manager = Arc::new(SafetyManager::new(&source, None));
        let ops_manager = Arc::new(OpsManager::with_safety(
            source.clone(),
            Some(store.clone()),
            reindex_sender.clone(),
            safety_manager.clone(),
        ));
        let semantic_manager = Arc::new(SemanticManager::with_ops(
            source.clone(),
            Some(store.clone()),
            Some(embedder.clone()),
            None,
            ops_manager.clone(),
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

    /// Resolve a `.reindex` path and reject anything that escapes the source root.
    fn jail_reindex_path(&self, path: &Path) -> Result<PathBuf, String> {
        ragfs_core::resolve_under_root(&self.source, path)
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
    ///
    /// This is mount wiring, not `~/.config/ragfs/config.toml`.
    fn get_config(&self) -> Vec<u8> {
        let json = serde_json::json!({
            "api_version": "0.2",
            "source": self.source.to_string_lossy(),
            "store_configured": self.store.is_some(),
            "query_executor_configured": self.query_executor.is_some(),
            "interfaces": {
                "query": true,
                "ops": true,
                "safety": true,
                "semantic": true
            },
            "note": "Mount wiring only. User TOML is not echoed here.",
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

The .ragfs directory is the agent control plane: search, file ops,
soft-delete, and propose/approve semantic plans.

Search and index
----------------
  .index          Read index statistics (JSON).
  .config         Read mount wiring (JSON). Includes api_version.
                  This is not ~/.config/ragfs/config.toml.
  .reindex        Write a path to queue reindex (relative paths join source).
                  Absolute paths and escapes that leave the source are rejected.
                  Example: echo "src/main.rs" > .ragfs/.reindex
  .query/<q>      Semantic search. Filename is the query.
                  Returns JSON results.
                  Example: cat ".ragfs/.query/authentication"
  .search/<q>     Symlinks to matching files.
  .similar/<path> Symlinks to files similar to <path>.
  .help           This file.

Agent operations (.ops/)
------------------------
  .ops/.create    Write: path<newline>content
  .ops/.delete    Write: path  (soft-delete when safety is on)
  .ops/.move      Write: src<newline>dst
  .ops/.batch     Write: JSON BatchRequest (create/delete/move/copy/write/mkdir/symlink)
  .ops/.result    Read:  JSON OperationResult of the last op (global; not per-agent)

  echo -e "notes.md\n# Hello" > .ragfs/.ops/.create
  cat .ragfs/.ops/.result

Safety (.safety/)
-----------------
  .safety/.trash/   Soft-deleted files (recoverable)
  .safety/.history  Audit log (JSONL)
  .safety/.undo     Write the history operation_id (same as undo_id in .result)

  echo "<undo_id>" > .ragfs/.safety/.undo

Semantic (.semantic/) — Beta
----------------------------
  .semantic/.organize  Write OrganizeRequest JSON → plan in .pending/
  .semantic/.similar   Write a path → similar files JSON
  .semantic/.cleanup   Read cleanup analysis (duplicates today)
  .semantic/.dedupe    Read duplicate groups
  .semantic/.pending/  Proposed plans
  .semantic/.approve   Write plan_id to execute
  .semantic/.reject    Write plan_id to cancel

Limits (honest)
---------------
  - .ops/.semantic/.reindex paths are jailed to the source root. Absolute paths
    and `..`/symlink hops that leave the source are rejected.
  - .ops/.result is the last operation only; concurrent agents overwrite it.
  - Semantic ByProject/cleanup is Beta: organize may only mkdir; cleanup is mostly duplicates.
  - allow_other (if enabled) exposes .ops/.safety/.semantic to every local user.
  - Code chunking is pattern-based (function/class signatures), not tree-sitter.
  - Default extractors: UTF-8 text/code, PDF, images. Binary .doc is not supported.
"#
        .as_bytes()
        .to_vec()
    }
}

pub(crate) fn reply_file_bytes(reply: ReplyData, content: &[u8], offset: i64, size: u32) {
    let offset = offset as usize;
    let size = size as usize;
    if offset >= content.len() {
        reply.data(&[]);
    } else {
        let end = (offset + size).min(content.len());
        reply.data(&content[offset..end]);
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

        match parent {
            ROOT_INO => self.lookup_root(&name_str, reply),
            RAGFS_DIR_INO => self.lookup_ragfs_dir(&name_str, reply),
            SEMANTIC_DIR_INO => self.lookup_semantic_dir(&name_str, reply),
            PENDING_DIR_INO => self.lookup_pending_dir(&name_str, reply),
            SAFETY_DIR_INO => self.lookup_safety_dir(&name_str, reply),
            OPS_DIR_INO => self.lookup_ops_dir(&name_str, reply),
            QUERY_DIR_INO => self.lookup_query_dir(&name_str, reply),
            _ => self.lookup_passthrough(parent, &name_str, reply),
        }
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
            QUERY_DIR_INO | SEARCH_DIR_INO | SIMILAR_DIR_INO | INDEX_FILE_INO | CONFIG_FILE_INO
            | REINDEX_FILE_INO | HELP_FILE_INO => self.getattr_query(ino, reply),
            OPS_DIR_INO | OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO
            | OPS_RESULT_INO => self.getattr_ops(ino, reply),
            SAFETY_DIR_INO | TRASH_DIR_INO | HISTORY_FILE_INO | UNDO_FILE_INO => {
                self.getattr_safety(ino, reply);
            }
            SEMANTIC_DIR_INO | PENDING_DIR_INO | SIMILAR_OPS_FILE_INO | CLEANUP_FILE_INO
            | DEDUPE_FILE_INO | ORGANIZE_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                self.getattr_semantic(ino, reply);
            }
            _ => self.getattr_cached_or_passthrough(ino, reply),
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

        let cache = self.runtime.block_on(self.content_cache.read());
        if let Some(content) = cache.get(&ino) {
            reply_file_bytes(reply, content, offset, size);
            return;
        }
        drop(cache);

        match ino {
            INDEX_FILE_INO | CONFIG_FILE_INO | REINDEX_FILE_INO | HELP_FILE_INO => {
                self.read_query(ino, offset, size, reply);
            }
            OPS_RESULT_INO | OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
                self.read_ops(ino, offset, size, reply);
            }
            HISTORY_FILE_INO | UNDO_FILE_INO => self.read_safety(ino, offset, size, reply),
            SIMILAR_OPS_FILE_INO | CLEANUP_FILE_INO | DEDUPE_FILE_INO | ORGANIZE_FILE_INO
            | APPROVE_FILE_INO | REJECT_FILE_INO => self.read_semantic(ino, offset, size, reply),
            _ => self.read_passthrough(ino, offset, size, reply),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        reply: ReplyDirectory,
    ) {
        debug!("readdir: ino={}, offset={}", ino, offset);

        match ino {
            ROOT_INO => self.readdir_root(offset, reply),
            RAGFS_DIR_INO => self.readdir_ragfs(offset, reply),
            QUERY_DIR_INO | SEARCH_DIR_INO | SIMILAR_DIR_INO => {
                self.readdir_query(ino, offset, reply);
            }
            OPS_DIR_INO => self.readdir_ops(offset, reply),
            SAFETY_DIR_INO | TRASH_DIR_INO => self.readdir_safety(ino, offset, reply),
            SEMANTIC_DIR_INO | PENDING_DIR_INO => self.readdir_semantic(ino, offset, reply),
            _ => self.readdir_passthrough(ino, offset, reply),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        debug!("open: ino={}, flags={}", ino, flags);
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

        match ino {
            REINDEX_FILE_INO => self.write_reindex(data, reply),
            OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
                self.write_ops(ino, data, reply);
            }
            UNDO_FILE_INO => self.write_safety(ino, data, reply),
            ORGANIZE_FILE_INO | SIMILAR_OPS_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                self.write_semantic(ino, data, reply);
            }
            _ => self.write_passthrough(ino, data, reply),
        }
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
        umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        self.create_passthrough(parent, name, mode, umask, flags, reply);
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        self.unlink_passthrough(parent, name, reply);
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        umask: u32,
        reply: ReplyEntry,
    ) {
        self.mkdir_passthrough(parent, name, mode, umask, reply);
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        self.rmdir_passthrough(parent, name, reply);
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        flags: u32,
        reply: ReplyEmpty,
    ) {
        self.rename_passthrough(parent, name, newparent, newname, flags, reply);
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        ctime: Option<SystemTime>,
        fh: Option<u64>,
        crtime: Option<SystemTime>,
        chgtime: Option<SystemTime>,
        bkuptime: Option<SystemTime>,
        flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        self.setattr_passthrough(
            ino, mode, uid, gid, size, atime, mtime, ctime, fh, crtime, chgtime, bkuptime, flags,
            reply,
        );
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
        assert_eq!(json["api_version"], "0.2");
        assert_eq!(json["interfaces"]["ops"], true);
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
        assert_eq!(json["api_version"], "0.2");
        assert!(json.get("interfaces").is_some());
    }

    #[tokio::test]
    async fn test_help_mentions_agent_interfaces() {
        let fs = RagFs::new(PathBuf::from("/tmp/test-help"));
        let help = String::from_utf8(fs.get_help_content()).expect("Valid UTF-8");
        assert!(help.contains(".ops/"), "help must document .ops/");
        assert!(help.contains(".safety/"), "help must document .safety/");
        assert!(help.contains(".semantic/"), "help must document .semantic/");
        assert!(help.contains(".undo"), "help must document undo");
        assert!(help.contains("api_version"));
        assert!(
            help.contains("jailed to the source root"),
            "help must document the source-root path jail"
        );
        assert!(
            !help.contains("not path-jailed"),
            "help must not claim .reindex is unjailed"
        );
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

    #[tokio::test]
    async fn test_jail_reindex_rejects_absolute_outside_source() {
        let temp = tempfile::TempDir::new().unwrap();
        let fs = RagFs::new(temp.path().to_path_buf());
        let outside = tempfile::TempDir::new().unwrap();
        let err = fs
            .jail_reindex_path(&outside.path().join("secret.txt"))
            .unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[tokio::test]
    async fn test_jail_reindex_allows_relative_inside_source() {
        let temp = tempfile::TempDir::new().unwrap();
        let fs = RagFs::new(temp.path().to_path_buf());
        let resolved = fs.jail_reindex_path(Path::new("src/main.rs")).unwrap();
        assert!(resolved.starts_with(temp.path().canonicalize().unwrap()));
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
