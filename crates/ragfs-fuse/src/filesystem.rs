//! FUSE filesystem implementation.

use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, ReplyOpen,
    ReplyWrite, Request,
};
use libc::{ENOENT, ENOSYS, EIO};
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
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::inode::{InodeKind, InodeTable, RAGFS_DIR_INO, ROOT_INO,
    INDEX_FILE_INO, CONFIG_FILE_INO, REINDEX_FILE_INO, QUERY_DIR_INO, SEARCH_DIR_INO, SIMILAR_DIR_INO};

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
}

impl RagFs {
    /// Create a new RAGFS filesystem (basic, for passthrough only).
    pub fn new(source: PathBuf) -> Self {
        Self {
            source,
            inodes: Arc::new(RwLock::new(InodeTable::new())),
            store: None,
            query_executor: None,
            runtime: Handle::current(),
            content_cache: Arc::new(RwLock::new(HashMap::new())),
            reindex_sender: None,
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
            embedder,
            10, // default limit
            false, // hybrid search
        ));

        Self {
            source,
            inodes: Arc::new(RwLock::new(InodeTable::new())),
            store: Some(store),
            query_executor: Some(query_executor),
            runtime,
            content_cache: Arc::new(RwLock::new(HashMap::new())),
            reindex_sender,
        }
    }

    /// Get the source directory.
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
            blocks: (metadata.len() + BLOCK_SIZE - 1) / BLOCK_SIZE,
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
            blocks: (size + BLOCK_SIZE - 1) / BLOCK_SIZE,
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
            let result = self.runtime.block_on(async {
                store.stats().await
            });

            match result {
                Ok(stats) => {
                    let json = serde_json::json!({
                        "status": "indexed",
                        "total_files": stats.total_files,
                        "total_chunks": stats.total_chunks,
                        "index_size_bytes": stats.index_size_bytes,
                        "last_updated": stats.last_updated.map(|t| t.to_rfc3339()),
                    });
                    serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
                }
                Err(e) => {
                    let json = serde_json::json!({
                        "status": "error",
                        "error": e.to_string(),
                    });
                    serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
                }
            }
        } else {
            let json = serde_json::json!({
                "status": "not_initialized",
                "message": "No store configured",
            });
            serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
        }
    }

    /// Execute a query and return results as JSON.
    fn execute_query(&self, query: &str) -> Vec<u8> {
        if let Some(ref executor) = self.query_executor {
            let executor = executor.clone();
            let query_str = query.to_string();
            let query_for_result = query_str.clone();
            let result = self.runtime.block_on(async move {
                executor.execute(&query_str).await
            });

            match result {
                Ok(results) => {
                    let json_results: Vec<_> = results.iter().map(|r| {
                        serde_json::json!({
                            "file": r.file_path.to_string_lossy(),
                            "score": r.score,
                            "content": truncate(&r.content, 500),
                            "byte_range": [r.byte_range.start, r.byte_range.end],
                            "line_range": r.line_range.as_ref().map(|lr| [lr.start, lr.end]),
                        })
                    }).collect();

                    let json = serde_json::json!({
                        "query": query_for_result,
                        "results": json_results,
                    });
                    serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
                }
                Err(e) => {
                    let json = serde_json::json!({
                        "query": query_for_result,
                        "error": e.to_string(),
                    });
                    serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
                }
            }
        } else {
            let json = serde_json::json!({
                "error": "Query executor not configured",
            });
            serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
        }
    }

    /// Get configuration as JSON.
    fn get_config(&self) -> Vec<u8> {
        let json = serde_json::json!({
            "source": self.source.to_string_lossy(),
            "store_configured": self.store.is_some(),
            "query_executor_configured": self.query_executor.is_some(),
        });
        serde_json::to_string_pretty(&json).unwrap_or_default().into_bytes()
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
                let metadata = match fs::metadata(&real_path) {
                    Ok(m) => m,
                    Err(_) => {
                        reply.error(ENOENT);
                        return;
                    }
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
                    let attr = self.make_attr(INDEX_FILE_INO, FileType::RegularFile, content.len() as u64);
                    let mut cache = self.runtime.block_on(self.content_cache.write());
                    cache.insert(INDEX_FILE_INO, content);
                    reply.entry(&TTL, &attr, 0);
                    return;
                }
                ".config" => {
                    let content = self.get_config();
                    let attr = self.make_attr(CONFIG_FILE_INO, FileType::RegularFile, content.len() as u64);
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
        if let Some(entry) = inodes.get(parent) {
            if let InodeKind::Real { path, .. } = &entry.kind {
                let real_path = path.join(&*name_str);
                if real_path.exists() {
                    drop(inodes);
                    let metadata = match fs::metadata(&real_path) {
                        Ok(m) => m,
                        Err(_) => {
                            reply.error(ENOENT);
                            return;
                        }
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
                if let Some(entry) = inodes.get(ino) {
                    if let InodeKind::Real { path, .. } = &entry.kind {
                        if let Some(attr) = self.real_path_to_attr(path, ino) {
                            reply.attr(&TTL, &attr);
                            return;
                        }
                    }
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
            _ => {}
        }

        // Try to read real file
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(ino) {
            if let InodeKind::Real { path, .. } = &entry.kind {
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
                let entries = vec![
                    (RAGFS_DIR_INO, FileType::Directory, "."),
                    (ROOT_INO, FileType::Directory, ".."),
                    (QUERY_DIR_INO, FileType::Directory, ".query"),
                    (SEARCH_DIR_INO, FileType::Directory, ".search"),
                    (INDEX_FILE_INO, FileType::RegularFile, ".index"),
                    (CONFIG_FILE_INO, FileType::RegularFile, ".config"),
                    (REINDEX_FILE_INO, FileType::RegularFile, ".reindex"),
                    (SIMILAR_DIR_INO, FileType::Directory, ".similar"),
                ];

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            QUERY_DIR_INO | SEARCH_DIR_INO | SIMILAR_DIR_INO => {
                // These directories are empty - files are created dynamically on lookup
                let entries = vec![
                    (ino, FileType::Directory, "."),
                    (RAGFS_DIR_INO, FileType::Directory, ".."),
                ];

                for (i, (entry_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*entry_ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            _ => {
                // Try to read real directory
                let inodes = self.runtime.block_on(self.inodes.read());
                if let Some(entry) = inodes.get(ino) {
                    if let InodeKind::Real { path, .. } = &entry.kind {
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
                                    let entry_ino = inodes.get_or_create_real(dir_entry.path(), metadata.ino());
                                    entries.push((entry_ino, file_type, name));
                                }
                            }

                            for (i, (entry_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                                if reply.add(*entry_ino, (i + 1) as i64, *kind, name) {
                                    break;
                                }
                            }
                            reply.ok();
                            return;
                        }
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

        // Real file writes (passthrough)
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(ino) {
            if let InodeKind::Real { path, .. } = &entry.kind {
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
        }

        reply.error(ENOSYS);
    }

    fn forget(&mut self, _req: &Request<'_>, ino: u64, nlookup: u64) {
        debug!("forget: ino={}, nlookup={}", ino, nlookup);
        let mut inodes = self.runtime.block_on(self.inodes.write());
        inodes.forget(ino, nlookup);
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
