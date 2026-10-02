//! `.safety/` virtual files.

use fuser::{FileType, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, ReplyWrite};
use libc::ENOENT;
use tracing::{info, warn};

use super::{RagFs, TTL, reply_file_bytes};
use crate::inode::{
    FIRST_REAL_INO, HISTORY_FILE_INO, RAGFS_DIR_INO, SAFETY_DIR_INO, TRASH_DIR_INO, UNDO_FILE_INO,
};

impl RagFs {
    pub(crate) fn lookup_safety_dir(&mut self, name_str: &str, reply: ReplyEntry) {
        match name_str {
            ".trash" => {
                let attr = self.make_attr(TRASH_DIR_INO, FileType::Directory, 0);
                reply.entry(&TTL, &attr, 0);
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
            }
            ".undo" => {
                let attr = self.make_attr(UNDO_FILE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn getattr_safety(&mut self, ino: u64, reply: ReplyAttr) {
        match ino {
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
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn read_safety(&mut self, ino: u64, offset: i64, size: u32, reply: ReplyData) {
        match ino {
            HISTORY_FILE_INO => {
                let content = self.safety_manager.get_history_json(Some(100));
                reply_file_bytes(reply, &content, offset, size);
            }
            UNDO_FILE_INO => {
                reply.data(&[]);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn readdir_safety(&mut self, ino: u64, offset: i64, mut reply: ReplyDirectory) {
        match ino {
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
                let mut entries: Vec<(u64, FileType, String)> = vec![
                    (TRASH_DIR_INO, FileType::Directory, ".".to_string()),
                    (SAFETY_DIR_INO, FileType::Directory, "..".to_string()),
                ];

                let trash = self.runtime.block_on(self.safety_manager.list_trash());
                for entry in trash {
                    let entry_ino =
                        FIRST_REAL_INO + 500_000 + (entry.id.as_u128() % 100_000) as u64;
                    entries.push((entry_ino, FileType::RegularFile, entry.id.to_string()));
                }

                for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                    if reply.add(*ino, (i + 1) as i64, *kind, name) {
                        break;
                    }
                }
                reply.ok();
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn write_safety(&mut self, _ino: u64, data: &[u8], reply: ReplyWrite) {
        let data_str = String::from_utf8_lossy(data).trim().to_string();
        if let Ok(operation_id) = uuid::Uuid::parse_str(&data_str) {
            let safety_manager = self.safety_manager.clone();
            let result = self
                .runtime
                .block_on(async move { safety_manager.undo(operation_id).await });
            match result {
                Ok(msg) => info!("Undo successful: {}", msg),
                Err(e) => warn!("Undo failed: {}", e),
            }
        } else {
            warn!("Invalid operation ID for undo: {}", data_str);
        }
        reply.written(data.len() as u32);
    }
}
