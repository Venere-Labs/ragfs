//! `.ops/` virtual files.

use fuser::{FileType, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, ReplyWrite};
use libc::ENOENT;

use super::{RagFs, TTL, reply_file_bytes};
use crate::inode::{
    OPS_BATCH_INO, OPS_CREATE_INO, OPS_DELETE_INO, OPS_DIR_INO, OPS_MOVE_INO, OPS_RESULT_INO,
    RAGFS_DIR_INO,
};

impl RagFs {
    pub(crate) fn lookup_ops_dir(&mut self, name_str: &str, reply: ReplyEntry) {
        match name_str {
            ".create" => {
                let attr = self.make_attr(OPS_CREATE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".delete" => {
                let attr = self.make_attr(OPS_DELETE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".move" => {
                let attr = self.make_attr(OPS_MOVE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".batch" => {
                let attr = self.make_attr(OPS_BATCH_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".result" => {
                let content = self.runtime.block_on(self.ops_manager.get_last_result());
                let attr =
                    self.make_attr(OPS_RESULT_INO, FileType::RegularFile, content.len() as u64);
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(OPS_RESULT_INO, content);
                reply.entry(&TTL, &attr, 0);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn getattr_ops(&mut self, ino: u64, reply: ReplyAttr) {
        match ino {
            OPS_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
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
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn read_ops(&mut self, ino: u64, offset: i64, size: u32, reply: ReplyData) {
        match ino {
            OPS_RESULT_INO => {
                let content = self.runtime.block_on(self.ops_manager.get_last_result());
                reply_file_bytes(reply, &content, offset, size);
            }
            OPS_CREATE_INO | OPS_DELETE_INO | OPS_MOVE_INO | OPS_BATCH_INO => {
                reply.data(&[]);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn readdir_ops(&mut self, offset: i64, mut reply: ReplyDirectory) {
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

    pub(crate) fn write_ops(&mut self, ino: u64, data: &[u8], reply: ReplyWrite) {
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
        }
    }
}
