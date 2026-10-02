//! `.semantic/` virtual files.

use fuser::{FileType, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, ReplyWrite};
use libc::ENOENT;
use std::path::PathBuf;
use tracing::{info, warn};

use super::{RagFs, TTL, reply_file_bytes};
use crate::inode::{
    APPROVE_FILE_INO, CLEANUP_FILE_INO, DEDUPE_FILE_INO, ORGANIZE_FILE_INO, PENDING_DIR_INO,
    RAGFS_DIR_INO, REJECT_FILE_INO, SEMANTIC_DIR_INO, SIMILAR_OPS_FILE_INO,
};

impl RagFs {
    pub(crate) fn lookup_semantic_dir(&mut self, name_str: &str, reply: ReplyEntry) {
        match name_str {
            ".organize" => {
                let attr = self.make_attr(ORGANIZE_FILE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".similar" => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_similar_json());
                let attr = self.make_attr(
                    SIMILAR_OPS_FILE_INO,
                    FileType::RegularFile,
                    content.len() as u64,
                );
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(SIMILAR_OPS_FILE_INO, content);
                reply.entry(&TTL, &attr, 0);
            }
            ".cleanup" => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_cleanup_json());
                let attr = self.make_attr(
                    CLEANUP_FILE_INO,
                    FileType::RegularFile,
                    content.len() as u64,
                );
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(CLEANUP_FILE_INO, content);
                reply.entry(&TTL, &attr, 0);
            }
            ".dedupe" => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_dedupe_json());
                let attr =
                    self.make_attr(DEDUPE_FILE_INO, FileType::RegularFile, content.len() as u64);
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(DEDUPE_FILE_INO, content);
                reply.entry(&TTL, &attr, 0);
            }
            ".pending" => {
                let attr = self.make_attr(PENDING_DIR_INO, FileType::Directory, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".approve" => {
                let attr = self.make_attr(APPROVE_FILE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            ".reject" => {
                let attr = self.make_attr(REJECT_FILE_INO, FileType::RegularFile, 0);
                reply.entry(&TTL, &attr, 0);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn lookup_pending_dir(&mut self, name_str: &str, reply: ReplyEntry) {
        let plan_ids = self
            .runtime
            .block_on(self.semantic_manager.get_pending_plan_ids());
        if plan_ids.contains(&name_str.to_string()) {
            let content = self
                .runtime
                .block_on(self.semantic_manager.get_plan_json(name_str));
            let mut inodes = self.runtime.block_on(self.inodes.write());
            let ino = inodes.get_or_create_query_result(PENDING_DIR_INO, name_str.to_string());
            let attr = self.make_attr(ino, FileType::RegularFile, content.len() as u64);
            let mut cache = self.runtime.block_on(self.content_cache.write());
            cache.insert(ino, content);
            reply.entry(&TTL, &attr, 0);
            return;
        }
        reply.error(ENOENT);
    }

    pub(crate) fn getattr_semantic(&mut self, ino: u64, reply: ReplyAttr) {
        match ino {
            SEMANTIC_DIR_INO | PENDING_DIR_INO => {
                let attr = self.make_attr(ino, FileType::Directory, 0);
                reply.attr(&TTL, &attr);
            }
            SIMILAR_OPS_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_similar_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(SIMILAR_OPS_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            CLEANUP_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_cleanup_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(CLEANUP_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            DEDUPE_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_dedupe_json());
                let size = content.len() as u64;
                let mut cache = self.runtime.block_on(self.content_cache.write());
                cache.insert(DEDUPE_FILE_INO, content);
                let attr = self.make_attr(ino, FileType::RegularFile, size);
                reply.attr(&TTL, &attr);
            }
            ORGANIZE_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                let attr = self.make_attr(ino, FileType::RegularFile, 0);
                reply.attr(&TTL, &attr);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn read_semantic(&mut self, ino: u64, offset: i64, size: u32, reply: ReplyData) {
        match ino {
            SIMILAR_OPS_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_similar_json());
                reply_file_bytes(reply, &content, offset, size);
            }
            CLEANUP_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_cleanup_json());
                reply_file_bytes(reply, &content, offset, size);
            }
            DEDUPE_FILE_INO => {
                let content = self
                    .runtime
                    .block_on(self.semantic_manager.get_dedupe_json());
                reply_file_bytes(reply, &content, offset, size);
            }
            ORGANIZE_FILE_INO | APPROVE_FILE_INO | REJECT_FILE_INO => {
                reply.data(&[]);
            }
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn readdir_semantic(&mut self, ino: u64, offset: i64, mut reply: ReplyDirectory) {
        match ino {
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
                let mut entries: Vec<(u64, FileType, String)> = vec![
                    (PENDING_DIR_INO, FileType::Directory, ".".to_string()),
                    (SEMANTIC_DIR_INO, FileType::Directory, "..".to_string()),
                ];

                let plan_ids = self
                    .runtime
                    .block_on(self.semantic_manager.get_pending_plan_ids());
                for plan_id in plan_ids {
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
            _ => reply.error(ENOENT),
        }
    }

    pub(crate) fn write_semantic(&mut self, ino: u64, data: &[u8], reply: ReplyWrite) {
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
        if ino == SIMILAR_OPS_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            let path = PathBuf::from(&data_str);
            let semantic_manager = self.semantic_manager.clone();
            let result = self
                .runtime
                .block_on(async move { semantic_manager.find_similar(&path).await });
            match result {
                Ok(r) => info!("Found {} similar files to {}", r.similar.len(), data_str),
                Err(e) => warn!("Failed to find similar files: {}", e),
            }
            reply.written(data.len() as u32);
            return;
        }
        if ino == APPROVE_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            if let Ok(plan_id) = uuid::Uuid::parse_str(&data_str) {
                let semantic_manager = self.semantic_manager.clone();
                let result = self
                    .runtime
                    .block_on(async move { semantic_manager.approve_plan(plan_id).await });
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
        if ino == REJECT_FILE_INO {
            let data_str = String::from_utf8_lossy(data).trim().to_string();
            if let Ok(plan_id) = uuid::Uuid::parse_str(&data_str) {
                let semantic_manager = self.semantic_manager.clone();
                let result = self
                    .runtime
                    .block_on(async move { semantic_manager.reject_plan(plan_id).await });
                match result {
                    Ok(plan) => info!("Rejected plan: {}", plan.id),
                    Err(e) => warn!("Failed to reject plan: {}", e),
                }
            } else {
                warn!("Invalid plan ID for reject: {}", data_str);
            }
            reply.written(data.len() as u32);
        }
    }
}
