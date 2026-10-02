//! Real-file passthrough (root and nested directories).

use fuser::{
    FileType, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry,
    ReplyWrite, TimeOrNow,
};
use libc::{EEXIST, EINVAL, EIO, EISDIR, ENOENT, ENOSYS, ENOTDIR, ENOTEMPTY, EPERM};
use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::time::SystemTime;
use tracing::{info, warn};

use super::{RagFs, TTL, reply_file_bytes};
use crate::inode::{FIRST_REAL_INO, InodeKind, RAGFS_DIR_INO, ROOT_INO};

impl RagFs {
    pub(crate) fn lookup_root(&mut self, name_str: &str, reply: ReplyEntry) {
        if name_str == ".ragfs" {
            let attr = self.make_attr(RAGFS_DIR_INO, FileType::Directory, 0);
            reply.entry(&TTL, &attr, 0);
            return;
        }

        let real_path = self.source.join(name_str);
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

        reply.error(ENOENT);
    }

    pub(crate) fn lookup_passthrough(&mut self, parent: u64, name_str: &str, reply: ReplyEntry) {
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(parent)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            let real_path = path.join(name_str);
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

    pub(crate) fn getattr_cached_or_passthrough(&mut self, ino: u64, reply: ReplyAttr) {
        let cache = self.runtime.block_on(self.content_cache.read());
        if let Some(content) = cache.get(&ino) {
            let attr = self.make_attr(ino, FileType::RegularFile, content.len() as u64);
            reply.attr(&TTL, &attr);
            return;
        }
        drop(cache);

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

    pub(crate) fn read_passthrough(&mut self, ino: u64, offset: i64, size: u32, reply: ReplyData) {
        let inodes = self.runtime.block_on(self.inodes.read());
        if let Some(entry) = inodes.get(ino)
            && let InodeKind::Real { path, .. } = &entry.kind
        {
            let path = path.clone();
            drop(inodes);

            match fs::read(&path) {
                Ok(content) => {
                    reply_file_bytes(reply, &content, offset, size);
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

    pub(crate) fn readdir_root(&mut self, offset: i64, mut reply: ReplyDirectory) {
        let mut entries = vec![
            (ROOT_INO, FileType::Directory, ".".to_string()),
            (ROOT_INO, FileType::Directory, "..".to_string()),
            (RAGFS_DIR_INO, FileType::Directory, ".ragfs".to_string()),
        ];

        if let Ok(read_dir) = fs::read_dir(&self.source) {
            for entry in read_dir.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
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

    pub(crate) fn readdir_passthrough(&mut self, ino: u64, offset: i64, mut reply: ReplyDirectory) {
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
                        let entry_ino = inodes.get_or_create_real(dir_entry.path(), metadata.ino());
                        entries.push((entry_ino, file_type, name));
                    }
                }

                for (i, (entry_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize)
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

    pub(crate) fn write_passthrough(&mut self, ino: u64, data: &[u8], reply: ReplyWrite) {
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

    pub(crate) fn create_passthrough(
        &mut self,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let name_str = name.to_string_lossy();
        tracing::debug!(
            "create: parent={}, name={}, mode={:o}",
            parent,
            name_str,
            mode
        );

        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let new_path = parent_path.join(&*name_str);

        if new_path.exists() {
            reply.error(EEXIST);
            return;
        }

        match fs::File::create(&new_path) {
            Ok(_) => {
                let metadata = match fs::metadata(&new_path) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for new file {:?}: {}", new_path, e);
                        reply.error(EIO);
                        return;
                    }
                };

                let mut inodes = self.runtime.block_on(self.inodes.write());
                let ino = inodes.get_or_create_real(new_path.clone(), metadata.ino());
                drop(inodes);

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

    pub(crate) fn unlink_passthrough(&mut self, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy();
        tracing::debug!("unlink: parent={}, name={}", parent, name_str);

        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let file_path = parent_path.join(&*name_str);

        if !file_path.exists() {
            reply.error(ENOENT);
            return;
        }

        if file_path.is_dir() {
            reply.error(EISDIR);
            return;
        }

        if let Some(ref store) = self.store {
            let store = store.clone();
            let path_for_delete = file_path.clone();
            let result = self
                .runtime
                .block_on(async move { store.delete_by_file_path(&path_for_delete).await });
            if let Err(e) = result {
                warn!("Failed to delete from store {:?}: {}", file_path, e);
            }
        }

        match fs::remove_file(&file_path) {
            Ok(()) => {
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

    pub(crate) fn mkdir_passthrough(
        &mut self,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let name_str = name.to_string_lossy();
        tracing::debug!(
            "mkdir: parent={}, name={}, mode={:o}",
            parent,
            name_str,
            mode
        );

        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let new_path = parent_path.join(&*name_str);

        if new_path.exists() {
            reply.error(EEXIST);
            return;
        }

        match fs::create_dir(&new_path) {
            Ok(()) => {
                let metadata = match fs::metadata(&new_path) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for new dir {:?}: {}", new_path, e);
                        reply.error(EIO);
                        return;
                    }
                };

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

    pub(crate) fn rmdir_passthrough(&mut self, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy();
        tracing::debug!("rmdir: parent={}, name={}", parent, name_str);

        if parent < FIRST_REAL_INO && parent != ROOT_INO {
            reply.error(EPERM);
            return;
        }

        let Some(parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let dir_path = parent_path.join(&*name_str);

        if !dir_path.exists() {
            reply.error(ENOENT);
            return;
        }

        if !dir_path.is_dir() {
            reply.error(ENOTDIR);
            return;
        }

        match fs::remove_dir(&dir_path) {
            Ok(()) => {
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

    pub(crate) fn rename_passthrough(
        &mut self,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let name_str = name.to_string_lossy();
        let newname_str = newname.to_string_lossy();
        tracing::debug!(
            "rename: parent={}, name={}, newparent={}, newname={}",
            parent,
            name_str,
            newparent,
            newname_str
        );

        if (parent < FIRST_REAL_INO && parent != ROOT_INO)
            || (newparent < FIRST_REAL_INO && newparent != ROOT_INO)
        {
            reply.error(EPERM);
            return;
        }

        let Some(src_parent_path) = self.resolve_parent_path(parent) else {
            reply.error(ENOENT);
            return;
        };

        let Some(dst_parent_path) = self.resolve_parent_path(newparent) else {
            reply.error(ENOENT);
            return;
        };

        let src_path = src_parent_path.join(&*name_str);
        let dst_path = dst_parent_path.join(&*newname_str);

        if !src_path.exists() {
            reply.error(ENOENT);
            return;
        }

        match fs::rename(&src_path, &dst_path) {
            Ok(()) => {
                if let Some(ref store) = self.store {
                    let store = store.clone();
                    let src = src_path.clone();
                    let dst = dst_path.clone();
                    let result = self
                        .runtime
                        .block_on(async move { store.update_file_path(&src, &dst).await });
                    if let Err(e) = result {
                        warn!(
                            "Failed to update store path {:?} -> {:?}: {}",
                            src_path, dst_path, e
                        );
                    }
                }

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

    pub(crate) fn setattr_passthrough(
        &mut self,
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
        tracing::debug!("setattr: ino={}, size={:?}", ino, size);

        if ino < FIRST_REAL_INO {
            reply.error(EPERM);
            return;
        }

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

        if let Some(new_size) = size {
            match fs::OpenOptions::new().write(true).open(&path) {
                Ok(file) => {
                    if let Err(e) = file.set_len(new_size) {
                        warn!("Failed to truncate {:?}: {}", path, e);
                        reply.error(EIO);
                        return;
                    }

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

        if let Some(attr) = self.real_path_to_attr(&path, ino) {
            reply.attr(&TTL, &attr);
        } else {
            reply.error(EIO);
        }
    }
}
