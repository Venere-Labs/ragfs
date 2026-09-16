//! Operations manager for agent file management.
//!
//! This module provides a structured interface for file operations with JSON feedback,
//! designed for AI agents to manage files through the virtual `.ops/` directory.

use chrono::{DateTime, Utc};
use ragfs_core::VectorStore;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::safety::{HistoryOperation, SafetyManager, UndoData};

/// A single file operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Operation {
    /// Create a new file with content
    Create { path: PathBuf, content: String },
    /// Delete a file (soft delete if safety layer is enabled)
    Delete { path: PathBuf },
    /// Move/rename a file
    Move { src: PathBuf, dst: PathBuf },
    /// Copy a file
    Copy { src: PathBuf, dst: PathBuf },
    /// Write content to a file (overwrite or append)
    Write {
        path: PathBuf,
        content: String,
        #[serde(default)]
        append: bool,
    },
    /// Create a directory
    Mkdir { path: PathBuf },
    /// Create a symbolic link
    Symlink { target: PathBuf, link: PathBuf },
}

/// Batch operation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// List of operations to perform
    pub operations: Vec<Operation>,
    /// If true, rollback all operations if any fails
    #[serde(default)]
    pub atomic: bool,
    /// If true, only validate without executing
    #[serde(default)]
    pub dry_run: bool,
}

/// Result of a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResult {
    /// Unique identifier for this operation
    pub id: Uuid,
    /// Whether the operation succeeded
    pub success: bool,
    /// Type of operation performed
    pub operation: String,
    /// Primary path involved
    pub path: PathBuf,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Timestamp of the operation
    pub timestamp: DateTime<Utc>,
    /// Whether the file was indexed/reindexed
    pub indexed: bool,
    /// `SafetyManager` history id — write this to `.safety/.undo`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub undo_id: Option<Uuid>,
    /// Soft-delete trash entry used by atomic batch rollback (not in `.result` JSON)
    #[serde(skip)]
    trash_id: Option<Uuid>,
}

impl OperationResult {
    /// Create a successful operation result.
    pub fn success(operation: &str, path: PathBuf, indexed: bool) -> Self {
        Self {
            id: Uuid::new_v4(),
            success: true,
            operation: operation.to_string(),
            path,
            error: None,
            timestamp: Utc::now(),
            indexed,
            undo_id: None,
            trash_id: None,
        }
    }

    /// Create a failed operation result.
    pub fn failure(operation: &str, path: PathBuf, error: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            success: false,
            operation: operation.to_string(),
            path,
            error: Some(error),
            timestamp: Utc::now(),
            indexed: false,
            undo_id: None,
            trash_id: None,
        }
    }
}

/// Result of batch operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Unique identifier for this batch
    pub id: Uuid,
    /// Whether all operations succeeded
    pub success: bool,
    /// Total number of operations
    pub total: usize,
    /// Number of successful operations
    pub succeeded: usize,
    /// Number of failed operations
    pub failed: usize,
    /// Individual operation results
    pub results: Vec<OperationResult>,
    /// Timestamp of the batch
    pub timestamp: DateTime<Utc>,
    /// Whether a rollback was performed (for atomic batches)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollback_performed: Option<bool>,
    /// Details about rollback if it was attempted
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollback_details: Option<RollbackDetails>,
}

/// Details about rollback execution for atomic batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackDetails {
    /// Number of operations rolled back successfully
    pub rolled_back: usize,
    /// Number of operations that failed to rollback
    pub rollback_failures: usize,
    /// Errors encountered during rollback
    pub errors: Vec<RollbackError>,
}

/// Error encountered during a rollback operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackError {
    /// Index of the operation in the original batch
    pub operation_index: usize,
    /// Error message
    pub error: String,
}

/// Data needed to rollback an operation.
/// This is internal to batch operations and more detailed than `UndoData`.
#[derive(Debug, Clone)]
enum RollbackData {
    /// Rollback a create by deleting the created file
    Create { created_path: PathBuf },
    /// Rollback a delete by restoring from trash or backup content
    Delete {
        original_path: PathBuf,
        trash_id: Option<Uuid>,
        content_backup: Option<Vec<u8>>,
    },
    /// Rollback a move by moving back
    Move { src: PathBuf, dst: PathBuf },
    /// Rollback a copy by deleting the copy
    Copy { copied_path: PathBuf },
    /// Rollback a write by restoring previous content
    Write {
        path: PathBuf,
        previous_content: Option<Vec<u8>>,
        file_existed: bool,
    },
    /// Rollback a mkdir by removing the directory
    Mkdir { created_path: PathBuf },
    /// Rollback a symlink by removing the link
    Symlink { link_path: PathBuf },
}

/// Journal entry tracking an executed operation for potential rollback.
#[derive(Debug, Clone)]
struct JournalEntry {
    /// Index of the operation in the batch
    operation_index: usize,
    /// Data needed to rollback this operation
    rollback_data: RollbackData,
}

/// Operations manager for file operations with feedback.
pub struct OpsManager {
    /// Source directory root
    source: PathBuf,
    /// Vector store for indexing operations
    store: Option<Arc<dyn VectorStore>>,
    /// Channel to send reindex requests
    reindex_sender: Option<mpsc::Sender<PathBuf>>,
    /// Safety manager for trash/history/undo (optional)
    safety_manager: Option<Arc<SafetyManager>>,
    /// Last operation result (for .result file)
    last_result: Arc<RwLock<Option<OperationResult>>>,
    /// Last batch result
    last_batch_result: Arc<RwLock<Option<BatchResult>>>,
}

impl OpsManager {
    /// Create a new operations manager.
    pub fn new(
        source: PathBuf,
        store: Option<Arc<dyn VectorStore>>,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
    ) -> Self {
        Self {
            source,
            store,
            reindex_sender,
            safety_manager: None,
            last_result: Arc::new(RwLock::new(None)),
            last_batch_result: Arc::new(RwLock::new(None)),
        }
    }

    /// Create operations manager with safety manager.
    pub fn with_safety(
        source: PathBuf,
        store: Option<Arc<dyn VectorStore>>,
        reindex_sender: Option<mpsc::Sender<PathBuf>>,
        safety_manager: Arc<SafetyManager>,
    ) -> Self {
        Self {
            source,
            store,
            reindex_sender,
            safety_manager: Some(safety_manager),
            last_result: Arc::new(RwLock::new(None)),
            last_batch_result: Arc::new(RwLock::new(None)),
        }
    }

    /// Set the safety manager.
    pub fn set_safety_manager(&mut self, safety_manager: Arc<SafetyManager>) {
        self.safety_manager = Some(safety_manager);
    }

    /// Log operation to history if safety manager is available.
    /// Returns the history entry id used as `OperationResult.undo_id`.
    fn log_to_history(
        &self,
        operation: HistoryOperation,
        undo_data: Option<UndoData>,
    ) -> Option<Uuid> {
        self.safety_manager
            .as_ref()
            .map(|safety| safety.log_success(operation, undo_data))
    }

    /// Log failure to history if safety manager is available.
    fn log_failure_to_history(&self, operation: HistoryOperation, error: &str) {
        if let Some(ref safety) = self.safety_manager {
            safety.log_failure(operation, error.to_string());
        }
    }

    /// Resolve a path relative to the source directory.
    /// Rejects paths that escape the source root (absolute, `..`, symlink).
    fn resolve_path(&self, path: &PathBuf) -> Result<PathBuf, String> {
        crate::path_jail::resolve_under_root(&self.source, path)
    }

    /// Jail-check a symlink target the way Unix will resolve it: relative
    /// targets are interpreted from the link's parent, not the source root.
    fn jail_symlink_target(
        &self,
        target: &PathBuf,
        resolved_link: &Path,
    ) -> Result<PathBuf, String> {
        let validation_target = if target.is_absolute() {
            target.clone()
        } else if let Some(parent) = resolved_link.parent() {
            parent.join(target)
        } else {
            target.clone()
        };
        self.resolve_path(&validation_target)
    }

    async fn fail_and_store(
        &self,
        operation: &str,
        path: PathBuf,
        error: String,
    ) -> OperationResult {
        let result = OperationResult::failure(operation, path, error);
        *self.last_result.write().await = Some(result.clone());
        result
    }

    /// Trigger reindexing for a path.
    async fn trigger_reindex(&self, path: &PathBuf) -> bool {
        if let Some(ref sender) = self.reindex_sender {
            match sender.send(path.clone()).await {
                Ok(()) => {
                    debug!("Reindex triggered for {:?}", path);
                    true
                }
                Err(e) => {
                    warn!("Failed to trigger reindex for {:?}: {}", path, e);
                    false
                }
            }
        } else {
            false
        }
    }

    /// Delete from vector store.
    async fn delete_from_store(&self, path: &PathBuf) {
        if let Some(ref store) = self.store
            && let Err(e) = store.delete_by_file_path(path).await
        {
            warn!("Failed to delete {:?} from store: {e}", path);
        }
    }

    /// Update path in vector store.
    async fn update_store_path(&self, from: &PathBuf, to: &PathBuf) {
        if let Some(ref store) = self.store
            && let Err(e) = store.update_file_path(from, to).await
        {
            warn!("Failed to update path in store {:?} -> {:?}: {e}", from, to);
        }
    }

    /// Execute an operation and capture rollback data for atomic batches.
    /// Returns (result, `rollback_data`) tuple.
    async fn execute_with_rollback(
        &self,
        op: &Operation,
    ) -> (OperationResult, Option<RollbackData>) {
        match op {
            Operation::Create { path, content } => {
                let result = self.create(path, content).await;
                let rollback = if result.success {
                    self.resolve_path(path)
                        .ok()
                        .map(|created_path| RollbackData::Create { created_path })
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Delete { path } => {
                let resolved = match self.resolve_path(path) {
                    Ok(p) => p,
                    Err(_) => return (self.delete(path).await, None),
                };
                // Capture content before delete for rollback (in case of hard delete)
                let content_backup = if resolved.exists() && resolved.is_file() {
                    fs::read(&resolved).ok()
                } else {
                    None
                };

                let result = self.delete(path).await;
                let rollback = if result.success {
                    Some(RollbackData::Delete {
                        original_path: resolved,
                        trash_id: result.trash_id,
                        content_backup: if result.trash_id.is_none() {
                            content_backup
                        } else {
                            None
                        },
                    })
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Move { src, dst } => {
                let result = self.move_file(src, dst).await;
                let rollback = if result.success {
                    match (self.resolve_path(src), self.resolve_path(dst)) {
                        (Ok(resolved_src), Ok(resolved_dst)) => Some(RollbackData::Move {
                            src: resolved_dst,
                            dst: resolved_src,
                        }),
                        _ => None,
                    }
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Copy { src, dst } => {
                let result = self.copy(src, dst).await;
                let rollback = if result.success {
                    self.resolve_path(dst)
                        .ok()
                        .map(|copied_path| RollbackData::Copy { copied_path })
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Write {
                path,
                content,
                append,
            } => {
                let (file_existed, previous_content, resolved) = match self.resolve_path(path) {
                    Ok(resolved) => {
                        let file_existed = resolved.exists();
                        let previous_content = if file_existed {
                            fs::read(&resolved).ok()
                        } else {
                            None
                        };
                        (file_existed, previous_content, Some(resolved))
                    }
                    Err(_) => (false, None, None),
                };

                let result = self.write(path, content, *append).await;
                let rollback = if result.success {
                    resolved.map(|path| RollbackData::Write {
                        path,
                        previous_content,
                        file_existed,
                    })
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Mkdir { path } => {
                let result = self.mkdir(path).await;
                let rollback = if result.success {
                    self.resolve_path(path)
                        .ok()
                        .map(|created_path| RollbackData::Mkdir { created_path })
                } else {
                    None
                };
                (result, rollback)
            }
            Operation::Symlink { target, link } => {
                let result = self.symlink(target, link).await;
                let rollback = if result.success {
                    self.resolve_path(link)
                        .ok()
                        .map(|link_path| RollbackData::Symlink { link_path })
                } else {
                    None
                };
                (result, rollback)
            }
        }
    }

    /// Rollback a single operation.
    async fn rollback_operation(&self, rollback_data: &RollbackData) -> Result<(), String> {
        match rollback_data {
            RollbackData::Create { created_path } => {
                // Undo create by deleting the file
                if created_path.exists() {
                    fs::remove_file(created_path)
                        .map_err(|e| format!("Failed to rollback create: {e}"))?;
                    // Remove from vector store
                    self.delete_from_store(created_path).await;
                }
                Ok(())
            }
            RollbackData::Delete {
                original_path,
                trash_id,
                content_backup,
            } => {
                // Try to restore from trash first
                if let Some(id) = trash_id
                    && let Some(ref safety) = self.safety_manager
                {
                    return safety
                        .restore(*id)
                        .await
                        .map(|_| ())
                        .map_err(|e| format!("Failed to restore from trash: {e}"));
                }
                // Fall back to content backup
                if let Some(content) = content_backup {
                    if let Some(parent) = original_path.parent() {
                        fs::create_dir_all(parent)
                            .map_err(|e| format!("Failed to create parent dir: {e}"))?;
                    }
                    fs::write(original_path, content)
                        .map_err(|e| format!("Failed to restore content: {e}"))?;
                    self.trigger_reindex(original_path).await;
                    Ok(())
                } else {
                    Err("Cannot rollback delete: no trash entry or content backup".into())
                }
            }
            RollbackData::Move { src, dst } => {
                // Move back (src is current location, dst is original location)
                if src.exists() {
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent)
                            .map_err(|e| format!("Failed to create parent dir: {e}"))?;
                    }
                    fs::rename(src, dst).map_err(|e| format!("Failed to rollback move: {e}"))?;
                    self.update_store_path(src, dst).await;
                }
                Ok(())
            }
            RollbackData::Copy { copied_path } => {
                // Delete the copy
                if copied_path.exists() {
                    fs::remove_file(copied_path)
                        .map_err(|e| format!("Failed to rollback copy: {e}"))?;
                    self.delete_from_store(copied_path).await;
                }
                Ok(())
            }
            RollbackData::Write {
                path,
                previous_content,
                file_existed,
            } => {
                if *file_existed {
                    if let Some(content) = previous_content {
                        fs::write(path, content)
                            .map_err(|e| format!("Failed to rollback write: {e}"))?;
                    } else {
                        return Err("Cannot rollback write: no previous content saved".into());
                    }
                } else {
                    // File didn't exist before, delete it
                    if path.exists() {
                        fs::remove_file(path)
                            .map_err(|e| format!("Failed to rollback write (delete): {e}"))?;
                        self.delete_from_store(path).await;
                    }
                }
                self.trigger_reindex(path).await;
                Ok(())
            }
            RollbackData::Mkdir { created_path } => {
                // Undo mkdir by removing the directory (only if empty)
                if created_path.exists() && created_path.is_dir() {
                    fs::remove_dir(created_path)
                        .map_err(|e| format!("Failed to rollback mkdir: {e}"))?;
                }
                Ok(())
            }
            RollbackData::Symlink { link_path } => {
                // Undo symlink by removing the link
                if link_path.exists() || link_path.symlink_metadata().is_ok() {
                    fs::remove_file(link_path)
                        .map_err(|e| format!("Failed to rollback symlink: {e}"))?;
                }
                Ok(())
            }
        }
    }

    /// Perform rollback of journal entries in reverse order.
    async fn perform_rollback(&self, journal: &[JournalEntry]) -> RollbackDetails {
        let mut rolled_back = 0;
        let mut rollback_failures = 0;
        let mut errors = Vec::new();

        // Rollback in reverse order
        for entry in journal.iter().rev() {
            match self.rollback_operation(&entry.rollback_data).await {
                Ok(()) => {
                    rolled_back += 1;
                    info!("Rolled back operation index {}", entry.operation_index);
                }
                Err(e) => {
                    rollback_failures += 1;
                    warn!(
                        "Failed to rollback operation index {}: {}",
                        entry.operation_index, e
                    );
                    errors.push(RollbackError {
                        operation_index: entry.operation_index,
                        error: e,
                    });
                }
            }
        }

        RollbackDetails {
            rolled_back,
            rollback_failures,
            errors,
        }
    }

    /// Create a new file with content.
    pub async fn create(&self, path: &PathBuf, content: &str) -> OperationResult {
        let resolved = match self.resolve_path(path) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("create", path.clone(), e).await,
        };
        debug!("ops::create {:?}", resolved);

        // Check if file already exists
        if resolved.exists() {
            return OperationResult::failure("create", path.clone(), "File already exists".into());
        }

        // Ensure parent directory exists
        if let Some(parent) = resolved.parent()
            && !parent.exists()
            && let Err(e) = fs::create_dir_all(parent)
        {
            return OperationResult::failure(
                "create",
                path.clone(),
                format!("Failed to create parent directory: {e}"),
            );
        }

        // Create the file
        match fs::write(&resolved, content) {
            Ok(()) => {
                info!("Created file: {:?}", resolved);
                let indexed = self.trigger_reindex(&resolved).await;
                let undo_id = self.log_to_history(
                    HistoryOperation::Create {
                        path: resolved.clone(),
                    },
                    Some(UndoData::Create { path: resolved }),
                );
                let mut result = OperationResult::success("create", path.clone(), indexed);
                result.undo_id = undo_id;

                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to create file: {e}");
                self.log_failure_to_history(
                    HistoryOperation::Create { path: resolved },
                    &error_msg,
                );

                let result = OperationResult::failure("create", path.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Delete a file.
    /// Uses soft delete (move to trash) if safety manager is available.
    pub async fn delete(&self, path: &PathBuf) -> OperationResult {
        let resolved = match self.resolve_path(path) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("delete", path.clone(), e).await,
        };
        debug!("ops::delete {:?}", resolved);

        if !resolved.exists() {
            return OperationResult::failure("delete", path.clone(), "File not found".into());
        }

        if resolved.is_dir() {
            return OperationResult::failure(
                "delete",
                path.clone(),
                "Cannot delete directory with .delete, use rmdir".into(),
            );
        }

        // Delete from store first
        self.delete_from_store(&resolved).await;

        // Try soft delete if safety manager is available
        if let Some(ref safety) = self.safety_manager {
            match safety.soft_delete(&resolved).await {
                Ok(entry) => {
                    info!("Soft deleted file: {:?} -> trash/{}", resolved, entry.id);

                    let undo_id = self.log_to_history(
                        HistoryOperation::Delete {
                            path: resolved,
                            trash_id: Some(entry.id),
                        },
                        Some(UndoData::Delete { trash_id: entry.id }),
                    );

                    let mut result = OperationResult::success("delete", path.clone(), false);
                    result.undo_id = undo_id;
                    result.trash_id = Some(entry.id);
                    *self.last_result.write().await = Some(result.clone());
                    return result;
                }
                Err(e) => {
                    warn!("Soft delete failed, falling back to hard delete: {e}");
                }
            }
        }

        // Hard delete (fallback or no safety manager)
        match fs::remove_file(&resolved) {
            Ok(()) => {
                info!("Hard deleted file: {:?}", resolved);

                let undo_id = self.log_to_history(
                    HistoryOperation::Delete {
                        path: resolved,
                        trash_id: None,
                    },
                    None, // Hard delete is not reversible
                );

                let mut result = OperationResult::success("delete", path.clone(), false);
                result.undo_id = undo_id;
                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to delete file: {e}");
                self.log_failure_to_history(
                    HistoryOperation::Delete {
                        path: resolved,
                        trash_id: None,
                    },
                    &error_msg,
                );

                let result = OperationResult::failure("delete", path.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Move/rename a file.
    pub async fn move_file(&self, src: &PathBuf, dst: &PathBuf) -> OperationResult {
        let resolved_src = match self.resolve_path(src) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("move", src.clone(), e).await,
        };
        let resolved_dst = match self.resolve_path(dst) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("move", src.clone(), e).await,
        };
        debug!("ops::move {:?} -> {:?}", resolved_src, resolved_dst);

        if !resolved_src.exists() {
            return OperationResult::failure("move", src.clone(), "Source file not found".into());
        }

        if resolved_dst.exists() {
            return OperationResult::failure(
                "move",
                src.clone(),
                "Destination already exists".into(),
            );
        }

        // Ensure parent directory exists
        if let Some(parent) = resolved_dst.parent()
            && !parent.exists()
            && let Err(e) = fs::create_dir_all(parent)
        {
            return OperationResult::failure(
                "move",
                src.clone(),
                format!("Failed to create parent directory: {e}"),
            );
        }

        // Move the file
        match fs::rename(&resolved_src, &resolved_dst) {
            Ok(()) => {
                info!("Moved: {:?} -> {:?}", resolved_src, resolved_dst);
                self.update_store_path(&resolved_src, &resolved_dst).await;

                let undo_id = self.log_to_history(
                    HistoryOperation::Move {
                        src: resolved_src.clone(),
                        dst: resolved_dst.clone(),
                    },
                    Some(UndoData::Move {
                        src: resolved_src,
                        dst: resolved_dst,
                    }),
                );

                let mut result = OperationResult::success("move", dst.clone(), true);
                result.undo_id = undo_id;
                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to move file: {e}");
                self.log_failure_to_history(
                    HistoryOperation::Move {
                        src: resolved_src,
                        dst: resolved_dst,
                    },
                    &error_msg,
                );

                let result = OperationResult::failure("move", src.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Copy a file.
    pub async fn copy(&self, src: &PathBuf, dst: &PathBuf) -> OperationResult {
        let resolved_src = match self.resolve_path(src) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("copy", src.clone(), e).await,
        };
        let resolved_dst = match self.resolve_path(dst) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("copy", src.clone(), e).await,
        };
        debug!("ops::copy {:?} -> {:?}", resolved_src, resolved_dst);

        if !resolved_src.exists() {
            return OperationResult::failure("copy", src.clone(), "Source file not found".into());
        }

        if resolved_dst.exists() {
            return OperationResult::failure(
                "copy",
                src.clone(),
                "Destination already exists".into(),
            );
        }

        // Ensure parent directory exists
        if let Some(parent) = resolved_dst.parent()
            && !parent.exists()
            && let Err(e) = fs::create_dir_all(parent)
        {
            return OperationResult::failure(
                "copy",
                src.clone(),
                format!("Failed to create parent directory: {e}"),
            );
        }

        // Copy the file
        match fs::copy(&resolved_src, &resolved_dst) {
            Ok(_) => {
                info!("Copied: {:?} -> {:?}", resolved_src, resolved_dst);
                let indexed = self.trigger_reindex(&resolved_dst).await;

                let undo_id = self.log_to_history(
                    HistoryOperation::Copy {
                        src: resolved_src,
                        dst: resolved_dst.clone(),
                    },
                    Some(UndoData::Copy { path: resolved_dst }),
                );

                let mut result = OperationResult::success("copy", dst.clone(), indexed);
                result.undo_id = undo_id;
                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to copy file: {e}");
                self.log_failure_to_history(
                    HistoryOperation::Copy {
                        src: resolved_src,
                        dst: resolved_dst,
                    },
                    &error_msg,
                );

                let result = OperationResult::failure("copy", src.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Write content to a file.
    pub async fn write(&self, path: &PathBuf, content: &str, append: bool) -> OperationResult {
        let resolved = match self.resolve_path(path) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("write", path.clone(), e).await,
        };
        debug!("ops::write {:?} (append={})", resolved, append);

        let write_result = if append {
            use std::io::Write;
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&resolved)
                .and_then(|mut f| f.write_all(content.as_bytes()))
        } else {
            fs::write(&resolved, content)
        };

        match write_result {
            Ok(()) => {
                info!("Wrote to file: {:?}", resolved);
                let indexed = self.trigger_reindex(&resolved).await;

                let undo_id = self.log_to_history(
                    HistoryOperation::Write {
                        path: resolved,
                        append,
                    },
                    None, // No undo data - would need to store previous content
                );

                let mut result = OperationResult::success("write", path.clone(), indexed);
                result.undo_id = undo_id;
                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to write file: {e}");
                self.log_failure_to_history(
                    HistoryOperation::Write {
                        path: resolved,
                        append,
                    },
                    &error_msg,
                );

                let result = OperationResult::failure("write", path.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Create a directory.
    pub async fn mkdir(&self, path: &PathBuf) -> OperationResult {
        let resolved = match self.resolve_path(path) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("mkdir", path.clone(), e).await,
        };
        debug!("ops::mkdir {:?}", resolved);

        if resolved.exists() {
            return OperationResult::failure("mkdir", path.clone(), "Path already exists".into());
        }

        match fs::create_dir_all(&resolved) {
            Ok(()) => {
                info!("Created directory: {:?}", resolved);

                // Note: Directories are not indexed; no HistoryOperation for mkdir.
                let result = OperationResult::success("mkdir", path.clone(), false);

                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to create directory: {e}");
                let result = OperationResult::failure("mkdir", path.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Create a symbolic link.
    #[cfg(unix)]
    pub async fn symlink(&self, target: &PathBuf, link: &PathBuf) -> OperationResult {
        let resolved_link = match self.resolve_path(link) {
            Ok(p) => p,
            Err(e) => return self.fail_and_store("symlink", link.clone(), e).await,
        };
        if let Err(e) = self.jail_symlink_target(target, &resolved_link) {
            return self.fail_and_store("symlink", link.clone(), e).await;
        }
        debug!("ops::symlink {:?} -> {:?}", resolved_link, target);

        if resolved_link.exists() {
            return OperationResult::failure(
                "symlink",
                link.clone(),
                "Link path already exists".into(),
            );
        }

        // Ensure parent directory exists
        if let Some(parent) = resolved_link.parent()
            && !parent.exists()
            && let Err(e) = fs::create_dir_all(parent)
        {
            return OperationResult::failure(
                "symlink",
                link.clone(),
                format!("Failed to create parent directory: {e}"),
            );
        }

        // Pass the original target so a relative link stays relative.
        match std::os::unix::fs::symlink(target, &resolved_link) {
            Ok(()) => {
                info!("Created symlink: {:?} -> {:?}", resolved_link, target);

                let result = OperationResult::success("symlink", link.clone(), false);

                *self.last_result.write().await = Some(result.clone());
                result
            }
            Err(e) => {
                let error_msg = format!("Failed to create symlink: {e}");
                let result = OperationResult::failure("symlink", link.clone(), error_msg);
                *self.last_result.write().await = Some(result.clone());
                result
            }
        }
    }

    /// Create a symbolic link (non-Unix fallback - not supported).
    #[cfg(not(unix))]
    pub async fn symlink(&self, _target: &PathBuf, link: &PathBuf) -> OperationResult {
        OperationResult::failure(
            "symlink",
            link.clone(),
            "Symlinks are not supported on this platform".into(),
        )
    }

    /// Execute a single operation.
    pub async fn execute_operation(&self, op: &Operation) -> OperationResult {
        match op {
            Operation::Create { path, content } => self.create(path, content).await,
            Operation::Delete { path } => self.delete(path).await,
            Operation::Move { src, dst } => self.move_file(src, dst).await,
            Operation::Copy { src, dst } => self.copy(src, dst).await,
            Operation::Write {
                path,
                content,
                append,
            } => self.write(path, content, *append).await,
            Operation::Mkdir { path } => self.mkdir(path).await,
            Operation::Symlink { target, link } => self.symlink(target, link).await,
        }
    }

    /// Execute a batch of operations.
    pub async fn batch(&self, request: BatchRequest) -> BatchResult {
        let batch_id = Uuid::new_v4();
        let total = request.operations.len();
        let mut results = Vec::with_capacity(total);
        let mut succeeded = 0;
        let mut failed = 0;
        let mut journal: Vec<JournalEntry> = Vec::new();

        debug!(
            "ops::batch {} operations (atomic={}, dry_run={})",
            total, request.atomic, request.dry_run
        );

        if request.dry_run {
            // Just validate operations without executing
            for op in &request.operations {
                let result = self.validate_operation(op);
                if result.success {
                    succeeded += 1;
                } else {
                    failed += 1;
                }
                results.push(result);
            }

            let batch_result = BatchResult {
                id: batch_id,
                success: failed == 0,
                total,
                succeeded,
                failed,
                results,
                timestamp: Utc::now(),
                rollback_performed: None,
                rollback_details: None,
            };

            *self.last_batch_result.write().await = Some(batch_result.clone());
            return batch_result;
        }

        // Execute operations with journal for atomic rollback
        for (index, op) in request.operations.iter().enumerate() {
            let (result, rollback_data) = self.execute_with_rollback(op).await;

            if result.success {
                succeeded += 1;
                // Record in journal for potential rollback
                if request.atomic
                    && let Some(rd) = rollback_data
                {
                    journal.push(JournalEntry {
                        operation_index: index,
                        rollback_data: rd,
                    });
                }
                results.push(result);
            } else {
                failed += 1;
                results.push(result);

                if request.atomic && !journal.is_empty() {
                    // Perform rollback of all previously successful operations
                    info!(
                        "Batch failed at operation {}, rolling back {} operations",
                        index,
                        journal.len()
                    );
                    let rollback_details = self.perform_rollback(&journal).await;
                    let rollback_success = rollback_details.rollback_failures == 0;

                    let batch_result = BatchResult {
                        id: batch_id,
                        success: false,
                        total,
                        succeeded,
                        failed,
                        results,
                        timestamp: Utc::now(),
                        rollback_performed: Some(rollback_success),
                        rollback_details: Some(rollback_details),
                    };

                    *self.last_batch_result.write().await = Some(batch_result.clone());
                    return batch_result;
                } else if request.atomic {
                    // First operation failed, no rollback needed
                    let batch_result = BatchResult {
                        id: batch_id,
                        success: false,
                        total,
                        succeeded,
                        failed,
                        results,
                        timestamp: Utc::now(),
                        rollback_performed: None,
                        rollback_details: None,
                    };

                    *self.last_batch_result.write().await = Some(batch_result.clone());
                    return batch_result;
                }
                // Non-atomic: continue with next operation
            }
        }

        let batch_result = BatchResult {
            id: batch_id,
            success: failed == 0,
            total,
            succeeded,
            failed,
            results,
            timestamp: Utc::now(),
            rollback_performed: None,
            rollback_details: None,
        };

        *self.last_batch_result.write().await = Some(batch_result.clone());
        batch_result
    }

    /// Validate an operation without executing it.
    fn validate_operation(&self, op: &Operation) -> OperationResult {
        match op {
            Operation::Create { path, .. } => {
                let resolved = match self.resolve_path(path) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("create", path.clone(), e),
                };
                if resolved.exists() {
                    OperationResult::failure("create", path.clone(), "File already exists".into())
                } else {
                    OperationResult::success("create", path.clone(), false)
                }
            }
            Operation::Delete { path } => {
                let resolved = match self.resolve_path(path) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("delete", path.clone(), e),
                };
                if !resolved.exists() {
                    OperationResult::failure("delete", path.clone(), "File not found".into())
                } else if resolved.is_dir() {
                    OperationResult::failure(
                        "delete",
                        path.clone(),
                        "Cannot delete directory".into(),
                    )
                } else {
                    OperationResult::success("delete", path.clone(), false)
                }
            }
            Operation::Move { src, dst } => {
                let resolved_src = match self.resolve_path(src) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("move", src.clone(), e),
                };
                let resolved_dst = match self.resolve_path(dst) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("move", src.clone(), e),
                };
                if !resolved_src.exists() {
                    OperationResult::failure("move", src.clone(), "Source not found".into())
                } else if resolved_dst.exists() {
                    OperationResult::failure(
                        "move",
                        src.clone(),
                        "Destination already exists".into(),
                    )
                } else {
                    OperationResult::success("move", dst.clone(), false)
                }
            }
            Operation::Copy { src, dst } => {
                let resolved_src = match self.resolve_path(src) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("copy", src.clone(), e),
                };
                let resolved_dst = match self.resolve_path(dst) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("copy", src.clone(), e),
                };
                if !resolved_src.exists() {
                    OperationResult::failure("copy", src.clone(), "Source not found".into())
                } else if resolved_dst.exists() {
                    OperationResult::failure(
                        "copy",
                        src.clone(),
                        "Destination already exists".into(),
                    )
                } else {
                    OperationResult::success("copy", dst.clone(), false)
                }
            }
            Operation::Write { path, .. } => match self.resolve_path(path) {
                Ok(_) => OperationResult::success("write", path.clone(), false),
                Err(e) => OperationResult::failure("write", path.clone(), e),
            },
            Operation::Mkdir { path } => {
                let resolved = match self.resolve_path(path) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("mkdir", path.clone(), e),
                };
                if resolved.exists() {
                    OperationResult::failure("mkdir", path.clone(), "Path already exists".into())
                } else {
                    OperationResult::success("mkdir", path.clone(), false)
                }
            }
            Operation::Symlink { target, link } => {
                let resolved_link = match self.resolve_path(link) {
                    Ok(p) => p,
                    Err(e) => return OperationResult::failure("symlink", link.clone(), e),
                };
                if let Err(e) = self.jail_symlink_target(target, &resolved_link) {
                    return OperationResult::failure("symlink", link.clone(), e);
                }
                if resolved_link.exists() {
                    OperationResult::failure(
                        "symlink",
                        link.clone(),
                        "Link path already exists".into(),
                    )
                } else {
                    OperationResult::success("symlink", link.clone(), false)
                }
            }
        }
    }

    /// Get the last operation result as JSON.
    pub async fn get_last_result(&self) -> Vec<u8> {
        let result = self.last_result.read().await;
        if let Some(r) = &*result {
            serde_json::to_string_pretty(r)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
        } else {
            let empty = serde_json::json!({
                "message": "No operations performed yet"
            });
            serde_json::to_string_pretty(&empty)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Get the last batch result as JSON.
    #[allow(dead_code)]
    pub async fn get_last_batch_result(&self) -> Vec<u8> {
        let result = self.last_batch_result.read().await;
        if let Some(r) = &*result {
            serde_json::to_string_pretty(r)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
        } else {
            let empty = serde_json::json!({
                "message": "No batch operations performed yet"
            });
            serde_json::to_string_pretty(&empty)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Parse and execute a create operation from string input.
    /// Format: "path\ncontent"
    pub async fn parse_and_create(&self, input: &str) -> OperationResult {
        let parts: Vec<&str> = input.splitn(2, '\n').collect();
        if parts.len() < 2 {
            return OperationResult::failure(
                "create",
                PathBuf::new(),
                "Invalid format. Expected: path\\ncontent".into(),
            );
        }
        let path = PathBuf::from(parts[0].trim());
        let content = parts[1];
        self.create(&path, content).await
    }

    /// Parse and execute a delete operation from string input.
    /// Format: "path"
    pub async fn parse_and_delete(&self, input: &str) -> OperationResult {
        let path = PathBuf::from(input.trim());
        if path.as_os_str().is_empty() {
            return OperationResult::failure(
                "delete",
                PathBuf::new(),
                "Invalid format. Expected: path".into(),
            );
        }
        self.delete(&path).await
    }

    /// Parse and execute a move operation from string input.
    /// Format: "src\ndst"
    pub async fn parse_and_move(&self, input: &str) -> OperationResult {
        let parts: Vec<&str> = input.splitn(2, '\n').collect();
        if parts.len() < 2 {
            return OperationResult::failure(
                "move",
                PathBuf::new(),
                "Invalid format. Expected: src\\ndst".into(),
            );
        }
        let src = PathBuf::from(parts[0].trim());
        let dst = PathBuf::from(parts[1].trim());
        self.move_file(&src, &dst).await
    }

    /// Parse and execute a batch operation from JSON input.
    pub async fn parse_and_batch(&self, input: &str) -> BatchResult {
        match serde_json::from_str::<BatchRequest>(input) {
            Ok(request) => self.batch(request).await,
            Err(e) => BatchResult {
                id: Uuid::new_v4(),
                success: false,
                total: 0,
                succeeded: 0,
                failed: 1,
                results: vec![OperationResult::failure(
                    "batch",
                    PathBuf::new(),
                    format!("Invalid JSON: {e}"),
                )],
                timestamp: Utc::now(),
                rollback_performed: None,
                rollback_details: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_manager() -> (OpsManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let manager = OpsManager::new(temp_dir.path().to_path_buf(), None, None);
        (manager, temp_dir)
    }

    #[tokio::test]
    async fn test_create_file() {
        let (manager, _temp) = create_test_manager();
        let path = PathBuf::from("test.txt");

        let result = manager.create(&path, "Hello, World!").await;

        assert!(result.success);
        assert_eq!(result.operation, "create");
        assert!(manager.resolve_path(&path).unwrap().exists());
    }

    #[tokio::test]
    async fn test_create_file_already_exists() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("existing.txt");
        fs::write(temp.path().join("existing.txt"), "content").unwrap();

        let result = manager.create(&path, "new content").await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("already exists"));
    }

    #[tokio::test]
    async fn test_delete_file() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("to_delete.txt");
        fs::write(temp.path().join("to_delete.txt"), "content").unwrap();

        let result = manager.delete(&path).await;

        assert!(result.success);
        assert!(!manager.resolve_path(&path).unwrap().exists());
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let (manager, _temp) = create_test_manager();
        let path = PathBuf::from("nonexistent.txt");

        let result = manager.delete(&path).await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn test_move_file() {
        let (manager, temp) = create_test_manager();
        let src = PathBuf::from("source.txt");
        let dst = PathBuf::from("dest.txt");
        fs::write(temp.path().join("source.txt"), "content").unwrap();

        let result = manager.move_file(&src, &dst).await;

        assert!(result.success);
        assert!(!manager.resolve_path(&src).unwrap().exists());
        assert!(manager.resolve_path(&dst).unwrap().exists());
    }

    #[tokio::test]
    async fn test_copy_file() {
        let (manager, temp) = create_test_manager();
        let src = PathBuf::from("original.txt");
        let dst = PathBuf::from("copy.txt");
        fs::write(temp.path().join("original.txt"), "content").unwrap();

        let result = manager.copy(&src, &dst).await;

        assert!(result.success);
        assert!(manager.resolve_path(&src).unwrap().exists());
        assert!(manager.resolve_path(&dst).unwrap().exists());
    }

    #[tokio::test]
    async fn test_write_file() {
        let (manager, _temp) = create_test_manager();
        let path = PathBuf::from("write_test.txt");

        let result = manager.write(&path, "content", false).await;

        assert!(result.success);
        let content = fs::read_to_string(manager.resolve_path(&path).unwrap()).unwrap();
        assert_eq!(content, "content");
    }

    #[tokio::test]
    async fn test_write_append() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("append_test.txt");
        fs::write(temp.path().join("append_test.txt"), "first").unwrap();

        let result = manager.write(&path, " second", true).await;

        assert!(result.success);
        let content = fs::read_to_string(manager.resolve_path(&path).unwrap()).unwrap();
        assert_eq!(content, "first second");
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let (manager, _temp) = create_test_manager();

        let request = BatchRequest {
            operations: vec![
                Operation::Create {
                    path: PathBuf::from("file1.txt"),
                    content: "content1".to_string(),
                },
                Operation::Create {
                    path: PathBuf::from("file2.txt"),
                    content: "content2".to_string(),
                },
            ],
            atomic: false,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(result.success);
        assert_eq!(result.total, 2);
        assert_eq!(result.succeeded, 2);
        assert_eq!(result.failed, 0);
    }

    #[tokio::test]
    async fn test_batch_dry_run() {
        let (manager, _temp) = create_test_manager();

        let request = BatchRequest {
            operations: vec![Operation::Create {
                path: PathBuf::from("dry_run.txt"),
                content: "content".to_string(),
            }],
            atomic: false,
            dry_run: true,
        };

        let result = manager.batch(request).await;

        assert!(result.success);
        // File should NOT be created in dry run
        assert!(
            !manager
                .resolve_path(&PathBuf::from("dry_run.txt"))
                .unwrap()
                .exists()
        );
    }

    #[tokio::test]
    async fn test_parse_and_create() {
        let (manager, _temp) = create_test_manager();

        let result = manager.parse_and_create("test.txt\nHello!").await;

        assert!(result.success);
        let content =
            fs::read_to_string(manager.resolve_path(&PathBuf::from("test.txt")).unwrap()).unwrap();
        assert_eq!(content, "Hello!");
    }

    #[tokio::test]
    async fn test_parse_and_move() {
        let (manager, temp) = create_test_manager();
        fs::write(temp.path().join("src.txt"), "content").unwrap();

        let result = manager.parse_and_move("src.txt\ndst.txt").await;

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_parse_and_batch() {
        let (manager, _temp) = create_test_manager();

        let json = r#"{"operations":[{"op":"create","path":"batch.txt","content":"test"}]}"#;
        let result = manager.parse_and_batch(json).await;

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_last_result() {
        let (manager, _temp) = create_test_manager();

        // Initially no result
        let initial = manager.get_last_result().await;
        let initial_str = String::from_utf8(initial).unwrap();
        assert!(initial_str.contains("No operations"));

        // After an operation
        manager.create(&PathBuf::from("test.txt"), "content").await;
        let after = manager.get_last_result().await;
        let after_str = String::from_utf8(after).unwrap();
        assert!(after_str.contains("create"));
        assert!(after_str.contains("success"));
    }

    #[test]
    fn test_operation_result_success() {
        let result = OperationResult::success("test", PathBuf::from("/test"), true);
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(
            result.undo_id.is_none(),
            "undo_id is assigned from SafetyManager history, not generated here"
        );
    }

    #[test]
    fn test_operation_result_failure() {
        let result = OperationResult::failure("test", PathBuf::from("/test"), "error".into());
        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.undo_id.is_none());
    }

    #[tokio::test]
    async fn test_atomic_batch_rollback_on_failure() {
        let (manager, temp) = create_test_manager();

        // Create a file that will cause the second create to fail
        fs::write(temp.path().join("existing.txt"), "exists").unwrap();

        let request = BatchRequest {
            operations: vec![
                Operation::Create {
                    path: PathBuf::from("new.txt"),
                    content: "content".to_string(),
                },
                Operation::Create {
                    path: PathBuf::from("existing.txt"), // Will fail
                    content: "content".to_string(),
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.succeeded, 1);
        assert_eq!(result.failed, 1);
        assert_eq!(result.rollback_performed, Some(true));
        assert!(result.rollback_details.is_some());
        let details = result.rollback_details.unwrap();
        assert_eq!(details.rolled_back, 1);
        assert_eq!(details.rollback_failures, 0);
        // First file should be rolled back (deleted)
        assert!(!temp.path().join("new.txt").exists());
    }

    #[tokio::test]
    async fn test_atomic_batch_rollback_move_operations() {
        let (manager, temp) = create_test_manager();

        fs::write(temp.path().join("file1.txt"), "content1").unwrap();

        let request = BatchRequest {
            operations: vec![
                Operation::Move {
                    src: PathBuf::from("file1.txt"),
                    dst: PathBuf::from("moved1.txt"),
                },
                Operation::Delete {
                    path: PathBuf::from("nonexistent.txt"), // Will fail
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.rollback_performed, Some(true));
        // Move should be rolled back
        assert!(temp.path().join("file1.txt").exists());
        assert!(!temp.path().join("moved1.txt").exists());
    }

    #[tokio::test]
    async fn test_atomic_batch_rollback_write_restores_content() {
        let (manager, temp) = create_test_manager();

        fs::write(temp.path().join("existing.txt"), "original content").unwrap();

        let request = BatchRequest {
            operations: vec![
                Operation::Write {
                    path: PathBuf::from("existing.txt"),
                    content: "modified content".to_string(),
                    append: false,
                },
                Operation::Delete {
                    path: PathBuf::from("nonexistent.txt"), // Will fail
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.rollback_performed, Some(true));
        // Content should be restored
        let content = fs::read_to_string(temp.path().join("existing.txt")).unwrap();
        assert_eq!(content, "original content");
    }

    #[tokio::test]
    async fn test_non_atomic_batch_no_rollback() {
        let (manager, temp) = create_test_manager();

        let request = BatchRequest {
            operations: vec![
                Operation::Create {
                    path: PathBuf::from("file1.txt"),
                    content: "content".to_string(),
                },
                Operation::Delete {
                    path: PathBuf::from("nonexistent.txt"), // Will fail
                },
                Operation::Create {
                    path: PathBuf::from("file2.txt"),
                    content: "content".to_string(),
                },
            ],
            atomic: false, // Non-atomic
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert!(result.rollback_performed.is_none());
        // Both files should exist (no rollback)
        assert!(temp.path().join("file1.txt").exists());
        assert!(temp.path().join("file2.txt").exists());
    }

    #[tokio::test]
    async fn test_atomic_batch_first_op_fails_no_rollback_needed() {
        let (manager, temp) = create_test_manager();

        // Create a file that will cause the first create to fail
        fs::write(temp.path().join("existing.txt"), "exists").unwrap();

        let request = BatchRequest {
            operations: vec![
                Operation::Create {
                    path: PathBuf::from("existing.txt"), // Will fail immediately
                    content: "content".to_string(),
                },
                Operation::Create {
                    path: PathBuf::from("new.txt"),
                    content: "content".to_string(),
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.succeeded, 0);
        assert_eq!(result.failed, 1);
        // No rollback needed since first operation failed
        assert!(result.rollback_performed.is_none());
        assert!(result.rollback_details.is_none());
        // Second operation should not have been attempted
        assert!(!temp.path().join("new.txt").exists());
    }

    #[tokio::test]
    async fn test_atomic_batch_copy_rollback() {
        let (manager, temp) = create_test_manager();

        fs::write(temp.path().join("source.txt"), "source content").unwrap();

        let request = BatchRequest {
            operations: vec![
                Operation::Copy {
                    src: PathBuf::from("source.txt"),
                    dst: PathBuf::from("copied.txt"),
                },
                Operation::Delete {
                    path: PathBuf::from("nonexistent.txt"), // Will fail
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.rollback_performed, Some(true));
        // Copy should be rolled back (deleted)
        assert!(temp.path().join("source.txt").exists());
        assert!(!temp.path().join("copied.txt").exists());
    }

    #[tokio::test]
    async fn test_mkdir() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("new_directory");

        let result = manager.mkdir(&path).await;

        assert!(result.success);
        assert_eq!(result.operation, "mkdir");
        assert!(temp.path().join("new_directory").is_dir());
    }

    #[tokio::test]
    async fn test_mkdir_nested() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("parent/child/grandchild");

        let result = manager.mkdir(&path).await;

        assert!(result.success);
        assert!(temp.path().join("parent/child/grandchild").is_dir());
    }

    #[tokio::test]
    async fn test_mkdir_already_exists() {
        let (manager, temp) = create_test_manager();
        let path = PathBuf::from("existing_dir");
        fs::create_dir(temp.path().join("existing_dir")).unwrap();

        let result = manager.mkdir(&path).await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("already exists"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_symlink() {
        let (manager, temp) = create_test_manager();
        let target = PathBuf::from("target_file.txt");
        let link = PathBuf::from("link_to_target");

        // Create target file
        fs::write(temp.path().join("target_file.txt"), "content").unwrap();

        let result = manager.symlink(&target, &link).await;

        assert!(result.success);
        assert_eq!(result.operation, "symlink");
        let link_path = temp.path().join("link_to_target");
        assert!(link_path.symlink_metadata().is_ok());
        assert_eq!(
            std::fs::read_link(&link_path).unwrap(),
            PathBuf::from("target_file.txt")
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_symlink_relative_target_from_link_parent() {
        let (manager, temp) = create_test_manager();
        fs::create_dir(temp.path().join("sub")).unwrap();
        fs::write(temp.path().join("ok.txt"), "ok").unwrap();

        let target = PathBuf::from("../ok.txt");
        let link = PathBuf::from("sub/link");
        let result = manager.symlink(&target, &link).await;

        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            std::fs::read_link(temp.path().join("sub/link")).unwrap(),
            PathBuf::from("../ok.txt")
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_symlink_rejects_parent_relative_escape_via_existing_link() {
        let (manager, temp) = create_test_manager();
        let outside = TempDir::new().unwrap();
        fs::create_dir(temp.path().join("sub")).unwrap();
        std::os::unix::fs::symlink(outside.path(), temp.path().join("sub/out")).unwrap();

        let target = PathBuf::from("out/secret.txt");
        let link = PathBuf::from("sub/link");
        let result = manager.symlink(&target, &link).await;

        assert!(!result.success);
        assert!(
            result.error.as_ref().unwrap().contains("escapes"),
            "{:?}",
            result.error
        );
        assert!(!temp.path().join("sub/link").exists());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_symlink_already_exists() {
        let (manager, temp) = create_test_manager();
        let target = PathBuf::from("target.txt");
        let link = PathBuf::from("existing_link");

        // Create existing link path
        fs::write(temp.path().join("existing_link"), "content").unwrap();

        let result = manager.symlink(&target, &link).await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("already exists"));
    }

    #[tokio::test]
    async fn test_atomic_batch_mkdir_rollback() {
        let (manager, temp) = create_test_manager();

        let request = BatchRequest {
            operations: vec![
                Operation::Mkdir {
                    path: PathBuf::from("new_dir"),
                },
                Operation::Delete {
                    path: PathBuf::from("nonexistent.txt"), // Will fail
                },
            ],
            atomic: true,
            dry_run: false,
        };

        let result = manager.batch(request).await;

        assert!(!result.success);
        assert_eq!(result.rollback_performed, Some(true));
        // Directory should be rolled back (removed)
        assert!(!temp.path().join("new_dir").exists());
    }

    fn create_test_manager_with_safety() -> (OpsManager, Arc<SafetyManager>, TempDir, TempDir) {
        let source_dir = TempDir::new().unwrap();
        let data_dir = TempDir::new().unwrap();
        let safety = Arc::new(SafetyManager::new(
            &source_dir.path().to_path_buf(),
            Some(crate::safety::SafetyConfig {
                data_dir: data_dir.path().to_path_buf(),
                trash_retention_days: 7,
                soft_delete: true,
            }),
        ));
        let manager = OpsManager::with_safety(
            source_dir.path().to_path_buf(),
            None,
            None,
            Arc::clone(&safety),
        );
        (manager, safety, source_dir, data_dir)
    }

    #[tokio::test]
    async fn test_resolve_path_rejects_relative_escape() {
        let (manager, _temp) = create_test_manager();
        let err = manager
            .resolve_path(&PathBuf::from("../escape.txt"))
            .unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[tokio::test]
    async fn test_resolve_path_rejects_absolute_outside_root() {
        let (manager, _temp) = create_test_manager();
        let outside = TempDir::new().unwrap();
        let err = manager
            .resolve_path(&outside.path().join("secret.txt"))
            .unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[tokio::test]
    async fn test_create_rejects_path_jail_escape() {
        let (manager, temp) = create_test_manager();
        let result = manager
            .create(&PathBuf::from("../jailbreak.txt"), "nope")
            .await;
        assert!(!result.success);
        assert!(result.error.unwrap().contains("escapes"));
        assert!(!temp.path().parent().unwrap().join("jailbreak.txt").exists());
    }

    #[tokio::test]
    async fn test_create_allows_absolute_path_inside_root() {
        let (manager, temp) = create_test_manager();
        let inside = temp.path().join("inside.txt");
        let result = manager.create(&inside, "ok").await;
        assert!(result.success);
        assert!(inside.exists());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_create_rejects_symlink_escape() {
        let (manager, temp) = create_test_manager();
        let outside = TempDir::new().unwrap();
        std::os::unix::fs::symlink(outside.path(), temp.path().join("out")).unwrap();

        let result = manager
            .create(&PathBuf::from("out/evil.txt"), "pwned")
            .await;
        assert!(!result.success);
        assert!(result.error.unwrap().contains("escapes"));
        assert!(!outside.path().join("evil.txt").exists());
    }

    #[tokio::test]
    async fn test_create_undo_id_is_history_id_and_undo_uses_trash() {
        let (manager, safety, source_dir, _data) = create_test_manager_with_safety();
        let path = PathBuf::from("created.txt");

        let result = manager.create(&path, "hello undo").await;
        assert!(result.success);
        let undo_id = result.undo_id.expect("create must expose history undo_id");

        let history = safety.find_operation(undo_id).expect("history entry");
        assert_eq!(history.id, undo_id);
        assert!(history.reversible);

        let file = source_dir.path().join("created.txt");
        assert!(file.exists());

        let msg = safety.undo(undo_id).await.unwrap();
        assert!(msg.contains("trash"), "{msg}");
        assert!(!file.exists(), "undo create must not permanently delete");

        let trash = safety.list_trash().await;
        assert_eq!(trash.len(), 1);
        let content = safety.get_trash_content(trash[0].id).unwrap();
        assert_eq!(content, b"hello undo");
    }

    #[tokio::test]
    async fn test_delete_undo_id_is_history_id_not_trash_id() {
        let (manager, safety, source_dir, _data) = create_test_manager_with_safety();
        let path = PathBuf::from("to_soft_delete.txt");
        fs::write(source_dir.path().join("to_soft_delete.txt"), "keep me").unwrap();

        let result = manager.delete(&path).await;
        assert!(result.success);
        let undo_id = result.undo_id.expect("delete must expose history undo_id");

        let history = safety.find_operation(undo_id).expect("history entry");
        assert_eq!(history.id, undo_id);
        let trash_id = match history.undo_data {
            Some(UndoData::Delete { trash_id }) => trash_id,
            other => panic!("expected delete undo data, got {other:?}"),
        };
        assert_ne!(
            undo_id, trash_id,
            "undo_id must be the history id, not the trash id"
        );
        assert!(!source_dir.path().join("to_soft_delete.txt").exists());

        safety.undo(undo_id).await.unwrap();
        assert!(source_dir.path().join("to_soft_delete.txt").exists());
        let content = fs::read_to_string(source_dir.path().join("to_soft_delete.txt")).unwrap();
        assert_eq!(content, "keep me");
    }

    #[tokio::test]
    async fn test_move_undo_id_is_history_id() {
        let (manager, safety, source_dir, _data) = create_test_manager_with_safety();
        fs::write(source_dir.path().join("src.txt"), "moved").unwrap();

        let result = manager
            .move_file(&PathBuf::from("src.txt"), &PathBuf::from("dst.txt"))
            .await;
        assert!(result.success);
        let undo_id = result.undo_id.expect("move must expose history undo_id");
        assert_eq!(safety.find_operation(undo_id).unwrap().id, undo_id);

        safety.undo(undo_id).await.unwrap();
        assert!(source_dir.path().join("src.txt").exists());
        assert!(!source_dir.path().join("dst.txt").exists());
    }

    #[tokio::test]
    async fn test_result_json_omits_internal_trash_id() {
        let (manager, _safety, source_dir, _data) = create_test_manager_with_safety();
        fs::write(source_dir.path().join("gone.txt"), "x").unwrap();
        manager.delete(&PathBuf::from("gone.txt")).await;
        let json = String::from_utf8(manager.get_last_result().await).unwrap();
        assert!(json.contains("undo_id"));
        assert!(!json.contains("trash_id"));
    }
}
