//! Safety layer for agent file operations.
//!
//! This module provides protection against destructive operations through:
//! - Soft delete with trash (files can be recovered)
//! - Audit history logging
//! - Undo support for reversible operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Entry in the trash directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrashEntry {
    /// Unique identifier for this trash entry
    pub id: Uuid,
    /// Original path of the file
    pub original_path: PathBuf,
    /// Path in trash storage
    pub trash_path: PathBuf,
    /// When the file was deleted
    pub deleted_at: DateTime<Utc>,
    /// When the trash entry expires (auto-purge)
    pub expires_at: DateTime<Utc>,
    /// Blake3 hash of the content
    pub content_hash: String,
    /// Original file size in bytes
    pub size: u64,
}

/// Type of operation for history logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryOperation {
    Create { path: PathBuf },
    Delete { path: PathBuf, trash_id: Option<Uuid> },
    Move { src: PathBuf, dst: PathBuf },
    Copy { src: PathBuf, dst: PathBuf },
    Write { path: PathBuf, append: bool },
    Restore { trash_id: Uuid, path: PathBuf },
}

/// Entry in the history log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Unique identifier for this operation
    pub id: Uuid,
    /// Type of operation
    pub operation: HistoryOperation,
    /// When the operation occurred
    pub timestamp: DateTime<Utc>,
    /// Whether the operation succeeded
    pub success: bool,
    /// Whether this operation can be undone
    pub reversible: bool,
    /// Data needed to undo this operation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub undo_data: Option<UndoData>,
    /// Error message if the operation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Data needed to undo an operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoData {
    /// Undo a create by deleting the file
    Create { path: PathBuf },
    /// Undo a delete by restoring from trash
    Delete { trash_id: Uuid },
    /// Undo a move by moving back
    Move { src: PathBuf, dst: PathBuf },
    /// Undo a copy by deleting the copy
    Copy { path: PathBuf },
}

/// Configuration for the safety manager.
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Base directory for safety data
    pub data_dir: PathBuf,
    /// How long to keep trash entries (in days)
    pub trash_retention_days: u32,
    /// Whether to enable soft delete (move to trash instead of delete)
    pub soft_delete: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ragfs");

        Self {
            data_dir,
            trash_retention_days: 7,
            soft_delete: true,
        }
    }
}

/// Safety manager for protecting file operations.
pub struct SafetyManager {
    /// Configuration
    config: SafetyConfig,
    /// Index hash (used for separating trash/history by index)
    index_hash: String,
    /// Trash directory
    trash_dir: PathBuf,
    /// History file
    history_file: PathBuf,
    /// In-memory cache of trash entries
    trash_cache: Arc<RwLock<Vec<TrashEntry>>>,
}

impl SafetyManager {
    /// Create a new safety manager.
    pub fn new(source: &PathBuf, config: Option<SafetyConfig>) -> Self {
        let config = config.unwrap_or_default();

        // Create a hash of the source path for isolation
        let index_hash = blake3::hash(source.to_string_lossy().as_bytes())
            .to_hex()
            .chars()
            .take(16)
            .collect::<String>();

        let trash_dir = config.data_dir.join("trash").join(&index_hash);
        let history_file = config.data_dir.join("history").join(format!("{index_hash}.jsonl"));

        // Ensure directories exist
        if let Err(e) = fs::create_dir_all(&trash_dir) {
            warn!("Failed to create trash directory: {e}");
        }
        if let Some(parent) = history_file.parent()
            && let Err(e) = fs::create_dir_all(parent)
        {
            warn!("Failed to create history directory: {e}");
        }

        // Load existing trash entries synchronously
        let entries = Self::load_trash_entries(&trash_dir).unwrap_or_default();

        Self {
            config,
            index_hash,
            trash_dir,
            history_file,
            trash_cache: Arc::new(RwLock::new(entries)),
        }
    }

    /// Get the index hash.
    #[must_use]
    pub fn index_hash(&self) -> &str {
        &self.index_hash
    }

    /// Load trash entries from disk.
    fn load_trash_entries(trash_dir: &PathBuf) -> std::io::Result<Vec<TrashEntry>> {
        let manifest_path = trash_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&manifest_path)?;
        let entries: Vec<TrashEntry> = serde_json::from_str(&content).unwrap_or_default();
        Ok(entries)
    }

    /// Save trash entries to disk.
    async fn save_trash_entries(&self) -> std::io::Result<()> {
        let entries = self.trash_cache.read().await;
        let manifest_path = self.trash_dir.join("manifest.json");
        let content = serde_json::to_string_pretty(&*entries)?;
        fs::write(&manifest_path, content)?;
        Ok(())
    }

    /// Move a file to trash (soft delete).
    pub async fn soft_delete(&self, path: &PathBuf) -> Result<TrashEntry, String> {
        if !path.exists() {
            return Err("File not found".into());
        }

        if path.is_dir() {
            return Err("Cannot soft delete directories".into());
        }

        // Read file content for hash
        let content = fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;
        let content_hash = blake3::hash(&content).to_hex().to_string();
        let size = content.len() as u64;

        // Create trash entry
        let id = Uuid::new_v4();
        let trash_entry_dir = self.trash_dir.join(id.to_string());
        fs::create_dir_all(&trash_entry_dir)
            .map_err(|e| format!("Failed to create trash entry dir: {e}"))?;

        let trash_content_path = trash_entry_dir.join("content");
        let trash_meta_path = trash_entry_dir.join("meta.json");

        // Move content to trash
        fs::rename(path, &trash_content_path)
            .or_else(|_| {
                // If rename fails (cross-device), copy and delete
                fs::copy(path, &trash_content_path)?;
                fs::remove_file(path)
            })
            .map_err(|e| format!("Failed to move file to trash: {e}"))?;

        let now = Utc::now();
        let expires_at = now + chrono::Duration::days(i64::from(self.config.trash_retention_days));

        let entry = TrashEntry {
            id,
            original_path: path.clone(),
            trash_path: trash_content_path,
            deleted_at: now,
            expires_at,
            content_hash,
            size,
        };

        // Save metadata
        let meta_content = serde_json::to_string_pretty(&entry)
            .map_err(|e| format!("Failed to serialize meta: {e}"))?;
        fs::write(&trash_meta_path, meta_content)
            .map_err(|e| format!("Failed to write meta: {e}"))?;

        // Update cache
        {
            let mut cache = self.trash_cache.write().await;
            cache.push(entry.clone());
        }

        // Save manifest
        if let Err(e) = self.save_trash_entries().await {
            warn!("Failed to save trash manifest: {e}");
        }

        info!("Soft deleted {:?} -> trash/{}", path, id);
        Ok(entry)
    }

    /// Restore a file from trash.
    pub async fn restore(&self, trash_id: Uuid) -> Result<PathBuf, String> {
        let entry = {
            let cache = self.trash_cache.read().await;
            cache.iter().find(|e| e.id == trash_id).cloned()
        };

        let entry = entry.ok_or_else(|| "Trash entry not found".to_string())?;

        if !entry.trash_path.exists() {
            return Err("Trash content not found".into());
        }

        // Restore to original path
        let restore_path = &entry.original_path;

        // Ensure parent directory exists
        if let Some(parent) = restore_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directory: {e}"))?;
        }

        // Check if destination already exists
        if restore_path.exists() {
            return Err("Destination already exists".into());
        }

        // Move content back
        fs::rename(&entry.trash_path, restore_path)
            .or_else(|_| {
                fs::copy(&entry.trash_path, restore_path)?;
                fs::remove_file(&entry.trash_path)
            })
            .map_err(|e| format!("Failed to restore file: {e}"))?;

        // Remove trash entry directory
        let trash_entry_dir = self.trash_dir.join(trash_id.to_string());
        if let Err(e) = fs::remove_dir_all(&trash_entry_dir) {
            warn!("Failed to remove trash entry dir: {e}");
        }

        // Update cache
        {
            let mut cache = self.trash_cache.write().await;
            cache.retain(|e| e.id != trash_id);
        }

        // Save manifest
        if let Err(e) = self.save_trash_entries().await {
            warn!("Failed to save trash manifest: {e}");
        }

        info!("Restored {:?} from trash/{}", restore_path, trash_id);
        Ok(restore_path.clone())
    }

    /// List all trash entries.
    pub async fn list_trash(&self) -> Vec<TrashEntry> {
        self.trash_cache.read().await.clone()
    }

    /// Get a specific trash entry.
    pub async fn get_trash_entry(&self, id: Uuid) -> Option<TrashEntry> {
        self.trash_cache.read().await.iter().find(|e| e.id == id).cloned()
    }

    /// Get trash content by ID.
    pub fn get_trash_content(&self, id: Uuid) -> Result<Vec<u8>, String> {
        let trash_content_path = self.trash_dir.join(id.to_string()).join("content");
        if !trash_content_path.exists() {
            return Err("Trash content not found".into());
        }
        fs::read(&trash_content_path).map_err(|e| format!("Failed to read trash content: {e}"))
    }

    /// Purge expired trash entries.
    pub async fn purge_expired(&self) -> usize {
        let now = Utc::now();
        let expired: Vec<Uuid> = {
            let cache = self.trash_cache.read().await;
            cache.iter()
                .filter(|e| e.expires_at < now)
                .map(|e| e.id)
                .collect()
        };

        let mut purged = 0;
        for id in expired {
            let trash_entry_dir = self.trash_dir.join(id.to_string());
            if let Err(e) = fs::remove_dir_all(&trash_entry_dir) {
                warn!("Failed to purge trash entry {}: {e}", id);
            } else {
                purged += 1;
            }
        }

        // Update cache
        {
            let mut cache = self.trash_cache.write().await;
            cache.retain(|e| e.expires_at >= now);
        }

        // Save manifest
        if let Err(e) = self.save_trash_entries().await {
            warn!("Failed to save trash manifest: {e}");
        }

        if purged > 0 {
            info!("Purged {} expired trash entries", purged);
        }

        purged
    }

    /// Log an operation to history.
    pub fn log(&self, entry: HistoryEntry) -> std::io::Result<()> {
        let line = serde_json::to_string(&entry)?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.history_file)?;

        writeln!(file, "{line}")?;
        debug!("Logged history entry: {:?}", entry.operation);
        Ok(())
    }

    /// Log a successful operation.
    pub fn log_success(&self, operation: HistoryOperation, undo_data: Option<UndoData>) {
        let entry = HistoryEntry {
            id: Uuid::new_v4(),
            operation,
            timestamp: Utc::now(),
            success: true,
            reversible: undo_data.is_some(),
            undo_data,
            error: None,
        };

        if let Err(e) = self.log(entry) {
            warn!("Failed to log history: {e}");
        }
    }

    /// Log a failed operation.
    pub fn log_failure(&self, operation: HistoryOperation, error: String) {
        let entry = HistoryEntry {
            id: Uuid::new_v4(),
            operation,
            timestamp: Utc::now(),
            success: false,
            reversible: false,
            undo_data: None,
            error: Some(error),
        };

        if let Err(e) = self.log(entry) {
            warn!("Failed to log history: {e}");
        }
    }

    /// Read history entries.
    pub fn read_history(&self, limit: Option<usize>) -> Vec<HistoryEntry> {
        let file = match File::open(&self.history_file) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let reader = BufReader::new(file);
        let mut entries: Vec<HistoryEntry> = reader
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str(&line).ok())
            .collect();

        // Return most recent first
        entries.reverse();

        if let Some(limit) = limit {
            entries.truncate(limit);
        }

        entries
    }

    /// Get history as JSON bytes (for FUSE read).
    pub fn get_history_json(&self, limit: Option<usize>) -> Vec<u8> {
        let entries = self.read_history(limit);
        serde_json::to_string_pretty(&entries)
            .unwrap_or_else(|_| "[]".to_string())
            .into_bytes()
    }

    /// Find an operation by ID for undo.
    pub fn find_operation(&self, id: Uuid) -> Option<HistoryEntry> {
        self.read_history(None)
            .into_iter()
            .find(|e| e.id == id)
    }

    /// Undo an operation.
    pub async fn undo(&self, operation_id: Uuid) -> Result<String, String> {
        let entry = self.find_operation(operation_id)
            .ok_or_else(|| "Operation not found".to_string())?;

        if !entry.reversible {
            return Err("Operation is not reversible".into());
        }

        let undo_data = entry.undo_data.ok_or_else(|| "No undo data available".to_string())?;

        match undo_data {
            UndoData::Create { path } => {
                // Undo create by deleting the file
                if path.exists() {
                    fs::remove_file(&path)
                        .map_err(|e| format!("Failed to undo create: {e}"))?;
                    self.log_success(
                        HistoryOperation::Delete { path: path.clone(), trash_id: None },
                        None,
                    );
                    Ok(format!("Undone: deleted {}", path.display()))
                } else {
                    Err("File no longer exists".into())
                }
            }
            UndoData::Delete { trash_id } => {
                // Undo delete by restoring from trash
                let restored = self.restore(trash_id).await?;
                Ok(format!("Undone: restored {}", restored.display()))
            }
            UndoData::Move { src, dst } => {
                // Undo move by moving back
                if dst.exists() {
                    fs::rename(&dst, &src)
                        .map_err(|e| format!("Failed to undo move: {e}"))?;
                    self.log_success(
                        HistoryOperation::Move { src: dst.clone(), dst: src.clone() },
                        Some(UndoData::Move { src: src.clone(), dst }),
                    );
                    Ok(format!("Undone: moved back to {}", src.display()))
                } else {
                    Err("Destination file no longer exists".into())
                }
            }
            UndoData::Copy { path } => {
                // Undo copy by deleting the copy
                if path.exists() {
                    fs::remove_file(&path)
                        .map_err(|e| format!("Failed to undo copy: {e}"))?;
                    self.log_success(
                        HistoryOperation::Delete { path: path.clone(), trash_id: None },
                        None,
                    );
                    Ok(format!("Undone: deleted copy {}", path.display()))
                } else {
                    Err("Copy file no longer exists".into())
                }
            }
        }
    }

    /// Check if soft delete is enabled.
    #[must_use]
    pub fn soft_delete_enabled(&self) -> bool {
        self.config.soft_delete
    }

    /// Get trash directory path.
    #[must_use]
    pub fn trash_dir(&self) -> &PathBuf {
        &self.trash_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_manager() -> (SafetyManager, TempDir, TempDir) {
        let source_dir = TempDir::new().unwrap();
        let data_dir = TempDir::new().unwrap();

        let config = SafetyConfig {
            data_dir: data_dir.path().to_path_buf(),
            trash_retention_days: 7,
            soft_delete: true,
        };

        let manager = SafetyManager::new(&source_dir.path().to_path_buf(), Some(config));
        (manager, source_dir, data_dir)
    }

    #[tokio::test]
    async fn test_soft_delete_and_restore() {
        let (manager, source_dir, _data_dir) = create_test_manager();

        // Create a test file
        let test_file = source_dir.path().join("test.txt");
        fs::write(&test_file, "Hello, World!").unwrap();
        assert!(test_file.exists());

        // Soft delete
        let entry = manager.soft_delete(&test_file).await.unwrap();
        assert!(!test_file.exists());
        assert!(entry.trash_path.exists());

        // Verify in trash list
        let trash = manager.list_trash().await;
        assert_eq!(trash.len(), 1);
        assert_eq!(trash[0].id, entry.id);

        // Restore
        let restored = manager.restore(entry.id).await.unwrap();
        assert_eq!(restored, test_file);
        assert!(test_file.exists());

        // Verify content
        let content = fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "Hello, World!");

        // Trash should be empty
        let trash = manager.list_trash().await;
        assert!(trash.is_empty());
    }

    #[tokio::test]
    async fn test_soft_delete_nonexistent() {
        let (manager, _source_dir, _data_dir) = create_test_manager();

        let result = manager.soft_delete(&PathBuf::from("/nonexistent.txt")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_restore_nonexistent() {
        let (manager, _source_dir, _data_dir) = create_test_manager();

        let result = manager.restore(Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_history_logging() {
        let (manager, _source_dir, _data_dir) = create_test_manager();

        // Log some operations
        manager.log_success(
            HistoryOperation::Create { path: PathBuf::from("/test.txt") },
            Some(UndoData::Create { path: PathBuf::from("/test.txt") }),
        );

        manager.log_failure(
            HistoryOperation::Delete { path: PathBuf::from("/fail.txt"), trash_id: None },
            "Permission denied".to_string(),
        );

        // Read history
        let history = manager.read_history(None);
        assert_eq!(history.len(), 2);

        // Most recent first
        assert!(!history[0].success);
        assert!(history[1].success);
    }

    #[test]
    fn test_get_history_json() {
        let (manager, _source_dir, _data_dir) = create_test_manager();

        manager.log_success(
            HistoryOperation::Create { path: PathBuf::from("/test.txt") },
            None,
        );

        let json = manager.get_history_json(None);
        let json_str = String::from_utf8(json).unwrap();
        assert!(json_str.contains("create"));
        assert!(json_str.contains("/test.txt"));
    }

    #[tokio::test]
    async fn test_get_trash_content() {
        let (manager, source_dir, _data_dir) = create_test_manager();

        let test_file = source_dir.path().join("content_test.txt");
        fs::write(&test_file, "Test content for trash").unwrap();

        let entry = manager.soft_delete(&test_file).await.unwrap();

        let content = manager.get_trash_content(entry.id).unwrap();
        assert_eq!(String::from_utf8(content).unwrap(), "Test content for trash");
    }

    #[test]
    fn test_safety_config_default() {
        let config = SafetyConfig::default();
        assert_eq!(config.trash_retention_days, 7);
        assert!(config.soft_delete);
    }

    #[test]
    fn test_trash_entry_serialization() {
        let entry = TrashEntry {
            id: Uuid::new_v4(),
            original_path: PathBuf::from("/test.txt"),
            trash_path: PathBuf::from("/trash/test.txt"),
            deleted_at: Utc::now(),
            expires_at: Utc::now(),
            content_hash: "abc123".to_string(),
            size: 1024,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: TrashEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, entry.id);
        assert_eq!(parsed.original_path, entry.original_path);
    }

    #[test]
    fn test_history_entry_serialization() {
        let entry = HistoryEntry {
            id: Uuid::new_v4(),
            operation: HistoryOperation::Create { path: PathBuf::from("/test.txt") },
            timestamp: Utc::now(),
            success: true,
            reversible: true,
            undo_data: Some(UndoData::Create { path: PathBuf::from("/test.txt") }),
            error: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: HistoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, entry.id);
        assert!(parsed.success);
    }
}
