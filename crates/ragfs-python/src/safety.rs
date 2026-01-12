//! Python bindings for RAGFS safety layer.
//!
//! This module provides Python access to the safety manager for:
//! - Soft delete with trash (files can be recovered)
//! - Audit history logging
//! - Undo support for reversible operations

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_fuse::safety::{HistoryEntry, HistoryOperation, SafetyConfig, SafetyManager, TrashEntry};
use std::path::PathBuf;
use std::sync::Arc;

/// Python wrapper for TrashEntry.
///
/// Represents a file that has been soft-deleted and can be restored.
#[pyclass(name = "TrashEntry")]
#[derive(Clone)]
pub struct PyTrashEntry {
    /// Unique identifier for this trash entry
    #[pyo3(get)]
    pub id: String,
    /// Original path of the file
    #[pyo3(get)]
    pub original_path: String,
    /// Path in trash storage
    #[pyo3(get)]
    pub trash_path: String,
    /// When the file was deleted (ISO 8601)
    #[pyo3(get)]
    pub deleted_at: String,
    /// When the trash entry expires (ISO 8601)
    #[pyo3(get)]
    pub expires_at: String,
    /// Blake3 hash of the content
    #[pyo3(get)]
    pub content_hash: String,
    /// Original file size in bytes
    #[pyo3(get)]
    pub size: u64,
}

#[pymethods]
impl PyTrashEntry {
    fn __repr__(&self) -> String {
        format!(
            "TrashEntry(id='{}', original_path='{}', size={})",
            self.id, self.original_path, self.size
        )
    }

    /// Convert to dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("original_path", &self.original_path)?;
        dict.set_item("trash_path", &self.trash_path)?;
        dict.set_item("deleted_at", &self.deleted_at)?;
        dict.set_item("expires_at", &self.expires_at)?;
        dict.set_item("content_hash", &self.content_hash)?;
        dict.set_item("size", self.size)?;
        Ok(dict.into())
    }
}

impl From<TrashEntry> for PyTrashEntry {
    fn from(entry: TrashEntry) -> Self {
        Self {
            id: entry.id.to_string(),
            original_path: entry.original_path.to_string_lossy().to_string(),
            trash_path: entry.trash_path.to_string_lossy().to_string(),
            deleted_at: entry.deleted_at.to_rfc3339(),
            expires_at: entry.expires_at.to_rfc3339(),
            content_hash: entry.content_hash,
            size: entry.size,
        }
    }
}

/// Python wrapper for HistoryOperation.
///
/// Represents the type of operation recorded in history.
#[pyclass(name = "HistoryOperation")]
#[derive(Clone)]
pub struct PyHistoryOperation {
    /// Operation type: "create", "delete", "move", "copy", "write", "restore"
    #[pyo3(get)]
    pub operation_type: String,
    /// Path affected by the operation
    #[pyo3(get)]
    pub path: Option<String>,
    /// Source path (for move/copy)
    #[pyo3(get)]
    pub src: Option<String>,
    /// Destination path (for move/copy)
    #[pyo3(get)]
    pub dst: Option<String>,
    /// Trash ID (for delete with soft delete)
    #[pyo3(get)]
    pub trash_id: Option<String>,
    /// Whether this was an append operation (for write)
    #[pyo3(get)]
    pub append: Option<bool>,
}

#[pymethods]
impl PyHistoryOperation {
    fn __repr__(&self) -> String {
        match self.operation_type.as_str() {
            "create" | "delete" | "write" => {
                format!(
                    "HistoryOperation(type='{}', path='{}')",
                    self.operation_type,
                    self.path.as_deref().unwrap_or("")
                )
            }
            "move" | "copy" => {
                format!(
                    "HistoryOperation(type='{}', src='{}', dst='{}')",
                    self.operation_type,
                    self.src.as_deref().unwrap_or(""),
                    self.dst.as_deref().unwrap_or("")
                )
            }
            "restore" => {
                format!(
                    "HistoryOperation(type='restore', trash_id='{}')",
                    self.trash_id.as_deref().unwrap_or("")
                )
            }
            _ => format!("HistoryOperation(type='{}')", self.operation_type),
        }
    }
}

impl From<HistoryOperation> for PyHistoryOperation {
    fn from(op: HistoryOperation) -> Self {
        match op {
            HistoryOperation::Create { path } => Self {
                operation_type: "create".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                src: None,
                dst: None,
                trash_id: None,
                append: None,
            },
            HistoryOperation::Delete { path, trash_id } => Self {
                operation_type: "delete".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                src: None,
                dst: None,
                trash_id: trash_id.map(|id| id.to_string()),
                append: None,
            },
            HistoryOperation::Move { src, dst } => Self {
                operation_type: "move".to_string(),
                path: None,
                src: Some(src.to_string_lossy().to_string()),
                dst: Some(dst.to_string_lossy().to_string()),
                trash_id: None,
                append: None,
            },
            HistoryOperation::Copy { src, dst } => Self {
                operation_type: "copy".to_string(),
                path: None,
                src: Some(src.to_string_lossy().to_string()),
                dst: Some(dst.to_string_lossy().to_string()),
                trash_id: None,
                append: None,
            },
            HistoryOperation::Write { path, append } => Self {
                operation_type: "write".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                src: None,
                dst: None,
                trash_id: None,
                append: Some(append),
            },
            HistoryOperation::Restore { trash_id, path } => Self {
                operation_type: "restore".to_string(),
                path: Some(path.to_string_lossy().to_string()),
                src: None,
                dst: None,
                trash_id: Some(trash_id.to_string()),
                append: None,
            },
        }
    }
}

/// Python wrapper for HistoryEntry.
///
/// Represents a single operation in the audit history.
#[pyclass(name = "HistoryEntry")]
#[derive(Clone)]
pub struct PyHistoryEntry {
    /// Unique identifier for this operation (can be used for undo)
    #[pyo3(get)]
    pub id: String,
    /// The operation that was performed
    #[pyo3(get)]
    pub operation: PyHistoryOperation,
    /// When the operation occurred (ISO 8601)
    #[pyo3(get)]
    pub timestamp: String,
    /// Whether the operation succeeded
    #[pyo3(get)]
    pub success: bool,
    /// Whether this operation can be undone
    #[pyo3(get)]
    pub reversible: bool,
    /// Error message if the operation failed
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyHistoryEntry {
    fn __repr__(&self) -> String {
        format!(
            "HistoryEntry(id='{}', operation={}, success={}, reversible={})",
            self.id,
            self.operation.__repr__(),
            self.success,
            self.reversible
        )
    }

    /// Convert to dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("operation_type", &self.operation.operation_type)?;
        dict.set_item("timestamp", &self.timestamp)?;
        dict.set_item("success", self.success)?;
        dict.set_item("reversible", self.reversible)?;
        if let Some(ref err) = self.error {
            dict.set_item("error", err)?;
        }
        Ok(dict.into())
    }
}

impl From<HistoryEntry> for PyHistoryEntry {
    fn from(entry: HistoryEntry) -> Self {
        Self {
            id: entry.id.to_string(),
            operation: entry.operation.into(),
            timestamp: entry.timestamp.to_rfc3339(),
            success: entry.success,
            reversible: entry.reversible,
            error: entry.error,
        }
    }
}

/// Safety manager for protecting file operations.
///
/// Provides soft delete (trash), audit history, and undo capabilities.
/// This is a core RAGFS innovation for safe AI agent file operations.
///
/// Example:
///
/// ```python
/// from ragfs import RagfsSafetyManager
///
/// # Create manager for a source directory
/// safety = RagfsSafetyManager("/path/to/source")
///
/// # Soft delete a file (can be undone)
/// entry = await safety.delete_to_trash("/path/to/file.txt")
/// print(f"Deleted to trash: {entry.id}")
///
/// # List trash contents
/// trash = await safety.list_trash()
/// for item in trash:
///     print(f"{item.original_path} deleted at {item.deleted_at}")
///
/// # Restore from trash
/// restored_path = await safety.restore_from_trash(entry.id)
///
/// # View operation history
/// history = safety.get_history(limit=10)
/// for h in history:
///     print(f"{h.operation} at {h.timestamp}")
///
/// # Undo any reversible operation
/// result = await safety.undo(history[0].id)
/// ```
#[pyclass(name = "RagfsSafetyManager")]
pub struct RagfsSafetyManager {
    manager: Arc<SafetyManager>,
}

#[pymethods]
impl RagfsSafetyManager {
    /// Create a new safety manager.
    ///
    /// Args:
    ///     source_path: Path to the source directory being managed.
    ///     data_dir: Optional path for safety data (trash, history).
    ///               Defaults to ~/.local/share/ragfs.
    ///     trash_retention_days: How long to keep trash entries. Defaults to 7.
    ///     soft_delete_enabled: Whether soft delete is enabled. Defaults to True.
    #[new]
    #[pyo3(signature = (source_path, data_dir=None, trash_retention_days=7, soft_delete_enabled=true))]
    fn new(
        source_path: String,
        data_dir: Option<String>,
        trash_retention_days: u32,
        soft_delete_enabled: bool,
    ) -> Self {
        let config = SafetyConfig {
            data_dir: data_dir.map(PathBuf::from).unwrap_or_else(|| {
                directories::ProjectDirs::from("", "", "ragfs")
                    .map(|dirs| dirs.data_local_dir().to_path_buf())
                    .unwrap_or_else(|| PathBuf::from(".ragfs"))
            }),
            trash_retention_days,
            soft_delete: soft_delete_enabled,
        };

        let manager = SafetyManager::new(&PathBuf::from(source_path), Some(config));

        Self {
            manager: Arc::new(manager),
        }
    }

    /// Delete a file to trash (soft delete).
    ///
    /// The file is moved to trash and can be restored later.
    /// Returns a TrashEntry with the undo ID.
    ///
    /// Args:
    ///     path: Path to the file to delete.
    ///
    /// Returns:
    ///     TrashEntry with id that can be used to restore.
    ///
    /// Raises:
    ///     RuntimeError: If the file doesn't exist or deletion fails.
    fn delete_to_trash<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let entry = manager
                .soft_delete(&path)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to soft delete: {e}")))?;

            Ok(PyTrashEntry::from(entry))
        })
    }

    /// List all files in trash.
    ///
    /// Returns:
    ///     List of TrashEntry objects.
    fn list_trash<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let entries = manager.list_trash().await;
            let py_entries: Vec<PyTrashEntry> =
                entries.into_iter().map(PyTrashEntry::from).collect();
            Ok(py_entries)
        })
    }

    /// Restore a file from trash.
    ///
    /// Args:
    ///     trash_id: The ID of the trash entry (from TrashEntry.id).
    ///
    /// Returns:
    ///     The path where the file was restored.
    ///
    /// Raises:
    ///     RuntimeError: If the trash entry doesn't exist or restore fails.
    fn restore_from_trash<'py>(
        &self,
        py: Python<'py>,
        trash_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let uuid = uuid::Uuid::parse_str(&trash_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid trash ID: {e}")))?;

            let restored_path = manager
                .restore(uuid)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to restore: {e}")))?;

            Ok(restored_path.to_string_lossy().to_string())
        })
    }

    /// Get a specific trash entry by ID.
    ///
    /// Args:
    ///     trash_id: The ID of the trash entry.
    ///
    /// Returns:
    ///     TrashEntry or None if not found.
    fn get_trash_entry<'py>(
        &self,
        py: Python<'py>,
        trash_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let uuid = uuid::Uuid::parse_str(&trash_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid trash ID: {e}")))?;

            let entry = manager.get_trash_entry(uuid).await;
            Ok(entry.map(PyTrashEntry::from))
        })
    }

    /// Get the content of a trashed file.
    ///
    /// Args:
    ///     trash_id: The ID of the trash entry.
    ///
    /// Returns:
    ///     The file content as bytes.
    ///
    /// Raises:
    ///     RuntimeError: If the trash entry doesn't exist.
    fn get_trash_content(&self, trash_id: String) -> PyResult<Vec<u8>> {
        let uuid = uuid::Uuid::parse_str(&trash_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid trash ID: {e}")))?;

        self.manager
            .get_trash_content(uuid)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trash content: {e}")))
    }

    /// Empty the trash by removing expired entries.
    ///
    /// Returns:
    ///     Number of entries purged.
    fn purge_expired<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let count = manager.purge_expired().await;
            Ok(count)
        })
    }

    /// Get operation history.
    ///
    /// Args:
    ///     limit: Maximum number of entries to return. None for all.
    ///
    /// Returns:
    ///     List of HistoryEntry objects (most recent first).
    #[pyo3(signature = (limit=None))]
    fn get_history(&self, limit: Option<usize>) -> Vec<PyHistoryEntry> {
        self.manager
            .read_history(limit)
            .into_iter()
            .map(PyHistoryEntry::from)
            .collect()
    }

    /// Get history as JSON string.
    ///
    /// Args:
    ///     limit: Maximum number of entries to return.
    ///
    /// Returns:
    ///     JSON string of history entries.
    #[pyo3(signature = (limit=None))]
    fn get_history_json(&self, limit: Option<usize>) -> String {
        String::from_utf8_lossy(&self.manager.get_history_json(limit)).to_string()
    }

    /// Undo an operation by its ID.
    ///
    /// Args:
    ///     operation_id: The ID of the operation to undo (from HistoryEntry.id).
    ///
    /// Returns:
    ///     A message describing what was undone.
    ///
    /// Raises:
    ///     RuntimeError: If the operation can't be found or undone.
    fn undo<'py>(&self, py: Python<'py>, operation_id: String) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let uuid = uuid::Uuid::parse_str(&operation_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid operation ID: {e}")))?;

            let result = manager
                .undo(uuid)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to undo: {e}")))?;

            Ok(result)
        })
    }

    /// Check if an operation can be undone.
    ///
    /// Args:
    ///     operation_id: The ID of the operation.
    ///
    /// Returns:
    ///     True if the operation is reversible, False otherwise.
    fn can_undo(&self, operation_id: String) -> PyResult<bool> {
        let uuid = uuid::Uuid::parse_str(&operation_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid operation ID: {e}")))?;

        Ok(self
            .manager
            .find_operation(uuid)
            .is_some_and(|e| e.reversible))
    }

    /// Check if soft delete is enabled.
    #[getter]
    fn soft_delete_enabled(&self) -> bool {
        self.manager.soft_delete_enabled()
    }

    /// Get the index hash (unique identifier for this source).
    #[getter]
    fn index_hash(&self) -> String {
        self.manager.index_hash().to_string()
    }

    /// Get the trash directory path.
    #[getter]
    fn trash_dir(&self) -> String {
        self.manager.trash_dir().to_string_lossy().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RagfsSafetyManager(index_hash='{}', soft_delete={})",
            self.manager.index_hash(),
            self.manager.soft_delete_enabled()
        )
    }
}
