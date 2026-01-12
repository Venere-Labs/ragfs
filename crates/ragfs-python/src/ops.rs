//! Python bindings for RAGFS operations manager.
//!
//! This module provides Python access to structured file operations with JSON feedback:
//! - Create, delete, move, copy files
//! - Write content with overwrite or append
//! - Create directories and symbolic links
//! - Batch operations with atomic rollback support

use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_fuse::ops::{BatchRequest, BatchResult, Operation, OperationResult, OpsManager};
use ragfs_fuse::safety::SafetyManager;
use std::path::PathBuf;
use std::sync::Arc;

// ============================================================================
// Python Types
// ============================================================================

/// Python wrapper for Operation.
#[pyclass(name = "Operation")]
#[derive(Clone)]
pub struct PyOperation {
    inner: Operation,
}

#[pymethods]
impl PyOperation {
    /// Create a "create" operation.
    #[staticmethod]
    fn create(path: String, content: String) -> Self {
        Self {
            inner: Operation::Create {
                path: PathBuf::from(path),
                content,
            },
        }
    }

    /// Create a "delete" operation.
    #[staticmethod]
    fn delete(path: String) -> Self {
        Self {
            inner: Operation::Delete {
                path: PathBuf::from(path),
            },
        }
    }

    /// Create a "move" operation.
    #[staticmethod]
    #[pyo3(name = "move")]
    fn move_op(src: String, dst: String) -> Self {
        Self {
            inner: Operation::Move {
                src: PathBuf::from(src),
                dst: PathBuf::from(dst),
            },
        }
    }

    /// Create a "copy" operation.
    #[staticmethod]
    fn copy(src: String, dst: String) -> Self {
        Self {
            inner: Operation::Copy {
                src: PathBuf::from(src),
                dst: PathBuf::from(dst),
            },
        }
    }

    /// Create a "write" operation.
    #[staticmethod]
    #[pyo3(signature = (path, content, append=false))]
    fn write(path: String, content: String, append: bool) -> Self {
        Self {
            inner: Operation::Write {
                path: PathBuf::from(path),
                content,
                append,
            },
        }
    }

    /// Create a "mkdir" operation.
    #[staticmethod]
    fn mkdir(path: String) -> Self {
        Self {
            inner: Operation::Mkdir {
                path: PathBuf::from(path),
            },
        }
    }

    /// Create a "symlink" operation.
    #[staticmethod]
    fn symlink(target: String, link: String) -> Self {
        Self {
            inner: Operation::Symlink {
                target: PathBuf::from(target),
                link: PathBuf::from(link),
            },
        }
    }

    #[getter]
    fn operation_type(&self) -> String {
        match &self.inner {
            Operation::Create { .. } => "create".to_string(),
            Operation::Delete { .. } => "delete".to_string(),
            Operation::Move { .. } => "move".to_string(),
            Operation::Copy { .. } => "copy".to_string(),
            Operation::Write { .. } => "write".to_string(),
            Operation::Mkdir { .. } => "mkdir".to_string(),
            Operation::Symlink { .. } => "symlink".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Operation::Create { path, .. } => {
                format!("Operation.create(path='{}')", path.display())
            }
            Operation::Delete { path } => {
                format!("Operation.delete(path='{}')", path.display())
            }
            Operation::Move { src, dst } => {
                format!(
                    "Operation.move(src='{}', dst='{}')",
                    src.display(),
                    dst.display()
                )
            }
            Operation::Copy { src, dst } => {
                format!(
                    "Operation.copy(src='{}', dst='{}')",
                    src.display(),
                    dst.display()
                )
            }
            Operation::Write { path, append, .. } => {
                format!(
                    "Operation.write(path='{}', append={})",
                    path.display(),
                    append
                )
            }
            Operation::Mkdir { path } => {
                format!("Operation.mkdir(path='{}')", path.display())
            }
            Operation::Symlink { target, link } => {
                format!(
                    "Operation.symlink(target='{}', link='{}')",
                    target.display(),
                    link.display()
                )
            }
        }
    }
}

/// Python wrapper for OperationResult.
#[pyclass(name = "OperationResult")]
#[derive(Clone)]
pub struct PyOperationResult {
    /// Unique identifier for this operation
    #[pyo3(get)]
    pub id: String,
    /// Whether the operation succeeded
    #[pyo3(get)]
    pub success: bool,
    /// Type of operation performed
    #[pyo3(get)]
    pub operation: String,
    /// Primary path involved
    #[pyo3(get)]
    pub path: String,
    /// Error message if failed
    #[pyo3(get)]
    pub error: Option<String>,
    /// Timestamp of the operation (ISO 8601)
    #[pyo3(get)]
    pub timestamp: String,
    /// Whether the file was indexed/reindexed
    #[pyo3(get)]
    pub indexed: bool,
    /// ID for undoing this operation (if reversible)
    #[pyo3(get)]
    pub undo_id: Option<String>,
}

#[pymethods]
impl PyOperationResult {
    fn __repr__(&self) -> String {
        if self.success {
            format!(
                "OperationResult(op='{}', path='{}', success=True, undo_id={:?})",
                self.operation, self.path, self.undo_id
            )
        } else {
            format!(
                "OperationResult(op='{}', path='{}', success=False, error={:?})",
                self.operation, self.path, self.error
            )
        }
    }

    /// Convert to dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("success", self.success)?;
        dict.set_item("operation", &self.operation)?;
        dict.set_item("path", &self.path)?;
        dict.set_item("timestamp", &self.timestamp)?;
        dict.set_item("indexed", self.indexed)?;
        if let Some(ref err) = self.error {
            dict.set_item("error", err)?;
        }
        if let Some(ref undo) = self.undo_id {
            dict.set_item("undo_id", undo)?;
        }
        Ok(dict.into())
    }
}

impl From<&OperationResult> for PyOperationResult {
    fn from(result: &OperationResult) -> Self {
        Self {
            id: result.id.to_string(),
            success: result.success,
            operation: result.operation.clone(),
            path: result.path.to_string_lossy().to_string(),
            error: result.error.clone(),
            timestamp: result.timestamp.to_rfc3339(),
            indexed: result.indexed,
            undo_id: result.undo_id.map(|id| id.to_string()),
        }
    }
}

/// Python wrapper for BatchResult.
#[pyclass(name = "BatchResult")]
#[derive(Clone)]
pub struct PyBatchResult {
    /// Unique identifier for this batch
    #[pyo3(get)]
    pub id: String,
    /// Whether all operations succeeded
    #[pyo3(get)]
    pub success: bool,
    /// Total number of operations
    #[pyo3(get)]
    pub total: usize,
    /// Number of successful operations
    #[pyo3(get)]
    pub succeeded: usize,
    /// Number of failed operations
    #[pyo3(get)]
    pub failed: usize,
    /// Individual operation results
    #[pyo3(get)]
    pub results: Vec<PyOperationResult>,
    /// Timestamp of the batch (ISO 8601)
    #[pyo3(get)]
    pub timestamp: String,
    /// Whether a rollback was performed (for atomic batches)
    #[pyo3(get)]
    pub rollback_performed: Option<bool>,
    /// Number of operations rolled back
    #[pyo3(get)]
    pub rolled_back: Option<usize>,
    /// Number of rollback failures
    #[pyo3(get)]
    pub rollback_failures: Option<usize>,
}

#[pymethods]
impl PyBatchResult {
    fn __repr__(&self) -> String {
        if self.success {
            format!(
                "BatchResult(success=True, total={}, succeeded={})",
                self.total, self.succeeded
            )
        } else {
            format!(
                "BatchResult(success=False, total={}, succeeded={}, failed={}, rollback={:?})",
                self.total, self.succeeded, self.failed, self.rollback_performed
            )
        }
    }

    /// Convert to dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("success", self.success)?;
        dict.set_item("total", self.total)?;
        dict.set_item("succeeded", self.succeeded)?;
        dict.set_item("failed", self.failed)?;
        dict.set_item("timestamp", &self.timestamp)?;
        if let Some(rollback) = self.rollback_performed {
            dict.set_item("rollback_performed", rollback)?;
        }
        if let Some(rolled_back) = self.rolled_back {
            dict.set_item("rolled_back", rolled_back)?;
        }
        if let Some(failures) = self.rollback_failures {
            dict.set_item("rollback_failures", failures)?;
        }
        Ok(dict.into())
    }
}

impl From<&BatchResult> for PyBatchResult {
    fn from(result: &BatchResult) -> Self {
        let (rolled_back, rollback_failures) = if let Some(ref details) = result.rollback_details {
            (Some(details.rolled_back), Some(details.rollback_failures))
        } else {
            (None, None)
        };

        Self {
            id: result.id.to_string(),
            success: result.success,
            total: result.total,
            succeeded: result.succeeded,
            failed: result.failed,
            results: result.results.iter().map(PyOperationResult::from).collect(),
            timestamp: result.timestamp.to_rfc3339(),
            rollback_performed: result.rollback_performed,
            rolled_back,
            rollback_failures,
        }
    }
}

// ============================================================================
// Operations Manager
// ============================================================================

/// Operations manager for structured file operations with JSON feedback.
///
/// Provides file operations designed for AI agents with:
/// - Structured JSON feedback for each operation
/// - Undo support via safety layer integration
/// - Atomic batch operations with rollback
/// - Dry-run validation
///
/// Example:
///
/// ```python
/// from ragfs import RagfsOpsManager, Operation
///
/// # Create manager
/// ops = RagfsOpsManager("/path/to/source")
///
/// # Single operations
/// result = await ops.create_file("newfile.txt", "Hello, World!")
/// if result.success:
///     print(f"Created with undo_id: {result.undo_id}")
///
/// result = await ops.move_file("old.txt", "new.txt")
/// result = await ops.delete_file("unwanted.txt")  # Soft delete if safety enabled
///
/// # Batch operations
/// operations = [
///     Operation.create("file1.txt", "content1"),
///     Operation.create("file2.txt", "content2"),
///     Operation.mkdir("new_directory"),
/// ]
/// batch_result = await ops.batch(operations, atomic=True)
/// if not batch_result.success and batch_result.rollback_performed:
///     print(f"Batch failed, rolled back {batch_result.rolled_back} operations")
///
/// # Dry run to validate without executing
/// validation = await ops.dry_run(operations)
/// if validation.success:
///     print("All operations would succeed")
/// ```
#[pyclass(name = "RagfsOpsManager")]
pub struct RagfsOpsManager {
    manager: Arc<OpsManager>,
}

#[pymethods]
impl RagfsOpsManager {
    /// Create a new operations manager.
    ///
    /// Args:
    ///     source_path: Root directory for resolving relative paths.
    ///     safety_source_path: Optional path for safety manager. If provided,
    ///                         enables soft delete (trash) and undo support.
    #[new]
    #[pyo3(signature = (source_path, safety_source_path=None))]
    fn new(source_path: String, safety_source_path: Option<String>) -> Self {
        let source = PathBuf::from(&source_path);

        let manager = if let Some(safety_path) = safety_source_path {
            let safety_manager = SafetyManager::new(&PathBuf::from(safety_path), None);
            OpsManager::with_safety(source, None, None, Arc::new(safety_manager))
        } else {
            OpsManager::new(source, None, None)
        };

        Self {
            manager: Arc::new(manager),
        }
    }

    /// Create a new file with content.
    ///
    /// Args:
    ///     path: File path (relative to source or absolute).
    ///     content: Content to write to the file.
    ///
    /// Returns:
    ///     OperationResult with success status and undo_id.
    fn create_file<'py>(
        &self,
        py: Python<'py>,
        path: String,
        content: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let result = manager.create(&path, &content).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Delete a file.
    ///
    /// Uses soft delete (move to trash) if safety manager is enabled,
    /// otherwise performs a hard delete.
    ///
    /// Args:
    ///     path: File path to delete.
    ///
    /// Returns:
    ///     OperationResult with undo_id if soft delete was used.
    fn delete_file<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let result = manager.delete(&path).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Move/rename a file.
    ///
    /// Args:
    ///     src: Source file path.
    ///     dst: Destination file path.
    ///
    /// Returns:
    ///     OperationResult with undo_id for reversal.
    fn move_file<'py>(
        &self,
        py: Python<'py>,
        src: String,
        dst: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let src = PathBuf::from(src);
        let dst = PathBuf::from(dst);

        future_into_py(py, async move {
            let result = manager.move_file(&src, &dst).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Copy a file.
    ///
    /// Args:
    ///     src: Source file path.
    ///     dst: Destination file path.
    ///
    /// Returns:
    ///     OperationResult with undo_id.
    fn copy_file<'py>(
        &self,
        py: Python<'py>,
        src: String,
        dst: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let src = PathBuf::from(src);
        let dst = PathBuf::from(dst);

        future_into_py(py, async move {
            let result = manager.copy(&src, &dst).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Write content to a file.
    ///
    /// Args:
    ///     path: File path (creates if not exists).
    ///     content: Content to write.
    ///     append: If True, append to existing content. Default is overwrite.
    ///
    /// Returns:
    ///     OperationResult.
    #[pyo3(signature = (path, content, append=false))]
    fn write_file<'py>(
        &self,
        py: Python<'py>,
        path: String,
        content: String,
        append: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let result = manager.write(&path, &content, append).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Create a directory.
    ///
    /// Creates parent directories as needed.
    ///
    /// Args:
    ///     path: Directory path to create.
    ///
    /// Returns:
    ///     OperationResult.
    fn mkdir<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let result = manager.mkdir(&path).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Create a symbolic link.
    ///
    /// Args:
    ///     target: Path that the link points to.
    ///     link: Path of the symbolic link to create.
    ///
    /// Returns:
    ///     OperationResult.
    fn symlink<'py>(
        &self,
        py: Python<'py>,
        target: String,
        link: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let target = PathBuf::from(target);
        let link = PathBuf::from(link);

        future_into_py(py, async move {
            let result = manager.symlink(&target, &link).await;
            Ok(PyOperationResult::from(&result))
        })
    }

    /// Execute a batch of operations.
    ///
    /// Args:
    ///     operations: List of Operation objects to execute.
    ///     atomic: If True, rollback all operations if any fails. Default False.
    ///
    /// Returns:
    ///     BatchResult with individual operation results.
    #[pyo3(signature = (operations, atomic=false))]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        operations: Vec<PyOperation>,
        atomic: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let ops: Vec<Operation> = operations.into_iter().map(|o| o.inner).collect();

        future_into_py(py, async move {
            let request = BatchRequest {
                operations: ops,
                atomic,
                dry_run: false,
            };
            let result = manager.batch(request).await;
            Ok(PyBatchResult::from(&result))
        })
    }

    /// Validate operations without executing them (dry run).
    ///
    /// Args:
    ///     operations: List of Operation objects to validate.
    ///
    /// Returns:
    ///     BatchResult indicating which operations would succeed/fail.
    fn dry_run<'py>(
        &self,
        py: Python<'py>,
        operations: Vec<PyOperation>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();
        let ops: Vec<Operation> = operations.into_iter().map(|o| o.inner).collect();

        future_into_py(py, async move {
            let request = BatchRequest {
                operations: ops,
                atomic: false,
                dry_run: true,
            };
            let result = manager.batch(request).await;
            Ok(PyBatchResult::from(&result))
        })
    }

    /// Get the last operation result as JSON string.
    ///
    /// Returns:
    ///     JSON string of the last operation result.
    fn get_last_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.manager.clone();

        future_into_py(py, async move {
            let bytes = manager.get_last_result().await;
            Ok(String::from_utf8_lossy(&bytes).to_string())
        })
    }

    fn __repr__(&self) -> String {
        "RagfsOpsManager(...)".to_string()
    }
}
