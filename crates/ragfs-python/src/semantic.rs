//! Python bindings for RAGFS semantic operations.
//!
//! This module provides Python access to AI-powered file operations:
//! - File organization by topic/similarity
//! - Duplicate detection
//! - Cleanup analysis
//! - Similar file discovery
//!
//! All operations follow the Propose-Review-Apply pattern for safety.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ragfs_core::{Embedder, VectorStore};
use ragfs_embed::CandleEmbedder;
use ragfs_fuse::semantic::{
    ActionType, CleanupAnalysis, CleanupCandidate, CleanupReason, DuplicateEntry, DuplicateGroup,
    DuplicateGroups, OrganizeRequest, OrganizeStrategy, PlanAction, PlanImpact, PlanStatus,
    SemanticConfig, SemanticManager, SemanticPlan, SimilarFile, SimilarFilesResult,
};
use ragfs_store::LanceStore;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Python Types
// ============================================================================

/// Python wrapper for OrganizeStrategy.
#[pyclass(name = "OrganizeStrategy")]
#[derive(Clone)]
pub struct PyOrganizeStrategy {
    inner: OrganizeStrategy,
}

#[pymethods]
impl PyOrganizeStrategy {
    /// Create "by_topic" strategy - groups files by semantic similarity.
    #[staticmethod]
    fn by_topic() -> Self {
        Self {
            inner: OrganizeStrategy::ByTopic,
        }
    }

    /// Create "by_type" strategy - groups files by file type.
    #[staticmethod]
    fn by_type() -> Self {
        Self {
            inner: OrganizeStrategy::ByType,
        }
    }

    /// Create "by_project" strategy - groups files by project structure.
    #[staticmethod]
    fn by_project() -> Self {
        Self {
            inner: OrganizeStrategy::ByProject,
        }
    }

    /// Create "custom" strategy with specific categories.
    #[staticmethod]
    fn custom(categories: Vec<String>) -> Self {
        Self {
            inner: OrganizeStrategy::Custom { categories },
        }
    }

    #[getter]
    fn name(&self) -> String {
        match &self.inner {
            OrganizeStrategy::ByTopic => "by_topic".to_string(),
            OrganizeStrategy::ByType => "by_type".to_string(),
            OrganizeStrategy::ByProject => "by_project".to_string(),
            OrganizeStrategy::Custom { .. } => "custom".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("OrganizeStrategy('{}')", self.name())
    }
}

/// Python wrapper for OrganizeRequest.
#[pyclass(name = "OrganizeRequest")]
#[derive(Clone)]
pub struct PyOrganizeRequest {
    /// Directory scope (relative to source root)
    #[pyo3(get, set)]
    pub scope: String,
    /// Organization strategy
    #[pyo3(get)]
    pub strategy: PyOrganizeStrategy,
    /// Maximum number of groups to create
    #[pyo3(get, set)]
    pub max_groups: usize,
    /// Minimum similarity threshold for grouping (0.0-1.0)
    #[pyo3(get, set)]
    pub similarity_threshold: f32,
}

#[pymethods]
impl PyOrganizeRequest {
    #[new]
    #[pyo3(signature = (scope, strategy, max_groups=10, similarity_threshold=0.7))]
    fn new(
        scope: String,
        strategy: PyOrganizeStrategy,
        max_groups: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            scope,
            strategy,
            max_groups,
            similarity_threshold,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OrganizeRequest(scope='{}', strategy={}, max_groups={}, similarity_threshold={})",
            self.scope,
            self.strategy.name(),
            self.max_groups,
            self.similarity_threshold
        )
    }
}

impl From<&PyOrganizeRequest> for OrganizeRequest {
    fn from(req: &PyOrganizeRequest) -> Self {
        OrganizeRequest {
            scope: PathBuf::from(&req.scope),
            strategy: req.strategy.inner.clone(),
            max_groups: req.max_groups,
            similarity_threshold: req.similarity_threshold,
        }
    }
}

/// Python wrapper for ActionType.
#[pyclass(name = "ActionType")]
#[derive(Clone)]
pub struct PyActionType {
    /// Action type: "move", "mkdir", "delete", "symlink"
    #[pyo3(get)]
    pub action_type: String,
    /// Source path (for move)
    #[pyo3(get)]
    pub from_path: Option<String>,
    /// Target path (for move, mkdir, delete, symlink)
    #[pyo3(get)]
    pub to_path: Option<String>,
    /// Link path (for symlink)
    #[pyo3(get)]
    pub link_path: Option<String>,
}

#[pymethods]
impl PyActionType {
    fn __repr__(&self) -> String {
        match self.action_type.as_str() {
            "move" => format!(
                "ActionType.Move(from='{}', to='{}')",
                self.from_path.as_deref().unwrap_or(""),
                self.to_path.as_deref().unwrap_or("")
            ),
            "mkdir" => format!(
                "ActionType.Mkdir(path='{}')",
                self.to_path.as_deref().unwrap_or("")
            ),
            "delete" => format!(
                "ActionType.Delete(path='{}')",
                self.to_path.as_deref().unwrap_or("")
            ),
            "symlink" => format!(
                "ActionType.Symlink(target='{}', link='{}')",
                self.to_path.as_deref().unwrap_or(""),
                self.link_path.as_deref().unwrap_or("")
            ),
            _ => format!("ActionType('{}')", self.action_type),
        }
    }
}

impl From<&ActionType> for PyActionType {
    fn from(action: &ActionType) -> Self {
        match action {
            ActionType::Move { from, to } => Self {
                action_type: "move".to_string(),
                from_path: Some(from.to_string_lossy().to_string()),
                to_path: Some(to.to_string_lossy().to_string()),
                link_path: None,
            },
            ActionType::Mkdir { path } => Self {
                action_type: "mkdir".to_string(),
                from_path: None,
                to_path: Some(path.to_string_lossy().to_string()),
                link_path: None,
            },
            ActionType::Delete { path } => Self {
                action_type: "delete".to_string(),
                from_path: None,
                to_path: Some(path.to_string_lossy().to_string()),
                link_path: None,
            },
            ActionType::Symlink { target, link } => Self {
                action_type: "symlink".to_string(),
                from_path: None,
                to_path: Some(target.to_string_lossy().to_string()),
                link_path: Some(link.to_string_lossy().to_string()),
            },
        }
    }
}

/// Python wrapper for PlanAction.
#[pyclass(name = "PlanAction")]
#[derive(Clone)]
pub struct PyPlanAction {
    /// The action to perform
    #[pyo3(get)]
    pub action: PyActionType,
    /// Confidence score (0.0-1.0)
    #[pyo3(get)]
    pub confidence: f32,
    /// Reason for this action
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl PyPlanAction {
    fn __repr__(&self) -> String {
        format!(
            "PlanAction(action={}, confidence={:.2}, reason='{}')",
            self.action.__repr__(),
            self.confidence,
            self.reason
        )
    }
}

impl From<&PlanAction> for PyPlanAction {
    fn from(action: &PlanAction) -> Self {
        Self {
            action: PyActionType::from(&action.action),
            confidence: action.confidence,
            reason: action.reason.clone(),
        }
    }
}

/// Python wrapper for PlanImpact.
#[pyclass(name = "PlanImpact")]
#[derive(Clone)]
pub struct PyPlanImpact {
    /// Total files affected
    #[pyo3(get)]
    pub files_affected: usize,
    /// Directories created
    #[pyo3(get)]
    pub dirs_created: usize,
    /// Files moved
    #[pyo3(get)]
    pub files_moved: usize,
    /// Files deleted
    #[pyo3(get)]
    pub files_deleted: usize,
}

#[pymethods]
impl PyPlanImpact {
    fn __repr__(&self) -> String {
        format!(
            "PlanImpact(files_affected={}, dirs_created={}, files_moved={}, files_deleted={})",
            self.files_affected, self.dirs_created, self.files_moved, self.files_deleted
        )
    }
}

impl From<&PlanImpact> for PyPlanImpact {
    fn from(impact: &PlanImpact) -> Self {
        Self {
            files_affected: impact.files_affected,
            dirs_created: impact.dirs_created,
            files_moved: impact.files_moved,
            files_deleted: impact.files_deleted,
        }
    }
}

/// Python wrapper for SemanticPlan.
#[pyclass(name = "SemanticPlan")]
#[derive(Clone)]
pub struct PySemanticPlan {
    /// Unique plan identifier
    #[pyo3(get)]
    pub id: String,
    /// When the plan was created (ISO 8601)
    #[pyo3(get)]
    pub created_at: String,
    /// Human-readable description
    #[pyo3(get)]
    pub description: String,
    /// Status: "pending", "approved", "rejected", "completed", "failed"
    #[pyo3(get)]
    pub status: String,
    /// Error message (if failed)
    #[pyo3(get)]
    pub error: Option<String>,
    /// Proposed actions
    #[pyo3(get)]
    pub actions: Vec<PyPlanAction>,
    /// Impact summary
    #[pyo3(get)]
    pub impact: PyPlanImpact,
}

#[pymethods]
impl PySemanticPlan {
    fn __repr__(&self) -> String {
        format!(
            "SemanticPlan(id='{}', status='{}', actions={}, description='{}')",
            self.id,
            self.status,
            self.actions.len(),
            self.description
        )
    }

    /// Convert to dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("created_at", &self.created_at)?;
        dict.set_item("description", &self.description)?;
        dict.set_item("status", &self.status)?;
        dict.set_item("num_actions", self.actions.len())?;
        if let Some(ref err) = self.error {
            dict.set_item("error", err)?;
        }
        Ok(dict.into())
    }

    /// Check if the plan is pending approval.
    fn is_pending(&self) -> bool {
        self.status == "pending"
    }

    /// Check if the plan completed successfully.
    fn is_completed(&self) -> bool {
        self.status == "completed"
    }

    /// Check if the plan failed.
    fn is_failed(&self) -> bool {
        self.status == "failed"
    }
}

impl From<&SemanticPlan> for PySemanticPlan {
    fn from(plan: &SemanticPlan) -> Self {
        let (status, error) = match &plan.status {
            PlanStatus::Pending => ("pending".to_string(), None),
            PlanStatus::Approved => ("approved".to_string(), None),
            PlanStatus::Rejected => ("rejected".to_string(), None),
            PlanStatus::Completed => ("completed".to_string(), None),
            PlanStatus::Failed { error } => ("failed".to_string(), Some(error.clone())),
        };

        Self {
            id: plan.id.to_string(),
            created_at: plan.created_at.to_rfc3339(),
            description: plan.description.clone(),
            status,
            error,
            actions: plan.actions.iter().map(PyPlanAction::from).collect(),
            impact: PyPlanImpact::from(&plan.impact),
        }
    }
}

/// Python wrapper for SimilarFile.
#[pyclass(name = "SimilarFile")]
#[derive(Clone)]
pub struct PySimilarFile {
    /// File path
    #[pyo3(get)]
    pub path: String,
    /// Similarity score (0.0-1.0)
    #[pyo3(get)]
    pub similarity: f32,
    /// Content preview
    #[pyo3(get)]
    pub preview: Option<String>,
}

#[pymethods]
impl PySimilarFile {
    fn __repr__(&self) -> String {
        format!(
            "SimilarFile(path='{}', similarity={:.3})",
            self.path, self.similarity
        )
    }
}

impl From<&SimilarFile> for PySimilarFile {
    fn from(sf: &SimilarFile) -> Self {
        Self {
            path: sf.path.to_string_lossy().to_string(),
            similarity: sf.similarity,
            preview: sf.preview.clone(),
        }
    }
}

/// Python wrapper for SimilarFilesResult.
#[pyclass(name = "SimilarFilesResult")]
#[derive(Clone)]
pub struct PySimilarFilesResult {
    /// Source file path
    #[pyo3(get)]
    pub source: String,
    /// Similar files found
    #[pyo3(get)]
    pub similar: Vec<PySimilarFile>,
}

#[pymethods]
impl PySimilarFilesResult {
    fn __repr__(&self) -> String {
        format!(
            "SimilarFilesResult(source='{}', found={})",
            self.source,
            self.similar.len()
        )
    }
}

impl From<&SimilarFilesResult> for PySimilarFilesResult {
    fn from(result: &SimilarFilesResult) -> Self {
        Self {
            source: result.source.to_string_lossy().to_string(),
            similar: result.similar.iter().map(PySimilarFile::from).collect(),
        }
    }
}

/// Python wrapper for DuplicateEntry.
#[pyclass(name = "DuplicateEntry")]
#[derive(Clone)]
pub struct PyDuplicateEntry {
    /// File path
    #[pyo3(get)]
    pub path: String,
    /// Similarity to representative (0.0-1.0)
    #[pyo3(get)]
    pub similarity: f32,
    /// File size in bytes
    #[pyo3(get)]
    pub size_bytes: u64,
}

#[pymethods]
impl PyDuplicateEntry {
    fn __repr__(&self) -> String {
        format!(
            "DuplicateEntry(path='{}', similarity={:.3}, size={})",
            self.path, self.similarity, self.size_bytes
        )
    }
}

impl From<&DuplicateEntry> for PyDuplicateEntry {
    fn from(entry: &DuplicateEntry) -> Self {
        Self {
            path: entry.path.to_string_lossy().to_string(),
            similarity: entry.similarity,
            size_bytes: entry.size_bytes,
        }
    }
}

/// Python wrapper for DuplicateGroup.
#[pyclass(name = "DuplicateGroup")]
#[derive(Clone)]
pub struct PyDuplicateGroup {
    /// Group identifier
    #[pyo3(get)]
    pub id: String,
    /// Representative file (keep this one)
    #[pyo3(get)]
    pub representative: String,
    /// Similar files (candidates for removal)
    #[pyo3(get)]
    pub duplicates: Vec<PyDuplicateEntry>,
    /// Total wasted bytes
    #[pyo3(get)]
    pub wasted_bytes: u64,
}

#[pymethods]
impl PyDuplicateGroup {
    fn __repr__(&self) -> String {
        format!(
            "DuplicateGroup(representative='{}', duplicates={}, wasted_bytes={})",
            self.representative,
            self.duplicates.len(),
            self.wasted_bytes
        )
    }
}

impl From<&DuplicateGroup> for PyDuplicateGroup {
    fn from(group: &DuplicateGroup) -> Self {
        Self {
            id: group.id.to_string(),
            representative: group.representative.to_string_lossy().to_string(),
            duplicates: group
                .duplicates
                .iter()
                .map(PyDuplicateEntry::from)
                .collect(),
            wasted_bytes: group.wasted_bytes,
        }
    }
}

/// Python wrapper for DuplicateGroups.
#[pyclass(name = "DuplicateGroups")]
#[derive(Clone)]
pub struct PyDuplicateGroups {
    /// When the analysis was performed (ISO 8601)
    #[pyo3(get)]
    pub analyzed_at: String,
    /// Similarity threshold used
    #[pyo3(get)]
    pub threshold: f32,
    /// Groups of similar files
    #[pyo3(get)]
    pub groups: Vec<PyDuplicateGroup>,
    /// Total potential savings in bytes
    #[pyo3(get)]
    pub potential_savings_bytes: u64,
}

#[pymethods]
impl PyDuplicateGroups {
    fn __repr__(&self) -> String {
        format!(
            "DuplicateGroups(groups={}, potential_savings={})",
            self.groups.len(),
            self.potential_savings_bytes
        )
    }
}

impl From<&DuplicateGroups> for PyDuplicateGroups {
    fn from(groups: &DuplicateGroups) -> Self {
        Self {
            analyzed_at: groups.analyzed_at.to_rfc3339(),
            threshold: groups.threshold,
            groups: groups.groups.iter().map(PyDuplicateGroup::from).collect(),
            potential_savings_bytes: groups.potential_savings_bytes,
        }
    }
}

/// Python wrapper for CleanupCandidate.
#[pyclass(name = "CleanupCandidate")]
#[derive(Clone)]
pub struct PyCleanupCandidate {
    /// File path
    #[pyo3(get)]
    pub path: String,
    /// Cleanup reason type
    #[pyo3(get)]
    pub reason_type: String,
    /// Additional reason details
    #[pyo3(get)]
    pub reason_details: Option<String>,
    /// Confidence score (0.0-1.0)
    #[pyo3(get)]
    pub confidence: f32,
    /// File size in bytes
    #[pyo3(get)]
    pub size_bytes: u64,
}

#[pymethods]
impl PyCleanupCandidate {
    fn __repr__(&self) -> String {
        format!(
            "CleanupCandidate(path='{}', reason='{}', confidence={:.2}, size={})",
            self.path, self.reason_type, self.confidence, self.size_bytes
        )
    }
}

impl From<&CleanupCandidate> for PyCleanupCandidate {
    fn from(candidate: &CleanupCandidate) -> Self {
        let (reason_type, reason_details) = match &candidate.reason {
            CleanupReason::Duplicate {
                similar_to,
                similarity,
            } => (
                "duplicate".to_string(),
                Some(format!(
                    "similar to {} ({}%)",
                    similar_to.display(),
                    (similarity * 100.0) as u32
                )),
            ),
            CleanupReason::Stale { last_accessed } => (
                "stale".to_string(),
                Some(format!("last accessed: {}", last_accessed.to_rfc3339())),
            ),
            CleanupReason::Temporary => ("temporary".to_string(), None),
            CleanupReason::Generated { source } => (
                "generated".to_string(),
                Some(format!("from {}", source.display())),
            ),
            CleanupReason::Empty => ("empty".to_string(), None),
        };

        Self {
            path: candidate.path.to_string_lossy().to_string(),
            reason_type,
            reason_details,
            confidence: candidate.confidence,
            size_bytes: candidate.size_bytes,
        }
    }
}

/// Python wrapper for CleanupAnalysis.
#[pyclass(name = "CleanupAnalysis")]
#[derive(Clone)]
pub struct PyCleanupAnalysis {
    /// When the analysis was performed (ISO 8601)
    #[pyo3(get)]
    pub analyzed_at: String,
    /// Total files analyzed
    #[pyo3(get)]
    pub total_files: usize,
    /// Cleanup candidates
    #[pyo3(get)]
    pub candidates: Vec<PyCleanupCandidate>,
    /// Potential savings in bytes
    #[pyo3(get)]
    pub potential_savings_bytes: u64,
}

#[pymethods]
impl PyCleanupAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "CleanupAnalysis(total_files={}, candidates={}, potential_savings={})",
            self.total_files,
            self.candidates.len(),
            self.potential_savings_bytes
        )
    }
}

impl From<&CleanupAnalysis> for PyCleanupAnalysis {
    fn from(analysis: &CleanupAnalysis) -> Self {
        Self {
            analyzed_at: analysis.analyzed_at.to_rfc3339(),
            total_files: analysis.total_files,
            candidates: analysis
                .candidates
                .iter()
                .map(PyCleanupCandidate::from)
                .collect(),
            potential_savings_bytes: analysis.potential_savings_bytes,
        }
    }
}

// ============================================================================
// Semantic Manager
// ============================================================================

/// Semantic manager for AI-powered file operations.
///
/// This is the core RAGFS innovation: AI-powered file organization with
/// the Propose-Review-Apply pattern for safety.
///
/// Example:
///
/// ```python
/// from ragfs import RagfsSemanticManager, OrganizeStrategy, OrganizeRequest
///
/// # Create manager (requires initialization)
/// semantic = RagfsSemanticManager("/path/to/source", "/path/to/db")
/// await semantic.init()
///
/// # Find similar files
/// result = await semantic.find_similar("/path/to/file.txt", k=5)
/// for f in result.similar:
///     print(f"{f.path}: {f.similarity:.2%}")
///
/// # Create an organization plan (NOT executed yet!)
/// request = OrganizeRequest(
///     scope="./docs",
///     strategy=OrganizeStrategy.by_topic(),
///     max_groups=5
/// )
/// plan = await semantic.create_organize_plan(request)
/// print(f"Plan {plan.id} proposes {len(plan.actions)} actions")
///
/// # Review the plan
/// for action in plan.actions:
///     print(f"  {action.action} - {action.reason}")
///
/// # Approve and execute (or reject)
/// if user_approves:
///     result = await semantic.approve_plan(plan.id)
///     print(f"Executed: {result.status}")
/// else:
///     await semantic.reject_plan(plan.id)
/// ```
#[pyclass(name = "RagfsSemanticManager")]
pub struct RagfsSemanticManager {
    manager: Arc<RwLock<Option<SemanticManager>>>,
    source_path: PathBuf,
    db_path: PathBuf,
    model_path: PathBuf,
    embedder: Arc<RwLock<Option<Arc<CandleEmbedder>>>>,
    store: Arc<LanceStore>,
    config: SemanticConfig,
}

#[pymethods]
impl RagfsSemanticManager {
    /// Create a new semantic manager.
    ///
    /// Args:
    ///     source_path: Path to the source directory to manage.
    ///     db_path: Path to the LanceDB database.
    ///     model_path: Optional path for embedding models.
    ///     duplicate_threshold: Similarity threshold for duplicates (default: 0.95).
    ///     similar_limit: Max results for similar search (default: 10).
    #[new]
    #[pyo3(signature = (source_path, db_path, model_path=None, duplicate_threshold=0.95, similar_limit=10))]
    fn new(
        source_path: String,
        db_path: String,
        model_path: Option<String>,
        duplicate_threshold: f32,
        similar_limit: usize,
    ) -> Self {
        let model_path = model_path.map(PathBuf::from).unwrap_or_else(|| {
            directories::ProjectDirs::from("", "", "ragfs")
                .map(|dirs| dirs.data_dir().join("models"))
                .unwrap_or_else(|| PathBuf::from(".ragfs/models"))
        });

        let data_dir = directories::ProjectDirs::from("", "", "ragfs")
            .map(|dirs| dirs.data_local_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".ragfs"));

        let config = SemanticConfig {
            duplicate_threshold,
            similar_limit,
            plan_retention_hours: 24,
            data_dir,
        };

        let store = LanceStore::new(PathBuf::from(&db_path), 384); // gte-small dimension

        Self {
            manager: Arc::new(RwLock::new(None)),
            source_path: PathBuf::from(source_path),
            db_path: PathBuf::from(db_path),
            model_path,
            embedder: Arc::new(RwLock::new(None)),
            store: Arc::new(store),
            config,
        }
    }

    /// Initialize the semantic manager (loads model and connects to store).
    fn init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let embedder_lock = self.embedder.clone();
        let manager_lock = self.manager.clone();
        let store = self.store.clone();
        let model_path = self.model_path.clone();
        let source_path = self.source_path.clone();
        let config = self.config.clone();

        future_into_py(py, async move {
            // Initialize embedder
            let embedder: Arc<dyn Embedder> = {
                let mut guard = embedder_lock.write().await;
                if guard.is_none() {
                    let new_embedder = CandleEmbedder::new(model_path);
                    new_embedder.init().await.map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to initialize embedder: {e}"))
                    })?;
                    let arc_embedder: Arc<CandleEmbedder> = Arc::new(new_embedder);
                    *guard = Some(arc_embedder.clone());
                    arc_embedder as Arc<dyn Embedder>
                } else {
                    guard.as_ref().unwrap().clone() as Arc<dyn Embedder>
                }
            };

            // Initialize store
            store
                .init()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize store: {e}")))?;

            let store_arc: Arc<dyn VectorStore> = store;

            // Create semantic manager
            let semantic_manager =
                SemanticManager::new(source_path, Some(store_arc), Some(embedder), Some(config));

            *manager_lock.write().await = Some(semantic_manager);

            Ok(())
        })
    }

    /// Check if the manager is initialized and available.
    fn is_available<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            Ok(guard
                .as_ref()
                .is_some_and(ragfs_fuse::SemanticManager::is_available))
        })
    }

    /// Find files similar to a given path.
    ///
    /// Args:
    ///     path: Path to the file to find similar files for.
    ///     k: Number of similar files to return (optional).
    ///
    /// Returns:
    ///     SimilarFilesResult with list of similar files.
    #[pyo3(signature = (path, k=None))]
    fn find_similar<'py>(
        &self,
        py: Python<'py>,
        path: String,
        k: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();
        let path = PathBuf::from(path);

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let result = manager.find_similar(&path).await.map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to find similar files: {e}"))
            })?;

            let mut py_result = PySimilarFilesResult::from(&result);

            // Apply k limit if specified
            if let Some(limit) = k {
                py_result.similar.truncate(limit);
            }

            Ok(py_result)
        })
    }

    /// Analyze files for cleanup candidates.
    ///
    /// Returns:
    ///     CleanupAnalysis with candidates for cleanup.
    fn analyze_cleanup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let analysis = manager
                .analyze_cleanup()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to analyze cleanup: {e}")))?;

            Ok(PyCleanupAnalysis::from(&analysis))
        })
    }

    /// Find duplicate file groups.
    ///
    /// Returns:
    ///     DuplicateGroups with groups of similar files.
    fn find_duplicates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let groups = manager
                .find_duplicates()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to find duplicates: {e}")))?;

            Ok(PyDuplicateGroups::from(&groups))
        })
    }

    /// Create an organization plan (NOT executed until approved).
    ///
    /// Args:
    ///     request: OrganizeRequest specifying scope and strategy.
    ///
    /// Returns:
    ///     SemanticPlan with proposed actions.
    fn create_organize_plan<'py>(
        &self,
        py: Python<'py>,
        request: PyOrganizeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let rust_request = OrganizeRequest::from(&request);
            let plan = manager
                .create_organize_plan(rust_request)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create plan: {e}")))?;

            Ok(PySemanticPlan::from(&plan))
        })
    }

    /// List all pending plans awaiting approval.
    ///
    /// Returns:
    ///     List of SemanticPlan objects.
    fn list_pending_plans<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let plans = manager.list_pending_plans().await;
            let py_plans: Vec<PySemanticPlan> = plans.iter().map(PySemanticPlan::from).collect();

            Ok(py_plans)
        })
    }

    /// Get a specific plan by ID.
    ///
    /// Args:
    ///     plan_id: The plan ID.
    ///
    /// Returns:
    ///     SemanticPlan or None if not found.
    fn get_plan<'py>(&self, py: Python<'py>, plan_id: String) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let uuid = uuid::Uuid::parse_str(&plan_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid plan ID: {e}")))?;

            let plan = manager.get_plan(uuid).await;
            Ok(plan.as_ref().map(PySemanticPlan::from))
        })
    }

    /// Approve and execute a plan.
    ///
    /// This executes all proposed actions. Each action is reversible via
    /// the safety layer (undo support).
    ///
    /// Args:
    ///     plan_id: The plan ID to approve.
    ///
    /// Returns:
    ///     SemanticPlan with updated status.
    fn approve_plan<'py>(&self, py: Python<'py>, plan_id: String) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let uuid = uuid::Uuid::parse_str(&plan_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid plan ID: {e}")))?;

            let plan = manager
                .approve_plan(uuid)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to approve plan: {e}")))?;

            Ok(PySemanticPlan::from(&plan))
        })
    }

    /// Reject a plan (no changes made).
    ///
    /// Args:
    ///     plan_id: The plan ID to reject.
    ///
    /// Returns:
    ///     SemanticPlan with updated status.
    fn reject_plan<'py>(&self, py: Python<'py>, plan_id: String) -> PyResult<Bound<'py, PyAny>> {
        let manager_lock = self.manager.clone();

        future_into_py(py, async move {
            let guard = manager_lock.read().await;
            let manager = guard.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Manager not initialized. Call init() first.")
            })?;

            let uuid = uuid::Uuid::parse_str(&plan_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid plan ID: {e}")))?;

            let plan = manager
                .reject_plan(uuid)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to reject plan: {e}")))?;

            Ok(PySemanticPlan::from(&plan))
        })
    }

    #[getter]
    fn source_path(&self) -> String {
        self.source_path.to_string_lossy().to_string()
    }

    #[getter]
    fn db_path(&self) -> String {
        self.db_path.to_string_lossy().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RagfsSemanticManager(source='{}', db='{}')",
            self.source_path.display(),
            self.db_path.display()
        )
    }
}
