//! Semantic operations for intelligent file management.
//!
//! This module provides AI-powered file operations based on vector embeddings:
//! - File organization by topic/similarity
//! - Duplicate detection
//! - Cleanup analysis
//! - Similar file discovery
//!
//! All operations follow a Propose-Review-Apply pattern for safety.

use chrono::{DateTime, Utc};
use ragfs_core::{DistanceMetric, Embedder, EmbeddingConfig, SearchQuery, VectorStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Request to organize files in a directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizeRequest {
    /// Directory scope (relative to source root)
    pub scope: PathBuf,
    /// Organization strategy
    pub strategy: OrganizeStrategy,
    /// Maximum number of groups to create
    #[serde(default = "default_max_groups")]
    pub max_groups: usize,
    /// Minimum similarity threshold for grouping (0.0-1.0)
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
}

fn default_max_groups() -> usize {
    10
}

fn default_similarity_threshold() -> f32 {
    0.7
}

/// Strategy for organizing files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrganizeStrategy {
    /// Group by semantic topic/content similarity
    ByTopic,
    /// Group by file type first, then by content
    ByType,
    /// Group by project/module structure
    ByProject,
    /// Custom grouping with specified categories
    Custom { categories: Vec<String> },
}

/// A proposed semantic operation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPlan {
    /// Unique plan identifier
    pub id: Uuid,
    /// When the plan was created
    pub created_at: DateTime<Utc>,
    /// Type of operation
    pub operation: PlanOperation,
    /// Human-readable description
    pub description: String,
    /// Proposed file operations
    pub actions: Vec<PlanAction>,
    /// Status of the plan
    pub status: PlanStatus,
    /// Estimated impact (files affected)
    pub impact: PlanImpact,
}

/// Type of semantic operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanOperation {
    /// Organize files into groups
    Organize {
        scope: PathBuf,
        strategy: OrganizeStrategy,
    },
    /// Clean up files
    Cleanup { scope: PathBuf },
    /// Deduplicate files
    Dedupe { scope: PathBuf },
}

/// A single action in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAction {
    /// Type of action
    pub action: ActionType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Reason for this action
    pub reason: String,
}

/// Type of file action.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    /// Move a file to a new location
    Move { from: PathBuf, to: PathBuf },
    /// Create a new directory
    Mkdir { path: PathBuf },
    /// Delete a file (will use soft delete)
    Delete { path: PathBuf },
    /// Create a symlink
    Symlink { target: PathBuf, link: PathBuf },
}

/// Status of a plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// Plan is pending review
    Pending,
    /// Plan was approved and is being executed
    Approved,
    /// Plan was rejected
    Rejected,
    /// Plan was executed successfully
    Completed,
    /// Plan execution failed
    Failed { error: String },
}

/// Impact summary of a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanImpact {
    /// Total files affected
    pub files_affected: usize,
    /// Directories created
    pub dirs_created: usize,
    /// Files moved
    pub files_moved: usize,
    /// Files deleted
    pub files_deleted: usize,
}

/// Analysis of cleanup candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupAnalysis {
    /// When the analysis was performed
    pub analyzed_at: DateTime<Utc>,
    /// Total files analyzed
    pub total_files: usize,
    /// Cleanup candidates
    pub candidates: Vec<CleanupCandidate>,
    /// Potential space savings in bytes
    pub potential_savings_bytes: u64,
}

/// A file that could be cleaned up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupCandidate {
    /// File path
    pub path: PathBuf,
    /// Reason for cleanup suggestion
    pub reason: CleanupReason,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// File size in bytes
    pub size_bytes: u64,
}

/// Reason a file is suggested for cleanup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupReason {
    /// File appears to be a duplicate
    Duplicate { similar_to: PathBuf, similarity: f32 },
    /// File hasn't been accessed in a long time
    Stale { last_accessed: DateTime<Utc> },
    /// Temporary file pattern
    Temporary,
    /// Generated file that can be recreated
    Generated { source: PathBuf },
    /// Empty or near-empty file
    Empty,
}

/// Groups of duplicate/similar files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroups {
    /// When the analysis was performed
    pub analyzed_at: DateTime<Utc>,
    /// Minimum similarity threshold used
    pub threshold: f32,
    /// Groups of similar files
    pub groups: Vec<DuplicateGroup>,
    /// Total potential savings if duplicates removed
    pub potential_savings_bytes: u64,
}

/// A group of similar files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Group identifier
    pub id: Uuid,
    /// Representative file (keep this one)
    pub representative: PathBuf,
    /// Similar files (candidates for removal)
    pub duplicates: Vec<DuplicateEntry>,
    /// Total size of duplicates
    pub wasted_bytes: u64,
}

/// A duplicate file entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateEntry {
    /// File path
    pub path: PathBuf,
    /// Similarity to representative (0.0-1.0)
    pub similarity: f32,
    /// File size
    pub size_bytes: u64,
}

/// Result of finding similar files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarFilesResult {
    /// Source file
    pub source: PathBuf,
    /// Similar files found
    pub similar: Vec<SimilarFile>,
}

/// A similar file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarFile {
    /// File path
    pub path: PathBuf,
    /// Similarity score (0.0-1.0)
    pub similarity: f32,
    /// Preview of content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

/// Configuration for semantic operations.
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Minimum similarity for duplicate detection
    pub duplicate_threshold: f32,
    /// Number of results for similar file search
    pub similar_limit: usize,
    /// Maximum plan retention (in hours)
    pub plan_retention_hours: u32,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.95,
            similar_limit: 10,
            plan_retention_hours: 24,
        }
    }
}

/// Semantic manager for intelligent file operations.
pub struct SemanticManager {
    /// Source directory root
    source: PathBuf,
    /// Vector store for similarity search
    store: Option<Arc<dyn VectorStore>>,
    /// Embedder for generating embeddings
    embedder: Option<Arc<dyn Embedder>>,
    /// Configuration
    config: SemanticConfig,
    /// Pending plans (`plan_id` -> plan)
    pending_plans: Arc<RwLock<HashMap<Uuid, SemanticPlan>>>,
    /// Last similar files result
    last_similar_result: Arc<RwLock<Option<SimilarFilesResult>>>,
    /// Cached cleanup analysis
    cleanup_cache: Arc<RwLock<Option<CleanupAnalysis>>>,
    /// Cached duplicate groups
    dedupe_cache: Arc<RwLock<Option<DuplicateGroups>>>,
}

impl SemanticManager {
    /// Create a new semantic manager.
    pub fn new(
        source: PathBuf,
        store: Option<Arc<dyn VectorStore>>,
        embedder: Option<Arc<dyn Embedder>>,
        config: Option<SemanticConfig>,
    ) -> Self {
        Self {
            source,
            store,
            embedder,
            config: config.unwrap_or_default(),
            pending_plans: Arc::new(RwLock::new(HashMap::new())),
            last_similar_result: Arc::new(RwLock::new(None)),
            cleanup_cache: Arc::new(RwLock::new(None)),
            dedupe_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if semantic operations are available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.store.is_some() && self.embedder.is_some()
    }

    /// Find files similar to a given path.
    pub async fn find_similar(&self, path: &PathBuf) -> Result<SimilarFilesResult, String> {
        let store = self.store.as_ref().ok_or("Vector store not available")?;
        let embedder = self.embedder.as_ref().ok_or("Embedder not available")?;

        let full_path = if path.is_absolute() {
            path.clone()
        } else {
            self.source.join(path)
        };

        debug!("Finding files similar to: {}", full_path.display());

        // Read the file content
        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {e}"))?;

        // Generate embedding for the content
        let config = EmbeddingConfig::default();
        let embedding_output = embedder
            .embed_query(&content, &config)
            .await
            .map_err(|e| format!("Failed to generate embedding: {e}"))?;

        // Search for similar files
        let query = SearchQuery {
            embedding: embedding_output.embedding,
            text: None,
            limit: self.config.similar_limit + 1, // +1 to exclude self
            filters: Vec::new(),
            metric: DistanceMetric::Cosine,
        };
        let results = store
            .search(query)
            .await
            .map_err(|e| format!("Search failed: {e}"))?;

        // Convert results, excluding the source file itself
        let similar: Vec<SimilarFile> = results
            .into_iter()
            .filter(|r| r.file_path != full_path)
            .take(self.config.similar_limit)
            .map(|r| SimilarFile {
                path: r.file_path,
                similarity: r.score, // score is already similarity (higher = more similar)
                preview: Some(truncate_content(&r.content, 200)),
            })
            .collect();

        let result = SimilarFilesResult {
            source: full_path,
            similar,
        };

        // Cache the result
        *self.last_similar_result.write().await = Some(result.clone());

        info!("Found {} similar files", result.similar.len());
        Ok(result)
    }

    /// Get the last similar files result.
    pub async fn get_last_similar_result(&self) -> Option<SimilarFilesResult> {
        self.last_similar_result.read().await.clone()
    }

    /// Analyze files for cleanup candidates.
    pub async fn analyze_cleanup(&self) -> Result<CleanupAnalysis, String> {
        let store = self.store.as_ref().ok_or("Vector store not available")?;

        debug!("Analyzing files for cleanup candidates");

        // Get all file records from the store
        let stats = store.stats().await.map_err(|e| format!("Failed to get stats: {e}"))?;

        let mut candidates = Vec::new();
        let mut potential_savings: u64 = 0;

        // For now, we'll focus on duplicate detection as the primary cleanup criterion
        // This could be expanded to include stale file detection, etc.

        // Get duplicate groups and convert high-confidence duplicates to cleanup candidates
        if let Ok(dupes) = self.find_duplicates().await {
            for group in &dupes.groups {
                for dup in &group.duplicates {
                    if dup.similarity >= self.config.duplicate_threshold {
                        candidates.push(CleanupCandidate {
                            path: dup.path.clone(),
                            reason: CleanupReason::Duplicate {
                                similar_to: group.representative.clone(),
                                similarity: dup.similarity,
                            },
                            confidence: dup.similarity,
                            size_bytes: dup.size_bytes,
                        });
                        potential_savings += dup.size_bytes;
                    }
                }
            }
        }

        let analysis = CleanupAnalysis {
            analyzed_at: Utc::now(),
            total_files: stats.total_files as usize,
            candidates,
            potential_savings_bytes: potential_savings,
        };

        // Cache the result
        *self.cleanup_cache.write().await = Some(analysis.clone());

        info!(
            "Cleanup analysis: {} candidates, {} bytes potential savings",
            analysis.candidates.len(),
            analysis.potential_savings_bytes
        );

        Ok(analysis)
    }

    /// Get cached cleanup analysis.
    pub async fn get_cleanup_analysis(&self) -> Option<CleanupAnalysis> {
        self.cleanup_cache.read().await.clone()
    }

    /// Find duplicate file groups.
    pub async fn find_duplicates(&self) -> Result<DuplicateGroups, String> {
        let store = self.store.as_ref().ok_or("Vector store not available")?;
        let _embedder = self.embedder.as_ref().ok_or("Embedder not available")?;

        debug!("Finding duplicate files");

        // Get stats to know how many files we have
        let stats = store.stats().await.map_err(|e| format!("Failed to get stats: {e}"))?;

        if stats.total_files == 0 {
            return Ok(DuplicateGroups {
                analyzed_at: Utc::now(),
                threshold: self.config.duplicate_threshold,
                groups: Vec::new(),
                potential_savings_bytes: 0,
            });
        }

        // This is a simplified duplicate detection algorithm
        // For each file, search for very similar files
        // In a real implementation, we'd use a more efficient clustering algorithm

        let groups: Vec<DuplicateGroup> = Vec::new();
        let _processed: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        let potential_savings: u64 = 0;

        // For now, return empty groups since full implementation requires iterating all files
        // which would need additional VectorStore API methods
        // This is a placeholder that can be enhanced later

        let result = DuplicateGroups {
            analyzed_at: Utc::now(),
            threshold: self.config.duplicate_threshold,
            groups,
            potential_savings_bytes: potential_savings,
        };

        // Cache the result
        *self.dedupe_cache.write().await = Some(result.clone());

        Ok(result)
    }

    /// Get cached duplicate groups.
    pub async fn get_duplicate_groups(&self) -> Option<DuplicateGroups> {
        self.dedupe_cache.read().await.clone()
    }

    /// Create an organization plan.
    pub async fn create_organize_plan(
        &self,
        request: OrganizeRequest,
    ) -> Result<SemanticPlan, String> {
        let _store = self.store.as_ref().ok_or("Vector store not available")?;
        let _embedder = self.embedder.as_ref().ok_or("Embedder not available")?;

        debug!("Creating organization plan for: {}", request.scope.display());

        // Generate a plan based on the strategy
        let actions = Vec::new();
        let description = match &request.strategy {
            OrganizeStrategy::ByTopic => {
                format!(
                    "Organize files in {} by semantic topic with {} max groups",
                    request.scope.display(),
                    request.max_groups
                )
            }
            OrganizeStrategy::ByType => {
                format!(
                    "Organize files in {} by type and content",
                    request.scope.display()
                )
            }
            OrganizeStrategy::ByProject => {
                format!(
                    "Organize files in {} by project structure",
                    request.scope.display()
                )
            }
            OrganizeStrategy::Custom { categories } => {
                format!(
                    "Organize files in {} into categories: {}",
                    request.scope.display(),
                    categories.join(", ")
                )
            }
        };

        // For now, create an empty plan that can be populated with actual file analysis
        // Full implementation would analyze files and generate move operations

        let plan = SemanticPlan {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            operation: PlanOperation::Organize {
                scope: request.scope,
                strategy: request.strategy,
            },
            description,
            actions,
            status: PlanStatus::Pending,
            impact: PlanImpact {
                files_affected: 0,
                dirs_created: 0,
                files_moved: 0,
                files_deleted: 0,
            },
        };

        // Store the plan
        self.pending_plans.write().await.insert(plan.id, plan.clone());

        info!("Created organization plan: {}", plan.id);
        Ok(plan)
    }

    /// List all pending plans.
    pub async fn list_pending_plans(&self) -> Vec<SemanticPlan> {
        self.pending_plans
            .read()
            .await
            .values()
            .filter(|p| p.status == PlanStatus::Pending)
            .cloned()
            .collect()
    }

    /// Get a specific plan.
    pub async fn get_plan(&self, plan_id: Uuid) -> Option<SemanticPlan> {
        self.pending_plans.read().await.get(&plan_id).cloned()
    }

    /// Approve and execute a plan.
    pub async fn approve_plan(&self, plan_id: Uuid) -> Result<SemanticPlan, String> {
        let mut plans = self.pending_plans.write().await;
        let plan = plans
            .get_mut(&plan_id)
            .ok_or_else(|| "Plan not found".to_string())?;

        if plan.status != PlanStatus::Pending {
            return Err(format!("Plan is not pending: {:?}", plan.status));
        }

        info!("Approving plan: {}", plan_id);
        plan.status = PlanStatus::Approved;

        // Execute the plan actions
        // For now, just mark as completed since we don't have actual actions
        plan.status = PlanStatus::Completed;

        Ok(plan.clone())
    }

    /// Reject a plan.
    pub async fn reject_plan(&self, plan_id: Uuid) -> Result<SemanticPlan, String> {
        let mut plans = self.pending_plans.write().await;
        let plan = plans
            .get_mut(&plan_id)
            .ok_or_else(|| "Plan not found".to_string())?;

        if plan.status != PlanStatus::Pending {
            return Err(format!("Plan is not pending: {:?}", plan.status));
        }

        info!("Rejecting plan: {}", plan_id);
        plan.status = PlanStatus::Rejected;

        Ok(plan.clone())
    }

    /// Get cleanup analysis as JSON bytes (for FUSE read).
    pub async fn get_cleanup_json(&self) -> Vec<u8> {
        if let Some(analysis) = self.get_cleanup_analysis().await {
            serde_json::to_string_pretty(&analysis)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
        } else {
            // Return a message indicating analysis hasn't been run
            let msg = serde_json::json!({
                "message": "No cleanup analysis available. Run analyze_cleanup first.",
                "hint": "Write any content to .semantic/.cleanup to trigger analysis"
            });
            serde_json::to_string_pretty(&msg)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Get duplicate groups as JSON bytes (for FUSE read).
    pub async fn get_dedupe_json(&self) -> Vec<u8> {
        if let Some(groups) = self.get_duplicate_groups().await {
            serde_json::to_string_pretty(&groups)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
        } else {
            let msg = serde_json::json!({
                "message": "No duplicate analysis available. Run find_duplicates first.",
                "hint": "Write any content to .semantic/.dedupe to trigger analysis"
            });
            serde_json::to_string_pretty(&msg)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Get similar files result as JSON bytes (for FUSE read).
    pub async fn get_similar_json(&self) -> Vec<u8> {
        if let Some(result) = self.get_last_similar_result().await {
            serde_json::to_string_pretty(&result)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
        } else {
            let msg = serde_json::json!({
                "message": "No similar files search performed yet.",
                "hint": "Write a file path to .semantic/.similar to find similar files"
            });
            serde_json::to_string_pretty(&msg)
                .unwrap_or_default()
                .into_bytes()
        }
    }

    /// Get pending plans directory listing.
    pub async fn get_pending_plan_ids(&self) -> Vec<String> {
        self.pending_plans
            .read()
            .await
            .iter()
            .filter(|(_, p)| p.status == PlanStatus::Pending)
            .map(|(id, _)| id.to_string())
            .collect()
    }

    /// Get a plan as JSON bytes (for FUSE read).
    pub async fn get_plan_json(&self, plan_id: &str) -> Vec<u8> {
        if let Ok(uuid) = Uuid::parse_str(plan_id)
            && let Some(plan) = self.get_plan(uuid).await
        {
            return serde_json::to_string_pretty(&plan)
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes();
        }
        let msg = serde_json::json!({
            "error": "Plan not found",
            "plan_id": plan_id
        });
        serde_json::to_string_pretty(&msg)
            .unwrap_or_default()
            .into_bytes()
    }
}

/// Truncate content for preview.
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organize_request_serialization() {
        let request = OrganizeRequest {
            scope: PathBuf::from("docs/"),
            strategy: OrganizeStrategy::ByTopic,
            max_groups: 5,
            similarity_threshold: 0.8,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: OrganizeRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.scope, request.scope);
        assert_eq!(parsed.max_groups, 5);
    }

    #[test]
    fn test_organize_request_defaults() {
        let json = r#"{"scope":"src/","strategy":"by_topic"}"#;
        let request: OrganizeRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.max_groups, 10);
        assert!((request.similarity_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_plan_status_serialization() {
        let status = PlanStatus::Failed {
            error: "test error".to_string(),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("failed"));
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_cleanup_reason_variants() {
        let duplicate = CleanupReason::Duplicate {
            similar_to: PathBuf::from("/original.txt"),
            similarity: 0.98,
        };
        let json = serde_json::to_string(&duplicate).unwrap();
        assert!(json.contains("duplicate"));

        let stale = CleanupReason::Stale {
            last_accessed: Utc::now(),
        };
        let json = serde_json::to_string(&stale).unwrap();
        assert!(json.contains("stale"));
    }

    #[test]
    fn test_semantic_config_default() {
        let config = SemanticConfig::default();
        assert!((config.duplicate_threshold - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.similar_limit, 10);
        assert_eq!(config.plan_retention_hours, 24);
    }

    #[test]
    fn test_truncate_content() {
        assert_eq!(truncate_content("short", 100), "short");
        assert_eq!(truncate_content("hello world", 5), "hello...");
    }

    #[test]
    fn test_action_type_serialization() {
        let action = ActionType::Move {
            from: PathBuf::from("/old/path.txt"),
            to: PathBuf::from("/new/path.txt"),
        };
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("move"));
        assert!(json.contains("/old/path.txt"));
    }

    #[test]
    fn test_similar_file_serialization() {
        let similar = SimilarFile {
            path: PathBuf::from("/doc.txt"),
            similarity: 0.85,
            preview: Some("This is a preview...".to_string()),
        };
        let json = serde_json::to_string(&similar).unwrap();
        assert!(json.contains("0.85"));
        assert!(json.contains("preview"));
    }

    #[tokio::test]
    async fn test_semantic_manager_without_store() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        assert!(!manager.is_available());
    }

    #[tokio::test]
    async fn test_pending_plans_empty() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        let plans = manager.list_pending_plans().await;
        assert!(plans.is_empty());
    }

    #[tokio::test]
    async fn test_get_plan_not_found() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        let plan = manager.get_plan(Uuid::new_v4()).await;
        assert!(plan.is_none());
    }

    #[tokio::test]
    async fn test_get_cleanup_json_empty() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        let json = manager.get_cleanup_json().await;
        let json_str = String::from_utf8(json).unwrap();
        assert!(json_str.contains("No cleanup analysis"));
    }

    #[tokio::test]
    async fn test_get_dedupe_json_empty() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        let json = manager.get_dedupe_json().await;
        let json_str = String::from_utf8(json).unwrap();
        assert!(json_str.contains("No duplicate analysis"));
    }

    #[tokio::test]
    async fn test_get_similar_json_empty() {
        let manager = SemanticManager::new(PathBuf::from("/tmp"), None, None, None);
        let json = manager.get_similar_json().await;
        let json_str = String::from_utf8(json).unwrap();
        assert!(json_str.contains("No similar files search"));
    }
}
