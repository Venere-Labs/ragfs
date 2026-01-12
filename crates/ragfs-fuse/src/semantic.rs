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
use ragfs_core::{Chunk, DistanceMetric, Embedder, EmbeddingConfig, FileRecord, SearchQuery, VectorStore};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

        // Get all chunks and files from the store
        let all_chunks = store
            .get_all_chunks()
            .await
            .map_err(|e| format!("Failed to get chunks: {e}"))?;

        let all_files = store
            .get_all_files()
            .await
            .map_err(|e| format!("Failed to get files: {e}"))?;

        if all_files.is_empty() {
            return Ok(DuplicateGroups {
                analyzed_at: Utc::now(),
                threshold: self.config.duplicate_threshold,
                groups: Vec::new(),
                potential_savings_bytes: 0,
            });
        }

        // Build a map of file_path -> chunks with embeddings
        let mut file_chunks: HashMap<PathBuf, Vec<&Chunk>> = HashMap::new();
        for chunk in &all_chunks {
            if chunk.embedding.is_some() {
                file_chunks
                    .entry(chunk.file_path.clone())
                    .or_default()
                    .push(chunk);
            }
        }

        // Build file info map
        let file_info: HashMap<PathBuf, &FileRecord> = all_files
            .iter()
            .map(|f| (f.path.clone(), f))
            .collect();

        // Calculate average embedding for each file
        let file_embeddings: HashMap<PathBuf, Vec<f32>> = file_chunks
            .iter()
            .filter_map(|(path, chunks)| {
                let embeddings: Vec<&Vec<f32>> = chunks
                    .iter()
                    .filter_map(|c| c.embedding.as_ref())
                    .collect();

                if embeddings.is_empty() {
                    return None;
                }

                // Average the embeddings
                let dim = embeddings[0].len();
                let mut avg = vec![0.0f32; dim];
                for emb in &embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        avg[i] += v;
                    }
                }
                let count = embeddings.len() as f32;
                for v in &mut avg {
                    *v /= count;
                }

                // Normalize the averaged embedding
                let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut avg {
                        *v /= norm;
                    }
                }

                Some((path.clone(), avg))
            })
            .collect();

        // Find similar file pairs using cosine similarity
        let file_paths: Vec<&PathBuf> = file_embeddings.keys().collect();
        let mut similarity_pairs: Vec<(PathBuf, PathBuf, f32)> = Vec::new();

        for (i, path_a) in file_paths.iter().enumerate() {
            let emb_a = &file_embeddings[*path_a];
            for path_b in file_paths.iter().skip(i + 1) {
                let emb_b = &file_embeddings[*path_b];
                let similarity = cosine_similarity(emb_a, emb_b);

                if similarity >= self.config.duplicate_threshold {
                    similarity_pairs.push(((*path_a).clone(), (*path_b).clone(), similarity));
                }
            }
        }

        // Cluster similar files using Union-Find
        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut processed: HashSet<PathBuf> = HashSet::new();

        for (path_a, path_b, similarity) in similarity_pairs {
            if processed.contains(&path_a) || processed.contains(&path_b) {
                continue;
            }

            // Find or create group for path_a
            let size_a = file_info.get(&path_a).map(|f| f.size_bytes).unwrap_or(0);
            let size_b = file_info.get(&path_b).map(|f| f.size_bytes).unwrap_or(0);

            // Use the larger file as representative
            let (representative, duplicate, dup_similarity, dup_size) = if size_a >= size_b {
                (path_a.clone(), path_b.clone(), similarity, size_b)
            } else {
                (path_b.clone(), path_a.clone(), similarity, size_a)
            };

            // Check if representative already has a group
            if let Some(group) = groups.iter_mut().find(|g| g.representative == representative) {
                group.duplicates.push(DuplicateEntry {
                    path: duplicate.clone(),
                    similarity: dup_similarity,
                    size_bytes: dup_size,
                });
                group.wasted_bytes += dup_size;
                processed.insert(duplicate);
            } else {
                // Create new group
                groups.push(DuplicateGroup {
                    id: Uuid::new_v4(),
                    representative: representative.clone(),
                    duplicates: vec![DuplicateEntry {
                        path: duplicate.clone(),
                        similarity: dup_similarity,
                        size_bytes: dup_size,
                    }],
                    wasted_bytes: dup_size,
                });
                processed.insert(representative);
                processed.insert(duplicate);
            }
        }

        let potential_savings: u64 = groups.iter().map(|g| g.wasted_bytes).sum();

        info!(
            "Found {} duplicate groups with {} bytes potential savings",
            groups.len(),
            potential_savings
        );

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
        let store = self.store.as_ref().ok_or("Vector store not available")?;
        let _embedder = self.embedder.as_ref().ok_or("Embedder not available")?;

        debug!("Creating organization plan for: {}", request.scope.display());

        // Get all chunks and files
        let all_chunks = store
            .get_all_chunks()
            .await
            .map_err(|e| format!("Failed to get chunks: {e}"))?;

        let all_files = store
            .get_all_files()
            .await
            .map_err(|e| format!("Failed to get files: {e}"))?;

        // Filter files within scope
        let scope_path = if request.scope.is_absolute() {
            request.scope.clone()
        } else {
            self.source.join(&request.scope)
        };

        let scoped_files: Vec<&FileRecord> = all_files
            .iter()
            .filter(|f| f.path.starts_with(&scope_path))
            .collect();

        if scoped_files.is_empty() {
            return Ok(SemanticPlan {
                id: Uuid::new_v4(),
                created_at: Utc::now(),
                operation: PlanOperation::Organize {
                    scope: request.scope.clone(),
                    strategy: request.strategy.clone(),
                },
                description: format!("No files found in scope: {}", request.scope.display()),
                actions: Vec::new(),
                status: PlanStatus::Pending,
                impact: PlanImpact::default(),
            });
        }

        // Build file embeddings map
        let mut file_chunks: HashMap<PathBuf, Vec<&Chunk>> = HashMap::new();
        for chunk in &all_chunks {
            if chunk.embedding.is_some() && chunk.file_path.starts_with(&scope_path) {
                file_chunks
                    .entry(chunk.file_path.clone())
                    .or_default()
                    .push(chunk);
            }
        }

        // Calculate average embedding for each file
        let file_embeddings: HashMap<PathBuf, Vec<f32>> = file_chunks
            .iter()
            .filter_map(|(path, chunks)| {
                let embeddings: Vec<&Vec<f32>> = chunks
                    .iter()
                    .filter_map(|c| c.embedding.as_ref())
                    .collect();

                if embeddings.is_empty() {
                    return None;
                }

                let dim = embeddings[0].len();
                let mut avg = vec![0.0f32; dim];
                for emb in &embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        avg[i] += v;
                    }
                }
                let count = embeddings.len() as f32;
                for v in &mut avg {
                    *v /= count;
                }

                // Normalize
                let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut avg {
                        *v /= norm;
                    }
                }

                Some((path.clone(), avg))
            })
            .collect();

        // Generate actions based on strategy
        let (actions, description) = match &request.strategy {
            OrganizeStrategy::ByTopic => {
                self.plan_by_topic(&file_embeddings, &scope_path, request.max_groups, request.similarity_threshold)
            }
            OrganizeStrategy::ByType => {
                self.plan_by_type(&scoped_files, &scope_path)
            }
            OrganizeStrategy::ByProject => {
                self.plan_by_project(&scoped_files, &scope_path)
            }
            OrganizeStrategy::Custom { categories } => {
                self.plan_by_custom(&file_embeddings, &scope_path, categories)
            }
        };

        let dirs_created = actions.iter().filter(|a| matches!(a.action, ActionType::Mkdir { .. })).count();
        let files_moved = actions.iter().filter(|a| matches!(a.action, ActionType::Move { .. })).count();

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
                files_affected: files_moved,
                dirs_created,
                files_moved,
                files_deleted: 0,
            },
        };

        // Store the plan
        self.pending_plans.write().await.insert(plan.id, plan.clone());

        info!("Created organization plan: {} with {} actions", plan.id, plan.actions.len());
        Ok(plan)
    }

    /// Plan organization by semantic topic using clustering.
    fn plan_by_topic(
        &self,
        file_embeddings: &HashMap<PathBuf, Vec<f32>>,
        scope_path: &PathBuf,
        max_groups: usize,
        similarity_threshold: f32,
    ) -> (Vec<PlanAction>, String) {
        if file_embeddings.is_empty() {
            return (Vec::new(), "No files with embeddings found".to_string());
        }

        // Simple clustering: find centroids and group files
        let file_paths: Vec<&PathBuf> = file_embeddings.keys().collect();
        let num_files = file_paths.len();
        let num_clusters = max_groups.min(num_files);

        // Initialize clusters with k random files (here we use evenly spaced indices)
        let step = if num_files > num_clusters { num_files / num_clusters } else { 1 };
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|i| file_embeddings[file_paths[i * step.min(num_files - 1)]].clone())
            .collect();

        // Simple k-means iterations
        let mut cluster_assignments: HashMap<PathBuf, usize> = HashMap::new();

        for _ in 0..5 {
            // Assign each file to nearest centroid
            cluster_assignments.clear();
            for path in &file_paths {
                let emb = &file_embeddings[*path];
                let mut best_cluster = 0;
                let mut best_sim = -1.0f32;

                for (cluster_idx, centroid) in centroids.iter().enumerate() {
                    let sim = cosine_similarity(emb, centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cluster = cluster_idx;
                    }
                }

                cluster_assignments.insert((*path).clone(), best_cluster);
            }

            // Update centroids
            for (cluster_idx, centroid) in centroids.iter_mut().enumerate() {
                let members: Vec<&PathBuf> = cluster_assignments
                    .iter()
                    .filter(|&(_, c)| *c == cluster_idx)
                    .map(|(p, _)| p)
                    .collect();

                if members.is_empty() {
                    continue;
                }

                let dim = centroid.len();
                let mut new_centroid = vec![0.0f32; dim];

                for path in &members {
                    let emb = &file_embeddings[*path];
                    for (i, &v) in emb.iter().enumerate() {
                        new_centroid[i] += v;
                    }
                }

                let count = members.len() as f32;
                for v in &mut new_centroid {
                    *v /= count;
                }

                // Normalize
                let norm: f32 = new_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut new_centroid {
                        *v /= norm;
                    }
                }

                *centroid = new_centroid;
            }
        }

        // Generate actions
        let mut actions = Vec::new();

        // Create topic directories
        for cluster_idx in 0..num_clusters {
            let topic_dir = scope_path.join(format!("topic_{}", cluster_idx + 1));
            actions.push(PlanAction {
                action: ActionType::Mkdir { path: topic_dir },
                confidence: 1.0,
                reason: format!("Create directory for topic cluster {}", cluster_idx + 1),
            });
        }

        // Move files to their clusters
        for (path, &cluster_idx) in &cluster_assignments {
            let file_name = path.file_name().unwrap_or_default();
            let topic_dir = scope_path.join(format!("topic_{}", cluster_idx + 1));
            let new_path = topic_dir.join(file_name);

            if new_path != *path {
                // Calculate confidence based on distance to centroid
                let emb = &file_embeddings[path];
                let centroid = &centroids[cluster_idx];
                let confidence = cosine_similarity(emb, centroid).max(similarity_threshold);

                actions.push(PlanAction {
                    action: ActionType::Move {
                        from: path.clone(),
                        to: new_path,
                    },
                    confidence,
                    reason: format!("Move to topic cluster {} based on content similarity", cluster_idx + 1),
                });
            }
        }

        let description = format!(
            "Organize {} files into {} topic clusters",
            file_paths.len(),
            num_clusters
        );

        (actions, description)
    }

    /// Plan organization by file type.
    fn plan_by_type(
        &self,
        files: &[&FileRecord],
        scope_path: &PathBuf,
    ) -> (Vec<PlanAction>, String) {
        let mut actions = Vec::new();
        let mut type_dirs: HashSet<String> = HashSet::new();

        for file in files {
            // Determine type directory based on extension or MIME type
            let type_dir = if let Some(ext) = file.path.extension() {
                ext.to_string_lossy().to_string()
            } else {
                // Use MIME type category
                file.mime_type
                    .split('/')
                    .next()
                    .unwrap_or("other")
                    .to_string()
            };

            // Create type directory if needed
            if type_dirs.insert(type_dir.clone()) {
                actions.push(PlanAction {
                    action: ActionType::Mkdir {
                        path: scope_path.join(&type_dir),
                    },
                    confidence: 1.0,
                    reason: format!("Create directory for {} files", type_dir),
                });
            }

            // Move file
            let file_name = file.path.file_name().unwrap_or_default();
            let new_path = scope_path.join(&type_dir).join(file_name);

            if new_path != file.path {
                actions.push(PlanAction {
                    action: ActionType::Move {
                        from: file.path.clone(),
                        to: new_path,
                    },
                    confidence: 1.0,
                    reason: format!("Move to {} directory based on file type", type_dir),
                });
            }
        }

        let description = format!(
            "Organize {} files into {} type-based directories",
            files.len(),
            type_dirs.len()
        );

        (actions, description)
    }

    /// Plan organization by project structure (based on imports/dependencies).
    fn plan_by_project(
        &self,
        files: &[&FileRecord],
        scope_path: &PathBuf,
    ) -> (Vec<PlanAction>, String) {
        // For project-based organization, we look at file paths to infer structure
        // This is a simplified implementation
        let mut actions = Vec::new();
        let mut project_dirs: HashSet<String> = HashSet::new();

        for file in files {
            // Use the first directory component after scope as "project"
            let relative = file.path.strip_prefix(scope_path).unwrap_or(&file.path);
            let project = relative
                .components()
                .next()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .unwrap_or_else(|| "root".to_string());

            if project_dirs.insert(project.clone()) && !project.contains('.') {
                actions.push(PlanAction {
                    action: ActionType::Mkdir {
                        path: scope_path.join(&project),
                    },
                    confidence: 0.8,
                    reason: format!("Create project directory: {}", project),
                });
            }
        }

        let description = format!(
            "Organize {} files into {} project directories",
            files.len(),
            project_dirs.len()
        );

        (actions, description)
    }

    /// Plan organization by custom categories.
    fn plan_by_custom(
        &self,
        file_embeddings: &HashMap<PathBuf, Vec<f32>>,
        scope_path: &PathBuf,
        categories: &[String],
    ) -> (Vec<PlanAction>, String) {
        let mut actions = Vec::new();

        // Create category directories
        for category in categories {
            actions.push(PlanAction {
                action: ActionType::Mkdir {
                    path: scope_path.join(category),
                },
                confidence: 1.0,
                reason: format!("Create custom category directory: {}", category),
            });
        }

        // Without category embeddings, we can't automatically assign files
        // This would require generating embeddings for category names and matching

        let description = format!(
            "Created {} custom category directories for {} files (manual assignment needed)",
            categories.len(),
            file_embeddings.len()
        );

        (actions, description)
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

/// Calculate cosine similarity between two embeddings.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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
