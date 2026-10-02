//! FUSE filesystem implementation for RAGFS.
//!
//! This crate provides a FUSE (Filesystem in Userspace) interface that allows
//! semantic search capabilities and agent file operations to be accessed through
//! standard filesystem operations.
//!
//! # Features
//!
//! - **Passthrough**: Real files are accessible at their original locations
//! - **Virtual Query Interface**: Special `.ragfs/` directory for semantic queries
//! - **Agent Operations**: Structured file operations via `.ops/` with JSON feedback
//! - **Safety Layer**: Soft delete, audit logging, and undo via `.safety/`
//! - **Semantic Operations**: AI-powered file organization via `.semantic/`
//!
//! # Virtual Directory Structure
//!
//! ```text
//! /mountpoint/
//! ├── real_files/                # Passthrough to source directory
//! │
//! └── .ragfs/                    # Virtual control directory
//!     ├── .query/<text>          # Semantic query → JSON results
//!     ├── .search/<text>         # Search results
//!     ├── .similar/<path>        # Find similar files
//!     ├── .index                 # Index statistics (JSON)
//!     ├── .config                # Current configuration (JSON)
//!     ├── .reindex               # Write path to trigger reindex
//!     ├── .help                  # Usage documentation
//!     │
//!     ├── .ops/                  # Agent file operations
//!     │   ├── .create            # Write: "path\ncontent"
//!     │   ├── .delete            # Write: "path"
//!     │   ├── .move              # Write: "src\ndst"
//!     │   ├── .batch             # Write: JSON BatchRequest
//!     │   └── .result            # Read: JSON OperationResult
//!     │
//!     ├── .safety/               # Protection layer
//!     │   ├── .trash/            # Soft-deleted files (recoverable)
//!     │   ├── .history           # Audit log (JSONL)
//!     │   └── .undo              # Write: operation_id to undo
//!     │
//!     └── .semantic/             # AI-powered operations
//!         ├── .organize          # Write: OrganizeRequest JSON
//!         ├── .similar           # Write: path → find similar
//!         ├── .cleanup           # Read: CleanupAnalysis JSON
//!         ├── .dedupe            # Read: DuplicateGroups JSON
//!         ├── .pending/          # Proposed plans directory
//!         ├── .approve           # Write: plan_id to execute
//!         └── .reject            # Write: plan_id to cancel
//! ```
//!
//! # Basic Usage
//!
//! ```bash
//! # Mount the filesystem
//! ragfs mount /source /mnt/ragfs -f
//!
//! # Query via filesystem
//! cat "/mnt/ragfs/.ragfs/.query/how to authenticate"
//!
//! # Check index status
//! cat /mnt/ragfs/.ragfs/.index
//! ```
//!
//! # Agent Operations (.ops/)
//!
//! ```bash
//! # Create a file with feedback
//! echo -e "docs/new.md\n# New Document" > /mnt/ragfs/.ragfs/.ops/.create
//! cat /mnt/ragfs/.ragfs/.ops/.result  # JSON result with undo_id
//!
//! # Delete a file (uses soft delete)
//! echo "docs/old.md" > /mnt/ragfs/.ragfs/.ops/.delete
//!
//! # Move/rename a file
//! echo -e "old/path.txt\nnew/path.txt" > /mnt/ragfs/.ragfs/.ops/.move
//!
//! # Batch operations
//! echo '{"operations":[{"Create":{"path":"a.txt","content":"A"}}],"atomic":true}' \
//!     > /mnt/ragfs/.ragfs/.ops/.batch
//! ```
//!
//! # Safety Layer (.safety/)
//!
//! ```bash
//! # View operation history
//! cat /mnt/ragfs/.ragfs/.safety/.history
//!
//! # List deleted files in trash
//! ls /mnt/ragfs/.ragfs/.safety/.trash/
//!
//! # Undo an operation
//! echo "550e8400-e29b-41d4-a716-446655440000" > /mnt/ragfs/.ragfs/.safety/.undo
//!
//! # Restore from trash (write "restore" to trash entry)
//! echo "restore" > /mnt/ragfs/.ragfs/.safety/.trash/<uuid>
//! ```
//!
//! # Semantic Operations (.semantic/)
//!
//! ```bash
//! # Find files similar to a given file
//! echo "src/main.rs" > /mnt/ragfs/.ragfs/.semantic/.similar
//! cat /mnt/ragfs/.ragfs/.semantic/.similar  # JSON results
//!
//! # Propose file organization
//! echo '{"scope":"docs/","strategy":"by_topic"}' > /mnt/ragfs/.ragfs/.semantic/.organize
//!
//! # Review pending plans
//! ls /mnt/ragfs/.ragfs/.semantic/.pending/
//! cat /mnt/ragfs/.ragfs/.semantic/.pending/<plan_id>
//!
//! # Approve or reject a plan
//! echo "<plan_id>" > /mnt/ragfs/.ragfs/.semantic/.approve
//! echo "<plan_id>" > /mnt/ragfs/.ragfs/.semantic/.reject
//!
//! # View cleanup analysis
//! cat /mnt/ragfs/.ragfs/.semantic/.cleanup
//!
//! # View duplicate detection
//! cat /mnt/ragfs/.ragfs/.semantic/.dedupe
//! ```
//!
//! # Rust API Example
//!
//! ```rust,ignore
//! use ragfs_fuse::RagFs;
//!
//! // Create filesystem with full RAG capabilities
//! let fs = RagFs::with_rag(source_path, store, embedder, runtime, reindex_sender);
//!
//! // Mount
//! fuser::mount2(fs, mountpoint, &options)?;
//! ```

pub mod filesystem;
pub mod inode;
pub mod ops;
pub mod safety;
pub mod semantic;

pub use filesystem::RagFs;
pub use inode::{InodeKind, InodeTable};
pub use ops::{BatchRequest, BatchResult, Operation, OperationResult, OpsManager};
pub use safety::{
    HistoryEntry, HistoryOperation, SafetyConfig, SafetyManager, TrashEntry, UndoData,
};
pub use semantic::{
    CleanupAnalysis, DuplicateGroups, OrganizeRequest, OrganizeStrategy, SemanticConfig,
    SemanticManager, SemanticPlan, SimilarFilesResult,
};
