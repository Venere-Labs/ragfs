//! FUSE filesystem implementation for RAGFS.
//!
//! This crate provides a FUSE (Filesystem in Userspace) interface that allows
//! semantic search capabilities to be accessed through standard filesystem
//! operations.
//!
//! # Features
//!
//! - **Passthrough**: Real files are accessible at their original locations
//! - **Virtual Query Interface**: Special `.ragfs/` directory for queries
//! - **Real-time Results**: Query results returned as file content
//!
//! # Virtual Directory Structure
//!
//! ```text
//! /mountpoint/
//! ├── .ragfs/                    # Virtual control directory
//! │   ├── .index                 # Index statistics (JSON)
//! │   └── .query/                # Query interface
//! │       └── <query>            # Read returns search results
//! ├── real_dir/                  # Passthrough to source
//! └── real_file.txt              # Passthrough to source
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ragfs_fuse::RagFs;
//!
//! // Create filesystem with RAG capabilities
//! let fs = RagFs::with_rag(source_path, store, embedder, runtime);
//!
//! // Mount
//! fuser::mount2(fs, mountpoint, &options)?;
//! ```
//!
//! # Usage
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

pub mod filesystem;
pub mod inode;

pub use filesystem::RagFs;
pub use inode::{InodeKind, InodeTable};
