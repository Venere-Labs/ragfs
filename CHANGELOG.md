# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Python bindings**: New `ragfs-python` crate with PyO3 bindings
  - `RAGFSIndex` class for indexing and querying
  - `RAGFSStore` for direct vector store access
  - Async support via `pyo3-async-runtimes`
  - Framework adapters for LangChain, LlamaIndex, Haystack
  - Build with `maturin build`
- **FUSE capabilities exposed to Python** (Propose-Review-Apply pattern for AI agents):
  - `RagfsSafetyManager`: Soft delete, trash, history, undo operations
  - `RagfsSemanticManager`: AI-powered file organization, similar/duplicate detection
  - `RagfsOpsManager`: Structured file operations with JSON feedback
  - `OrganizeStrategy`, `OrganizeRequest`, `SemanticPlan`, `PlanAction` types
- **LlamaIndex FUSE-aware integration**:
  - `RagfsSafeVectorStore`: VectorStore with safety layer integration
  - `RagfsOrganizer`: Semantic organizer with Propose-Review-Apply workflow
- **Haystack FUSE-aware integration**:
  - `RagfsSafeDocumentStore`: DocumentStore with soft delete and undo
  - `RagfsOrganizer`: Haystack component for AI-powered file organization
- **MCP Server with complete FUSE capabilities** (18 tools):
  - Safety Layer: `ragfs_delete_to_trash`, `ragfs_list_trash`, `ragfs_restore_from_trash`, `ragfs_get_history`, `ragfs_undo`
  - Semantic Operations: `ragfs_find_duplicates`, `ragfs_analyze_cleanup`
  - Approval Workflow: `ragfs_propose_organization`, `ragfs_propose_cleanup`, `ragfs_list_pending_plans`, `ragfs_get_plan`, `ragfs_approve_plan`, `ragfs_reject_plan`
  - Batch Operations: `ragfs_batch_operations`
- **CI/CD Python support**:
  - Python tests for all integrations (3.10-3.13)
  - Pre-release validation workflow
  - Python release workflow for PyPI publishing
- **Security hardening**:
  - `SECURITY.md` with vulnerability reporting guidelines
  - `deny.toml` for cargo-deny license and advisory checks
- **VectorStore iteration**: New methods for bulk operations
  - `get_all_chunks()`: Returns all chunks in the store
  - `get_all_files()`: Returns all file records in the store
- **BlipCaptioner**: BLIP-based image captioning using Candle ML
  - Auto-download from HuggingFace Hub (`Salesforce/blip-image-captioning-base`)
  - Image preprocessing with CLIP normalization
  - Autoregressive caption generation
  - Requires `vision` feature flag
- **Semantic organization**: AI-powered file management features
  - `find_duplicates()`: Detect similar files using embedding cosine similarity
  - `suggest_organization()`: AI-powered folder structure suggestions
  - Configurable similarity thresholds
- **Daemonization**: `ragfs mount` now properly forks to background when run without `--foreground`
  - PID file stored in `$XDG_RUNTIME_DIR/ragfs/` or `~/.cache/ragfs/run/`
  - Logs written to `~/.cache/ragfs/logs/`
  - Graceful unmount via `fusermount -u <mountpoint>`
- **TOML configuration**: Load settings from `~/.config/ragfs/config.toml`
  - `ragfs config show` - Display current configuration
  - `ragfs config init` - Generate sample config file
  - `ragfs config path` - Show config file location
  - `--config` global flag to specify custom config path
- **PDF image extraction**: Extract embedded images from PDFs using lopdf
  - Supports JPEG (DCTDecode), PNG (FlateDecode), JPEG2000 (JPXDecode)
  - CMYK to RGB conversion
  - Memory limits: 100 images max, 50MB total, 50px minimum dimension
- **Virtual `.ragfs/.help` file**: Usage documentation accessible via FUSE mount
- **NoopEmbedder**: Testing embedder available without Candle dependency
- **MemoryStore**: In-memory vector store available without LanceDB dependency
- **Atomic batch rollback**: Batch operations with `atomic: true` now automatically roll back on failure
  - All successful operations are undone if any operation fails
  - Rollback details included in batch result (`rollback_performed`, `rollback_details`)
- **Mkdir and Symlink operations**: New operation types in OpsManager
  - `mkdir`: Create directories via `.ops/` interface
  - `symlink`: Create symbolic links (Unix-only)
- **Semantic plan persistence**: Plans are saved to disk and survive restarts
  - Plans stored in `~/.local/share/ragfs/plans/{index_hash}/`
  - Automatic cleanup of expired plans (configurable retention)
- **Semantic plan execution**: Approved plans are now executed automatically
  - Actions run sequentially via OpsManager
  - Each action generates an undo_id for manual reversal if needed
  - Plan status reflects execution result (Completed/Failed)

### Changed
- **BREAKING**: Feature flags restructured for optional ML backends
  - `candle` feature (default): Enables `CandleEmbedder` in ragfs-embed
  - `lancedb` feature (default): Enables `LanceStore` in ragfs-store
  - `vision` feature: Enables `BlipCaptioner` for image captioning
  - `full` feature: Enables all optional features
  - Build profiles: `cargo build` (default), `--features full`, `--no-default-features` (minimal)

### Fixed
- N/A

## [0.2.0] - 2026-01-11

### Added
- **Multimodal support**: PDF extraction, image handling
- **Advanced chunking**: CodeChunker with tree-sitter for syntax-aware code splitting
- **Semantic chunking**: SemanticChunker for document structure-aware chunking
- **Comprehensive test suite**: 291 tests across all crates
  - ragfs-core: 59 tests (types, errors)
  - ragfs-extract: 56 tests (text, PDF, image extractors)
  - ragfs-chunker: 54 tests (fixed-size, code, semantic chunkers)
  - ragfs-fuse: 65 tests (inode management, filesystem helpers)
  - ragfs-index: 18 tests (indexing pipeline)
  - ragfs-store: 14 tests (LanceDB operations)
  - ragfs-embed: 13 tests (embeddings)
  - ragfs-query: 9 tests (query execution)
- **Reindex trigger**: `.ragfs/.reindex` file write support for on-demand reindexing
- **Data integrity**: Full round-trip for line ranges, embeddings, timestamps in LanceDB
- **Benchmarks**: Criterion-based benchmarks for embedding, search, and indexing

### Changed
- Improved error handling with proper error chain propagation
- Enhanced InodeTable with proper FUSE reference counting
- Better MIME type preservation through the indexing pipeline

### Fixed
- Line range parsing from LanceDB results
- Embedding vector round-trip in search results
- Timestamp parsing for file records

## [0.1.0] - 2025-01-11

### Added
- Initial release
- Core traits and types for RAG filesystem (`ragfs-core`)
- Content extraction for text files (`ragfs-extract`)
- Fixed-size chunking strategy (`ragfs-chunker`)
- Local embedding generation with Candle/gte-small (`ragfs-embed`)
- LanceDB-based vector storage (`ragfs-store`)
- Indexing pipeline with file watching (`ragfs-index`)
- Query execution with semantic search (`ragfs-query`)
- FUSE filesystem interface (`ragfs-fuse`)
- CLI with `index`, `query`, `mount`, and `status` commands (`ragfs`)
- JSON and text output formats
- Verbose logging support

### Technical Details
- Rust edition 2024, MSRV 1.88
- Async-first design with Tokio
- 384-dimensional embeddings (gte-small model)
- LanceDB for vector storage with ANN search
- Content-addressed storage with blake3 hashing
- Registry pattern for extensible extractors and chunkers

[Unreleased]: https://github.com/Venere-Labs/ragfs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Venere-Labs/ragfs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Venere-Labs/ragfs/releases/tag/v0.1.0
