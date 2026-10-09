# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Path jail (`resolve_under_root`) now lives in `ragfs-core` and is enforced by `SafetyManager` (`soft_delete` / `restore` / `undo`), so Python `RagfsSafetyManager` and MCP `ragfs_delete_to_trash` cannot escape the source root.
- Documented the advisory-response ladder, `ignore:` ownership, and §1.2 review cadence (next review by 2026-12-31) in `SECURITY.md`.

### Removed
- Optional `pdf_oxide` extractor feature. PDF text and images continue to use `PdfExtractor` (`pdf-extract` + `lopdf`). `--features pdf_oxide` is no longer valid.

### Changed
- Agent docs (`CLAUDE.md`, `AGENTS.md`) list all 10 Cargo workspace crates plus `ragfs-mcp`. `docs/PERFORMANCE.md` documents `[embedding].model`, CLI-only `--force`, and `--hybrid` vs `[query].hybrid`.
- Embedding dimension and model id for `LanceStore` come from `Embedder::dimension()` / `model_name()`. Opening an index whose `vector` width differs from the embedder fails with `StoreError::Schema`.
- Split `ragfs-fuse` FUSE handler `filesystem.rs` into `filesystem/` modules by virtual-directory family (query, ops, safety, semantic, passthrough). Issue #52.
- CI: `Swatinem/rust-cache` uses a shared key across Check/Clippy/Test/Docs/Python; pull requests run Python 3.12 only (3.10–3.13 stays on `main`); coverage and Python integration run on `main` only. Dependabot ignores 0.x minor bumps of `criterion`, `notify-debouncer-full`, and `tokenizers`; auto-merge skips Cargo 0.x minors.
- CI cost: deps compile at `opt-level=0` on runners; PRs drop `--all-features` (keep `ragfs/test-backends`); Python jobs skip when Python paths are unchanged; Coverage is weekly; `docs.yml` uses `cargo +nightly` so `rust-toolchain.toml` cannot pin it to stable. A `CI` aggregator treats skipped rust jobs as success on docs-only PRs.
- Project hygiene: MSRV docs 1.91, `Makefile` / `rust-toolchain.toml`, `CODEOWNERS`, GitHub private advisory as the security channel, `contents: read` on CI/coverage, pinned `ruff==0.15.9`.
- `docs/AUDIT.md`: RUSTSEC-2025-0119 (`number_prefix`) is resolved. `hf-hub` 0.5 pulls `indicatif` 0.18, which uses `unit-prefix` (issue #79). Three unmaintained rows remain (`paste` #80, `core2` #77, `ttf-parser` #82). `paste` also stays via `gemm`, `rav1e`, and `lance-bitpacking`, so a `tokenizers` bump alone does not clear it.

### Added
- CLI integration tests (`assert_cmd`) for config/`--force`/`--hybrid`/`max_file_size` and unsupported models, using `RAGFS_TEST_EMBEDDER=noop`.
- Repo social card (`docs/assets/social.svg` source, `docs/assets/social.png` 1280×640) as the README header. README is a front page (quick start, feature matrix, grouped docs); the CLI manpage stays in `docs/USER_GUIDE.md`.

### Fixed
- Revert Dependabot `lancedb` 0.39.0 (#72): `Error::Http` is `cfg(feature = "remote")` while `job.rs` uses it with default features. Pin stays 0.37.1; cargo Dependabot ignores `lancedb >=0.38` and Arrow 59+.
- MCP `ragfs_batch_operations` serializes a failed atomic batch: `BatchResult.rollback_id` aliases the batch `id` (the previous attribute miss turned a jail reject into `{"error": "... no attribute 'rollback_id'"}`).
- Align `candle-core` and `candle-nn` with `candle-transformers` 0.11 so `ragfs-embed` compiles (one `Tensor` type, not 0.9 + 0.11).
- Pin the MSRV job to rustc 1.91 via `dtolnay/rust-toolchain@stable` + `toolchain: "1.91"` (Dependabot #67 had moved the action tag to 1.120).

## [0.2.0] - 2026-09-25

### Security
- Bump `lopdf` to `0.42` (RUSTSEC-2026-0187 nesting-depth DoS). Already on the 0.42 API in `ragfs-extract`.
- Bump `pyo3` and `pyo3-async-runtimes` to `0.29` (RUSTSEC-2026-0176 / RUSTSEC-2026-0177).
- Replace unmaintained `daemonize` with `nix` (`fork` + `setsid`, stdio redirect, pid file) for `ragfs mount` without `--foreground`. rustix 1.x does not expose `fork` outside its unstable `runtime` feature.

### Changed
- Background `ragfs mount` still writes a PID file to `$XDG_RUNTIME_DIR/ragfs/<hash>.pid` (fallback `~/.cache/ragfs/run/`) and logs to `~/.cache/ragfs/logs/<hash>.log`. Unmount remains `fusermount -u <mountpoint>`.
- Documented actual code chunking (pattern matching) and removed unused `tree-sitter` dependency
- FUSE `.help` now covers `.ops/`, `.safety/`, `.semantic/` and product limits
- FUSE `.config` includes `api_version` and states it is mount wiring, not user TOML

### Added
- Directory-scoped vector search: persist relative `dir_path` on chunks and filter with `ragfs query --scope <dir>` (exact directory or subdirectory; IVF-PQ ANN is unchanged)
- Lance chunks schema v2 migration: existing indexes gain `dir_path` / `dir_depth` / `path_components` on open (sidecar `ragfs-schema.json`); `--force` is not a schema upgrade
- FUSE `mount2` runs on a dedicated OS thread so callbacks can `block_on` without panicking
- Best-effort IVF-PQ ANN index on `vector` when a Lance table has at least 256 rows (cosine-trained; incrementally refreshed after large appends; L2/Dot and failed builds stay exact)
- Office text extraction for `.docx`, `.xlsx`, `.pptx`, and `.odt` (ZIP+XML; not binary `.doc`)
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
