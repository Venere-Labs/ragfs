# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Daemonization**: `ragfs mount` now properly forks to background when run without `--foreground`
  - PID file stored in `$XDG_RUNTIME_DIR/ragfs/` or `~/.cache/ragfs/run/`
  - Logs written to `~/.cache/ragfs/logs/`
  - Graceful unmount via `fusermount -u <mountpoint>`

### Changed
- N/A

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

[Unreleased]: https://github.com/user/ragfs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/user/ragfs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/user/ragfs/releases/tag/v0.1.0
