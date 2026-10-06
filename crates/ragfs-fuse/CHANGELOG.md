# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-06

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz
- **fuse**: Add agent file management with ops, safety, and semantic layers
- **fuse**: Add duplicate detection and organization planning
- **mcp**: Add MCP server for AI assistant integration
- Add custom category embeddings, CI model caching, and pdf_oxide support
- Directory-scoped search and dedicated FUSE mount thread ([#60](https://github.com/Venere-Labs/ragfs/pull/60))

### Changed

- Restructure feature flags for optional ML backends
- Fix clippy warnings in ops.rs and semantic.rs
- Fix formatting and clippy warnings
- **fuse**: Split filesystem.rs by virtual-directory family ([#94](https://github.com/Venere-Labs/ragfs/pull/94))

### Fixed

- Resolve all clippy warnings and update toml to 0.9
- **fuse**: Path jail and history-aligned undo_id
- Apply config.toml to index, query, and embed
- Resolve CI failures for v0.2.0 release
- Align docs and FUSE help with real behavior
- **core**: Share path jail across fuse, python and mcp ([#90](https://github.com/Venere-Labs/ragfs/pull/90))

### Miscellaneous

- Release v0.2.0
- Add CI workflows, security policy, and dependency audit
- Unify project descriptions and metadata across packages

### Deps

- Bump dirs from 5.0.1 to 6.0.0 ([#104](https://github.com/Venere-Labs/ragfs/pull/104))
- Bump dirs from 6.0.0 to 7.0.0 ([#124](https://github.com/Venere-Labs/ragfs/pull/124))
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - 2026-01-12

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz

### Fixed

- Resolve all clippy warnings and update toml to 0.9

