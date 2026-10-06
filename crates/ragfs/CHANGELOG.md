# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-06

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz
- **extract**: Extract text from Office OOXML and ODT ([#55](https://github.com/Venere-Labs/ragfs/pull/55))
- Directory-scoped search and dedicated FUSE mount thread ([#60](https://github.com/Venere-Labs/ragfs/pull/60))

### CI/CD

- **security**: Fix audit reporting and reduce CI fan-out ([#43](https://github.com/Venere-Labs/ragfs/pull/43))

### Changed

- Restructure feature flags for optional ML backends
- Fix formatting and clippy warnings

### Documentation

- Align crate list and config docs with the tree ([#89](https://github.com/Venere-Labs/ragfs/pull/89))

### Fixed

- Resolve all clippy warnings and update toml to 0.9
- Apply config.toml to index, query, and embed
- **store**: Apply query filters and metric in LanceStore
- Align docs and FUSE help with real behavior
- **security**: Clear P0 direct-dep advisories for #44 ([#62](https://github.com/Venere-Labs/ragfs/pull/62))
- **core**: Share path jail across fuse, python and mcp ([#90](https://github.com/Venere-Labs/ragfs/pull/90))
- **store**: Derive embedding dim from the embedder ([#92](https://github.com/Venere-Labs/ragfs/pull/92))

### Miscellaneous

- Release v0.2.0
- Unify project descriptions and metadata across packages

### Testing

- **cli**: Assert_cmd coverage for config, force, hybrid ([#93](https://github.com/Venere-Labs/ragfs/pull/93))
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - 2026-01-12

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz

### Fixed

- Resolve all clippy warnings and update toml to 0.9

