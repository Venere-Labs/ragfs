# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-06

### Added

- Initial commit - RAGFS v0.2.0
- Add release automation with release-plz
- **core**: Add get_all_chunks and get_all_files to VectorStore trait
- Directory-scoped search and dedicated FUSE mount thread ([#60](https://github.com/Venere-Labs/ragfs/pull/60))

### Fixed

- Resolve all clippy warnings and update toml to 0.9
- **core**: Share path jail across fuse, python and mcp ([#90](https://github.com/Venere-Labs/ragfs/pull/90))

### Miscellaneous

- Release v0.2.0
- Unify project descriptions and metadata across packages
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - 2026-01-12

### Added

- Initial commit - RAGFS v0.2.0
- Add release automation with release-plz

### Fixed

- Resolve all clippy warnings and update toml to 0.9

