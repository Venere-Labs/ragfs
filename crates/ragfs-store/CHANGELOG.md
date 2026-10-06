# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-06

### Added

- Initial commit - RAGFS v0.2.0
- Add release automation with release-plz
- **core**: Add get_all_chunks and get_all_files to VectorStore trait
- **store**: Add IVF-PQ ANN index for large Lance tables ([#56](https://github.com/Venere-Labs/ragfs/pull/56))
- Directory-scoped search and dedicated FUSE mount thread ([#60](https://github.com/Venere-Labs/ragfs/pull/60))

### Changed

- Restructure feature flags for optional ML backends
- Fix formatting and clippy warnings

### Fixed

- Resolve all clippy warnings and update toml to 0.9
- **deps**: Resolve rand version conflict for lancedb 0.23
- **store**: Apply query filters and metric in LanceStore
- Align docs and FUSE help with real behavior
- **store**: Migrate to lancedb 0.37 and drop deny ignores ([#63](https://github.com/Venere-Labs/ragfs/pull/63))
- **store**: Derive embedding dim from the embedder ([#92](https://github.com/Venere-Labs/ragfs/pull/92))

### Miscellaneous

- Release v0.2.0
- Improve test coverage and code quality
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - 2026-01-12

### Added

- Initial commit - RAGFS v0.2.0
- Add release automation with release-plz

### Fixed

- Resolve all clippy warnings and update toml to 0.9

