# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-06

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz
- **extract**: Implement BlipCaptioner for image captioning
- Add custom category embeddings, CI model caching, and pdf_oxide support
- **extract**: Extract text from Office OOXML and ODT ([#55](https://github.com/Venere-Labs/ragfs/pull/55))

### CI/CD

- Unblock inherited rustc, cargo-deny, and maturin failures ([#42](https://github.com/Venere-Labs/ragfs/pull/42))
- **security**: Fix audit reporting and reduce CI fan-out ([#43](https://github.com/Venere-Labs/ragfs/pull/43))

### Changed

- Fix formatting and clippy warnings

### Documentation

- Add custom landing page and improve crate documentation

### Fixed

- Resolve all clippy warnings and update toml to 0.9
- **extract**: Correct pdf_oxide API for image extraction
- Align docs and FUSE help with real behavior

### Miscellaneous

- Release v0.2.0
- **extract**: Retire optional pdf_oxide feature ([#87](https://github.com/Venere-Labs/ragfs/pull/87))

### Deps

- Bump zip from 2.4.2 to 8.6.0 ([#130](https://github.com/Venere-Labs/ragfs/pull/130))
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - 2026-01-12

### Added

- Initial commit - RAGFS v0.2.0
- Add config loading, PDF image extraction, vision infrastructure, and .help file
- Add release automation with release-plz

### Documentation

- Add custom landing page and improve crate documentation

### Fixed

- Resolve all clippy warnings and update toml to 0.9

