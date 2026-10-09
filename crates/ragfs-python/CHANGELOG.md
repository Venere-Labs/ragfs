# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file documents the existing workspace version `0.2.0` for `ragfs-python`
(`version.workspace = true` / `pyproject.toml` `version = "0.2.0"` already on
`main`). It is **not** a crates.io or PyPI publication record.
`ragfs-python` is **not** listed in `release-plz.toml`. There is no git tag
and no GitHub Release for this version. Root `CHANGELOG.md` still uses
`[Unreleased]` for work after the dated `[0.2.0] - 2026-01-11` entry.

## [0.2.1] - 2026-10-09

### Fixed

- **core**: Share path jail across fuse, python and mcp ([#90](https://github.com/Venere-Labs/ragfs/pull/90))
- **store**: Derive embedding dim from the embedder ([#92](https://github.com/Venere-Labs/ragfs/pull/92))

### Deps

- **python**: Update langchain-anthropic requirement ([#34](https://github.com/Venere-Labs/ragfs/pull/34))
- **python**: Update langchain-openai requirement ([#31](https://github.com/Venere-Labs/ragfs/pull/31))
- **python**: Update langchain-ollama requirement ([#32](https://github.com/Venere-Labs/ragfs/pull/32))
- **python**: Update haystack-ai requirement in /crates/ragfs-python ([#33](https://github.com/Venere-Labs/ragfs/pull/33))
- **python**: Update pytest-asyncio requirement in /crates/ragfs-python ([#35](https://github.com/Venere-Labs/ragfs/pull/35))
[0.2.1]: https://github.com/Venere-Labs/ragfs/compare/0.2.0...0.2.1


## [0.2.0] - workspace (unpublished)

First changelog for the Python bindings at workspace `0.2.0`. Entries below
were collected from git history; they do not imply a registry or GitHub
release.

### Added

- **python**: Add PyO3 bindings for RAGFS
- **python**: Extend bindings with safety, semantic, and ops modules
- **examples**: Add LlamaIndex and Haystack RAG pipeline examples
- **extract**: Extract text from Office OOXML and ODT ([#55](https://github.com/Venere-Labs/ragfs/pull/55))
- Directory-scoped search and dedicated FUSE mount thread ([#60](https://github.com/Venere-Labs/ragfs/pull/60))

### CI/CD

- Unblock inherited rustc, cargo-deny, and maturin failures ([#42](https://github.com/Venere-Labs/ragfs/pull/42))

### Changed

- Fix formatting and clippy warnings

### Fixed

- **python**: Correct import names in __init__.py
- **deps**: Update fuser 0.16 and pyo3 0.27 for security fixes
- Resolve CI failures while the workspace version was 0.2.0
- Align docs and FUSE help with real behavior
- **security**: Clear P0 direct-dep advisories for #44 ([#62](https://github.com/Venere-Labs/ragfs/pull/62))

### Miscellaneous

- Improve project configuration
- Unify project descriptions and metadata across packages
