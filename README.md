# RAGFS

[![CI](https://github.com/Venere-Labs/ragfs/actions/workflows/ci.yml/badge.svg)](https://github.com/Venere-Labs/ragfs/actions/workflows/ci.yml)
[![Security Audit](https://github.com/Venere-Labs/ragfs/actions/workflows/security.yml/badge.svg)](https://github.com/Venere-Labs/ragfs/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/Venere-Labs/ragfs/branch/main/graph/badge.svg)](https://codecov.io/gh/Venere-Labs/ragfs)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Venere-Labs.github.io/ragfs/ragfs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange.svg)](https://www.rust-lang.org)

An agentic FUSE filesystem that makes file management safe and structured for LLM agents. Includes JSON-based operations with undo support, complete audit logging, and AI-powered features like semantic search, auto-organization, and deduplication.

## Features

- **Agent File Operations** - Structured file ops with JSON feedback via `.ops/` interface
- **Safety Layer** - Soft delete, audit logging, and undo support via `.safety/`
- **AI-Powered Management** - Auto-organization, deduplication, and cleanup via `.semantic/`
- **Semantic Search** - Query files by meaning using vector similarity search
- **Local Embeddings** - Runs entirely offline using the `gte-small` model via Candle
- **FUSE Integration** - Mount indexed directories as a virtual filesystem
- **Real-time Indexing** - Watch directories for changes and update the index automatically
- **Multimodal Support** - Extract content from text, code, markdown, PDF, images, and Office OOXML/ODT
- **Code-aware Chunking** - Splits source at function/class signatures (pattern matching, not tree-sitter)
- **Hybrid Search** - Combine vector similarity with full-text search (when enabled)
- **MCP Server** - Claude Desktop integration for AI assistants
- **Tests** - Unit and integration tests across crates (CI on every PR)

## Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| CLI (index, query, status) | Stable | Core functionality |
| FUSE mount | Stable | Linux only |
| Semantic search | Stable | Vector similarity with LanceDB |
| Hybrid search | Stable | Vector + full-text |
| Text extraction | Stable | UTF-8 text/code/markup, PDF, images, OOXML (docx/xlsx/pptx/odt). No binary `.doc` |
| Code chunking | Stable | Pattern-based function/class splits |
| PDF extraction | Stable | Text + embedded images |
| Agent operations (.ops/) | Stable | JSON feedback, batch support |
| Safety layer (.safety/) | Stable | Trash, history, undo |
| Semantic operations (.semantic/) | Beta | Organize, dedupe, cleanup |
| Python bindings | Beta | PyO3 based |
| MCP server | Beta | Claude Desktop integration |
| Image captioning | Experimental | Optional, requires `vision` feature |

## Use Cases

**Ideal for:**
- LLM agents managing files (Claude, GPT, local models)
- Automated file organization and cleanup
- Safe file operations with audit trail
- Code repositories (1K-50K files)
- Documentation collections
- Research notes and papers
- Local-first semantic search

**Limitations:**
- Linux only (FUSE requirement). macOS CI artifacts are not a supported FUSE product.
- `ragfs index` without `--watch` does not remove files that disappeared while the process was not running. `--watch` receives those deletions.
- Embedding model is `thenlper/gte-small` (~67–100MB download, hundreds of MB RAM).
- Code chunking is regex/signature based, not a tree-sitter AST.
- Default extractors do not parse binary `.doc` / RTF / EPUB.
- Vector search is an exact scan until cosine IVF-PQ is built (≥256 chunks). L2/Dot stay exact.
- Semantic organize/cleanup is Beta.
- Large repositories (100K+ files) may need tuning.

## Requirements

- Rust 1.91 or later
- Linux with FUSE support (`libfuse-dev` on Debian/Ubuntu, `fuse` on Arch)
- Disk space for the `gte-small` embedding model (downloaded on first run, ~67–100MB). `HF_HUB_OFFLINE=1` uses only a cache that is already present.

## Installation

GitHub Release `v0.2.0` publishes the CLI only. The crates are not on crates.io, and the Python bindings are not on PyPI.

Linux archives, each with a `.sha256` file:

- `ragfs-linux-x86_64.tar.gz`
- `ragfs-linux-aarch64.tar.gz`

`ragfs-macos-x86_64.tar.gz` and `ragfs-macos-aarch64.tar.gz` are CI binaries. They are not a supported FUSE product.

```bash
# From a Linux release archive
tar -xzf ragfs-linux-x86_64.tar.gz
sudo install ragfs /usr/local/bin/ragfs

# Or build from source
git clone https://github.com/Venere-Labs/ragfs.git
cd ragfs
cargo build --release
cargo install --path crates/ragfs
```

## Quick Start

### Index a directory

```bash
# Index all files in a directory
ragfs index ~/Documents

# Watch for changes (continuous indexing)
ragfs index ~/Documents --watch
```

### Search your files

```bash
# Semantic search
ragfs query ~/Documents "machine learning implementation"

# Get more results
ragfs query ~/Documents "authentication logic" --limit 20

# JSON output for scripting
ragfs query ~/Documents "database connection" --format json
```

### Mount as a filesystem

```bash
# Create a mount point
mkdir ~/ragfs-mount

# Mount the indexed directory
ragfs mount ~/Documents ~/ragfs-mount --foreground
```

### Check index status

```bash
ragfs status ~/Documents
```

### Agent file operations (via FUSE mount)

```bash
# Create a file with feedback
echo -e "docs/new.md\n# New Document" > ~/ragfs-mount/.ragfs/.ops/.create
cat ~/ragfs-mount/.ragfs/.ops/.result  # JSON with undo_id

# Delete a file (soft delete to trash)
echo "docs/old.md" > ~/ragfs-mount/.ragfs/.ops/.delete

# Find similar files
echo "src/main.rs" > ~/ragfs-mount/.ragfs/.semantic/.similar
cat ~/ragfs-mount/.ragfs/.semantic/.similar

# Undo an operation
echo "<undo_id>" > ~/ragfs-mount/.ragfs/.safety/.undo
```

## CLI Reference

```
ragfs [OPTIONS] <COMMAND>

Commands:
  mount   Mount a directory as a RAGFS filesystem
  index   Index a directory (without mounting)
  query   Query the index
  status  Show index status
  config  Manage configuration

Options:
  -c, --config <FILE>    Config file path [default: ~/.config/ragfs/config.toml]
  -v, --verbose          Enable verbose logging
  -f, --format <FORMAT>  Output format: text, json [default: text]
  -h, --help             Print help
  -V, --version          Print version
```

### mount

```
ragfs mount <SOURCE> <MOUNTPOINT> [OPTIONS]

Arguments:
  <SOURCE>      Source directory to index
  <MOUNTPOINT>  Mount point

Options:
  -f, --foreground  Run in foreground (don't daemonize)
      --allow-other Allow other users to access the mount
```

### index

```
ragfs index <PATH> [OPTIONS]

Arguments:
  <PATH>  Directory to index

Options:
  -f, --force  Force reindexing of all files (rewrites directory-scope fields;
               schema upgrades for old Lance indexes run automatically on open)
  -w, --watch  Watch for changes after initial indexing. Deletions are
               applied only while this process is running. A later
               `ragfs index` without `--watch` keeps chunks for files
               that disappeared in between.
```

### query

```
ragfs query <PATH> <QUERY> [OPTIONS]

Arguments:
  <PATH>   Path to indexed directory
  <QUERY>  Query string

Options:
  -l, --limit <LIMIT>  Maximum results [default: 10]
      --scope <DIR>    Restrict results to this directory and its subdirectories
                       (existing indexes are migrated on open; see USER_GUIDE)
```

### status

```
ragfs status <PATH>

Arguments:
  <PATH>  Path to indexed directory
```

### config

```
ragfs config <ACTION>

Actions:
  show  Display current configuration
  init  Print sample config file
  path  Print config file path
```

## Architecture

RAGFS is organized as a Rust workspace with specialized crates:

| Crate | Description |
|-------|-------------|
| `ragfs` | CLI application |
| `ragfs-core` | Core traits and types |
| `ragfs-fuse` | FUSE filesystem implementation |
| `ragfs-index` | File indexing engine |
| `ragfs-chunker` | Document chunking strategies |
| `ragfs-embed` | Embedding generation (Candle) |
| `ragfs-extract` | Content extraction |
| `ragfs-store` | Vector storage (LanceDB) |
| `ragfs-query` | Query execution |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - 5-minute tutorial
- [User Guide](docs/USER_GUIDE.md) - Complete CLI reference
- [Configuration](docs/CONFIGURATION.md) - All config options
- [Performance Guide](docs/PERFORMANCE.md) - Tuning and optimization
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Architecture](docs/ARCHITECTURE.md) - Technical deep-dive
- [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md) - Why we made these choices
- [API Reference](docs/API.md) - Library usage and types
- [Python Bindings](docs/PYTHON.md) - Python SDK and framework integrations
- [MCP Server](docs/MCP.md) - Claude Desktop integration
- [Development Guide](docs/DEVELOPMENT.md) - Contributing to RAGFS
- [Audit & Improvement Tracker](docs/AUDIT.md) - Security posture, accepted advisories, open work

## How It Works

1. **Extraction** - Content is extracted from files based on their MIME type
2. **Chunking** - Text is split into overlapping chunks (~512 tokens each)
3. **Embedding** - Each chunk is converted to a 384-dimensional vector using the `gte-small` model
4. **Storage** - Vectors are stored in LanceDB for efficient similarity search
5. **Search** - Queries are embedded and matched against stored vectors using cosine similarity

## Storage Locations

- **Indices**: `~/.local/share/ragfs/indices/{hash}/index.lance`
- **Models**: `~/.local/share/ragfs/models/`

`RAGFS_DATA_DIR` moves both. The default remains `~/.local/share/ragfs`.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
