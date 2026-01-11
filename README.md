# RAGFS

[![CI](https://github.com/Venere-Labs/ragfs/actions/workflows/ci.yml/badge.svg)](https://github.com/Venere-Labs/ragfs/actions/workflows/ci.yml)
[![Security Audit](https://github.com/Venere-Labs/ragfs/actions/workflows/security.yml/badge.svg)](https://github.com/Venere-Labs/ragfs/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/Venere-Labs/ragfs/branch/main/graph/badge.svg)](https://codecov.io/gh/Venere-Labs/ragfs)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Venere-Labs.github.io/ragfs/ragfs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A FUSE filesystem for RAG (Retrieval-Augmented Generation) architectures. RAGFS enables semantic search over your files using vector embeddings, allowing you to find content by meaning rather than keywords.

## Features

- **Semantic Search** - Query files by meaning using vector similarity search
- **Local Embeddings** - Runs entirely offline using the `gte-small` model via Candle
- **FUSE Integration** - Mount indexed directories as a searchable virtual filesystem
- **Real-time Indexing** - Watch directories for changes and update the index automatically
- **Multimodal Support** - Extract content from text, code, markdown, PDF, and images
- **Code-aware Chunking** - Syntax-aware splitting using tree-sitter for source code
- **Hybrid Search** - Combine vector similarity with full-text search
- **JSON Output** - Machine-readable output for scripting and integration
- **Comprehensive Testing** - 291 tests across all crates ensuring reliability

## Requirements

- Rust 1.85 or later
- Linux with FUSE support (`libfuse-dev` on Debian/Ubuntu, `fuse` on Arch)
- ~500MB disk space for the embedding model (downloaded on first run)

## Installation

```bash
# Clone the repository
git clone https://github.com/Venere-Labs/ragfs.git
cd ragfs

# Build in release mode
cargo build --release

# Install to ~/.cargo/bin
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

## CLI Reference

```
ragfs [OPTIONS] <COMMAND>

Commands:
  mount   Mount a directory as a RAGFS filesystem
  index   Index a directory (without mounting)
  query   Query the index
  status  Show index status

Options:
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
  -f, --force  Force reindexing of all files
  -w, --watch  Watch for changes after initial indexing
```

### query

```
ragfs query <PATH> <QUERY> [OPTIONS]

Arguments:
  <PATH>   Path to indexed directory
  <QUERY>  Query string

Options:
  -l, --limit <LIMIT>  Maximum results [default: 10]
```

### status

```
ragfs status <PATH>

Arguments:
  <PATH>  Path to indexed directory
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

- [Architecture Overview](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [API Reference](docs/API.md)

## How It Works

1. **Extraction** - Content is extracted from files based on their MIME type
2. **Chunking** - Text is split into overlapping chunks (~512 tokens each)
3. **Embedding** - Each chunk is converted to a 384-dimensional vector using the `gte-small` model
4. **Storage** - Vectors are stored in LanceDB for efficient similarity search
5. **Search** - Queries are embedded and matched against stored vectors using cosine similarity

## Storage Locations

- **Indices**: `~/.local/share/ragfs/indices/{hash}/index.lance`
- **Models**: `~/.local/share/ragfs/models/`

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
