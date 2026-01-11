# RAGFS User Guide

This guide provides comprehensive documentation for using RAGFS.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Storage Locations](#storage-locations)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- **Rust 1.88+**: Install via [rustup](https://rustup.rs/)
- **FUSE libraries**: Required for filesystem mounting
  - Debian/Ubuntu: `sudo apt install libfuse-dev pkg-config`
  - Fedora: `sudo dnf install fuse-devel`
  - Arch Linux: `sudo pacman -S fuse2`
- **Build tools**: `build-essential` or equivalent

### Building from Source

```bash
# Clone the repository
git clone https://github.com/user/ragfs.git
cd ragfs

# Build in release mode (recommended)
cargo build --release

# The binary is at target/release/ragfs
./target/release/ragfs --help

# Or install to ~/.cargo/bin
cargo install --path crates/ragfs
```

### First Run

On first run, RAGFS downloads the embedding model (~100MB) from Hugging Face Hub:

```bash
ragfs index /path/to/directory
# Output: Initializing embedder (this may download the model on first run)...
```

The model is cached at `~/.local/share/ragfs/models/` for subsequent runs.

## Quick Start

### 1. Index a Directory

```bash
# Index all supported files
ragfs index ~/Documents

# Force reindex everything
ragfs index ~/Documents --force

# Continuous indexing (watch for changes)
ragfs index ~/Documents --watch
```

### 2. Search Your Files

```bash
# Basic semantic search
ragfs query ~/Documents "how to authenticate users"

# Get more results
ragfs query ~/Documents "database schema" --limit 20
```

### 3. Check Index Status

```bash
ragfs status ~/Documents
```

Output:
```
Index Status for "/home/user/Documents"
  Files:  142
  Chunks: 1893
  Updated: 2025-01-11 14:32:15
```

## CLI Reference

### Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Enable debug-level logging |
| `--format` | `-f` | Output format: `text` (default), `json` |
| `--help` | `-h` | Print help information |
| `--version` | `-V` | Print version |

### ragfs index

Index a directory for semantic search.

```
ragfs index <PATH> [OPTIONS]
```

**Arguments:**
- `<PATH>`: Directory to index

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Force reindexing of all files |
| `--watch` | `-w` | Watch for changes after initial indexing |

**Examples:**

```bash
# Index once
ragfs index ./src

# Force reindex
ragfs index ./src --force

# Continuous mode
ragfs index ./src --watch
```

**Default Exclusions:**

The following patterns are excluded by default:
- `**/.*` (hidden files)
- `**/.git/**`
- `**/node_modules/**`
- `**/target/**`
- `**/__pycache__/**`
- `**/*.lock`

### ragfs query

Execute a semantic search query.

```
ragfs query <PATH> <QUERY> [OPTIONS]
```

**Arguments:**
- `<PATH>`: Path to indexed directory
- `<QUERY>`: Natural language query string

**Options:**
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--limit` | `-l` | 10 | Maximum number of results |

**Examples:**

```bash
# Basic query
ragfs query ./src "error handling implementation"

# More results
ragfs query ./src "API endpoint" --limit 25

# JSON output for scripting
ragfs query ./src "configuration" -f json
```

**Text Output Format:**

```
Query: error handling implementation

1. src/lib.rs (score: 0.847)
   Lines: 45-52
   Handle errors gracefully by wrapping results in a custom Error type...

2. src/handlers/error.rs (score: 0.812)
   Lines: 12-28
   The ErrorHandler middleware intercepts all errors and formats...
```

**JSON Output Format:**

```json
{
  "query": "error handling implementation",
  "results": [
    {
      "file": "src/lib.rs",
      "score": 0.847,
      "content": "Handle errors gracefully by...",
      "lines": "45:52"
    }
  ]
}
```

### ragfs mount

Mount a directory as a RAGFS filesystem.

```
ragfs mount <SOURCE> <MOUNTPOINT> [OPTIONS]
```

**Arguments:**
- `<SOURCE>`: Source directory to index
- `<MOUNTPOINT>`: Mount point (must exist)

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--foreground` | `-f` | Run in foreground (don't daemonize) |
| `--allow-other` | | Allow other users to access the mount |

**Examples:**

```bash
# Create mount point
mkdir ~/ragfs-mount

# Mount in foreground
ragfs mount ~/Documents ~/ragfs-mount --foreground

# Unmount (in another terminal or after Ctrl+C)
fusermount -u ~/ragfs-mount
```

**FUSE Requirements:**

- User must be in the `fuse` group: `sudo usermod -aG fuse $USER`
- For `--allow-other`, edit `/etc/fuse.conf` and uncomment `user_allow_other`

### ragfs status

Show index status for a directory.

```
ragfs status <PATH>
```

**Arguments:**
- `<PATH>`: Path to indexed directory

**Examples:**

```bash
# Text output
ragfs status ./src

# JSON output
ragfs status ./src -f json
```

**JSON Output:**

```json
{
  "path": "/home/user/src",
  "total_files": 42,
  "total_chunks": 583,
  "index_size_bytes": 12582912,
  "last_updated": "2025-01-11T14:32:15Z"
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAGFS_DATA_DIR` | Override data directory | `~/.local/share/ragfs` |
| `RAGFS_CONFIG_DIR` | Override config directory | `~/.config/ragfs` |

### Chunking Parameters

Default chunking configuration:
- **Target size**: 512 tokens
- **Max size**: 1024 tokens
- **Overlap**: 64 tokens

### Embedding Model

RAGFS uses the `thenlper/gte-small` model:
- **Dimension**: 384
- **Max tokens**: 512
- **Architecture**: BERT-based

## Output Formats

### Text Format (default)

Human-readable output suitable for terminal viewing.

```bash
ragfs query ./src "authentication"
```

### JSON Format

Structured output for scripting and integration.

```bash
ragfs query ./src "authentication" --format json
```

Use with `jq` for processing:

```bash
# Get just the file paths
ragfs query ./src "auth" -f json | jq -r '.results[].file'

# Get top result score
ragfs query ./src "auth" -f json | jq '.results[0].score'
```

## Storage Locations

### Index Database

```
~/.local/share/ragfs/indices/{hash}/index.lance
```

Each indexed directory gets a unique index based on its path hash (blake3).

### Embedding Model Cache

```
~/.local/share/ragfs/models/
```

Contains the downloaded Hugging Face model files.

### Viewing Index Location

```bash
# The index path for a directory
echo ~/.local/share/ragfs/indices/$(echo -n "/path/to/dir" | blake3 | head -c 16)/
```

## Troubleshooting

### "Index not found"

The directory hasn't been indexed yet.

```bash
# Solution: Index the directory first
ragfs index /path/to/directory
```

### "Model download failed"

Network issues during first run.

```bash
# Check your internet connection and try again
ragfs index /path/to/directory

# Or manually download:
# Visit huggingface.co/thenlper/gte-small
```

### "Permission denied" on mount

FUSE permissions not configured.

```bash
# Add yourself to the fuse group
sudo usermod -aG fuse $USER

# Log out and back in, then try again
```

### "fusermount: user has no write access"

Mount point doesn't exist or is not writable.

```bash
# Create the mount point
mkdir ~/ragfs-mount
chmod 755 ~/ragfs-mount
```

### High memory usage during indexing

Large files or many files being processed.

```bash
# Index in smaller batches by using include patterns
# (feature planned for future release)
```

### Stale index after file changes

Index not updated after files were modified.

```bash
# Solution 1: Force reindex
ragfs index /path/to/directory --force

# Solution 2: Use watch mode for automatic updates
ragfs index /path/to/directory --watch
```

## Tips and Best Practices

### Query Writing

- Use natural language: "how does authentication work" rather than keywords
- Be specific: "JWT token validation in login handler" rather than "JWT"
- Include context: "database connection pooling in the API layer"

### Performance

- Start with `--watch` mode for directories that change frequently
- Use `--force` sparingly; incremental indexing is more efficient
- Keep index directories on fast storage (SSD recommended)

### Integration

```bash
# Find files matching a concept
ragfs query ./src "error handling" -f json | jq -r '.results[].file' | xargs code

# Search and open in vim
vim $(ragfs query ./src "config" -f json | jq -r '.results[0].file')
```
