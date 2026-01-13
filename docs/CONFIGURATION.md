# RAGFS Configuration Reference

This document provides a complete reference for all RAGFS configuration options.

## Table of Contents

- [Configuration File](#configuration-file)
- [Precedence Rules](#precedence-rules)
- [All Configuration Options](#all-configuration-options)
- [Environment Variables](#environment-variables)
- [Feature Flags](#feature-flags)
- [Example Configurations](#example-configurations)

---

## Configuration File

### Location

RAGFS uses a TOML configuration file at:

```
~/.config/ragfs/config.toml
```

### Creating a Config File

```bash
# Generate sample config
ragfs config init > ~/.config/ragfs/config.toml

# View current settings
ragfs config show

# Get config file path
ragfs config path
```

### Override Location

Use the `--config` CLI flag or `RAGFS_CONFIG_DIR` environment variable:

```bash
# CLI override
ragfs --config /custom/path/config.toml index ~/Documents

# Environment variable
export RAGFS_CONFIG_DIR=/custom/path
ragfs index ~/Documents
```

---

## Precedence Rules

Settings are applied in this order (later overrides earlier):

1. **Built-in defaults** (hardcoded)
2. **Config file** (`~/.config/ragfs/config.toml`)
3. **CLI arguments** (highest priority)

Example:
```bash
# Config file says limit=10, CLI says limit=20
ragfs query ~/docs "search" --limit 20  # Uses 20
```

---

## All Configuration Options

### `[mount]` - Mount Settings

Options for FUSE filesystem mounting.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `allow_other` | bool | `false` | Allow other users to access the mount |

```toml
[mount]
allow_other = false
```

**Note:** Using `allow_other = true` requires `user_allow_other` in `/etc/fuse.conf`.

---

### `[index]` - Indexing Settings

Options controlling which files are indexed.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include` | string[] | `["**/*"]` | Glob patterns for files to include |
| `exclude` | string[] | (see below) | Glob patterns for files to exclude |
| `max_file_size` | int | `52428800` (50MB) | Maximum file size to index (bytes) |
| `debounce_ms` | int | `500` | Debounce for file watcher (milliseconds) |

**Default exclude patterns:**
```toml
exclude = [
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/venv/**",
    "**/.env"
]
```

**Full example:**
```toml
[index]
include = ["**/*.rs", "**/*.py", "**/*.md"]
exclude = [
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/venv/**",
    "**/.env",
    "**/dist/**",
    "**/build/**",
    "**/*.lock"
]
max_file_size = 10485760  # 10MB
debounce_ms = 1000
```

---

### `[embedding]` - Embedding Settings

Options for the embedding model.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | `"jina-embeddings-v3"` | Embedding model to use |
| `batch_size` | int | `32` | Batch size for embedding generation |
| `use_gpu` | bool | `true` | Use GPU if available |
| `max_concurrent` | int | `4` | Maximum concurrent embedding jobs |

```toml
[embedding]
model = "jina-embeddings-v3"  # Or "gte-small" for 384-dim
batch_size = 32
use_gpu = true
max_concurrent = 4
```

**Models available:**
- `jina-embeddings-v3` - Default, high quality (requires API or local model)
- `gte-small` - Local, 384 dimensions, ~100MB download

---

### `[chunking]` - Chunking Settings

Options for how content is split into chunks.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `target_size` | int | `512` | Target chunk size (tokens) |
| `max_size` | int | `1024` | Maximum chunk size (tokens) |
| `overlap` | int | `64` | Overlap between chunks (tokens) |
| `hierarchical` | bool | `true` | Enable hierarchical chunking |
| `max_depth` | int | `2` | Maximum hierarchy depth |

```toml
[chunking]
target_size = 512
max_size = 1024
overlap = 64
hierarchical = true
max_depth = 2
```

**Tuning guidance:**
- **Larger chunks** (768-1024): Better for long documents, fewer chunks
- **Smaller chunks** (256-384): Better for granular search, more precise results
- **More overlap** (128+): Better recall, prevents context loss at boundaries
- **Less overlap** (32-64): Smaller index, faster indexing

---

### `[query]` - Query Settings

Options for search behavior.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_limit` | int | `10` | Default number of results |
| `max_limit` | int | `100` | Maximum allowed results |
| `hybrid` | bool | `true` | Enable hybrid search (vector + full-text) |
| `rerank` | bool | `false` | Enable result reranking |

```toml
[query]
default_limit = 10
max_limit = 100
hybrid = true
rerank = false
```

**Hybrid search:**
When enabled, combines vector similarity with full-text search using reciprocal rank fusion. Improves results when exact keyword matches are important.

---

### `[logging]` - Logging Settings

Options for log output.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | `"info"` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `file` | string | `null` | Optional log file path |

```toml
[logging]
level = "info"
# file = "/var/log/ragfs.log"  # Optional file logging
```

**Log levels:**
- `error` - Only errors
- `warn` - Warnings and errors
- `info` - General operational info (default)
- `debug` - Detailed debugging info
- `trace` - Very verbose, includes all internal operations

---

## Environment Variables

### Directory Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `RAGFS_DATA_DIR` | Data directory (indices, models, trash) | `~/.local/share/ragfs` |
| `RAGFS_CONFIG_DIR` | Config directory | `~/.config/ragfs` |

```bash
# Use custom data location
export RAGFS_DATA_DIR=/mnt/fast-ssd/ragfs
ragfs index ~/Documents
```

### Logging

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Override log level (standard Rust logging) |

```bash
# Debug logging
RUST_LOG=debug ragfs index ~/Documents

# Trace specific modules
RUST_LOG=ragfs_embed=trace,ragfs_store=debug ragfs index ~/Documents
```

### Proxy (for model downloads)

| Variable | Description |
|----------|-------------|
| `HTTPS_PROXY` | HTTPS proxy for model downloads |
| `HTTP_PROXY` | HTTP proxy for model downloads |

```bash
export HTTPS_PROXY=http://proxy.company.com:8080
ragfs index ~/Documents  # First run downloads model through proxy
```

---

## Feature Flags

RAGFS uses Cargo feature flags to control optional functionality.

### Available Features

| Feature | Default | Description |
|---------|---------|-------------|
| `candle` | Yes | Enables `CandleEmbedder` with GTE-small model (384 dim) |
| `lancedb` | Yes | Enables `LanceStore` for vector storage |
| `vision` | No | Enables BLIP image captioning (~400MB model) |
| `full` | No | Enables all optional features |

### Build Examples

```bash
# Default build (candle + lancedb)
cargo build --release

# Build with all features
cargo build --release --features full

# Minimal build (NoopEmbedder + MemoryStore only)
cargo build --release --no-default-features

# Add vision support
cargo build --release --features vision
```

### Feature Detection

```bash
# Check which features are enabled
ragfs --version
# Output includes: "Features: candle, lancedb"
```

---

## Example Configurations

### Minimal Config

For most users, defaults work well:

```toml
# ~/.config/ragfs/config.toml
# Empty file or minimal overrides

[query]
default_limit = 15
```

### Development Project

Optimized for code repositories:

```toml
[index]
include = ["**/*.rs", "**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.md"]
exclude = [
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/__pycache__/**",
    "**/dist/**",
    "**/build/**",
    "**/*.lock",
    "**/vendor/**"
]
max_file_size = 5242880  # 5MB - skip large generated files

[chunking]
target_size = 384       # Smaller chunks for code
overlap = 96            # More overlap for function context
hierarchical = true

[query]
hybrid = true
default_limit = 15
```

### Documentation Repository

Optimized for markdown and documentation:

```toml
[index]
include = ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt"]
exclude = ["**/node_modules/**", "**/.git/**"]
max_file_size = 10485760  # 10MB

[chunking]
target_size = 768       # Larger chunks for prose
overlap = 128           # Good overlap for context
hierarchical = true
max_depth = 3           # Deeper hierarchy for long docs

[query]
hybrid = true
default_limit = 10
```

### Low-Memory System

For systems with limited RAM:

```toml
[embedding]
batch_size = 8           # Smaller batches
max_concurrent = 1       # Single embedding job
use_gpu = false          # CPU only

[index]
max_file_size = 2097152  # 2MB limit
debounce_ms = 2000       # Less frequent updates

[chunking]
target_size = 256        # Smaller chunks = less memory per batch
max_size = 512
```

### High-Performance Server

For dedicated indexing servers:

```toml
[embedding]
batch_size = 64          # Larger batches
max_concurrent = 8       # More parallel jobs
use_gpu = true           # Use GPU

[index]
max_file_size = 104857600  # 100MB
debounce_ms = 100          # Fast response

[chunking]
target_size = 512
max_size = 1024
overlap = 64
hierarchical = true

[query]
max_limit = 500          # Allow more results
hybrid = true
```

---

## Storage Locations

| Data | Default Location |
|------|-----------------|
| Vector indices | `~/.local/share/ragfs/indices/{hash}/` |
| Embedding models | `~/.local/share/ragfs/models/` |
| Trash (soft deletes) | `~/.local/share/ragfs/trash/` |
| Audit history | `~/.local/share/ragfs/{hash}.history` |
| Config file | `~/.config/ragfs/config.toml` |
| Log files | `~/.cache/ragfs/logs/` |
| PID files | `~/.cache/ragfs/run/` or `$XDG_RUNTIME_DIR/ragfs/` |

Override with `RAGFS_DATA_DIR`:
```bash
export RAGFS_DATA_DIR=/mnt/ssd/ragfs
# Now uses:
# /mnt/ssd/ragfs/indices/
# /mnt/ssd/ragfs/models/
# /mnt/ssd/ragfs/trash/
```

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Quick start guide
- [User Guide](USER_GUIDE.md) - Complete usage documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
