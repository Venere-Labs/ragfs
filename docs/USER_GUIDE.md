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
- [Agent File Operations](#agent-file-operations)
- [Tips and Best Practices](#tips-and-best-practices)

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
| `--config` | `-c` | Config file path (default: `~/.config/ragfs/config.toml`) |
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

### ragfs config

Manage RAGFS configuration.

```
ragfs config <ACTION>
```

**Actions:**
| Action | Description |
|--------|-------------|
| `show` | Display current configuration |
| `init` | Print sample config file |
| `path` | Print config file path |

**Examples:**

```bash
# Generate sample config
ragfs config init > ~/.config/ragfs/config.toml

# View current settings
ragfs config show

# Get config file location
ragfs config path
```

## Configuration

### Configuration File

RAGFS can be configured via `~/.config/ragfs/config.toml`. Generate a sample config:

```bash
ragfs config init > ~/.config/ragfs/config.toml
```

**Sample configuration:**

```toml
[mount]
allow_other = false

[index]
include = ["**/*"]
exclude = ["**/node_modules/**", "**/.git/**", "**/target/**"]
max_file_size = 52428800  # 50MB
debounce_ms = 500

[embedding]
# Model: thenlper/gte-small (384 dimensions, 512 max tokens)
# Downloaded automatically on first use
# GPU auto-detected by Candle framework
# batch_size = 32       # Default, handled internally
# max_concurrent = 4    # Default worker pool size

[chunking]
target_size = 512
max_size = 1024
overlap = 64
hierarchical = true
max_depth = 2

[query]
default_limit = 10
max_limit = 100
hybrid = true
rerank = false

[logging]
level = "info"
# file = "/path/to/ragfs.log"
```

**Precedence:** CLI arguments > config file > defaults

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

## Agent File Operations

RAGFS provides a filesystem-based API for AI agents to perform file operations safely and autonomously. These features are accessible through the `.ragfs/` virtual directory when using the FUSE mount.

### Operations Interface (`.ops/`)

The `.ops/` directory provides structured file operations with JSON feedback.

#### Creating Files

```bash
# Write: "relative/path\ncontent"
echo -e "docs/new-feature.md\n# New Feature\n\nDescription here..." > /mnt/ragfs/.ragfs/.ops/.create

# Check the result
cat /mnt/ragfs/.ragfs/.ops/.result
```

**Result:**
```json
{
  "success": true,
  "operation": "create",
  "path": "docs/new-feature.md",
  "message": null,
  "indexed": true,
  "undo_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Deleting Files

Files are soft-deleted (moved to trash) for safety:

```bash
# Write the path to delete
echo "docs/old-file.md" > /mnt/ragfs/.ragfs/.ops/.delete

# Check result
cat /mnt/ragfs/.ragfs/.ops/.result
```

#### Moving/Renaming Files

```bash
# Write: "source\ndestination"
echo -e "old/location.txt\nnew/location.txt" > /mnt/ragfs/.ragfs/.ops/.move

cat /mnt/ragfs/.ragfs/.ops/.result
```

#### Batch Operations

Execute multiple operations atomically:

```bash
# Prepare batch request
cat << 'EOF' > /tmp/batch.json
{
  "operations": [
    {"Create": {"path": "src/module_a.rs", "content": "// Module A"}},
    {"Create": {"path": "src/module_b.rs", "content": "// Module B"}},
    {"Move": {"src": "old/readme.md", "dst": "docs/readme.md"}}
  ],
  "atomic": true,
  "dry_run": false
}
EOF

# Execute batch
cat /tmp/batch.json > /mnt/ragfs/.ragfs/.ops/.batch
cat /mnt/ragfs/.ragfs/.ops/.result
```

**Dry run mode:**

```bash
# Set dry_run: true to validate without executing
echo '{"operations":[...],"atomic":true,"dry_run":true}' > /mnt/ragfs/.ragfs/.ops/.batch
```

**Atomic rollback:**

When `atomic: true` and an operation fails, all previously successful operations are automatically rolled back:

```bash
# This will fail on the second create (file already exists from first)
echo '{"operations":[
  {"Create": {"path": "test.txt", "content": "hello"}},
  {"Create": {"path": "test.txt", "content": "duplicate"}}
],"atomic":true}' > /mnt/ragfs/.ragfs/.ops/.batch

# Result shows rollback happened
cat /mnt/ragfs/.ragfs/.ops/.result
```

**Rollback result:**
```json
{
  "success": false,
  "operations_attempted": 2,
  "operations_succeeded": 1,
  "rollback_performed": true,
  "rollback_details": {
    "rolled_back": 1,
    "rollback_failures": 0
  }
}
```

#### Creating Directories

```bash
# Create a directory (including nested paths)
echo '{"operations":[{"Mkdir": {"path": "new/nested/dir"}}],"atomic":false}' > /mnt/ragfs/.ragfs/.ops/.batch
```

#### Creating Symlinks (Unix only)

```bash
# Create a symbolic link
echo '{"operations":[{"Symlink": {"target": "original.txt", "link": "link.txt"}}],"atomic":false}' > /mnt/ragfs/.ragfs/.ops/.batch
```

### Safety Features (`.safety/`)

The safety layer protects against accidental data loss with soft delete, audit logging, and undo support.

#### Viewing Operation History

```bash
# View all operations (JSONL format)
cat /mnt/ragfs/.ragfs/.safety/.history

# Parse with jq
cat /mnt/ragfs/.ragfs/.safety/.history | jq -s '.'
```

**History entry format:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-01-11T14:32:15Z",
  "operation": {"Delete": {"path": "/docs/old.md", "trash_id": "..."}},
  "undo_data": {"DeletedFile": {"trash_id": "..."}}
}
```

#### Managing Trash

```bash
# List deleted files
ls /mnt/ragfs/.ragfs/.safety/.trash/

# View deleted file info
cat /mnt/ragfs/.ragfs/.safety/.trash/<uuid>

# Restore a file from trash
echo "restore" > /mnt/ragfs/.ragfs/.safety/.trash/<uuid>
```

**Trash entry info:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "original_path": "/home/user/docs/old-file.md",
  "deleted_at": "2025-01-11T14:32:15Z",
  "size": 1234,
  "content_hash": "abc123..."
}
```

#### Undoing Operations

Most operations can be undone using the operation ID:

```bash
# Get the undo_id from the operation result or history
echo "550e8400-e29b-41d4-a716-446655440000" > /mnt/ragfs/.ragfs/.safety/.undo

cat /mnt/ragfs/.ragfs/.ops/.result
```

**Note:** Not all operations are undoable. Check `undo_data` in the history to see if an operation can be reversed.

### Semantic Operations (`.semantic/`)

AI-powered file operations using vector embeddings for intelligent file management.

#### Finding Similar Files

```bash
# Write a file path to find similar files
echo "src/auth/login.rs" > /mnt/ragfs/.ragfs/.semantic/.similar

# Read results
cat /mnt/ragfs/.ragfs/.semantic/.similar
```

**Result:**
```json
{
  "query_path": "src/auth/login.rs",
  "similar": [
    {"path": "src/auth/oauth.rs", "score": 0.89, "shared_topics": ["authentication", "user session"]},
    {"path": "src/middleware/auth.rs", "score": 0.82, "shared_topics": ["authentication"]},
    {"path": "src/handlers/session.rs", "score": 0.75, "shared_topics": ["user session"]}
  ]
}
```

#### Organizing Files

Request intelligent file organization:

```bash
# Request organization by topic
echo '{"scope":"docs/","strategy":"ByTopic","dry_run":false}' \
    > /mnt/ragfs/.ragfs/.semantic/.organize

# A plan is created in .pending/
ls /mnt/ragfs/.ragfs/.semantic/.pending/
```

**Organization strategies:**
- `ByTopic` - Group files by semantic similarity/topic
- `ByType` - Group by file type (code, docs, config, etc.)
- `ByDate` - Group by modification date
- `Flatten` - Flatten nested directories
- `Custom` - Custom grouping logic (JSON description)

#### Reviewing Plans

All semantic operations that modify files go through a propose-review-apply workflow:

```bash
# List pending plans
ls /mnt/ragfs/.ragfs/.semantic/.pending/

# Review a specific plan
cat /mnt/ragfs/.ragfs/.semantic/.pending/<plan_id>
```

**Note:** Plans are automatically persisted to disk and survive filesystem restarts. Pending plans will still be available after unmounting and remounting.

**Plan format:**
```json
{
  "id": "plan-uuid-here",
  "created_at": "2025-01-11T14:32:15Z",
  "description": "Organize docs/ by topic into 3 groups",
  "actions": [
    {
      "operation": {"Move": {"src": "docs/api.md", "dst": "docs/api/api.md"}},
      "reason": "Groups with other API documentation",
      "confidence": 0.92
    }
  ],
  "impact_summary": {
    "files_affected": 12,
    "moves": 10,
    "deletes": 0,
    "creates": 2
  }
}
```

#### Approving or Rejecting Plans

```bash
# Approve a plan (executes ALL actions automatically)
echo "<plan_id>" > /mnt/ragfs/.ragfs/.semantic/.approve

# Reject a plan (discards without execution)
echo "<plan_id>" > /mnt/ragfs/.ragfs/.semantic/.reject
```

**Execution behavior:**
- Actions are executed sequentially in order
- If any action fails, execution stops and plan status becomes `Failed`
- Each successful action generates an `undo_id` for manual reversal via `.safety/.undo`
- Check the plan status after approval to verify execution result

#### Cleanup Analysis

View suggestions for cleaning up your project:

```bash
cat /mnt/ragfs/.ragfs/.semantic/.cleanup
```

**Output:**
```json
{
  "duplicates": [
    {"hash": "abc123", "files": ["a.txt", "backup/a.txt"], "size": 1024}
  ],
  "stale_files": [
    {"path": "old/unused.md", "last_modified": "2024-01-01T00:00:00Z", "days_stale": 376}
  ],
  "empty_dirs": ["src/deprecated/", "old/"],
  "large_files": [
    {"path": "data/dump.sql", "size": 52428800}
  ]
}
```

#### Duplicate Detection

Find duplicate files by content:

```bash
cat /mnt/ragfs/.ragfs/.semantic/.dedupe
```

**Output:**
```json
{
  "groups": [
    {
      "hash": "abc123def456...",
      "files": ["src/utils.rs", "lib/utils.rs", "backup/utils.rs"],
      "size": 2048
    }
  ],
  "total_recoverable_bytes": 4096
}
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

### Agent Operations

- Always check `.ops/.result` after operations for feedback
- Use `dry_run: true` in batch operations to validate before executing
- Review semantic plans before approving to verify the proposed changes
- Keep the safety layer enabled to allow undo of accidental changes

### Integration

```bash
# Find files matching a concept
ragfs query ./src "error handling" -f json | jq -r '.results[].file' | xargs code

# Search and open in vim
vim $(ragfs query ./src "config" -f json | jq -r '.results[0].file')

# Agent workflow: create file and verify
echo -e "docs/api.md\n# API Reference" > /mnt/ragfs/.ragfs/.ops/.create
cat /mnt/ragfs/.ragfs/.ops/.result | jq '.success'
```
