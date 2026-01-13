# RAGFS Performance Guide

This guide covers performance characteristics, tuning, and optimization for RAGFS.

## Table of Contents

- [Expected Performance](#expected-performance)
- [Resource Usage](#resource-usage)
- [Tuning Parameters](#tuning-parameters)
- [Optimization Tips](#optimization-tips)
- [Benchmarking](#benchmarking)
- [Scaling Guidelines](#scaling-guidelines)

---

## Expected Performance

### Indexing Speed

Typical indexing throughput on modern hardware (SSD, 8+ cores):

| File Type | Files/min | Notes |
|-----------|-----------|-------|
| Plain text (.txt, .md) | 300-500 | Fastest |
| Source code (.rs, .py, .js) | 200-400 | Tree-sitter parsing adds overhead |
| PDF documents | 50-100 | Depends on page count and images |
| Large files (>1MB) | 10-50 | Limited by embedding batch size |

**First run penalty:** ~30-60 seconds for model initialization.

### Query Latency

| Query Type | Latency | Notes |
|------------|---------|-------|
| Vector search | 5-50ms | Depends on index size |
| Hybrid search | 10-100ms | Adds full-text component |
| Similar files | 20-100ms | Multiple queries per file |

### Memory Usage

| Component | Typical Usage | Notes |
|-----------|---------------|-------|
| Embedding model | 200-400 MB | Loaded once, shared |
| Vector index | 1-5 MB per 1000 chunks | In-memory index structure |
| Query processing | 50-100 MB | Per-query overhead |

---

## Resource Usage

### Disk Space

| Data | Approximate Size |
|------|------------------|
| Embedding model | ~100 MB |
| Vector per chunk | ~1.5 KB (384 dims x 4 bytes) |
| Metadata per chunk | ~500 bytes |
| Index overhead | ~20% of raw vectors |

**Rule of thumb:** Index size ≈ 2 KB × number of chunks

**Example:**
- 1000 files → ~5000 chunks → ~10 MB index
- 10000 files → ~50000 chunks → ~100 MB index

### CPU Usage

| Operation | CPU Pattern |
|-----------|-------------|
| Indexing | High, sustained (embedding generation) |
| Querying | Burst, short duration |
| FUSE mount | Low, idle most of time |
| File watching | Very low, event-driven |

### Memory Baseline

```
Base process:        ~50 MB
+ Embedding model:   ~300 MB
+ Index (1K files):  ~20 MB
+ Per query:         ~50 MB
------------------------
Typical total:       ~420 MB for small-medium projects
```

---

## Tuning Parameters

### Chunking Configuration

Chunk size affects index granularity and memory usage:

```toml
[chunking]
target_size = 512   # Tokens per chunk
max_size = 1024     # Maximum chunk size
overlap = 64        # Overlap between chunks
```

| Setting | Trade-off |
|---------|-----------|
| Smaller chunks (256-384) | More precise results, larger index, slower indexing |
| Larger chunks (768-1024) | Faster indexing, fewer chunks, less precise |
| More overlap (128+) | Better recall at boundaries, larger index |
| Less overlap (32-64) | Smaller index, risk of missing context |

**Recommended presets:**

```toml
# For code repositories (precise function-level search)
[chunking]
target_size = 384
overlap = 96

# For documentation (longer context)
[chunking]
target_size = 768
overlap = 128

# For mixed content (balanced)
[chunking]
target_size = 512
overlap = 64
```

### Embedding Configuration

```toml
[embedding]
batch_size = 32       # Texts per batch
max_concurrent = 4    # Parallel embedding jobs
use_gpu = true        # Use GPU if available
```

| Setting | Effect |
|---------|--------|
| `batch_size` | Higher = more memory, better throughput |
| `max_concurrent` | Higher = more CPU usage, faster indexing |
| `use_gpu` | Much faster if GPU available |

**Low memory configuration:**
```toml
[embedding]
batch_size = 8
max_concurrent = 1
```

**High performance configuration:**
```toml
[embedding]
batch_size = 64
max_concurrent = 8
```

### Index Configuration

```toml
[index]
max_file_size = 52428800  # 50 MB
debounce_ms = 500         # File watcher debounce
```

| Setting | Effect |
|---------|--------|
| `max_file_size` | Limits memory per file |
| `debounce_ms` | Batches rapid file changes |

### Query Configuration

```toml
[query]
default_limit = 10    # Default results
max_limit = 100       # Maximum results
hybrid = true         # Vector + full-text
```

Hybrid search adds ~50% latency but improves recall for keyword-heavy queries.

---

## Optimization Tips

### For Faster Indexing

1. **Use SSD storage** for both source and index directories.

2. **Exclude unnecessary files:**
   ```toml
   [index]
   exclude = [
       "**/node_modules/**",
       "**/target/**",
       "**/.git/**",
       "**/*.lock",
       "**/dist/**",
       "**/build/**"
   ]
   ```

3. **Limit file size:**
   ```toml
   [index]
   max_file_size = 5242880  # 5 MB
   ```

4. **Index specific file types:**
   ```toml
   [index]
   include = ["**/*.rs", "**/*.py", "**/*.md"]
   ```

5. **Increase concurrency** (if memory allows):
   ```toml
   [embedding]
   max_concurrent = 8
   batch_size = 64
   ```

### For Faster Queries

1. **Reduce result limit:**
   ```bash
   ragfs query ~/project "search" --limit 5
   ```

2. **Disable hybrid search** for pure semantic queries:
   ```toml
   [query]
   hybrid = false
   ```

3. **Use SSD storage** for index location.

4. **Pre-warm the index** by running a dummy query after mount:
   ```bash
   ragfs query ~/project "warm up" --limit 1 > /dev/null
   ```

### For Lower Memory Usage

1. **Reduce batch size:**
   ```toml
   [embedding]
   batch_size = 8
   max_concurrent = 1
   ```

2. **Use smaller chunks:**
   ```toml
   [chunking]
   target_size = 256
   max_size = 384
   ```

3. **Limit file size:**
   ```toml
   [index]
   max_file_size = 2097152  # 2 MB
   ```

4. **Index fewer file types** to reduce total chunks.

### For FUSE Mount

1. **Run in foreground** during development to see errors:
   ```bash
   ragfs mount ~/project ~/mnt --foreground
   ```

2. **Keep index on fast storage:**
   ```bash
   export RAGFS_DATA_DIR=/mnt/fast-ssd/ragfs
   ragfs mount ~/project ~/mnt
   ```

---

## Benchmarking

### Measure Indexing Speed

```bash
# Time indexing
time ragfs index ~/project --force

# With verbose output
RUST_LOG=info ragfs index ~/project --force 2>&1 | tee indexing.log
```

### Measure Query Latency

```bash
# Single query timing
time ragfs query ~/project "search term" --limit 10

# Multiple queries
for i in {1..10}; do
    time ragfs query ~/project "query $i" --limit 5 2>/dev/null
done
```

### Profile Memory Usage

```bash
# Monitor peak memory during indexing
/usr/bin/time -v ragfs index ~/project --force 2>&1 | grep "Maximum resident"

# Continuous monitoring
watch -n 1 'ps -o rss,vsz,comm -p $(pgrep ragfs)'
```

### Check Index Statistics

```bash
# Get index stats
ragfs status ~/project -f json | jq .

# Check index size on disk
du -sh ~/.local/share/ragfs/indices/*/
```

---

## Scaling Guidelines

### Small Projects (< 1,000 files)

Default settings work well. Expect:
- Indexing: < 2 minutes
- Query latency: < 50ms
- Memory: 400-500 MB
- Index size: 10-50 MB

### Medium Projects (1,000 - 10,000 files)

Recommended configuration:
```toml
[index]
max_file_size = 10485760  # 10 MB

[embedding]
batch_size = 32
max_concurrent = 4

[chunking]
target_size = 512
overlap = 64
```

Expect:
- Indexing: 5-15 minutes
- Query latency: 50-100ms
- Memory: 500-800 MB
- Index size: 50-200 MB

### Large Projects (10,000+ files)

Recommended configuration:
```toml
[index]
max_file_size = 5242880  # 5 MB
exclude = ["**/vendor/**", "**/node_modules/**", "**/*.min.js"]

[embedding]
batch_size = 64
max_concurrent = 8

[chunking]
target_size = 512
overlap = 32  # Reduced for index size
```

Consider:
- Index in batches by directory
- Use watch mode for incremental updates
- Place index on SSD
- Monitor memory usage

### Very Large Projects (100,000+ files)

Recommendations:
- Index specific directories, not entire repository
- Use aggressive exclusion patterns
- Consider multiple indices for different areas
- Monitor and tune based on actual usage

---

## Hardware Recommendations

### Minimum Requirements

- CPU: 4 cores
- RAM: 4 GB
- Storage: SSD with 10+ GB free

### Recommended

- CPU: 8+ cores
- RAM: 16 GB
- Storage: NVMe SSD
- GPU: Optional, improves embedding speed 5-10x

### For Production/Server Use

- CPU: 16+ cores
- RAM: 32+ GB
- Storage: NVMe SSD RAID
- GPU: NVIDIA GPU with 8+ GB VRAM (if using GPU acceleration)

---

## See Also

- [Configuration Reference](CONFIGURATION.md) - All config options
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [Architecture](ARCHITECTURE.md) - System design
