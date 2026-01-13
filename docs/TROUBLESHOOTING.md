# RAGFS Troubleshooting Guide

This guide helps you diagnose and resolve common issues with RAGFS.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Common Issues](#common-issues)
- [Error Reference](#error-reference)
- [Debug Techniques](#debug-techniques)
- [Recovery Procedures](#recovery-procedures)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run these commands to quickly assess the state of your RAGFS installation:

```bash
# Check RAGFS version
ragfs --version

# Check index status
ragfs status ~/your-directory

# Verify FUSE is available
which fusermount

# Check if user is in fuse group
groups | grep fuse

# Check logs
tail -100 ~/.cache/ragfs/logs/ragfs.log
```

---

## Common Issues

### Index Not Found

**Error message:**
```
Error: Index not found for "/path/to/directory"
```

**Cause:** The directory hasn't been indexed yet.

**Solution:**
```bash
# Index the directory first
ragfs index /path/to/directory

# Then query
ragfs query /path/to/directory "your search"
```

---

### Model Download Failed

**Error message:**
```
Error: embedding error: model loading failed: Failed to download model
```

**Cause:** Network issues during first run when downloading the embedding model.

**Solutions:**

1. **Check internet connection** and retry:
   ```bash
   ragfs index /path/to/directory
   ```

2. **Check proxy settings** if behind a corporate firewall:
   ```bash
   export HTTPS_PROXY=http://proxy:port
   ragfs index /path/to/directory
   ```

3. **Manual download** as fallback:
   - Visit [huggingface.co/thenlper/gte-small](https://huggingface.co/thenlper/gte-small)
   - Download model files to `~/.local/share/ragfs/models/`

---

### Permission Denied on Mount

**Error message:**
```
Error: Permission denied
```
or
```
fusermount: user has no write access to mountpoint
```

**Cause:** FUSE permissions not configured.

**Solution:**

1. Add yourself to the `fuse` group:
   ```bash
   sudo usermod -aG fuse $USER
   ```

2. Log out and log back in (or reboot).

3. Verify:
   ```bash
   groups | grep fuse
   ```

4. Try mounting again:
   ```bash
   ragfs mount ~/Documents ~/ragfs-mount --foreground
   ```

---

### Mount Point Doesn't Exist

**Error message:**
```
Error: Mount point does not exist: /path/to/mountpoint
```

**Solution:**
```bash
# Create the mount point
mkdir -p ~/ragfs-mount
chmod 755 ~/ragfs-mount

# Then mount
ragfs mount ~/Documents ~/ragfs-mount --foreground
```

---

### "Allow Other" Permission Error

**Error message:**
```
fusermount: option allow_other only allowed if 'user_allow_other' is set in /etc/fuse.conf
```

**Cause:** Trying to use `--allow-other` without system configuration.

**Solution:**
```bash
# Edit FUSE config
sudo nano /etc/fuse.conf

# Uncomment this line:
user_allow_other

# Save and retry
ragfs mount ~/Documents ~/mnt --allow-other
```

---

### Stale Index / Missing Files in Results

**Symptom:** Query results don't include recently modified or created files.

**Solutions:**

1. **Force reindex:**
   ```bash
   ragfs index /path/to/directory --force
   ```

2. **Use watch mode** for continuous updates:
   ```bash
   ragfs index /path/to/directory --watch
   ```

3. **With FUSE mount**, trigger reindex for specific file:
   ```bash
   echo "/path/to/file" > /mnt/ragfs/.ragfs/.reindex
   ```

---

### High Memory Usage

**Symptom:** RAGFS uses excessive memory during indexing.

**Causes:**
- Large files being processed
- Many files indexed simultaneously
- Embedding batch too large

**Solutions:**

1. **Exclude large binary files** in config:
   ```toml
   # ~/.config/ragfs/config.toml
   [index]
   exclude = ["**/*.zip", "**/*.tar.gz", "**/*.bin", "**/node_modules/**"]
   max_file_size = 10485760  # 10MB limit
   ```

2. **Reduce embedding concurrency**:
   ```toml
   [embedding]
   max_concurrent = 2  # Default is 4
   ```

3. **Index in smaller batches** using include patterns.

---

### Slow Query Performance

**Symptom:** Queries take too long to return results.

**Solutions:**

1. **Reduce result limit:**
   ```bash
   ragfs query ~/path "search" --limit 5
   ```

2. **Check index size:**
   ```bash
   ragfs status ~/path
   # If chunk count is very high, consider more selective indexing
   ```

3. **Use SSD storage** for the index directory.

4. **Check for filesystem issues:**
   ```bash
   df -h ~/.local/share/ragfs/
   ```

---

### Empty Query Results

**Symptom:** Query returns no results even for content that exists.

**Causes:**
- File not indexed (unsupported type or excluded)
- Index is empty or corrupted
- Query too specific

**Diagnosis:**
```bash
# Check index status
ragfs status ~/path

# Check with verbose logging
ragfs -v query ~/path "search term"
```

**Solutions:**

1. **Verify file is supported:**
   - Check the file extension is in the supported list
   - Check file isn't in exclude patterns

2. **Try broader query:**
   ```bash
   # Instead of very specific
   ragfs query ~/path "the specific function validateUserJWTToken"

   # Try more general
   ragfs query ~/path "JWT token validation"
   ```

3. **Force reindex:**
   ```bash
   ragfs index ~/path --force
   ```

---

### FUSE Mount Hangs

**Symptom:** Mount command doesn't return, or filesystem operations hang.

**Solutions:**

1. **Kill and unmount:**
   ```bash
   # Find the process
   ps aux | grep ragfs

   # Kill it
   kill -9 <pid>

   # Force unmount
   fusermount -u ~/ragfs-mount
   # or if that fails:
   sudo umount -l ~/ragfs-mount
   ```

2. **Check for stale mount:**
   ```bash
   mount | grep ragfs
   ```

3. **Use foreground mode** for debugging:
   ```bash
   ragfs mount ~/Documents ~/ragfs-mount --foreground
   ```

---

### Python Binding Import Error

**Error message:**
```python
ImportError: No module named 'ragfs'
```

**Solutions:**

1. **Install the Python package:**
   ```bash
   pip install ragfs
   # or
   pip install -e ./crates/ragfs-python
   ```

2. **Check virtual environment:**
   ```bash
   which python
   pip list | grep ragfs
   ```

3. **Verify Rust extension built:**
   ```bash
   cd crates/ragfs-python
   maturin develop
   ```

---

### MCP Server Connection Failed

**Error message:**
```
Error: Failed to connect to MCP server
```

**Solutions:**

1. **Start the MCP server:**
   ```bash
   ragfs mcp serve
   ```

2. **Check port availability:**
   ```bash
   lsof -i :8080  # Default MCP port
   ```

3. **Check environment variables:**
   ```bash
   export RAGFS_MCP_HOST=localhost
   export RAGFS_MCP_PORT=8080
   ```

---

## Error Reference

### Extraction Errors (`ExtractError`)

| Error | Message | Cause | Solution |
|-------|---------|-------|----------|
| `UnsupportedType` | `unsupported file type: {mime}` | File type not supported | Add custom extractor or skip file |
| `Parse` | `parse error: {details}` | File content couldn't be parsed | Check file encoding, may be corrupted |
| `Io` | `io error: {details}` | File read failed | Check permissions, disk space |
| `Failed` | `extraction failed: {details}` | General extraction failure | Check verbose logs for details |

### Chunking Errors (`ChunkError`)

| Error | Message | Cause | Solution |
|-------|---------|-------|----------|
| `Failed` | `chunking failed: {details}` | Content couldn't be chunked | File may be empty or binary |
| `InvalidConfig` | `invalid configuration: {details}` | Bad chunk config | Check config.toml chunking section |

### Embedding Errors (`EmbedError`)

| Error | Message | Cause | Solution |
|-------|---------|-------|----------|
| `ModelLoad` | `model loading failed: {details}` | Model couldn't be loaded | Check model files, redownload |
| `Inference` | `inference failed: {details}` | Embedding generation failed | Check memory, GPU drivers |
| `ModalityNotSupported` | `modality not supported: {type}` | Unsupported content type | Only text embeddings supported |
| `InputTooLong` | `input too long: {n} tokens, max {max}` | Text exceeds 512 tokens | Content will be chunked automatically |

### Store Errors (`StoreError`)

| Error | Message | Cause | Solution |
|-------|---------|-------|----------|
| `Init` | `store initialization failed: {details}` | Database couldn't start | Check disk space, permissions |
| `Insert` | `insert failed: {details}` | Data couldn't be stored | May be disk full or corruption |
| `Query` | `query failed: {details}` | Search failed | Check vector dimensions match |
| `Delete` | `delete failed: {details}` | Deletion failed | File may not be indexed |
| `Schema` | `schema error: {details}` | Database schema issue | May need to recreate index |

### I/O Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Permission denied` | No read/write access | Check file/directory permissions |
| `No such file or directory` | Path doesn't exist | Verify path is correct |
| `Is a directory` | Expected file, got directory | Use correct path |
| `Disk quota exceeded` | No disk space | Free up space |

---

## Debug Techniques

### Enable Verbose Logging

```bash
# CLI verbose mode
ragfs -v index ~/Documents

# Debug level logging
RUST_LOG=debug ragfs index ~/Documents

# Trace level (very verbose)
RUST_LOG=trace ragfs index ~/Documents
```

### Check Log Files

```bash
# View recent logs
tail -f ~/.cache/ragfs/logs/ragfs.log

# Search for errors
grep -i error ~/.cache/ragfs/logs/ragfs.log

# View with context
grep -B5 -A5 "error" ~/.cache/ragfs/logs/ragfs.log
```

### Inspect Index Database

```bash
# Check index location
ls -la ~/.local/share/ragfs/indices/

# Check index size
du -sh ~/.local/share/ragfs/indices/*

# View index status
ragfs status ~/your-directory -f json | jq .
```

### Test Individual Components

```bash
# Test extraction on a single file
ragfs -v index ~/Documents/test.md --force

# Check what files would be indexed
find ~/Documents -type f -name "*.rs" | head -10

# Verify MIME type detection
file --mime-type ~/Documents/example.pdf
```

### Profile Memory Usage

```bash
# Monitor memory during indexing
/usr/bin/time -v ragfs index ~/Documents --force 2>&1 | grep "Maximum resident"

# Or use htop/top during indexing
htop -p $(pgrep ragfs)
```

---

## Recovery Procedures

### Rebuild Corrupted Index

If the index is corrupted or behaving unexpectedly:

```bash
# Remove the index
rm -rf ~/.local/share/ragfs/indices/<hash>/

# Reindex from scratch
ragfs index ~/your-directory --force
```

To find the index hash:
```bash
ragfs status ~/your-directory -f json | jq .index_path
```

### Recover from Trash

Files deleted via `.ops/.delete` go to trash:

```bash
# List trash contents (via FUSE mount)
ls /mnt/ragfs/.ragfs/.safety/.trash/

# View file details
cat /mnt/ragfs/.ragfs/.safety/.trash/<uuid>

# Restore a file
echo "restore" > /mnt/ragfs/.ragfs/.safety/.trash/<uuid>
```

Or manually:
```bash
# Find in trash directory
ls ~/.local/share/ragfs/trash/<index-hash>/

# Copy back manually
cp ~/.local/share/ragfs/trash/<hash>/<uuid> ~/original/path/
```

### Undo Operations

If you have the operation ID:

```bash
# Via FUSE mount
echo "<operation-uuid>" > /mnt/ragfs/.ragfs/.safety/.undo

# Check result
cat /mnt/ragfs/.ragfs/.ops/.result
```

### Reset Configuration

```bash
# Backup current config
cp ~/.config/ragfs/config.toml ~/.config/ragfs/config.toml.bak

# Generate fresh config
ragfs config init > ~/.config/ragfs/config.toml
```

### Clean Up All RAGFS Data

**Warning:** This deletes all indices and cached models.

```bash
# Remove all RAGFS data
rm -rf ~/.local/share/ragfs/
rm -rf ~/.cache/ragfs/

# Remove config (optional)
rm -rf ~/.config/ragfs/
```

---

## Getting Help

### Collect Diagnostic Information

When reporting issues, include:

```bash
# System info
uname -a
rustc --version
ragfs --version

# Index status
ragfs status ~/your-directory -f json

# Recent errors
tail -50 ~/.cache/ragfs/logs/ragfs.log

# Disk space
df -h ~/.local/share/ragfs/
```

### Report Issues

- **GitHub Issues**: [github.com/Venere-Labs/ragfs/issues](https://github.com/Venere-Labs/ragfs/issues)
- Include: Error message, steps to reproduce, diagnostic info above

### Resources

- [Getting Started Guide](GETTING_STARTED.md)
- [User Guide](USER_GUIDE.md)
- [Architecture](ARCHITECTURE.md)
- [API Reference](API.md)
