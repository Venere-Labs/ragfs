# Getting Started with RAGFS

This guide will get you up and running with RAGFS in 5 minutes.

## Table of Contents

- [What is RAGFS?](#what-is-ragfs)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [5-Minute Tutorial](#5-minute-tutorial)
- [Three Ways to Use RAGFS](#three-ways-to-use-ragfs)
- [Next Steps](#next-steps)

---

## What is RAGFS?

RAGFS is a **semantic search filesystem** for your files. Instead of searching by filename or keywords, you can search by **meaning**.

**Traditional search:**
```bash
grep -r "authenticate" ~/Documents   # Finds literal string matches
find ~/Documents -name "*auth*"      # Finds filename patterns
```

**RAGFS semantic search:**
```bash
ragfs query ~/Documents "how does user login work"
# Finds files about authentication, even if they don't contain "login"
```

RAGFS converts your files into **vector embeddings** (numerical representations of meaning) and stores them locally. When you search, it finds files with similar meaning to your query.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Your      │     │    RAGFS     │     │   Vector    │
│   Files     │────▶│   Pipeline   │────▶│   Index     │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │ Extract │   │  Chunk  │
              │ Content │   │  Text   │
              └────┬────┘   └────┬────┘
                   │             │
                   ▼             ▼
              ┌─────────┐   ┌─────────┐
              │ Generate│   │  Store  │
              │Embedding│   │ Vectors │
              └─────────┘   └─────────┘
```

1. **Extract** - Content is pulled from files (text, code, PDFs, images)
2. **Chunk** - Large files are split into ~512 token segments
3. **Embed** - Each chunk becomes a 384-dimensional vector using the `gte-small` model
4. **Store** - Vectors are saved in LanceDB for fast similarity search
5. **Query** - Your question becomes a vector and finds similar content

**All processing happens locally** - no data leaves your machine.

---

## Installation

### Prerequisites

- **Rust 1.88+**: Install via [rustup.rs](https://rustup.rs)
- **FUSE libraries** (for mounting):
  - Debian/Ubuntu: `sudo apt install libfuse-dev pkg-config`
  - Fedora: `sudo dnf install fuse-devel`
  - Arch: `sudo pacman -S fuse2`

### Build from Source

```bash
# Clone the repository
git clone https://github.com/Venere-Labs/ragfs.git
cd ragfs

# Build in release mode
cargo build --release

# Install to ~/.cargo/bin
cargo install --path crates/ragfs

# Verify installation
ragfs --version
```

### First Run

The first time you run RAGFS, it downloads the embedding model (~100MB):

```bash
ragfs index ~/Documents
# Output: Initializing embedder (this may download the model on first run)...
```

After this, RAGFS works completely offline.

---

## 5-Minute Tutorial

Let's index a project and search it semantically.

### Step 1: Index Your Files

```bash
# Index a directory (e.g., a project folder)
ragfs index ~/my-project
```

Output:
```
Indexing /home/user/my-project...
  Processed: 47 files
  Chunks: 312
  Time: 8.2s
```

### Step 2: Search by Meaning

```bash
# Ask a question in natural language
ragfs query ~/my-project "how does authentication work"
```

Output:
```
Query: how does authentication work

1. src/auth/jwt.rs (score: 0.89)
   Lines: 23-45
   The JwtValidator verifies tokens using RS256 signatures...

2. src/middleware/auth.rs (score: 0.82)
   Lines: 12-28
   AuthMiddleware extracts the Bearer token from headers...

3. docs/security.md (score: 0.76)
   Lines: 45-62
   ## Authentication Flow
   Users authenticate via OAuth2...
```

### Step 3: Try Different Queries

```bash
# Find error handling patterns
ragfs query ~/my-project "error handling and recovery"

# Find database-related code
ragfs query ~/my-project "how to connect to the database"

# Get more results
ragfs query ~/my-project "configuration" --limit 20
```

### Step 4: Check Index Status

```bash
ragfs status ~/my-project
```

Output:
```
Index Status for "/home/user/my-project"
  Files:  47
  Chunks: 312
  Updated: 2025-01-13 10:15:32
```

**That's it!** You now have semantic search over your files.

---

## Three Ways to Use RAGFS

RAGFS supports three usage modes. Choose based on your needs:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAGFS Usage Modes                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
│  │   CLI Query   │   │  FUSE Mount   │   │  Watch Mode   │     │
│  │   (Simple)    │   │  (Advanced)   │   │ (Continuous)  │     │
│  └───────────────┘   └───────────────┘   └───────────────┘     │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  • One-time index    • Virtual filesystem  • Real-time sync    │
│  • Quick queries     • Agent operations    • Background daemon │
│  • Scripting         • Safety features     • Auto-reindex      │
│                      • Semantic ops                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mode 1: CLI Query (Simplest)

Best for: Quick searches, scripting, one-off queries.

```bash
# Index once
ragfs index ~/Documents

# Query whenever you need
ragfs query ~/Documents "how to deploy to production"

# Use JSON for scripts
ragfs query ~/Documents "config" -f json | jq '.results[].file'
```

### Mode 2: FUSE Mount (Most Powerful)

Best for: AI agents, continuous access, advanced operations.

```bash
# Create mount point
mkdir ~/ragfs-mount

# Mount the indexed directory
ragfs mount ~/Documents ~/ragfs-mount --foreground
```

Once mounted, you get:
- **Real files** accessible normally
- **Virtual `.ragfs/` directory** with special capabilities:
  - `.ragfs/.query/<text>` - Query by reading a virtual file
  - `.ragfs/.ops/` - Structured file operations with JSON feedback
  - `.ragfs/.safety/` - Undo, trash, audit history
  - `.ragfs/.semantic/` - AI-powered organize, dedupe, cleanup

```bash
# Query via filesystem
cat ~/ragfs-mount/.ragfs/.query/authentication

# Create file with feedback
echo -e "notes/idea.md\n# My Idea" > ~/ragfs-mount/.ragfs/.ops/.create
cat ~/ragfs-mount/.ragfs/.ops/.result

# Unmount when done
fusermount -u ~/ragfs-mount
```

### Mode 3: Watch Mode (Continuous)

Best for: Keeping index fresh during active development.

```bash
# Start watching (runs in foreground)
ragfs index ~/my-project --watch
```

Files are automatically re-indexed when they change. Press `Ctrl+C` to stop.

---

## Choosing the Right Mode

| Use Case | Recommended Mode |
|----------|-----------------|
| Quick one-time search | CLI Query |
| Script/automation | CLI Query with `-f json` |
| AI agent integration | FUSE Mount |
| Active development | Watch Mode |
| Long-running service | FUSE Mount (daemon) |

---

## Common First Questions

### "How do I update the index after editing files?"

Option 1: Force reindex
```bash
ragfs index ~/my-project --force
```

Option 2: Use watch mode
```bash
ragfs index ~/my-project --watch
```

Option 3: FUSE mount (auto-updates)
```bash
ragfs mount ~/Documents ~/ragfs-mount
```

### "Where is my data stored?"

| Data | Location |
|------|----------|
| Vector index | `~/.local/share/ragfs/indices/{hash}/` |
| Embedding model | `~/.local/share/ragfs/models/` |
| Trash (soft deletes) | `~/.local/share/ragfs/trash/` |
| Config | `~/.config/ragfs/config.toml` |

### "What files does RAGFS understand?"

- **Text**: `.txt`, `.md`, `.rst`
- **Code**: `.rs`, `.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, and 30+ more
- **Config**: `.json`, `.yaml`, `.toml`, `.xml`
- **Documents**: `.pdf` (text + embedded images)
- **Images**: `.png`, `.jpg`, `.gif` (metadata extraction)

### "How much disk space do I need?"

- **Embedding model**: ~100MB (downloaded once)
- **Vector index**: Roughly 1-2KB per chunk (varies by content)
- Example: 1000 files → ~500KB to 5MB index

### "Is my data sent anywhere?"

No. RAGFS processes everything locally:
- Embeddings generated on your machine
- No API calls after model download
- No telemetry or data collection

---

## Next Steps

You're ready to use RAGFS. Here's where to go next:

1. **[User Guide](USER_GUIDE.md)** - Complete CLI reference and examples
2. **[Agent Operations](USER_GUIDE.md#agent-file-operations)** - Using `.ops/`, `.safety/`, `.semantic/`
3. **[Architecture](ARCHITECTURE.md)** - Technical deep-dive
4. **[API Reference](API.md)** - Library usage and types
5. **[Configuration](USER_GUIDE.md#configuration)** - Customizing behavior

### Quick Tips

- Use natural language in queries: "how does X work" beats "X"
- Be specific: "JWT token validation" beats "authentication"
- Check index status before long-running queries: `ragfs status ~/path`

---

## Getting Help

- **Issues**: [github.com/Venere-Labs/ragfs/issues](https://github.com/Venere-Labs/ragfs/issues)
- **Verbose mode**: `ragfs -v index ~/path` for debug output
- **Logs**: `~/.cache/ragfs/logs/`
