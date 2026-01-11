# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build                          # Development build
cargo build --release                # Release build
cargo build -p <crate-name>          # Build specific crate
```

## Testing

```bash
cargo test --all                     # Run all tests
cargo test -p <crate-name>           # Run tests for specific crate
cargo test -- --nocapture            # Run tests with output
```

## Linting & Formatting

```bash
cargo fmt --all                      # Format code
cargo fmt --all -- --check           # Check formatting
cargo clippy --all-targets -- -D warnings  # Run clippy
```

## Running

```bash
cargo run -- -v index /path/to/dir   # Index with verbose logging
cargo run -- query /path "search"    # Query the index
RUST_LOG=debug cargo run -- ...      # Debug logging
```

## Architecture

RAGFS is a FUSE filesystem for semantic search using vector embeddings. It's organized as a Rust workspace with 9 crates following a pipeline pattern:

```
File → Extraction → Chunking → Embedding → Storage → Search
```

**Crate hierarchy:**

- `ragfs` - CLI binary (entry: `crates/ragfs/src/main.rs`)
- `ragfs-core` - Foundation traits and types (`traits.rs`, `types.rs`)
- `ragfs-extract` - Content extraction via `ExtractorRegistry`
- `ragfs-chunker` - Document chunking via `ChunkerRegistry`
- `ragfs-embed` - Local embeddings using Candle (`gte-small`, 384-dim)
- `ragfs-store` - Vector storage with LanceDB
- `ragfs-index` - Pipeline orchestration and file watching
- `ragfs-query` - Query execution
- `ragfs-fuse` - FUSE filesystem

**Key abstractions in ragfs-core:**
- `ContentExtractor` trait - Extract content from files
- `Chunker` trait - Split content into chunks
- `Embedder` trait - Generate vector embeddings
- `VectorStore` trait - Store and search vectors

**Extension points:** Add new extractors/chunkers by implementing the trait and registering with the appropriate registry.

## Key Details

- Rust 1.88+ required (edition 2024)
- Async-first design using Tokio
- Content-addressed storage using blake3 hashing
- Embeddings are generated locally (offline after first model download)
- Storage: `~/.local/share/ragfs/indices/` and `~/.local/share/ragfs/models/`
