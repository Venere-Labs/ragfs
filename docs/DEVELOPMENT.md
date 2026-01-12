# RAGFS Development Guide

This guide helps developers set up their environment and contribute to RAGFS.

## Prerequisites

### Rust

Install Rust 1.88 or later via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

Verify your installation:

```bash
rustc --version
# Should show 1.88.0 or later
```

### System Dependencies

#### Debian/Ubuntu

```bash
sudo apt update
sudo apt install build-essential pkg-config libfuse-dev
```

#### Fedora

```bash
sudo dnf install gcc make pkg-config fuse-devel
```

#### Arch Linux

```bash
sudo pacman -S base-devel fuse2
```

#### macOS

FUSE support requires macFUSE:

```bash
brew install macfuse
```

Note: macOS support is experimental.

## Building

### Development Build

```bash
git clone https://github.com/user/ragfs.git
cd ragfs

# Build all crates
cargo build

# Build specific crate
cargo build -p ragfs-core
```

### Release Build

```bash
cargo build --release
```

The binary will be at `target/release/ragfs`.

### Documentation

Generate and view rustdocs:

```bash
cargo doc --no-deps --open
```

## Running Tests

```bash
# Run all tests
cargo test --all

# Run tests for a specific crate
cargo test -p ragfs-core

# Run tests with output
cargo test -- --nocapture
```

## Project Structure

```
ragfs/
├── Cargo.toml           # Workspace configuration
├── crates/
│   ├── ragfs/           # CLI binary
│   │   ├── src/
│   │   │   ├── main.rs  # Entry point
│   │   │   └── config.rs
│   │   └── examples/
│   │
│   ├── ragfs-core/      # Core abstractions
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs
│   │       ├── types.rs
│   │       └── error.rs
│   │
│   ├── ragfs-extract/   # Content extraction
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── text.rs
│   │       └── registry.rs
│   │
│   ├── ragfs-chunker/   # Document chunking
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── fixed.rs
│   │       └── registry.rs
│   │
│   ├── ragfs-embed/     # Embedding generation
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── candle.rs
│   │       └── pool.rs
│   │
│   ├── ragfs-store/     # Vector storage
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── lancedb.rs
│   │       └── schema.rs
│   │
│   ├── ragfs-index/     # Indexing pipeline
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── indexer.rs
│   │       └── watcher.rs
│   │
│   ├── ragfs-query/     # Query execution
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── executor.rs
│   │       └── parser.rs
│   │
│   └── ragfs-fuse/      # FUSE filesystem
│       └── src/
│           ├── lib.rs
│           ├── filesystem.rs
│           ├── inode.rs
│           ├── ops.rs       # Agent file operations
│           ├── safety.rs    # Safety layer (trash, undo)
│           └── semantic.rs  # Semantic operations
│
└── docs/                # Documentation
```

## Adding a New Extractor

1. Create a new file in `crates/ragfs-extract/src/`:

```rust
// crates/ragfs-extract/src/pdf.rs
use async_trait::async_trait;
use ragfs_core::{ContentExtractor, ExtractedContent, ExtractError};
use std::path::Path;

pub struct PdfExtractor;

impl PdfExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContentExtractor for PdfExtractor {
    fn supported_types(&self) -> &[&str] {
        &["application/pdf"]
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent, ExtractError> {
        // Implementation here
        todo!()
    }
}
```

2. Export in `lib.rs`:

```rust
mod pdf;
pub use pdf::PdfExtractor;
```

3. Register in the CLI:

```rust
let mut extractors = ExtractorRegistry::new();
extractors.register("text", TextExtractor::new());
extractors.register("pdf", PdfExtractor::new());
```

## Adding a New Chunking Strategy

1. Create a new file in `crates/ragfs-chunker/src/`:

```rust
// crates/ragfs-chunker/src/semantic.rs
use async_trait::async_trait;
use ragfs_core::{Chunker, ChunkConfig, ChunkError, ChunkOutput, ContentType, ExtractedContent};

pub struct SemanticChunker;

impl SemanticChunker {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Chunker for SemanticChunker {
    fn name(&self) -> &str {
        "semantic"
    }

    fn content_types(&self) -> &[&str] {
        &["text", "markdown"]
    }

    fn can_chunk(&self, content_type: &ContentType) -> bool {
        matches!(content_type, ContentType::Text | ContentType::Markdown)
    }

    async fn chunk(
        &self,
        content: &ExtractedContent,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkOutput>, ChunkError> {
        // Implementation here
        todo!()
    }
}
```

2. Export and register similarly to extractors.

## Extending Agent Operations

### Adding a New Operation Type

To add a new file operation to OpsManager:

1. Add the variant to the `Operation` enum in `crates/ragfs-fuse/src/ops.rs`:

```rust
pub enum Operation {
    Create { path: String, content: String },
    Delete { path: String },
    Move { src: String, dst: String },
    Copy { src: String, dst: String },
    Write { path: String, content: String, mode: WriteMode },
    // Add new operation:
    Rename { path: String, new_name: String },
}
```

2. Handle the operation in `OpsManager::execute_operation`:

```rust
fn execute_operation(&self, op: Operation) -> OperationResult {
    match op {
        Operation::Rename { path, new_name } => {
            // 1. Validate the paths
            // 2. Execute the operation
            // 3. Log to safety history
            // 4. Trigger reindex if needed
            // 5. Return result with undo_id
        }
        // ... other cases
    }
}
```

3. Add the corresponding virtual file in `filesystem.rs` to handle writes to `.ops/.rename`.

### Adding a New Virtual File

To expose a new operation through the filesystem:

1. Add an inode kind in `inode.rs`:

```rust
pub enum InodeKind {
    // ...
    OpsRename,
}
```

2. Add to the inode table initialization in `filesystem.rs`:

```rust
// In RagFs::init_inode_table
table.add_virtual(OPS_RENAME_INO, InodeKind::OpsRename);
```

3. Handle writes in `filesystem.rs`:

```rust
fn write(&mut self, ...) -> Result<...> {
    match self.inodes.get(ino) {
        Some(InodeKind::OpsRename) => {
            let op = parse_rename_request(data)?;
            self.ops_manager.execute_operation(op);
        }
        // ...
    }
}
```

## Extending the Safety Layer

### Adding Undo Support for New Operations

1. Add the undo data variant in `crates/ragfs-fuse/src/safety.rs`:

```rust
pub enum UndoData {
    // ...
    RenamedFile { old_name: String, new_name: String },
}
```

2. Capture undo data when executing the operation:

```rust
fn execute_rename(&self, path: &str, new_name: &str) -> OperationResult {
    let old_name = path.file_name().unwrap().to_string();

    // Execute rename...

    // Log with undo data
    self.safety_manager.log_operation(
        HistoryOperation::Rename { path, new_name },
        Some(UndoData::RenamedFile { old_name, new_name }),
    );
}
```

3. Implement the undo handler:

```rust
fn undo(&self, entry: &HistoryEntry) -> Result<(), SafetyError> {
    match &entry.undo_data {
        Some(UndoData::RenamedFile { old_name, new_name }) => {
            // Rename back to original name
        }
        // ...
    }
}
```

### Customizing Trash Retention

The `SafetyConfig` allows customization:

```rust
let config = SafetyConfig {
    trash_dir: PathBuf::from("~/.local/share/ragfs/trash"),
    history_file: PathBuf::from("~/.local/share/ragfs/history.jsonl"),
    retention_days: 30,        // Keep trash for 30 days
    max_trash_size_mb: 2048,   // 2GB max trash size
};
```

## Extending Semantic Operations

### Adding a New Organization Strategy

1. Add the strategy variant in `crates/ragfs-fuse/src/semantic.rs`:

```rust
pub enum OrganizeStrategy {
    ByTopic,
    ByType,
    ByDate,
    Flatten,
    Custom(String),
    // Add new strategy:
    ByProject,  // Group files by detected project boundaries
}
```

2. Implement the strategy logic:

```rust
impl SemanticManager {
    fn plan_by_project(&self, scope: &str) -> Result<SemanticPlan, SemanticError> {
        // 1. Analyze files in scope using embeddings
        // 2. Detect project boundaries (package.json, Cargo.toml, etc.)
        // 3. Cluster files by project association
        // 4. Generate PlannedActions for the reorganization
        // 5. Return plan with impact summary
    }
}
```

3. Handle in the dispatcher:

```rust
fn generate_plan(&self, request: &OrganizeRequest) -> Result<SemanticPlan, SemanticError> {
    match request.strategy {
        OrganizeStrategy::ByProject => self.plan_by_project(&request.scope),
        // ...
    }
}
```

### Adding a New Analysis Type

To add new analysis (e.g., finding outdated dependencies):

1. Define the analysis result type:

```rust
pub struct DependencyAnalysis {
    pub outdated: Vec<OutdatedDependency>,
    pub unused: Vec<UnusedDependency>,
}
```

2. Add a virtual file for reading the analysis:

```rust
// In inode.rs
pub enum InodeKind {
    // ...
    SemanticDependencies,
}
```

3. Implement the analysis logic:

```rust
impl SemanticManager {
    pub fn analyze_dependencies(&self) -> DependencyAnalysis {
        // Analyze package manifests in scope
        // Use embeddings to find related code
        // Identify unused dependencies
    }
}
```

4. Handle reads in `filesystem.rs` to return JSON analysis results.

## Debugging

### Verbose Logging

```bash
# Enable debug output
cargo run -- -v index /path/to/dir

# Even more verbose with RUST_LOG
RUST_LOG=debug cargo run -- index /path/to/dir
```

### Inspecting the Index

The LanceDB index is stored at:
```
~/.local/share/ragfs/indices/{hash}/index.lance
```

You can inspect it using LanceDB tools or by adding debug endpoints.

### Common Issues

**Model download hangs**

Check network connectivity. The model is downloaded from Hugging Face Hub.

**FUSE mount fails**

Ensure you have FUSE permissions:
```bash
sudo usermod -aG fuse $USER
# Log out and back in
```

**Out of memory during embedding**

Reduce batch size or limit concurrent embeddings in `EmbedderPool`.

## Performance Profiling

### CPU Profiling

Use `cargo flamegraph`:

```bash
cargo install flamegraph
cargo flamegraph --root -- index /path/to/large/dir
```

### Memory Profiling

Use `heaptrack` or similar:

```bash
heaptrack ./target/release/ragfs index /path/to/dir
```

## Code Quality

### Formatting

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --all-targets --all-features
```

### Documentation Coverage

Check for missing docs:

```bash
RUSTDOCFLAGS="-D missing_docs" cargo doc --no-deps
```

## Continuous Integration

The CI pipeline runs:

1. `cargo fmt --check` - Formatting
2. `cargo clippy` - Linting
3. `cargo test` - All tests
4. `cargo doc` - Documentation build

Ensure all checks pass before submitting a PR.

## Release Process

1. Update version in workspace `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.x.y`
4. Push tag: `git push origin v0.x.y`
5. CI will build and publish

## Getting Help

- Open an issue for bugs or questions
- Check existing issues before creating new ones
- Join discussions in PRs for design decisions
