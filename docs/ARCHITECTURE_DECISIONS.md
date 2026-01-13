# Architecture Decision Records

This document captures key architectural decisions made in RAGFS and the rationale behind them.

## Table of Contents

- [ADR-001: Local-First Embeddings with Candle](#adr-001-local-first-embeddings-with-candle)
- [ADR-002: LanceDB for Vector Storage](#adr-002-lancedb-for-vector-storage)
- [ADR-003: Propose-Review-Apply Pattern](#adr-003-propose-review-apply-pattern)
- [ADR-004: FUSE-Based Interface](#adr-004-fuse-based-interface)
- [ADR-005: Registry Pattern for Extractors and Chunkers](#adr-005-registry-pattern-for-extractors-and-chunkers)
- [ADR-006: Content-Addressed Storage with Blake3](#adr-006-content-addressed-storage-with-blake3)
- [ADR-007: Async-First Design with Tokio](#adr-007-async-first-design-with-tokio)

---

## ADR-001: Local-First Embeddings with Candle

### Status
Accepted

### Context
RAGFS needs to generate vector embeddings for semantic search. Options considered:
1. **External API** (OpenAI, Cohere, etc.)
2. **Python ML framework** (sentence-transformers, etc.)
3. **Rust ML framework** (Candle, Burn, ONNX Runtime)

### Decision
Use **Candle** with the **gte-small** model for local embedding generation.

### Rationale

**Privacy:**
- All data stays on the user's machine
- No network requests after initial model download
- Suitable for sensitive codebases and documents

**Offline capability:**
- Works without internet after first run
- No dependency on external service availability
- Consistent performance regardless of network

**Cost:**
- No API fees or rate limits
- Predictable resource usage
- No per-query costs

**Candle specifically:**
- Pure Rust implementation
- Good performance on CPU
- Optional GPU acceleration
- Active development by Hugging Face
- Smaller binary than Python alternatives

### Trade-offs

**Pros:**
- Complete privacy
- No ongoing costs
- Works offline
- Fast local inference

**Cons:**
- Larger binary size (~100MB model)
- CPU-bound without GPU
- Model quality limited to local options
- Initial download required

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| OpenAI API | Privacy concerns, ongoing costs, network dependency |
| sentence-transformers | Python dependency, larger runtime overhead |
| ONNX Runtime | More complex setup, less Rust-native |
| Burn | Less mature ecosystem at decision time |

---

## ADR-002: LanceDB for Vector Storage

### Status
Accepted

### Context
RAGFS needs efficient storage and retrieval of vector embeddings. Options considered:
1. **In-memory store** (simple Vec<f32>)
2. **SQLite with vector extension** (sqlite-vss)
3. **Dedicated vector database** (Qdrant, Milvus, Weaviate)
4. **Embedded vector database** (LanceDB, Chroma)

### Decision
Use **LanceDB** as the vector storage backend.

### Rationale

**Embedded:**
- No separate server process required
- Single binary deployment
- Simpler operational model

**Performance:**
- Fast ANN (Approximate Nearest Neighbor) search
- Built-in HNSW indexing
- Efficient for medium-scale datasets (1M+ vectors)

**Features:**
- Hybrid search (vector + full-text) built-in
- Columnar storage (Lance format)
- Good Rust bindings

**Reliability:**
- ACID transactions
- Crash recovery
- Consistent state

### Trade-offs

**Pros:**
- No external dependencies
- Fast similarity search
- Built-in hybrid search
- Good Rust integration

**Cons:**
- Larger dependency tree
- Less flexible than raw SQL
- Learning curve for Lance format

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| In-memory | No persistence, memory limits |
| SQLite + vss | Less efficient for large vector sets |
| Qdrant | Requires separate server |
| Chroma | Python-first, less mature Rust bindings |

---

## ADR-003: Propose-Review-Apply Pattern

### Status
Accepted

### Context
RAGFS enables AI agents to perform file operations. We need a safety mechanism to prevent unintended changes.

### Decision
Implement a **Propose-Review-Apply** pattern for destructive semantic operations:
1. Agent proposes a plan (e.g., file organization)
2. User reviews the proposed actions
3. User explicitly approves or rejects
4. Only approved plans are executed

### Rationale

**Safety:**
- No automatic execution of destructive operations
- User retains final control
- Clear audit trail of proposed vs executed actions

**Reversibility:**
- All executed operations are reversible
- Undo support via operation IDs
- Trash-based soft delete

**Transparency:**
- Actions show reasoning and confidence scores
- Impact summary before execution
- Full visibility into what will change

### Implementation

```
Agent → create_plan() → Plan(pending)
                              ↓
User reviews .pending/<plan_id>
                              ↓
         approve() ──────────→ Execute with undo support
              │
              └── reject() ──→ Discard, no changes
```

### Trade-offs

**Pros:**
- Safe for autonomous agents
- User control preserved
- Auditable decisions

**Cons:**
- Extra step for simple operations
- Plans can expire or become stale
- Requires user interaction

---

## ADR-004: FUSE-Based Interface

### Status
Accepted

### Context
RAGFS needs an interface for users and agents to interact with semantic search and file operations.

Options:
1. CLI only
2. REST API server
3. FUSE filesystem
4. Library-only (SDK)

### Decision
Use **FUSE** (Filesystem in Userspace) as the primary advanced interface, with CLI for simple operations.

### Rationale

**Universal compatibility:**
- Works with any tool that reads/writes files
- No special SDK required
- Shell scripts, editors, agents all work

**Intuitive operations:**
- `cat .query/authentication` for search
- `echo "path" > .ops/.delete` for operations
- Standard Unix file operations

**Agent-friendly:**
- Agents can use standard file I/O
- JSON responses in files
- No HTTP client needed

### Implementation

```
/mountpoint/
├── real_files/           # Passthrough
└── .ragfs/               # Virtual control interface
    ├── .query/<text>     # Read: search results
    ├── .ops/             # Write: operations
    └── .result           # Read: operation results
```

### Trade-offs

**Pros:**
- Universal interface
- No new protocols
- Works with existing tools

**Cons:**
- Linux only (FUSE)
- Slight performance overhead
- Debugging more complex

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| REST API | Requires HTTP client, port management |
| CLI only | Not suitable for agent integration |
| Library only | Limits language accessibility |

---

## ADR-005: Registry Pattern for Extractors and Chunkers

### Status
Accepted

### Context
RAGFS needs to support multiple file formats and chunking strategies. The system should be extensible.

### Decision
Use a **Registry Pattern** for both extractors and chunkers:
- `ExtractorRegistry` maps MIME types to extractors
- `ChunkerRegistry` maps content types to chunkers

### Rationale

**Extensibility:**
- New formats added without modifying core code
- Custom extractors can be registered
- Plugin-like architecture

**Separation of concerns:**
- Each extractor handles one format
- Chunkers are independent of extractors
- Clear responsibilities

**Testability:**
- Individual components testable in isolation
- Mock implementations for testing
- Easy to swap implementations

### Implementation

```rust
// Register extractors
registry.register("text", TextExtractor::new());
registry.register("pdf", PdfExtractor::new());

// Extract based on MIME type
let content = registry.extract(path, "application/pdf")?;
```

### Trade-offs

**Pros:**
- Easy to extend
- Clean architecture
- Good testability

**Cons:**
- Runtime dispatch overhead
- Registration required
- More complex than direct calls

---

## ADR-006: Content-Addressed Storage with Blake3

### Status
Accepted

### Context
RAGFS needs to track file changes to avoid redundant processing and maintain consistent state.

### Decision
Use **Blake3** hashing for content-addressed identification of files.

### Rationale

**Efficiency:**
- Only re-process files that actually changed
- Hash comparison faster than content comparison
- Incremental indexing support

**Blake3 specifically:**
- Extremely fast (~3 GB/s on modern CPUs)
- Cryptographically secure
- Parallel-friendly
- Pure Rust implementation

**Consistency:**
- Deterministic file identification
- Reproducible across restarts
- Stable references for undo/trash

### Implementation

```rust
let hash = blake3::hash(content);
let file_id = format!("{:x}", hash)[..16]; // First 16 chars
```

### Trade-offs

**Pros:**
- Very fast hashing
- Reliable change detection
- Cryptographic security

**Cons:**
- Full file read required for hash
- Storage for hash values
- Hash collisions (extremely rare)

---

## ADR-007: Async-First Design with Tokio

### Status
Accepted

### Context
RAGFS performs many I/O operations: file reading, database queries, network (for model download), filesystem events.

### Decision
Use **Tokio** async runtime with async-first API design.

### Rationale

**Concurrency:**
- Parallel file processing
- Non-blocking I/O
- Efficient resource utilization

**Tokio specifically:**
- Mature, well-tested runtime
- Excellent ecosystem support
- Good performance characteristics
- Multi-threaded executor

**API design:**
- All public APIs are async
- Consistent async/await usage
- Natural integration with async ecosystem

### Implementation

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>>;
}
```

### Trade-offs

**Pros:**
- Efficient I/O handling
- Good concurrency model
- Modern Rust idioms

**Cons:**
- Async complexity
- Runtime overhead
- Learning curve

---

## Summary

| Decision | Key Driver |
|----------|-----------|
| Candle embeddings | Privacy, offline capability |
| LanceDB | Embedded, hybrid search |
| Propose-Review-Apply | Agent safety |
| FUSE interface | Universal compatibility |
| Registry pattern | Extensibility |
| Blake3 hashing | Performance, reliability |
| Tokio async | Concurrency, I/O efficiency |

These decisions collectively create a system that is:
- **Private**: All processing local
- **Safe**: Reversible operations with approval workflow
- **Fast**: Efficient async I/O and vector search
- **Extensible**: Plugin-like architecture for formats
- **Universal**: Filesystem interface works with any tool
