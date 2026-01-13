# RAGFS Python Bindings

This guide covers the Python bindings for RAGFS, enabling semantic search and RAG pipelines in Python applications.

## Installation

```bash
# From PyPI (when published)
pip install ragfs

# From source
cd crates/ragfs-python
pip install maturin
maturin develop --release
```

## Quick Start

```python
import asyncio
from ragfs import RagfsRetriever

async def main():
    # Initialize retriever with a database path
    retriever = RagfsRetriever("/path/to/index")
    await retriever.init()  # Downloads model on first use

    # Search for relevant documents
    docs = await retriever.get_relevant_documents("authentication implementation")

    for doc in docs:
        print(f"File: {doc.metadata['file_path']}")
        print(f"Content: {doc.page_content[:200]}...")

asyncio.run(main())
```

## Complete RAG Pipeline Example

This example shows the complete workflow: load files, chunk, embed, store, and query.

```python
import asyncio
from ragfs import (
    RagfsDocumentLoader,
    RagfsTextSplitter,
    RagfsEmbeddings,
    RagfsVectorStore,
    RagfsRetriever,
)

async def build_rag_index(source_dir: str, db_path: str):
    """Build a complete RAG index from a source directory."""

    # 1. Load documents from various formats
    print("Loading documents...")
    loader = RagfsDocumentLoader()
    await loader.init()
    docs = await loader.load_directory(source_dir, glob="**/*.{py,rs,md,txt}")
    print(f"  Loaded {len(docs)} documents")

    # 2. Split into chunks (code-aware for source files)
    print("Splitting into chunks...")
    splitter = RagfsTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        code_aware=True,  # Uses tree-sitter for code files
    )
    await splitter.init()
    chunks = await splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    # 3. Initialize embeddings (downloads model on first run)
    print("Initializing embeddings...")
    embeddings = RagfsEmbeddings()
    await embeddings.init()
    print(f"  Using {embeddings.model_name} ({embeddings.dimension} dimensions)")

    # 4. Store in vector database
    print("Storing vectors...")
    store = RagfsVectorStore(db_path=db_path)
    await store.init()
    await store.add_documents(chunks)
    print(f"  Stored {len(chunks)} vectors")

    return store

async def query_index(db_path: str, query: str, k: int = 5):
    """Query an existing index."""

    retriever = RagfsRetriever(db_path=db_path, hybrid=True, k=k)
    await retriever.init()

    results = await retriever.search(query)

    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.document.metadata.get('file_path', 'unknown')}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Preview: {result.document.page_content[:150]}...")

async def main():
    # Build the index
    store = await build_rag_index(
        source_dir="./my_project",
        db_path="./ragfs_index"
    )

    # Query the index
    await query_index(
        db_path="./ragfs_index",
        query="how does authentication work",
        k=5
    )

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
Loading documents...
  Loaded 47 documents
Splitting into chunks...
  Created 312 chunks
Initializing embeddings...
  Using gte-small (384 dimensions)
Storing vectors...
  Stored 312 vectors

Query: how does authentication work
--------------------------------------------------

1. src/auth/jwt.rs
   Score: 0.8542
   Preview: pub async fn validate_token(token: &str) -> Result<Claims> {
       let key = get_signing_key()?;
       let claims = decode::<Claims>(token, &key, &Validation::default())...

2. src/middleware/auth.rs
   Score: 0.8123
   Preview: impl<S> AuthMiddleware<S> {
       pub fn new(inner: S) -> Self {
           Self { inner, jwt_validator: JwtValidator::new() }
       }...
```

## Core Components

### RagfsEmbeddings

Generate embeddings using the local GTE-small model (384 dimensions).

```python
from ragfs import RagfsEmbeddings

embeddings = RagfsEmbeddings(model_path="/path/to/models")  # Optional
await embeddings.init()

# Embed text
vectors = await embeddings.embed_documents(["text 1", "text 2"])
query_vector = await embeddings.embed_query("search query")

print(f"Dimension: {embeddings.dimension}")  # 384
```

### RagfsVectorStore

Store and search vectors using LanceDB.

```python
from ragfs import RagfsVectorStore

store = RagfsVectorStore(db_path="/path/to/db")
await store.init()

# Add documents
await store.add_texts(["document 1", "document 2"], metadatas=[{"source": "a"}, {"source": "b"}])

# Search
results = await store.similarity_search("query", k=5)
results_with_scores = await store.similarity_search_with_score("query", k=5)
```

### RagfsDocumentLoader

Load documents from 40+ file formats.

```python
from ragfs import RagfsDocumentLoader

loader = RagfsDocumentLoader()
await loader.init()

# Load single file
docs = await loader.load("/path/to/file.pdf")

# Load directory
docs = await loader.load_directory("/path/to/dir", glob="**/*.py")
```

### RagfsTextSplitter

Split documents with code-aware chunking.

```python
from ragfs import RagfsTextSplitter

splitter = RagfsTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    code_aware=True,  # Uses tree-sitter for code files
)
await splitter.init()

chunks = await splitter.split_documents(docs)
```

### RagfsRetriever

Combined embeddings + vector search retriever.

```python
from ragfs import RagfsRetriever

retriever = RagfsRetriever(
    db_path="/path/to/db",
    model_path="/path/to/models",  # Optional
    hybrid=True,  # Enable hybrid search (vector + full-text)
    k=10,
)
await retriever.init()

# Search
docs = await retriever.get_relevant_documents("query")

# Search with scores
results = await retriever.search("query", hybrid=True, k=5)
for result in results:
    print(f"Score: {result.score}, Content: {result.document.page_content[:100]}")
```

---

## Framework Integrations

### LangChain

```python
from ragfs.langchain import RagfsEmbeddings, RagfsVectorStore, RagfsRetriever

# Embeddings
embeddings = RagfsEmbeddings()
await embeddings.ainit()

# Vector store
vectorstore = RagfsVectorStore("/path/to/db", embeddings=embeddings)
await vectorstore.ainit()

# Use with LangChain chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

retriever = RagfsRetriever("/path/to/db")
await retriever.ainit()

# RAG chain
prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)
chain = retriever | prompt | ChatOpenAI()
```

### LlamaIndex

```python
from ragfs.llamaindex import (
    RagfsEmbeddings,
    RagfsVectorStore,
    RagfsReader,
    create_ragfs_index,
)

# Quick setup with factory function
index = await create_ragfs_index(
    db_path="/path/to/db",
    source_dir="/path/to/documents",
)

# Query
query_engine = index.as_query_engine()
response = await query_engine.aquery("What is RAGFS?")

# Manual setup
embeddings = RagfsEmbeddings()
await embeddings.init()

vectorstore = RagfsVectorStore(db_path="/path/to/db")
await vectorstore.init()

# Load documents
reader = RagfsReader()
documents = await reader.aload_data("/path/to/dir")
```

### Haystack

```python
from ragfs.haystack import (
    RagfsTextEmbedder,
    RagfsDocumentEmbedder,
    RagfsDocumentStore,
    RagfsRetriever,
    RagfsDocumentConverter,
    RagfsDocumentSplitter,
)
from haystack import Pipeline

# Create pipeline
pipeline = Pipeline()

# Add components
pipeline.add_component("converter", RagfsDocumentConverter())
pipeline.add_component("splitter", RagfsDocumentSplitter(chunk_size=512))
pipeline.add_component("embedder", RagfsDocumentEmbedder())
pipeline.add_component("writer", RagfsDocumentWriter(document_store=RagfsDocumentStore("/path/to/db")))

# Connect
pipeline.connect("converter", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")

# Run
pipeline.run({"converter": {"sources": ["/path/to/file.pdf"]}})

# Query pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", RagfsTextEmbedder())
query_pipeline.add_component("retriever", RagfsRetriever(document_store=RagfsDocumentStore("/path/to/db")))
query_pipeline.connect("embedder", "retriever")

results = query_pipeline.run({"embedder": {"text": "search query"}})
```

---

## FUSE Capabilities for AI Agents

RAGFS exposes advanced filesystem capabilities designed for AI agent workflows. These features implement the **Propose-Review-Apply** pattern for safe, reversible operations.

### Safety Layer

The Safety Layer provides soft delete, audit history, and undo capabilities.

```python
from ragfs import RagfsSafetyManager

# Initialize safety manager
safety = RagfsSafetyManager("/path/to/source", "/path/to/db")

# Soft delete (moves to trash, returns undo_id)
trash_entry = await safety.delete_to_trash("/path/to/file.txt")
print(f"Undo ID: {trash_entry.id}")

# List trash contents
entries = await safety.list_trash()
for entry in entries:
    print(f"{entry.original_path} - deleted at {entry.deleted_at}")

# Restore from trash
await safety.restore_from_trash(trash_entry.id)

# Get operation history (audit log)
history = safety.get_history(limit=50)
for entry in history:
    print(f"{entry.timestamp}: {entry.operation.operation_type} on {entry.operation.path}")

# Undo any reversible operation
if safety.can_undo(operation_id):
    await safety.undo(operation_id)
```

### Semantic Operations

AI-powered file analysis and organization.

```python
from ragfs import RagfsSemanticManager, OrganizeRequest, OrganizeStrategy

# Initialize semantic manager
semantic = RagfsSemanticManager(
    source_path="/path/to/source",
    db_path="/path/to/db",
    duplicate_threshold=0.95,  # For duplicate detection
)
await semantic.init()

# Find similar files
similar = await semantic.find_similar("/path/to/file.txt", k=5)
for file in similar.similar:
    print(f"{file.path}: {file.similarity:.2%}")

# Find duplicates
duplicates = await semantic.find_duplicates()
for group in duplicates.groups:
    print(f"Duplicate group ({group.similarity:.2%}): {group.files}")

# Analyze for cleanup
analysis = await semantic.analyze_cleanup()
print(f"Potential savings: {analysis.potential_savings / 1024 / 1024:.1f} MB")
```

### Propose-Review-Apply Pattern

The key innovation for safe AI agent operations. Actions are proposed, reviewed, then approved/rejected.

```python
from ragfs import RagfsSemanticManager, OrganizeRequest, OrganizeStrategy

semantic = RagfsSemanticManager("/path/to/source", "/path/to/db")
await semantic.init()

# 1. PROPOSE: Create an organization plan (NOT executed yet!)
request = OrganizeRequest(
    scope="./docs",
    strategy=OrganizeStrategy.by_topic(),
    max_groups=5,
    similarity_threshold=0.7,
)
plan = await semantic.create_organize_plan(request)

print(f"Plan {plan.id} proposes {len(plan.actions)} actions:")
for action in plan.actions:
    print(f"  {action.action.action_type}: {action.action.source} -> {action.action.target}")
    print(f"    Reason: {action.reason} (confidence: {action.confidence:.1%})")

# 2. REVIEW: List all pending plans
pending = await semantic.list_pending_plans()
for p in pending:
    print(f"Plan {p.id}: {p.description}")

# 3. APPLY or REJECT
if user_approves:
    result = await semantic.approve_plan(plan.id)  # Execute!
    print(f"Executed: {result.status}")
else:
    result = await semantic.reject_plan(plan.id)  # Discard
    print(f"Rejected: {result.status}")
```

### Structured File Operations

File operations with JSON feedback and undo support.

```python
from ragfs import RagfsOpsManager, Operation

ops = RagfsOpsManager("/path/to/source", "/path/to/db")

# Single operations with feedback
result = await ops.create_file("/path/new_file.txt", "content")
print(f"Success: {result.success}, Undo ID: {result.undo_id}")

result = await ops.move_file("/path/old", "/path/new")
result = await ops.copy_file("/path/src", "/path/dst")
result = await ops.mkdir("/path/new_dir")

# Batch operations (atomic - all succeed or all fail)
operations = [
    Operation.create("/file1.txt", "content1"),
    Operation.create("/file2.txt", "content2"),
    Operation.mkdir("/new_dir"),
]

# Dry run to validate first
validation = await ops.dry_run(operations)
if validation.is_valid:
    batch_result = await ops.batch(operations, atomic=True)
    print(f"Batch ID: {batch_result.rollback_id}")  # For batch undo
```

### LlamaIndex with Safety Layer

```python
from ragfs.llamaindex import RagfsSafeVectorStore, RagfsOrganizer

# Safe vector store with undo support
store = RagfsSafeVectorStore("/path/to/db", safety_enabled=True)
await store.ainit()

# Safe delete with undo
undo_info = await store.safe_delete("doc_id")
print(f"Undo ID: {undo_info['undo_id']}")

# Restore
await store.undo_delete(undo_info["undo_id"])

# Semantic organizer
organizer = RagfsOrganizer("/path/to/source", "/path/to/db")
await organizer.init()

# Propose organization
plan = await organizer.propose_organization("./docs", strategy="by_topic")

# Approve or reject
await organizer.approve(plan.id)  # or organizer.reject(plan.id)
```

### Haystack with Safety Layer

```python
from ragfs.haystack import RagfsSafeDocumentStore, RagfsOrganizer

# Safe document store
store = RagfsSafeDocumentStore("/path/to/db", safety_enabled=True)

# Safe delete with undo
result = store.safe_delete_documents(["doc1", "doc2"])
print(f"Undo IDs: {result['undo_ids']}")

# Restore
store.restore_documents(list(result["undo_ids"].values()))

# Get history
history = store.get_history(limit=10)

# Semantic organizer as Haystack component
organizer = RagfsOrganizer("/path/to/source", "/path/to/db")

# Use in pipeline
result = organizer.run(
    operation="propose_organization",
    scope="./docs",
    strategy="by_topic",
)
plan = result["plan"]

# Approve
organizer.run(operation="approve", plan_id=plan.id)
```

---

## Configuration

### Environment Variables

```bash
# Custom model storage path
export RAGFS_MODEL_PATH=~/.local/share/ragfs/models

# Custom database path
export RAGFS_DB_PATH=~/.local/share/ragfs/indices/default
```

### Model Information

- **Model**: `thenlper/gte-small`
- **Dimensions**: 384
- **Max Tokens**: 512
- **Size**: ~100MB (downloaded on first use)

---

## API Reference

### Document

```python
@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]
```

### SearchResult

```python
@dataclass
class SearchResult:
    document: Document
    score: float  # Cosine similarity (0-1)
```

### PyChunk

```python
@dataclass
class PyChunk:
    id: str
    content: str
    file_path: str
    chunk_index: int
    metadata: dict[str, Any]
```

---

## Troubleshooting

### Model Download Issues

If model download fails:

```python
# Manually download model
from ragfs import RagfsEmbeddings

embeddings = RagfsEmbeddings(model_path="/custom/path")
await embeddings.init()  # Will download to custom path
```

### LanceDB Connection Errors

Ensure the database path is writable and not in use by another process:

```python
store = RagfsVectorStore(db_path="/path/to/writable/dir")
```

### Memory Usage

For large document sets, process in batches:

```python
splitter = RagfsTextSplitter(chunk_size=512)
for batch in chunked(documents, 100):
    chunks = await splitter.split_documents(batch)
    await store.add_documents(chunks)
```
