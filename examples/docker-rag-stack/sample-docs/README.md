# Sample Documents

This directory contains sample documents for the RAGFS Docker demo.

## Getting Started

Add your own documents to this directory to index them with RAGFS. Supported formats include:

- **Markdown** (`.md`)
- **Text** (`.txt`)
- **Code** (`.py`, `.rs`, `.js`, `.ts`, `.go`, `.java`, etc.)
- **PDF** (`.pdf`)
- **And 40+ more formats**

## Example: Project Documentation

Here's a sample document to demonstrate semantic search capabilities.

### Architecture Overview

The RAGFS system consists of several key components:

1. **Content Extraction**: Extracts text from various file formats using specialized extractors.
2. **Chunking**: Splits documents into semantically meaningful chunks using token-based or code-aware strategies.
3. **Embedding**: Generates vector embeddings using the GTE-small model (384 dimensions).
4. **Vector Storage**: Stores embeddings in LanceDB for efficient similarity search.
5. **Query Execution**: Performs hybrid search combining vector similarity and full-text matching.

### API Usage

To query the indexed documents programmatically:

```python
from ragfs.langchain import RagfsRetriever

# Initialize retriever
retriever = RagfsRetriever("/path/to/index", hybrid=True, k=4)
await retriever.ainit()

# Search for relevant documents
docs = await retriever.ainvoke("How does authentication work?")
for doc in docs:
    print(f"Source: {doc.metadata['file_path']}")
    print(doc.page_content[:200])
```

### Configuration

RAGFS can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RAGFS_DB_PATH` | Path to the index database | `~/.local/share/ragfs/indices/default` |
| `RUST_LOG` | Logging level | `info` |

### Troubleshooting

**Q: The indexer is slow on first run.**
A: The embedding model (~100MB) is downloaded on first use. Subsequent runs will be faster.

**Q: Search results are not relevant.**
A: Try using more specific queries or increase the `k` parameter for more results.

**Q: How do I re-index after adding new documents?**
A: Restart the `ragfs-indexer` container: `docker compose restart ragfs-indexer`
