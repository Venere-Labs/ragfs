# RAGFS Python

Python bindings for RAGFS - a high-performance local semantic search and RAG pipeline.

## Features

- **Local Embeddings**: GTE-small model (384 dimensions), no API calls
- **Vector Storage**: LanceDB with hybrid search (vector + full-text)
- **Multi-format Loading**: 40+ text formats, PDF, images
- **Code-aware Chunking**: AST-based splitting with tree-sitter
- **Framework Adapters**: LangChain, LlamaIndex, Haystack

## Installation

```bash
# Base package
pip install ragfs

# With framework support
pip install ragfs[langchain]      # LangChain
pip install ragfs[llamaindex]     # LlamaIndex
pip install ragfs[haystack]       # Haystack
pip install ragfs[all]            # All frameworks
```

## Quick Start

### Standalone Usage

```python
import asyncio
from ragfs import RagfsRetriever

async def main():
    # Initialize retriever
    retriever = RagfsRetriever("/path/to/db")
    await retriever.init()

    # Search
    docs = await retriever.get_relevant_documents("authentication error handling")
    for doc in docs:
        print(f"{doc.metadata['file_path']}: {doc.page_content[:100]}")

asyncio.run(main())
```

### With LangChain

```python
from ragfs.langchain import RagfsEmbeddings, RagfsRetriever

# Embeddings
embeddings = RagfsEmbeddings()
await embeddings.ainit()
vectors = await embeddings.aembed_documents(["Hello", "World"])

# Retriever
retriever = RagfsRetriever("/path/to/db")
await retriever.ainit()
docs = await retriever.aget_relevant_documents("my query")
```

### With LlamaIndex

```python
from ragfs.llamaindex import RagfsEmbeddings
from llama_index.core import VectorStoreIndex, Document

embed_model = RagfsEmbeddings()
index = VectorStoreIndex.from_documents(
    [Document(text="Hello world")],
    embed_model=embed_model,
)
```

### With Haystack

```python
from ragfs.haystack import RagfsTextEmbedder, RagfsRetriever

# Embedder
embedder = RagfsTextEmbedder()
result = embedder.run(text="my query")
embedding = result["embedding"]

# Retriever
retriever = RagfsRetriever("/path/to/db")
result = retriever.run(query="my question")
documents = result["documents"]
```

## Components

| Component | Description |
|-----------|-------------|
| `RagfsEmbeddings` | Local embeddings with GTE-small (384 dim) |
| `RagfsVectorStore` | LanceDB vector storage with hybrid search |
| `RagfsDocumentLoader` | Multi-format document extraction |
| `RagfsTextSplitter` | Token-aware, code-aware chunking |
| `RagfsRetriever` | Combined embeddings + search |

## Development

```bash
# Build
cd crates/ragfs-python
maturin develop

# Build release
maturin build --release
```

## License

MIT OR Apache-2.0
