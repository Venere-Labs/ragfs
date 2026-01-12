"""RAGFS - Local semantic search and RAG pipeline.

This package provides Python bindings for RAGFS, a high-performance
semantic search and RAG (Retrieval Augmented Generation) pipeline
written in Rust.

Features:
- Local embeddings using GTE-small (384 dimensions, no API calls)
- Vector storage with LanceDB (hybrid search support)
- Multi-format document loading (40+ text formats, PDF, images)
- Code-aware text splitting with tree-sitter
- Framework adapters for LangChain, LlamaIndex, and Haystack

Quick Start:
    from ragfs import RagfsRetriever

    # Initialize
    retriever = RagfsRetriever("/path/to/db")
    await retriever.init()

    # Search
    docs = await retriever.get_relevant_documents("my query")

Using with LangChain:
    from ragfs.langchain import RagfsEmbeddings, RagfsVectorStore

    embeddings = RagfsEmbeddings()
    await embeddings.init()

    vectorstore = RagfsVectorStore.from_embeddings(embeddings, "/path/to/db")
"""

from ragfs._core import (
    Document,
    SearchResultPy as SearchResult,
    PyChunk,
    RagfsEmbeddings,
    RagfsVectorStore,
    RagfsDocumentLoader,
    RagfsTextSplitter,
    RagfsRetriever,
)

__all__ = [
    # Core types
    "Document",
    "SearchResult",
    "PyChunk",
    # Components
    "RagfsEmbeddings",
    "RagfsVectorStore",
    "RagfsDocumentLoader",
    "RagfsTextSplitter",
    "RagfsRetriever",
]

__version__ = "0.2.0"
