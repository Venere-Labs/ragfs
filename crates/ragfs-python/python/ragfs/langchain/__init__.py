"""LangChain adapters for RAGFS.

This module provides LangChain-compatible wrappers for RAGFS components.

Example:
    from ragfs.langchain import RagfsEmbeddings, RagfsVectorStore

    # Initialize embeddings
    embeddings = RagfsEmbeddings()
    await embeddings.ainit()

    # Use with LangChain
    from langchain_core.vectorstores import InMemoryVectorStore
    store = InMemoryVectorStore(embeddings)
"""

from .embeddings import LangChainRagfsEmbeddings as RagfsEmbeddings
from .vectorstore import LangChainRagfsVectorStore as RagfsVectorStore
from .document_loader import LangChainRagfsLoader as RagfsLoader
from .text_splitter import LangChainRagfsTextSplitter as RagfsTextSplitter
from .retriever import LangChainRagfsRetriever as RagfsRetriever

__all__ = [
    "RagfsEmbeddings",
    "RagfsVectorStore",
    "RagfsLoader",
    "RagfsTextSplitter",
    "RagfsRetriever",
]
