"""LlamaIndex adapters for RAGFS.

This module provides LlamaIndex-compatible wrappers for RAGFS components.

Example:
    from ragfs.llamaindex import RagfsEmbeddings, RagfsRetriever

    embed_model = RagfsEmbeddings()
    retriever = RagfsRetriever("/path/to/db")
"""

from .embeddings import LlamaIndexRagfsEmbeddings as RagfsEmbeddings
from .retriever import LlamaIndexRagfsRetriever as RagfsRetriever

__all__ = [
    "RagfsEmbeddings",
    "RagfsRetriever",
]
