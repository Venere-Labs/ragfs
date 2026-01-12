"""Haystack adapters for RAGFS.

This module provides Haystack-compatible wrappers for RAGFS components.

Example:
    from ragfs.haystack import RagfsTextEmbedder, RagfsDocumentEmbedder

    text_embedder = RagfsTextEmbedder()
    doc_embedder = RagfsDocumentEmbedder()
"""

from .embedders import (
    HaystackRagfsTextEmbedder as RagfsTextEmbedder,
    HaystackRagfsDocumentEmbedder as RagfsDocumentEmbedder,
)
from .retriever import HaystackRagfsRetriever as RagfsRetriever

__all__ = [
    "RagfsTextEmbedder",
    "RagfsDocumentEmbedder",
    "RagfsRetriever",
]
