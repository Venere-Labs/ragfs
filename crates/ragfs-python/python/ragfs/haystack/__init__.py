"""Haystack adapters for RAGFS.

This module provides Haystack-compatible wrappers for RAGFS components.

Example:
    from haystack import Pipeline
    from ragfs.haystack import (
        RagfsDocumentConverter,
        RagfsDocumentSplitter,
        RagfsDocumentEmbedder,
        RagfsDocumentStore,
        RagfsDocumentWriter,
        RagfsRetriever,
    )

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("converter", RagfsDocumentConverter())
    pipeline.add_component("splitter", RagfsDocumentSplitter())
    pipeline.add_component("embedder", RagfsDocumentEmbedder())
    pipeline.add_component("writer", RagfsDocumentWriter(document_store))

    # Connect components
    pipeline.connect("converter", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # Run pipeline
    result = pipeline.run({"converter": {"sources": ["./docs"]}})

Safety Layer (soft delete, undo, history):
    from ragfs.haystack import RagfsSafeDocumentStore

    store = RagfsSafeDocumentStore("/path/to/db", safety_enabled=True)

    # Soft delete with undo
    result = store.safe_delete_documents(["doc_id"])
    store.restore_documents(list(result["undo_ids"].values()))

AI-Powered Organization (Propose-Review-Apply pattern):
    from ragfs.haystack import RagfsOrganizer

    organizer = RagfsOrganizer("/path/to/source", "/path/to/db")

    # Use in pipeline
    pipeline.add_component("organizer", organizer)
    result = pipeline.run({
        "organizer": {"operation": "propose_organization", "scope": "./docs"}
    })
"""

from .embedders import (
    HaystackRagfsTextEmbedder as RagfsTextEmbedder,
    HaystackRagfsDocumentEmbedder as RagfsDocumentEmbedder,
)
from .retriever import HaystackRagfsRetriever as RagfsRetriever
from .document_store import HaystackRagfsDocumentStore as RagfsDocumentStore
from .document_converter import HaystackRagfsDocumentConverter as RagfsDocumentConverter
from .document_splitter import HaystackRagfsDocumentSplitter as RagfsDocumentSplitter
from .document_writer import HaystackRagfsDocumentWriter as RagfsDocumentWriter
from .safe_document_store import HaystackRagfsSafeDocumentStore as RagfsSafeDocumentStore
from .organizer import HaystackRagfsOrganizer as RagfsOrganizer

__all__ = [
    # Embedders
    "RagfsTextEmbedder",
    "RagfsDocumentEmbedder",
    # Retriever
    "RagfsRetriever",
    # Document Store
    "RagfsDocumentStore",
    # Safe Document Store (with safety layer)
    "RagfsSafeDocumentStore",
    # Document Converter
    "RagfsDocumentConverter",
    # Document Splitter
    "RagfsDocumentSplitter",
    # Document Writer
    "RagfsDocumentWriter",
    # Semantic Organizer (Propose-Review-Apply)
    "RagfsOrganizer",
]
