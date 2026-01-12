"""Haystack DocumentWriter adapter for RAGFS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from haystack import component, Document
    from haystack.document_stores.types import DuplicatePolicy
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )

from .document_store import HaystackRagfsDocumentStore


@component
class HaystackRagfsDocumentWriter:
    """Haystack-compatible document writer for RAGFS.

    Writes documents to a RagfsDocumentStore.

    Example:
        from ragfs.haystack import RagfsDocumentStore, RagfsDocumentWriter
        from haystack import Document, Pipeline

        store = RagfsDocumentStore("/path/to/db")
        writer = RagfsDocumentWriter(document_store=store)

        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_component("writer", writer)
        result = pipeline.run({"writer": {"documents": [Document(content="...")]}})
    """

    def __init__(
        self,
        document_store: HaystackRagfsDocumentStore,
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ):
        """Initialize the document writer.

        Args:
            document_store: The RagfsDocumentStore to write to.
            policy: How to handle duplicate documents.
        """
        self._document_store = document_store
        self._policy = policy

    def warm_up(self) -> None:
        """Warm up the writer by initializing the document store."""
        self._document_store._ensure_initialized()

    @component.output_types(documents_written=int)
    def run(
        self,
        documents: List[Document],
        policy: Optional[DuplicatePolicy] = None,
    ) -> Dict[str, int]:
        """Write documents to the store.

        Args:
            documents: List of Haystack Documents to write.
            policy: Optional override for duplicate policy.

        Returns:
            Dictionary with 'documents_written' count.
        """
        effective_policy = policy or self._policy

        count = self._document_store.write_documents(
            documents=documents,
            policy=effective_policy,
        )

        return {"documents_written": count}


# Convenience alias
RagfsDocumentWriter = HaystackRagfsDocumentWriter
