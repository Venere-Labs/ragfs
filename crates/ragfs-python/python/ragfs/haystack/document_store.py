"""Haystack DocumentStore adapter for RAGFS."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from ragfs import RagfsVectorStore as CoreVectorStore, RagfsEmbeddings as CoreEmbeddings, PyChunk

try:
    from haystack import Document
    from haystack.document_stores.types import DuplicatePolicy
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


class HaystackRagfsDocumentStore:
    """Haystack-compatible document store using RAGFS/LanceDB.

    Provides document storage and retrieval with vector search support.

    Example:
        from ragfs.haystack import RagfsDocumentStore
        from haystack import Document

        store = RagfsDocumentStore("/path/to/db")
        store.write_documents([Document(content="Hello world")])
        docs = store.filter_documents()
    """

    def __init__(
        self,
        db_path: str,
        dimension: int = 384,
        embedding_model_path: Optional[str] = None,
    ):
        """Initialize the document store.

        Args:
            db_path: Path to the LanceDB database.
            dimension: Embedding dimension. Defaults to 384 (GTE-small).
            embedding_model_path: Optional path to store model files.
        """
        self._db_path = db_path
        self._dimension = dimension
        self._store = CoreVectorStore(db_path=db_path, dimension=dimension)
        self._embedder = CoreEmbeddings(model_path=embedding_model_path)
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure store is initialized."""
        if not self._initialized:
            asyncio.get_event_loop().run_until_complete(self._async_init())

    async def _async_init(self) -> None:
        """Initialize store and embedder asynchronously."""
        if not self._initialized:
            await self._store.init()
            await self._embedder.init()
            self._initialized = True

    def count_documents(self) -> int:
        """Return the number of documents in the store.

        Returns:
            Number of document chunks stored.
        """
        self._ensure_initialized()
        stats = asyncio.get_event_loop().run_until_complete(self._store.stats())
        return stats.get("total_chunks", 0)

    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve documents matching the given filters.

        Note: RAGFS stores chunks, so this returns all chunks as documents.
        For filtered retrieval, use the retriever component with search.

        Args:
            filters: Optional filters (currently not fully supported).

        Returns:
            List of Haystack Documents.
        """
        self._ensure_initialized()

        # RAGFS doesn't have direct filtering; return empty for now
        # Users should use retriever for search-based filtering
        if filters:
            # If filters are provided, we'd need to implement filter logic
            # For now, warn users to use retriever
            import warnings
            warnings.warn(
                "filter_documents with filters is not fully supported. "
                "Use RagfsRetriever for search-based retrieval.",
                UserWarning,
            )

        return []

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """Write documents to the store.

        Documents must have embeddings set, or they will be generated automatically.

        Args:
            documents: List of Haystack Documents to write.
            policy: How to handle duplicates (currently ignored).

        Returns:
            Number of documents written.
        """
        self._ensure_initialized()
        return asyncio.get_event_loop().run_until_complete(
            self._async_write_documents(documents, policy)
        )

    async def _async_write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy,
    ) -> int:
        """Write documents asynchronously."""
        if not documents:
            return 0

        # Generate embeddings for documents without them
        docs_needing_embedding = [
            doc for doc in documents if doc.embedding is None
        ]
        if docs_needing_embedding:
            texts = [doc.content or "" for doc in docs_needing_embedding]
            embeddings = await self._embedder.embed_documents(texts)
            for doc, embedding in zip(docs_needing_embedding, embeddings):
                doc.embedding = embedding

        # Convert to PyChunks
        file_id = str(uuid.uuid4())
        chunks = []

        for i, doc in enumerate(documents):
            chunk_id = doc.id or str(uuid.uuid4())
            content = doc.content or ""
            metadata = dict(doc.meta) if doc.meta else {}

            # Get source path from metadata
            source = metadata.get("file_path", f"haystack://documents/{file_id}")

            # Convert metadata to strings
            str_metadata = {k: str(v) for k, v in metadata.items()}

            chunk = PyChunk(
                id=chunk_id,
                file_id=file_id,
                file_path=str(source),
                content=content,
                content_type="text",
                mime_type="text/plain",
                chunk_index=i,
                start_byte=0,
                end_byte=len(content.encode("utf-8")),
                start_line=None,
                end_line=None,
                embedding=list(doc.embedding) if doc.embedding else None,
                metadata=str_metadata,
            )
            chunks.append(chunk)

        # Upsert to store
        await self._store.upsert_chunks(chunks)

        return len(documents)

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by their IDs.

        Note: RAGFS deletes by file path. Document IDs are treated as file paths.

        Args:
            document_ids: List of document IDs (file paths) to delete.
        """
        self._ensure_initialized()
        asyncio.get_event_loop().run_until_complete(
            self._async_delete_documents(document_ids)
        )

    async def _async_delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents asynchronously."""
        for doc_id in document_ids:
            await self._store.delete_by_path(doc_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the document store to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "type": "ragfs.haystack.RagfsDocumentStore",
            "init_parameters": {
                "db_path": self._db_path,
                "dimension": self._dimension,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HaystackRagfsDocumentStore":
        """Create a document store from a dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            New HaystackRagfsDocumentStore instance.
        """
        return cls(**data.get("init_parameters", {}))


# Convenience alias
RagfsDocumentStore = HaystackRagfsDocumentStore
