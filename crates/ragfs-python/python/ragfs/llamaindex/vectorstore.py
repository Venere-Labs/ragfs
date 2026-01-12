"""LlamaIndex VectorStore adapter for RAGFS."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ragfs import RagfsVectorStore as CoreVectorStore, PyChunk

try:
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
        VectorStoreQueryMode,
    )
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.bridge.pydantic import PrivateAttr
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsVectorStore(BasePydanticVectorStore):
    """LlamaIndex-compatible vector store using RAGFS/LanceDB.

    Supports both vector similarity search and hybrid search.

    Example:
        from ragfs.llamaindex import RagfsVectorStore, RagfsEmbeddings
        from llama_index.core import VectorStoreIndex, StorageContext

        # Create vector store
        vector_store = RagfsVectorStore("/path/to/db")
        await vector_store.ainit()

        # Create index with custom storage
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=RagfsEmbeddings(),
        )
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    _store: Any = PrivateAttr()
    _db_path: str = PrivateAttr()
    _dimension: int = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        db_path: str,
        dimension: int = 384,
        **kwargs: Any,
    ):
        """Initialize the vector store.

        Args:
            db_path: Path to the LanceDB database.
            dimension: Embedding dimension. Defaults to 384 (GTE-small).
        """
        super().__init__(**kwargs)
        self._db_path = db_path
        self._dimension = dimension
        self._store = CoreVectorStore(db_path=db_path, dimension=dimension)
        self._initialized = False

    async def ainit(self) -> None:
        """Initialize the store (creates tables if needed)."""
        if not self._initialized:
            await self._store.init()
            self._initialized = True

    def _ensure_init_sync(self) -> None:
        """Ensure store is initialized (sync version)."""
        if not self._initialized:
            import asyncio
            asyncio.get_event_loop().run_until_complete(self.ainit())

    @property
    def client(self) -> Any:
        """Return the underlying store."""
        return self._store

    def add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector store synchronously.

        Args:
            nodes: List of nodes to add.

        Returns:
            List of node IDs.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, **kwargs)
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector store asynchronously.

        Args:
            nodes: List of nodes to add.

        Returns:
            List of node IDs.
        """
        await self.ainit()

        if not nodes:
            return []

        # Group nodes by a virtual file_id (one per batch)
        file_id = str(uuid.uuid4())
        chunks = []

        for i, node in enumerate(nodes):
            node_id = node.node_id or str(uuid.uuid4())

            # Get embedding from node
            embedding = node.embedding
            if embedding is None:
                raise ValueError(
                    f"Node {node_id} has no embedding. "
                    "Ensure embeddings are generated before adding to store."
                )

            # Get text content
            content = node.get_content()

            # Get metadata
            metadata = dict(node.metadata) if node.metadata else {}

            # Get source path from metadata or use virtual path
            source = metadata.get("file_path", f"llamaindex://nodes/{file_id}")

            # Convert metadata values to strings
            str_metadata = {k: str(v) for k, v in metadata.items()}

            chunk = PyChunk(
                id=node_id,
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
                embedding=list(embedding),
                metadata=str_metadata,
            )
            chunks.append(chunk)

        # Upsert to store
        await self._store.upsert_chunks(chunks)

        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete nodes by reference document ID synchronously.

        Args:
            ref_doc_id: Reference document ID (file path).
        """
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **kwargs)
        )

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete nodes by reference document ID asynchronously.

        Args:
            ref_doc_id: Reference document ID (file path).
        """
        await self.ainit()
        await self._store.delete_by_path(ref_doc_id)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store synchronously.

        Args:
            query: VectorStoreQuery with embedding and parameters.

        Returns:
            VectorStoreQueryResult with nodes and scores.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.aquery(query, **kwargs)
        )

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store asynchronously.

        Args:
            query: VectorStoreQuery with embedding and parameters.

        Returns:
            VectorStoreQueryResult with nodes and scores.
        """
        await self.ainit()

        if query.query_embedding is None:
            raise ValueError("Query must have an embedding.")

        k = query.similarity_top_k or 4

        # Determine search mode
        use_hybrid = query.mode in (
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.TEXT_SEARCH,
        )

        if use_hybrid and query.query_str:
            # Hybrid search
            results = await self._store.hybrid_search(
                query.query_str,
                query.query_embedding,
                k=k,
            )
        else:
            # Vector-only search
            results = await self._store.similarity_search(
                query.query_embedding,
                k=k,
            )

        # Convert to LlamaIndex nodes
        nodes = []
        similarities = []
        ids = []

        for result in results:
            node = TextNode(
                text=result.document.page_content,
                metadata=result.document.metadata,
                id_=result.chunk_id,
            )
            nodes.append(node)
            similarities.append(result.score)
            ids.append(result.chunk_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path
