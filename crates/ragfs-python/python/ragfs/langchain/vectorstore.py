"""LangChain VectorStore adapter for RAGFS."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Type

from ragfs import RagfsVectorStore as CoreVectorStore, PyChunk

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install with: pip install ragfs[langchain]"
    )


class LangChainRagfsVectorStore(VectorStore):
    """LangChain-compatible vector store using RAGFS/LanceDB.

    Supports both vector similarity search and hybrid search.

    Example:
        from ragfs.langchain import RagfsEmbeddings, RagfsVectorStore

        embeddings = RagfsEmbeddings()
        await embeddings.ainit()

        store = RagfsVectorStore(embeddings, "/path/to/db")
        await store.ainit()

        # Add documents
        ids = await store.aadd_texts(["Hello world", "Goodbye world"])

        # Search
        results = await store.asimilarity_search("hello", k=4)
    """

    def __init__(
        self,
        embedding: Embeddings,
        db_path: str,
        dimension: int = 384,
    ):
        """Initialize the vector store.

        Args:
            embedding: Embeddings instance for generating vectors.
            db_path: Path to the LanceDB database.
            dimension: Embedding dimension.
        """
        self._embedding = embedding
        self._db_path = db_path
        self._dimension = dimension
        self._store = CoreVectorStore(db_path=db_path, dimension=dimension)
        self._initialized = False

    async def ainit(self) -> None:
        """Initialize the store (creates tables if needed)."""
        if not self._initialized:
            await self._store.init()
            self._initialized = True

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings instance."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.aadd_texts(texts, metadatas=metadatas, **kwargs)
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text. Generated if not provided.

        Returns:
            List of IDs for the added texts.
        """
        await self.ainit()

        texts_list = list(texts)
        if not texts_list:
            return []

        # Generate embeddings
        embeddings = await self._embedding.aembed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # Create PyChunk objects
        file_id = str(uuid.uuid4())  # Group all texts under one "virtual" file
        chunks = []

        for i, (text, emb, meta, chunk_id) in enumerate(
            zip(texts_list, embeddings, metadatas, ids)
        ):
            # Get source path from metadata or use virtual path
            source = meta.get("source", f"langchain://texts/{file_id}")

            # Convert metadata to string values (required by ragfs)
            str_metadata = {k: str(v) for k, v in meta.items()}

            chunk = PyChunk(
                id=chunk_id,
                file_id=file_id,
                file_path=str(source),
                content=text,
                content_type="text",
                mime_type="text/plain",
                chunk_index=i,
                start_byte=0,
                end_byte=len(text.encode("utf-8")),
                start_line=None,
                end_line=None,
                embedding=emb,
                metadata=str_metadata,
            )
            chunks.append(chunk)

        # Upsert to store
        await self._store.upsert_chunks(chunks)

        return ids

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add LangChain documents to the vector store.

        Args:
            documents: Documents to add.
            ids: Optional IDs for each document.

        Returns:
            List of IDs for the added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.aadd_documents(documents, ids=ids, **kwargs)
        )

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents by their file paths.

        Note: RAGFS deletes by file path, not by chunk ID.
        Use the 'file_path' key in kwargs or pass file paths as ids.

        Args:
            ids: File paths to delete (not chunk IDs).
            **kwargs: Additional arguments. Use 'file_paths' for explicit paths.

        Returns:
            True if deletion was successful.
        """
        await self.ainit()

        file_paths = kwargs.get("file_paths", ids)
        if file_paths is None:
            return None

        for path in file_paths:
            await self._store.delete_by_path(str(path))

        return True

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.adelete(ids=ids, **kwargs)
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.asimilarity_search(query, k, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents."""
        await self.ainit()

        # Embed query
        query_embedding = await self._embedding.aembed_query(query)

        # Search
        results = await self._store.similarity_search(query_embedding, k=k)

        # Convert to LangChain documents
        return [
            Document(
                page_content=r.document.page_content,
                metadata=r.document.metadata,
            )
            for r in results
        ]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
        await self.ainit()

        query_embedding = await self._embedding.aembed_query(query)
        results = await self._store.similarity_search(query_embedding, k=k)

        return [
            (
                Document(
                    page_content=r.document.page_content,
                    metadata=r.document.metadata,
                ),
                r.score,
            )
            for r in results
        ]

    async def ahybrid_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Hybrid search (vector + full-text)."""
        await self.ainit()

        query_embedding = await self._embedding.aembed_query(query)
        results = await self._store.hybrid_search(query, query_embedding, k=k)

        return [
            Document(
                page_content=r.document.page_content,
                metadata=r.document.metadata,
            )
            for r in results
        ]

    @classmethod
    def from_texts(
        cls: Type["LangChainRagfsVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "LangChainRagfsVectorStore":
        """Create from texts synchronously.

        Note: Use afrom_texts for async version.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)
        )

    @classmethod
    async def afrom_texts(
        cls: Type["LangChainRagfsVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        db_path: Optional[str] = None,
        dimension: int = 384,
        **kwargs: Any,
    ) -> "LangChainRagfsVectorStore":
        """Create a vector store from texts asynchronously.

        Args:
            texts: Texts to add.
            embedding: Embeddings instance.
            metadatas: Optional metadata for each text.
            db_path: Path to database. Creates temp dir if not provided.
            dimension: Embedding dimension.

        Returns:
            Initialized vector store with texts added.
        """
        if db_path is None:
            import tempfile

            db_path = tempfile.mkdtemp(prefix="ragfs_")

        store = cls(embedding=embedding, db_path=db_path, dimension=dimension)
        await store.ainit()
        await store.aadd_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_embeddings(
        cls,
        embedding: Embeddings,
        db_path: str,
        dimension: int = 384,
    ) -> "LangChainRagfsVectorStore":
        """Create a vector store from an embeddings instance."""
        return cls(embedding=embedding, db_path=db_path, dimension=dimension)

    @classmethod
    def from_documents(
        cls: Type["LangChainRagfsVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "LangChainRagfsVectorStore":
        """Create from documents synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            cls.afrom_documents(documents, embedding, **kwargs)
        )

    @classmethod
    async def afrom_documents(
        cls: Type["LangChainRagfsVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        db_path: Optional[str] = None,
        dimension: int = 384,
        **kwargs: Any,
    ) -> "LangChainRagfsVectorStore":
        """Create a vector store from documents asynchronously.

        Args:
            documents: Documents to add.
            embedding: Embeddings instance.
            db_path: Path to database. Creates temp dir if not provided.
            dimension: Embedding dimension.

        Returns:
            Initialized vector store with documents added.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await cls.afrom_texts(
            texts,
            embedding,
            metadatas=metadatas,
            db_path=db_path,
            dimension=dimension,
            **kwargs,
        )
