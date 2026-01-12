"""LangChain Retriever adapter for RAGFS."""

from __future__ import annotations

from typing import List, Optional

from ragfs import RagfsRetriever as CoreRetriever

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install with: pip install ragfs[langchain]"
    )


class LangChainRagfsRetriever(BaseRetriever):
    """LangChain-compatible retriever using RAGFS.

    Combines embeddings and vector search in a single component.
    Supports hybrid search (vector + full-text).

    Example:
        retriever = LangChainRagfsRetriever("/path/to/db")
        await retriever.ainit()
        docs = await retriever.aget_relevant_documents("my query")
    """

    db_path: str
    model_path: Optional[str] = None
    hybrid: bool = True
    k: int = 4

    _retriever: Optional[CoreRetriever] = None
    _initialized: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        db_path: str,
        model_path: Optional[str] = None,
        hybrid: bool = True,
        k: int = 4,
        **kwargs,
    ):
        """Initialize the retriever.

        Args:
            db_path: Path to the LanceDB database.
            model_path: Path to store model files.
            hybrid: Enable hybrid search.
            k: Number of results to return.
        """
        super().__init__(
            db_path=db_path,
            model_path=model_path,
            hybrid=hybrid,
            k=k,
            **kwargs,
        )
        self._retriever = CoreRetriever(
            db_path=db_path,
            model_path=model_path,
            hybrid=hybrid,
            k=k,
        )
        self._initialized = False

    async def ainit(self) -> None:
        """Initialize the retriever (downloads model if needed)."""
        if not self._initialized:
            await self._retriever.init()
            self._initialized = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents synchronously.

        Note: This blocks the event loop. Use aget_relevant_documents for async.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents asynchronously."""
        await self.ainit()

        core_docs = await self._retriever.get_relevant_documents(query)

        return [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in core_docs
        ]

    async def asearch(
        self,
        query: str,
        hybrid: Optional[bool] = None,
        k: Optional[int] = None,
    ) -> List[tuple[Document, float]]:
        """Search with explicit control and scores."""
        await self.ainit()

        results = await self._retriever.search(query, hybrid=hybrid, k=k)

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
