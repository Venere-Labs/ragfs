"""LlamaIndex Retriever adapter for RAGFS."""

from __future__ import annotations

from typing import Any, List, Optional

from ragfs import RagfsRetriever as CoreRetriever

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.callbacks import CallbackManager
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsRetriever(BaseRetriever):
    """LlamaIndex-compatible retriever using RAGFS.

    Combines embeddings and vector search.

    Example:
        from ragfs.llamaindex import RagfsRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine

        retriever = RagfsRetriever("/path/to/db")
        query_engine = RetrieverQueryEngine(retriever=retriever)
        response = query_engine.query("my question")
    """

    _retriever: Any = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        db_path: str,
        model_path: Optional[str] = None,
        hybrid: bool = True,
        k: int = 4,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        """Initialize the retriever.

        Args:
            db_path: Path to the LanceDB database.
            model_path: Path to store model files.
            hybrid: Enable hybrid search.
            k: Number of results to return.
        """
        super().__init__(callback_manager=callback_manager, **kwargs)
        self._retriever = CoreRetriever(
            db_path=db_path,
            model_path=model_path,
            hybrid=hybrid,
            k=k,
        )
        self._initialized = False
        self._k = k

    async def _ainit(self) -> None:
        """Initialize the retriever."""
        if not self._initialized:
            await self._retriever.init()
            self._initialized = True

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aretrieve(query_bundle)
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes asynchronously."""
        await self._ainit()

        query_str = query_bundle.query_str
        results = await self._retriever.search(query_str, k=self._k)

        nodes_with_scores = []
        for result in results:
            node = TextNode(
                text=result.document.page_content,
                metadata=result.document.metadata,
                id_=result.chunk_id,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=result.score))

        return nodes_with_scores
