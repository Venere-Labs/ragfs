"""LlamaIndex Embeddings adapter for RAGFS."""

from __future__ import annotations

from typing import Any, List, Optional

from ragfs import RagfsEmbeddings as CoreEmbeddings

try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsEmbeddings(BaseEmbedding):
    """LlamaIndex-compatible embeddings using RAGFS.

    Uses local GTE-small model (384 dimensions) for embedding generation.

    Example:
        from ragfs.llamaindex import RagfsEmbeddings
        from llama_index.core import VectorStoreIndex, Document

        embed_model = RagfsEmbeddings()
        index = VectorStoreIndex.from_documents(
            [Document(text="Hello world")],
            embed_model=embed_model,
        )
    """

    model_name: str = "thenlper/gte-small"
    embed_batch_size: int = 32

    _embedder: Any = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs: Any,
    ):
        """Initialize the embeddings.

        Args:
            model_path: Path to store/load model files.
            batch_size: Batch size for embedding.
            normalize: Whether to L2-normalize embeddings.
        """
        super().__init__(embed_batch_size=batch_size, **kwargs)
        self._embedder = CoreEmbeddings(
            model_path=model_path,
            batch_size=batch_size,
            normalize=normalize,
        )
        self._initialized = False

    async def _ainit(self) -> None:
        """Initialize the embedder."""
        if not self._initialized:
            await self._embedder.init()
            self._initialized = True

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (sync)."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aget_text_embedding(text)
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (async)."""
        await self._ainit()
        return await self._embedder.embed_query(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (sync)."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aget_text_embeddings(texts)
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (async)."""
        await self._ainit()
        return await self._embedder.embed_documents(texts)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (sync)."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (async)."""
        return await self._aget_text_embedding(query)
