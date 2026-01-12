"""LangChain Embeddings adapter for RAGFS."""

from __future__ import annotations

from typing import List

from ragfs import RagfsEmbeddings as CoreEmbeddings

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install with: pip install ragfs[langchain]"
    )


class LangChainRagfsEmbeddings(Embeddings):
    """LangChain-compatible embeddings using RAGFS.

    Uses local GTE-small model (384 dimensions) for embedding generation.
    No API calls required - runs entirely locally.

    Example:
        embeddings = LangChainRagfsEmbeddings()
        await embeddings.ainit()

        # Embed documents
        vectors = await embeddings.aembed_documents(["Hello", "World"])

        # Embed query
        query_vector = await embeddings.aembed_query("search query")
    """

    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """Initialize the embeddings.

        Args:
            model_path: Path to store/load model files.
            batch_size: Batch size for embedding.
            normalize: Whether to L2-normalize embeddings.
        """
        self._embedder = CoreEmbeddings(
            model_path=model_path,
            batch_size=batch_size,
            normalize=normalize,
        )
        self._initialized = False

    async def ainit(self) -> None:
        """Initialize the embedder (downloads model if needed)."""
        if not self._initialized:
            await self._embedder.init()
            self._initialized = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents synchronously.

        Note: This blocks the event loop. Use aembed_documents for async.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.aembed_documents(texts)
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a query synchronously.

        Note: This blocks the event loop. Use aembed_query for async.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.aembed_query(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents asynchronously."""
        await self.ainit()
        return await self._embedder.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query asynchronously."""
        await self.ainit()
        return await self._embedder.embed_query(text)

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._embedder.model_name
