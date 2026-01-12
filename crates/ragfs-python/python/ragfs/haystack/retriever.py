"""Haystack Retriever adapter for RAGFS."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ragfs import RagfsRetriever as CoreRetriever

try:
    from haystack import component, Document
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


@component
class HaystackRagfsRetriever:
    """Haystack-compatible retriever using RAGFS.

    Combines embeddings and vector search.

    Example:
        from ragfs.haystack import RagfsRetriever

        retriever = RagfsRetriever("/path/to/db")
        result = retriever.run(query="my question")
        documents = result["documents"]
    """

    def __init__(
        self,
        db_path: str,
        model_path: Optional[str] = None,
        hybrid: bool = True,
        top_k: int = 4,
    ):
        """Initialize the retriever.

        Args:
            db_path: Path to the LanceDB database.
            model_path: Path to store model files.
            hybrid: Enable hybrid search.
            top_k: Number of results to return.
        """
        self._retriever = CoreRetriever(
            db_path=db_path,
            model_path=model_path,
            hybrid=hybrid,
            k=top_k,
        )
        self._top_k = top_k
        self._initialized = False

    def warm_up(self) -> None:
        """Warm up the retriever (downloads model if needed)."""
        asyncio.get_event_loop().run_until_complete(self._retriever.init())
        self._initialized = True

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """Retrieve relevant documents.

        Args:
            query: The search query.
            top_k: Override the default top_k.

        Returns:
            Dictionary with 'documents' list.
        """
        if not self._initialized:
            self.warm_up()

        k = top_k or self._top_k

        results = asyncio.get_event_loop().run_until_complete(
            self._retriever.search(query, k=k)
        )

        documents = []
        for result in results:
            doc = Document(
                content=result.document.page_content,
                meta=result.document.metadata,
                score=result.score,
            )
            documents.append(doc)

        return {"documents": documents}
