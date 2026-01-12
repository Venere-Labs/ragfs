"""LangChain DocumentLoader adapter for RAGFS."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Union

from ragfs import RagfsDocumentLoader as CoreLoader

try:
    from langchain_core.documents import Document
    from langchain_core.document_loaders import BaseLoader
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install with: pip install ragfs[langchain]"
    )


class LangChainRagfsLoader(BaseLoader):
    """LangChain-compatible document loader using RAGFS.

    Supports 40+ text formats, PDFs, and images.

    Example:
        loader = LangChainRagfsLoader("/path/to/docs")
        documents = await loader.aload()
    """

    def __init__(
        self,
        path: Union[str, Path],
        extractors: Optional[List[str]] = None,
    ):
        """Initialize the document loader.

        Args:
            path: Path to file or directory to load.
            extractors: List of extractors to enable (text, pdf, image).
        """
        self._path = str(path)
        self._loader = CoreLoader(extractors=extractors)

    def load(self) -> List[Document]:
        """Load documents synchronously.

        Note: This blocks the event loop. Use aload for async.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.aload())

    async def aload(self) -> List[Document]:
        """Load documents asynchronously."""
        core_docs = await self._loader.load(self._path)

        return [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in core_docs
        ]

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load is not supported - falls back to load()."""
        yield from self.load()
