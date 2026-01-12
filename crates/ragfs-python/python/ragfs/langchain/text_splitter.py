"""LangChain TextSplitter adapter for RAGFS."""

from __future__ import annotations

from typing import Any, List, Optional

from ragfs import RagfsTextSplitter as CoreSplitter

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain_core.documents import Document
        from langchain.text_splitter import TextSplitter
    except ImportError:
        raise ImportError(
            "langchain-core or langchain-text-splitters is required. "
            "Install with: pip install ragfs[langchain]"
        )


class LangChainRagfsTextSplitter(TextSplitter):
    """LangChain-compatible text splitter using RAGFS.

    Supports multiple chunking strategies:
    - fixed: Token-based chunking with overlap
    - code: AST-aware chunking for source code
    - semantic: Structure-aware chunking on headings/paragraphs
    - auto: Automatically select based on content type

    Example:
        splitter = LangChainRagfsTextSplitter(chunk_size=512)
        chunks = await splitter.asplit_text("Long text...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunker_type: str = "auto",
        **kwargs: Any,
    ):
        """Initialize the text splitter.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            chunker_type: Strategy: auto, fixed, code, or semantic.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._splitter = CoreSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_type=chunker_type,
        )

    def split_text(self, text: str) -> List[str]:
        """Split text synchronously.

        Note: This blocks the event loop. Use asplit_text for async.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.asplit_text(text))

    async def asplit_text(self, text: str) -> List[str]:
        """Split text asynchronously."""
        return await self._splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.asplit_documents(documents)
        )

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents asynchronously."""
        from ragfs import Document as CoreDocument

        # Convert to core documents
        core_docs = [
            CoreDocument(
                page_content=doc.page_content,
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            for doc in documents
        ]

        # Split
        split_docs = await self._splitter.split_documents(core_docs)

        # Convert back
        return [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in split_docs
        ]
