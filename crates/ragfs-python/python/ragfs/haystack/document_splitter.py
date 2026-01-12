"""Haystack DocumentSplitter adapter for RAGFS."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ragfs import RagfsTextSplitter as CoreSplitter

try:
    from haystack import component, Document
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


@component
class HaystackRagfsDocumentSplitter:
    """Haystack-compatible document splitter using RAGFS.

    Splits documents into smaller chunks using code/semantic-aware strategies.

    Example:
        from ragfs.haystack import RagfsDocumentSplitter
        from haystack import Document

        splitter = RagfsDocumentSplitter(chunk_size=512, split_by="auto")
        result = splitter.run(documents=[Document(content="Long text...")])
        chunks = result["documents"]
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        split_by: str = "auto",
    ):
        """Initialize the document splitter.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            split_by: Splitting strategy. Options:
                - "auto": Automatically select based on content type
                - "fixed": Token-based chunking with overlap
                - "code": AST-aware chunking for source code
                - "semantic": Structure-aware chunking on headings/paragraphs
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._split_by = split_by
        self._splitter = CoreSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_type=split_by,
        )

    def warm_up(self) -> None:
        """Warm up the splitter (no-op for this component)."""
        pass

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """Split documents into smaller chunks.

        Args:
            documents: List of Haystack Documents to split.

        Returns:
            Dictionary with 'documents' list of split documents.
        """
        result_docs = asyncio.get_event_loop().run_until_complete(
            self._async_split(documents)
        )
        return {"documents": result_docs}

    async def _async_split(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents asynchronously."""
        result_docs = []

        for doc in documents:
            content = doc.content or ""
            if not content:
                result_docs.append(doc)
                continue

            # Detect language from metadata for code splitting
            language = None
            if doc.meta:
                language = doc.meta.get("language")
                # Also check file extension
                file_path = doc.meta.get("file_path", "")
                if not language and file_path:
                    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                    ext_to_lang = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "rs": "rust",
                        "go": "go",
                        "java": "java",
                        "c": "c",
                        "cpp": "cpp",
                        "h": "c",
                        "hpp": "cpp",
                        "rb": "ruby",
                        "php": "php",
                        "swift": "swift",
                        "kt": "kotlin",
                        "scala": "scala",
                        "cs": "csharp",
                    }
                    language = ext_to_lang.get(ext)

            # Split the content
            try:
                chunks = await self._splitter.split_text(content, language=language)
            except Exception:
                # If splitting fails, keep original document
                result_docs.append(doc)
                continue

            if len(chunks) <= 1:
                result_docs.append(doc)
                continue

            # Create new documents for each chunk
            for i, chunk_text in enumerate(chunks):
                # Copy and update metadata
                metadata = dict(doc.meta) if doc.meta else {}
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                metadata["original_id"] = doc.id

                new_doc = Document(
                    content=chunk_text,
                    meta=metadata,
                )
                result_docs.append(new_doc)

        return result_docs

    @property
    def chunk_size(self) -> int:
        """Get the configured chunk size."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get the configured chunk overlap."""
        return self._chunk_overlap

    @property
    def split_by(self) -> str:
        """Get the splitting strategy."""
        return self._split_by


# Convenience alias
RagfsDocumentSplitter = HaystackRagfsDocumentSplitter
