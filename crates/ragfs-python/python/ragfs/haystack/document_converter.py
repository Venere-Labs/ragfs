"""Haystack DocumentConverter adapter for RAGFS."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ragfs import RagfsDocumentLoader as CoreLoader

try:
    from haystack import component, Document
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


@component
class HaystackRagfsDocumentConverter:
    """Haystack-compatible document converter using RAGFS.

    Converts files to Haystack Documents. Supports 40+ text formats, PDFs, and images.

    Example:
        from ragfs.haystack import RagfsDocumentConverter

        converter = RagfsDocumentConverter()
        result = converter.run(sources=["./docs/readme.md", "./report.pdf"])
        documents = result["documents"]
    """

    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        """Initialize the document converter.

        Args:
            extractors: List of extractors to enable. Options: "text", "pdf", "image".
                       Defaults to all extractors.
            recursive: Whether to recursively process directories.
        """
        self._loader = CoreLoader(extractors=extractors)
        self._recursive = recursive

    def warm_up(self) -> None:
        """Warm up the converter (no-op for this component)."""
        pass

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Document]]:
        """Convert files to Haystack Documents.

        Args:
            sources: List of file or directory paths to convert.
            meta: Optional metadata to add to all documents.

        Returns:
            Dictionary with 'documents' list.
        """
        documents = asyncio.get_event_loop().run_until_complete(
            self._async_convert(sources, meta)
        )
        return {"documents": documents}

    async def _async_convert(
        self,
        sources: List[Union[str, Path]],
        meta: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Convert files asynchronously."""
        documents = []
        base_meta = meta or {}

        for source in sources:
            path = Path(source)

            if path.is_file():
                docs = await self._convert_file(path, base_meta)
                documents.extend(docs)
            elif path.is_dir():
                docs = await self._convert_directory(path, base_meta)
                documents.extend(docs)

        return documents

    async def _convert_file(
        self,
        file_path: Path,
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """Convert a single file."""
        if not self._loader.can_load(str(file_path)):
            return []

        try:
            ragfs_docs = await self._loader.load(str(file_path))
        except Exception:
            return []

        documents = []
        for doc in ragfs_docs:
            # Merge metadata
            metadata = dict(base_meta)
            if doc.metadata:
                metadata.update(doc.metadata)
            metadata["file_path"] = str(file_path)

            documents.append(Document(
                content=doc.page_content,
                meta=metadata,
            ))

        return documents

    async def _convert_directory(
        self,
        dir_path: Path,
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """Convert all files in a directory."""
        documents = []

        if self._recursive:
            file_iter = dir_path.rglob("*")
        else:
            file_iter = dir_path.iterdir()

        for file_path in file_iter:
            if file_path.is_file():
                docs = await self._convert_file(file_path, base_meta)
                documents.extend(docs)

        return documents

    @property
    def supported_types(self) -> List[str]:
        """Get list of supported MIME types."""
        return self._loader.supported_types()


# Convenience alias
RagfsDocumentConverter = HaystackRagfsDocumentConverter
