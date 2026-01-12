"""LlamaIndex Reader adapter for RAGFS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from ragfs import RagfsDocumentLoader as CoreLoader

try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsReader(BaseReader):
    """LlamaIndex-compatible document reader using RAGFS.

    Supports 40+ text formats, PDFs, and images.

    Example:
        from ragfs.llamaindex import RagfsReader

        reader = RagfsReader()
        documents = await reader.aload_data("/path/to/docs")

        # Use with LlamaIndex
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents(documents)
    """

    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        recursive: bool = True,
        **kwargs: Any,
    ):
        """Initialize the reader.

        Args:
            extractors: List of extractors to enable. Options: "text", "pdf", "image".
                       Defaults to all extractors.
            recursive: Whether to recursively load directories.
        """
        super().__init__()
        self._loader = CoreLoader(extractors=extractors)
        self._recursive = recursive

    def load_data(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> List[Document]:
        """Load documents from a file or directory synchronously.

        Args:
            file_path: Path to file or directory to load.

        Returns:
            List of LlamaIndex Document objects.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.aload_data(file_path, **kwargs)
        )

    async def aload_data(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> List[Document]:
        """Load documents from a file or directory asynchronously.

        Args:
            file_path: Path to file or directory to load.

        Returns:
            List of LlamaIndex Document objects.
        """
        path = Path(file_path)

        if path.is_file():
            return await self._load_file(path)
        elif path.is_dir():
            return await self._load_directory(path)
        else:
            raise ValueError(f"Path does not exist: {path}")

    async def _load_file(self, file_path: Path) -> List[Document]:
        """Load a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of LlamaIndex Document objects.
        """
        # Check if file is supported
        if not self._loader.can_load(str(file_path)):
            return []

        # Load with RAGFS loader
        ragfs_docs = await self._loader.load(str(file_path))

        # Convert to LlamaIndex documents
        documents = []
        for doc in ragfs_docs:
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["file_path"] = str(file_path)

            documents.append(Document(
                text=doc.page_content,
                metadata=metadata,
            ))

        return documents

    async def _load_directory(self, dir_path: Path) -> List[Document]:
        """Load all files from a directory.

        Args:
            dir_path: Path to the directory.

        Returns:
            List of LlamaIndex Document objects.
        """
        documents = []

        if self._recursive:
            # Recursively walk directory
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        docs = await self._load_file(file_path)
                        documents.extend(docs)
                    except Exception:
                        # Skip files that can't be loaded
                        continue
        else:
            # Only load files in the immediate directory
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    try:
                        docs = await self._load_file(file_path)
                        documents.extend(docs)
                    except Exception:
                        continue

        return documents

    def can_load(self, file_path: Union[str, Path]) -> bool:
        """Check if a file can be loaded.

        Args:
            file_path: Path to check.

        Returns:
            True if the file format is supported.
        """
        return self._loader.can_load(str(file_path))

    @property
    def supported_types(self) -> List[str]:
        """Get list of supported MIME types."""
        return self._loader.supported_types()


# Convenience alias
RagfsReader = LlamaIndexRagfsReader
