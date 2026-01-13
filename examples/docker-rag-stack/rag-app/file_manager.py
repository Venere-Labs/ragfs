"""
File Manager module for RAGFS Docker Stack

Provides file operations with RAGFS safety layer integration:
- Upload documents (triggers re-indexing via file watcher)
- Soft delete to trash
- Restore from trash
- View operation history
- Undo operations
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
import aiofiles.os

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from ragfs import (
    RagfsSafetyManager,
    TrashEntry,
    HistoryEntry,
)


@dataclass
class DocumentInfo:
    """Information about an indexed document."""
    path: str
    name: str
    size: int
    mime_type: str
    modified: datetime
    is_indexed: bool = True


@dataclass
class UploadResult:
    """Result of a file upload operation."""
    success: bool
    path: Optional[str] = None
    error: Optional[str] = None
    message: str = ""


@dataclass
class OperationResult:
    """Result of a file operation."""
    success: bool
    message: str
    undo_id: Optional[str] = None
    error: Optional[str] = None


# Allowed MIME types for upload
ALLOWED_MIME_TYPES = {
    # Text
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/html",
    "text/css",
    "text/csv",
    "text/xml",
    # Code
    "text/x-python",
    "text/x-java",
    "text/x-c",
    "text/x-c++",
    "text/x-rust",
    "text/x-go",
    "text/x-javascript",
    "text/x-typescript",
    "application/javascript",
    "application/typescript",
    "application/json",
    "application/x-yaml",
    "application/toml",
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    # Fallback for unknown text
    "application/octet-stream",
}

# File extensions that are always allowed
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".css",
    ".py", ".rs", ".go", ".java", ".js", ".ts", ".jsx", ".tsx",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
    ".pdf", ".doc", ".docx",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql",
    ".dockerfile", ".makefile",
}


class FileManager:
    """File operations with RAGFS safety layer integration."""

    def __init__(
        self,
        documents_path: Optional[str] = None,
        db_path: Optional[str] = None,
        trash_path: Optional[str] = None,
    ):
        """Initialize the file manager.

        Args:
            documents_path: Path to documents directory.
            db_path: Path to RAGFS index database.
            trash_path: Path to trash directory.
        """
        self.documents_path = Path(documents_path or os.environ.get("DOCUMENTS_PATH", "/data/docs"))
        self.db_path = Path(db_path or os.environ.get("RAGFS_DB_PATH", "/data/index"))
        self.trash_path = Path(trash_path or os.environ.get("RAGFS_TRASH_PATH", "/data/trash"))

        self._safety_manager: Optional[RagfsSafetyManager] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the safety manager."""
        if self._initialized:
            return

        # Ensure directories exist
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self.trash_path.mkdir(parents=True, exist_ok=True)

        # Initialize safety manager
        self._safety_manager = RagfsSafetyManager(
            str(self.documents_path),
            trash_path=str(self.trash_path),
        )

        self._initialized = True

    def _get_mime_type(self, file_path: Path, content: bytes) -> str:
        """Detect MIME type of a file."""
        if HAS_MAGIC:
            try:
                return magic.from_buffer(content, mime=True)
            except Exception:
                pass

        # Fallback to extension-based detection
        ext = file_path.suffix.lower()
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".py": "text/x-python",
            ".rs": "text/x-rust",
            ".js": "application/javascript",
            ".ts": "application/typescript",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".pdf": "application/pdf",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _is_allowed_file(self, filename: str, mime_type: str) -> bool:
        """Check if a file is allowed to be uploaded."""
        ext = Path(filename).suffix.lower()
        if ext in ALLOWED_EXTENSIONS:
            return True
        if mime_type in ALLOWED_MIME_TYPES:
            return True
        # Allow text files
        if mime_type.startswith("text/"):
            return True
        return False

    async def upload(
        self,
        filename: str,
        content: bytes,
        subfolder: Optional[str] = None,
    ) -> UploadResult:
        """Upload a file to the documents directory.

        The file watcher will automatically index it.

        Args:
            filename: Name of the file.
            content: File content as bytes.
            subfolder: Optional subfolder within documents.

        Returns:
            UploadResult with success status and path.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Validate file
            mime_type = self._get_mime_type(Path(filename), content)
            if not self._is_allowed_file(filename, mime_type):
                return UploadResult(
                    success=False,
                    error=f"File type not allowed: {mime_type}",
                    message=f"Cannot upload {filename}: unsupported file type",
                )

            # Determine target path
            if subfolder:
                target_dir = self.documents_path / subfolder
            else:
                target_dir = self.documents_path

            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            # Handle duplicates
            if target_path.exists():
                base = target_path.stem
                ext = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{base}_{counter}{ext}"
                    counter += 1

            # Write file
            async with aiofiles.open(target_path, "wb") as f:
                await f.write(content)

            return UploadResult(
                success=True,
                path=str(target_path),
                message=f"Uploaded {filename} successfully. It will be indexed automatically.",
            )

        except Exception as e:
            return UploadResult(
                success=False,
                error=str(e),
                message=f"Failed to upload {filename}: {e}",
            )

    async def delete(self, path: str) -> OperationResult:
        """Soft delete a file to trash.

        Args:
            path: Path to the file to delete.

        Returns:
            OperationResult with undo ID.
        """
        if not self._initialized:
            await self.initialize()

        try:
            file_path = Path(path)
            if not file_path.exists():
                return OperationResult(
                    success=False,
                    message=f"File not found: {path}",
                    error="File does not exist",
                )

            # Use safety manager for soft delete
            entry = await self._safety_manager.delete_to_trash(str(file_path))

            return OperationResult(
                success=True,
                message=f"Moved to trash: {file_path.name}",
                undo_id=str(entry.id),
            )

        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Failed to delete: {e}",
                error=str(e),
            )

    async def restore(self, trash_id: str) -> OperationResult:
        """Restore a file from trash.

        Args:
            trash_id: ID of the trash entry.

        Returns:
            OperationResult with status.
        """
        if not self._initialized:
            await self.initialize()

        try:
            restored_path = await self._safety_manager.restore_from_trash(trash_id)

            return OperationResult(
                success=True,
                message=f"Restored: {Path(restored_path).name}",
            )

        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Failed to restore: {e}",
                error=str(e),
            )

    async def list_documents(self) -> List[DocumentInfo]:
        """List all documents in the documents directory.

        Returns:
            List of DocumentInfo objects.
        """
        if not self._initialized:
            await self.initialize()

        documents = []

        for path in self.documents_path.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                try:
                    stat = path.stat()
                    content = b""
                    if path.stat().st_size < 1024 * 1024:  # Only read first 1MB for mime detection
                        async with aiofiles.open(path, "rb") as f:
                            content = await f.read(1024)

                    documents.append(DocumentInfo(
                        path=str(path),
                        name=path.name,
                        size=stat.st_size,
                        mime_type=self._get_mime_type(path, content),
                        modified=datetime.fromtimestamp(stat.st_mtime),
                    ))
                except Exception:
                    pass

        return sorted(documents, key=lambda d: d.modified, reverse=True)

    async def list_trash(self) -> List[TrashEntry]:
        """List all items in trash.

        Returns:
            List of TrashEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._safety_manager.list_trash()
        except Exception:
            return []

    async def get_history(self, limit: int = 50) -> List[HistoryEntry]:
        """Get operation history.

        Args:
            limit: Maximum number of entries.

        Returns:
            List of HistoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._safety_manager.get_history(limit=limit)
        except Exception:
            return []

    async def undo(self, operation_id: str) -> OperationResult:
        """Undo an operation.

        Args:
            operation_id: ID of the operation to undo.

        Returns:
            OperationResult with status.
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self._safety_manager.undo(operation_id)

            return OperationResult(
                success=True,
                message="Operation undone successfully",
            )

        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Failed to undo: {e}",
                error=str(e),
            )

    async def get_document_preview(self, path: str, max_chars: int = 500) -> str:
        """Get a preview of a document's content.

        Args:
            path: Path to the file.
            max_chars: Maximum characters to return.

        Returns:
            Preview string.
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return "[File not found]"

            # Don't preview binary files
            mime_type = self._get_mime_type(file_path, b"")
            if not (mime_type.startswith("text/") or mime_type in {"application/json", "application/x-yaml"}):
                return f"[Binary file: {mime_type}]"

            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read(max_chars + 100)

            if len(content) > max_chars:
                content = content[:max_chars] + "..."

            return content

        except Exception as e:
            return f"[Error reading file: {e}]"
