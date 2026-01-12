"""LlamaIndex VectorStore with RAGFS safety layer integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .vectorstore import LlamaIndexRagfsVectorStore
from ragfs import RagfsSafetyManager, TrashEntry, HistoryEntry

try:
    from llama_index.core.schema import BaseNode
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsSafeVectorStore(LlamaIndexRagfsVectorStore):
    """VectorStore with RAGFS safety layer integration.

    Extends the standard vector store with:
    - Soft delete support (files go to trash, can be recovered)
    - Audit history logging
    - Undo support for reversible operations

    This is a core RAGFS innovation for safe AI agent file operations.

    Example:
        from ragfs.llamaindex import RagfsSafeVectorStore

        # Create with safety enabled
        store = RagfsSafeVectorStore("/path/to/db", safety_enabled=True)
        await store.ainit()

        # Delete with soft-delete (returns undo capability)
        undo_info = await store.safe_delete("document_id")
        print(f"Deleted, can undo with: {undo_info['undo_id']}")

        # Restore from trash
        await store.undo_delete(undo_info["undo_id"])

        # View operation history
        history = await store.get_history(limit=10)
        for entry in history:
            print(f"{entry.operation.operation_type} at {entry.timestamp}")
    """

    _safety: Optional[RagfsSafetyManager] = None
    _safety_enabled: bool = False
    _source_path: str = ""

    def __init__(
        self,
        db_path: str,
        dimension: int = 384,
        safety_enabled: bool = True,
        source_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the safe vector store.

        Args:
            db_path: Path to the LanceDB database.
            dimension: Embedding dimension. Defaults to 384 (GTE-small).
            safety_enabled: Whether to enable safety layer. Defaults to True.
            source_path: Path to source directory for safety tracking.
                        Defaults to db_path parent directory.
        """
        super().__init__(db_path=db_path, dimension=dimension, **kwargs)

        self._safety_enabled = safety_enabled
        self._source_path = source_path or db_path

        if safety_enabled:
            self._safety = RagfsSafetyManager(self._source_path)

    async def safe_delete(
        self,
        ref_doc_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Delete with soft-delete support (can be undone).

        Unlike the standard delete, this moves the reference to trash
        and can be restored later.

        Args:
            ref_doc_id: Reference document ID (typically file path).

        Returns:
            Dictionary with:
            - undo_id: ID to use for restoring (if safety enabled)
            - soft_deleted: Whether soft delete was used
        """
        await self.ainit()

        result = {
            "ref_doc_id": ref_doc_id,
            "soft_deleted": False,
            "undo_id": None,
        }

        if self._safety and self._safety_enabled:
            try:
                # Use safety manager for soft delete
                trash_entry = await self._safety.delete_to_trash(ref_doc_id)
                result["soft_deleted"] = True
                result["undo_id"] = trash_entry.id
                result["trash_entry"] = trash_entry
            except Exception:
                # Fall back to hard delete if soft delete fails
                await self._store.delete_by_path(ref_doc_id)
        else:
            # Hard delete
            await self._store.delete_by_path(ref_doc_id)

        return result

    async def undo_delete(self, undo_id: str) -> str:
        """Restore a soft-deleted document from trash.

        Args:
            undo_id: The undo_id returned from safe_delete.

        Returns:
            The restored file path.

        Raises:
            RuntimeError: If safety is not enabled or restore fails.
        """
        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled. Cannot undo delete.")

        restored_path = await self._safety.restore_from_trash(undo_id)
        return restored_path

    async def get_trash_contents(self) -> List[TrashEntry]:
        """List all items in trash that can be restored.

        Returns:
            List of TrashEntry objects.

        Raises:
            RuntimeError: If safety is not enabled.
        """
        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled.")

        return await self._safety.list_trash()

    async def get_history(
        self,
        limit: Optional[int] = None,
    ) -> List[HistoryEntry]:
        """Get operation history for audit trail.

        Args:
            limit: Maximum number of entries to return. None for all.

        Returns:
            List of HistoryEntry objects (most recent first).

        Raises:
            RuntimeError: If safety is not enabled.
        """
        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled.")

        return self._safety.get_history(limit=limit)

    async def undo_operation(self, operation_id: str) -> str:
        """Undo any reversible operation by its ID.

        Args:
            operation_id: The operation ID from HistoryEntry.id.

        Returns:
            A message describing what was undone.

        Raises:
            RuntimeError: If safety is not enabled or undo fails.
        """
        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled.")

        return await self._safety.undo(operation_id)

    async def can_undo(self, operation_id: str) -> bool:
        """Check if an operation can be undone.

        Args:
            operation_id: The operation ID from HistoryEntry.id.

        Returns:
            True if the operation is reversible, False otherwise.
        """
        if not self._safety or not self._safety_enabled:
            return False

        return self._safety.can_undo(operation_id)

    @property
    def safety_enabled(self) -> bool:
        """Check if safety layer is enabled."""
        return self._safety_enabled

    @property
    def safety_manager(self) -> Optional[RagfsSafetyManager]:
        """Get the underlying safety manager."""
        return self._safety


# Convenience aliases
RagfsSafeVectorStore = LlamaIndexRagfsSafeVectorStore
