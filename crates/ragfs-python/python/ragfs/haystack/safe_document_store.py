"""Haystack DocumentStore with RAGFS safety layer integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .document_store import HaystackRagfsDocumentStore
from ragfs import RagfsSafetyManager, TrashEntry, HistoryEntry

try:
    from haystack.document_stores.types import DuplicatePolicy
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


class HaystackRagfsSafeDocumentStore(HaystackRagfsDocumentStore):
    """DocumentStore with RAGFS safety layer integration.

    Extends the standard document store with:
    - Soft delete support (documents go to trash, can be recovered)
    - Audit history logging
    - Undo support for reversible operations

    This is a core RAGFS innovation for safe AI agent file operations.

    Example:
        from ragfs.haystack import RagfsSafeDocumentStore

        # Create with safety enabled
        store = RagfsSafeDocumentStore("/path/to/db", safety_enabled=True)

        # Delete with soft-delete (returns undo capability)
        result = store.safe_delete_documents(["doc_id_1", "doc_id_2"])
        print(f"Deleted, undo IDs: {result['undo_ids']}")

        # Restore from trash
        store.restore_documents(list(result["undo_ids"].values()))

        # View operation history
        history = store.get_history(limit=10)
        for entry in history:
            print(f"{entry.operation.operation_type} at {entry.timestamp}")
    """

    def __init__(
        self,
        db_path: str,
        dimension: int = 384,
        safety_enabled: bool = True,
        source_path: Optional[str] = None,
    ):
        """Initialize the safe document store.

        Args:
            db_path: Path to the LanceDB database.
            dimension: Embedding dimension. Defaults to 384 (GTE-small).
            safety_enabled: Whether to enable safety layer. Defaults to True.
            source_path: Path to source directory for safety tracking.
                        Defaults to db_path parent directory.
        """
        super().__init__(db_path=db_path, dimension=dimension)

        self._safety_enabled = safety_enabled
        self._source_path = source_path or db_path

        if safety_enabled:
            self._safety = RagfsSafetyManager(self._source_path)
        else:
            self._safety = None

    def safe_delete_documents(
        self,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Soft delete documents (can be undone).

        Unlike the standard delete, this moves documents to trash
        and they can be restored later.

        Args:
            document_ids: List of document IDs to delete.

        Returns:
            Dictionary with:
            - undo_ids: Mapping of doc_id -> undo_id for restoration
            - soft_deleted: Whether soft delete was used
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._async_safe_delete_documents(document_ids)
        )

    async def _async_safe_delete_documents(
        self,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Async implementation of safe delete."""
        result = {
            "document_ids": document_ids,
            "soft_deleted": False,
            "undo_ids": {},
        }

        if self._safety and self._safety_enabled:
            for doc_id in document_ids:
                try:
                    # Use safety manager for soft delete
                    trash_entry = await self._safety.delete_to_trash(doc_id)
                    result["undo_ids"][doc_id] = trash_entry.id
                except Exception:
                    # Fall back to hard delete if soft delete fails
                    self.delete_documents([doc_id])

            if result["undo_ids"]:
                result["soft_deleted"] = True
        else:
            # Hard delete
            self.delete_documents(document_ids)

        return result

    def restore_documents(self, undo_ids: List[str]) -> List[str]:
        """Restore soft-deleted documents from trash.

        Args:
            undo_ids: List of undo IDs from safe_delete_documents.

        Returns:
            List of restored file paths.

        Raises:
            RuntimeError: If safety is not enabled.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._async_restore_documents(undo_ids)
        )

    async def _async_restore_documents(self, undo_ids: List[str]) -> List[str]:
        """Async implementation of restore."""
        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled. Cannot restore documents.")

        restored_paths = []
        for undo_id in undo_ids:
            restored_path = await self._safety.restore_from_trash(undo_id)
            restored_paths.append(restored_path)

        return restored_paths

    def get_trash_contents(self) -> List[TrashEntry]:
        """List all items in trash that can be restored.

        Returns:
            List of TrashEntry objects.

        Raises:
            RuntimeError: If safety is not enabled.
        """
        import asyncio

        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled.")

        return asyncio.get_event_loop().run_until_complete(
            self._safety.list_trash()
        )

    def get_history(self, limit: Optional[int] = None) -> List[HistoryEntry]:
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

    def undo_operation(self, operation_id: str) -> str:
        """Undo any reversible operation by its ID.

        Args:
            operation_id: The operation ID from HistoryEntry.id.

        Returns:
            A message describing what was undone.

        Raises:
            RuntimeError: If safety is not enabled or undo fails.
        """
        import asyncio

        if not self._safety or not self._safety_enabled:
            raise RuntimeError("Safety layer not enabled.")

        return asyncio.get_event_loop().run_until_complete(
            self._safety.undo(operation_id)
        )

    def can_undo(self, operation_id: str) -> bool:
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


# Convenience alias
RagfsSafeDocumentStore = HaystackRagfsSafeDocumentStore
