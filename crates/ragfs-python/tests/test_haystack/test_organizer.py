"""Tests for Haystack RagfsOrganizer component."""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.requires_model


class TestHaystackOrganizer:
    """Tests for the Haystack RagfsOrganizer."""

    def test_organizer_import(self):
        """Test that RagfsOrganizer can be imported."""
        from ragfs.haystack import RagfsOrganizer

        assert RagfsOrganizer is not None

    def test_organizer_creation(self, source_path: Path, db_path: Path):
        """Test creating an organizer instance."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        assert organizer is not None
        assert organizer.source_path == str(source_path)
        assert organizer.db_path == str(db_path)

    @pytest.mark.slow
    def test_organizer_warm_up(self, source_path: Path, db_path: Path):
        """Test warming up the organizer."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        organizer.warm_up()
        # Should not raise

    @pytest.mark.slow
    def test_organizer_find_similar(self, source_path: Path, db_path: Path):
        """Test finding similar files via run()."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        doc_path = source_path / "doc1.txt"

        result = organizer.run(
            operation="find_similar",
            file_path=str(doc_path),
            k=3,
        )

        assert "similar_files" in result
        # May be empty if not indexed yet

    @pytest.mark.slow
    def test_organizer_propose_organization(self, source_path: Path, db_path: Path):
        """Test proposing organization via run()."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))

        result = organizer.run(
            operation="propose_organization",
            scope="./",
            strategy="by_topic",
            max_groups=3,
        )

        assert "plan" in result
        plan = result["plan"]
        assert plan is not None
        assert plan.id is not None

    @pytest.mark.slow
    def test_organizer_list_pending(self, source_path: Path, db_path: Path):
        """Test listing pending plans via run()."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))

        result = organizer.run(operation="list_pending")
        assert "pending_plans" in result

    def test_organizer_invalid_operation(self, source_path: Path, db_path: Path):
        """Test that invalid operation raises error."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))

        with pytest.raises(ValueError, match="Unknown operation"):
            organizer.run(operation="invalid_operation")

    def test_organizer_missing_file_path(self, source_path: Path, db_path: Path):
        """Test that find_similar without file_path raises error."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))

        with pytest.raises(ValueError, match="file_path required"):
            organizer.run(operation="find_similar")

    def test_organizer_missing_scope(self, source_path: Path, db_path: Path):
        """Test that propose_organization without scope raises error."""
        from ragfs.haystack import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))

        with pytest.raises(ValueError, match="scope required"):
            organizer.run(operation="propose_organization")


class TestHaystackSafeDocumentStore:
    """Tests for the Haystack RagfsSafeDocumentStore."""

    def test_safe_document_store_import(self):
        """Test that RagfsSafeDocumentStore can be imported."""
        from ragfs.haystack import RagfsSafeDocumentStore

        assert RagfsSafeDocumentStore is not None

    def test_safe_document_store_creation(self, db_path: Path):
        """Test creating a safe document store."""
        from ragfs.haystack import RagfsSafeDocumentStore

        store = RagfsSafeDocumentStore(str(db_path), safety_enabled=True)
        assert store is not None
        assert store.safety_enabled

    def test_safe_document_store_safety_disabled(self, db_path: Path):
        """Test creating with safety disabled."""
        from ragfs.haystack import RagfsSafeDocumentStore

        store = RagfsSafeDocumentStore(str(db_path), safety_enabled=False)
        assert store is not None
        assert not store.safety_enabled

    def test_safe_document_store_get_history_disabled(self, db_path: Path):
        """Test that get_history raises when safety is disabled."""
        from ragfs.haystack import RagfsSafeDocumentStore

        store = RagfsSafeDocumentStore(str(db_path), safety_enabled=False)

        with pytest.raises(RuntimeError, match="Safety layer not enabled"):
            store.get_history()

    def test_safe_document_store_get_trash_disabled(self, db_path: Path):
        """Test that get_trash_contents raises when safety is disabled."""
        from ragfs.haystack import RagfsSafeDocumentStore

        store = RagfsSafeDocumentStore(str(db_path), safety_enabled=False)

        with pytest.raises(RuntimeError, match="Safety layer not enabled"):
            store.get_trash_contents()
