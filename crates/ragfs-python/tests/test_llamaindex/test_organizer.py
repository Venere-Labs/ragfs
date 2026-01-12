"""Tests for LlamaIndex RagfsOrganizer component."""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.requires_model


class TestLlamaIndexOrganizer:
    """Tests for the LlamaIndex RagfsOrganizer."""

    def test_organizer_import(self):
        """Test that RagfsOrganizer can be imported."""
        from ragfs.llamaindex import RagfsOrganizer

        assert RagfsOrganizer is not None

    def test_organizer_creation(self, source_path: Path, db_path: Path):
        """Test creating an organizer instance."""
        from ragfs.llamaindex import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        assert organizer is not None
        assert organizer.source_path == str(source_path)
        assert organizer.db_path == str(db_path)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_organizer_init(self, source_path: Path, db_path: Path):
        """Test initializing the organizer."""
        from ragfs.llamaindex import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        await organizer.init()
        assert await organizer.is_available()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_propose_organization(self, source_path: Path, db_path: Path):
        """Test proposing an organization plan."""
        from ragfs.llamaindex import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        await organizer.init()

        plan = await organizer.propose_organization(
            scope="./",
            strategy="by_topic",
            max_groups=3,
        )

        assert plan is not None
        assert plan.id is not None
        assert plan.status == "pending"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_list_pending_plans(self, source_path: Path, db_path: Path):
        """Test listing pending plans."""
        from ragfs.llamaindex import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        await organizer.init()

        # Create a plan first
        plan = await organizer.propose_organization(scope="./", strategy="by_topic")

        # List pending
        pending = await organizer.list_pending_plans()
        assert len(pending) >= 1
        assert any(p.id == plan.id for p in pending)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_reject_plan(self, source_path: Path, db_path: Path):
        """Test rejecting a plan."""
        from ragfs.llamaindex import RagfsOrganizer

        organizer = RagfsOrganizer(str(source_path), str(db_path))
        await organizer.init()

        # Create and reject a plan
        plan = await organizer.propose_organization(scope="./", strategy="by_type")
        result = await organizer.reject(plan.id)

        assert result.status == "rejected"


class TestLlamaIndexSafeVectorStore:
    """Tests for the LlamaIndex RagfsSafeVectorStore."""

    def test_safe_vectorstore_import(self):
        """Test that RagfsSafeVectorStore can be imported."""
        from ragfs.llamaindex import RagfsSafeVectorStore

        assert RagfsSafeVectorStore is not None

    def test_safe_vectorstore_creation(self, db_path: Path):
        """Test creating a safe vector store."""
        from ragfs.llamaindex import RagfsSafeVectorStore

        store = RagfsSafeVectorStore(str(db_path), safety_enabled=True)
        assert store is not None
        assert store.safety_enabled

    def test_safe_vectorstore_safety_disabled(self, db_path: Path):
        """Test creating a safe vector store with safety disabled."""
        from ragfs.llamaindex import RagfsSafeVectorStore

        store = RagfsSafeVectorStore(str(db_path), safety_enabled=False)
        assert store is not None
        assert not store.safety_enabled
