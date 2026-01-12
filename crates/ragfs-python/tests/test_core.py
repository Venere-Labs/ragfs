"""Tests for core RAGFS Python bindings."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDocument:
    """Tests for Document class."""

    def test_document_creation(self):
        """Test creating a Document."""
        from ragfs import Document

        doc = Document(page_content="Hello, world!", metadata={"key": "value"})
        assert doc.page_content == "Hello, world!"
        assert doc.metadata["key"] == "value"

    def test_document_empty_metadata(self):
        """Test Document with empty metadata."""
        from ragfs import Document

        doc = Document(page_content="Test content")
        assert doc.page_content == "Test content"
        assert doc.metadata == {} or doc.metadata is None


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        from ragfs import Document, SearchResult

        doc = Document(page_content="Content", metadata={"file": "test.txt"})
        result = SearchResult(document=doc, score=0.95)
        assert result.document.page_content == "Content"
        assert result.score == 0.95


class TestOperationTypes:
    """Tests for Operation type."""

    def test_operation_create(self):
        """Test Operation.create factory."""
        from ragfs import Operation

        op = Operation.create("/path/file.txt", "content")
        assert op.action_type == "create"
        assert op.target == "/path/file.txt"

    def test_operation_move(self):
        """Test Operation.move factory."""
        from ragfs import Operation

        op = Operation.move("/old/path", "/new/path")
        assert op.action_type == "move"
        assert op.source == "/old/path"
        assert op.target == "/new/path"

    def test_operation_copy(self):
        """Test Operation.copy factory."""
        from ragfs import Operation

        op = Operation.copy("/src", "/dst")
        assert op.action_type == "copy"
        assert op.source == "/src"
        assert op.target == "/dst"

    def test_operation_delete(self):
        """Test Operation.delete factory."""
        from ragfs import Operation

        op = Operation.delete("/path/to/delete")
        assert op.action_type == "delete"
        assert op.target == "/path/to/delete"

    def test_operation_mkdir(self):
        """Test Operation.mkdir factory."""
        from ragfs import Operation

        op = Operation.mkdir("/new/directory")
        assert op.action_type == "mkdir"
        assert op.target == "/new/directory"


class TestOrganizeTypes:
    """Tests for organize-related types."""

    def test_organize_strategy_by_topic(self):
        """Test OrganizeStrategy.by_topic factory."""
        from ragfs import OrganizeStrategy

        strat = OrganizeStrategy.by_topic()
        assert strat.strategy_type == "by_topic"

    def test_organize_strategy_by_type(self):
        """Test OrganizeStrategy.by_type factory."""
        from ragfs import OrganizeStrategy

        strat = OrganizeStrategy.by_type()
        assert strat.strategy_type == "by_type"

    def test_organize_strategy_by_project(self):
        """Test OrganizeStrategy.by_project factory."""
        from ragfs import OrganizeStrategy

        strat = OrganizeStrategy.by_project()
        assert strat.strategy_type == "by_project"

    def test_organize_request(self):
        """Test OrganizeRequest creation."""
        from ragfs import OrganizeRequest, OrganizeStrategy

        strat = OrganizeStrategy.by_topic()
        request = OrganizeRequest(
            scope="./docs",
            strategy=strat,
            max_groups=5,
            similarity_threshold=0.8,
        )
        assert request.scope == "./docs"
        assert request.max_groups == 5
        assert request.similarity_threshold == 0.8


@pytest.mark.requires_model
class TestSafetyManager:
    """Tests for RagfsSafetyManager (requires model for some operations)."""

    def test_safety_manager_creation(self, source_path: Path, db_path: Path):
        """Test creating a SafetyManager."""
        from ragfs import RagfsSafetyManager

        safety = RagfsSafetyManager(str(source_path), str(db_path))
        assert safety is not None

    @pytest.mark.asyncio
    async def test_list_trash_empty(self, source_path: Path, db_path: Path):
        """Test listing trash when empty."""
        from ragfs import RagfsSafetyManager

        safety = RagfsSafetyManager(str(source_path), str(db_path))
        entries = await safety.list_trash()
        assert entries == []

    def test_get_history_empty(self, source_path: Path, db_path: Path):
        """Test getting history when empty."""
        from ragfs import RagfsSafetyManager

        safety = RagfsSafetyManager(str(source_path), str(db_path))
        history = safety.get_history()
        assert history == []


@pytest.mark.requires_model
class TestOpsManager:
    """Tests for RagfsOpsManager (requires model for indexing)."""

    def test_ops_manager_creation(self, source_path: Path, db_path: Path):
        """Test creating an OpsManager."""
        from ragfs import RagfsOpsManager

        ops = RagfsOpsManager(str(source_path), str(db_path))
        assert ops is not None

    @pytest.mark.asyncio
    async def test_create_file(self, source_path: Path, db_path: Path):
        """Test creating a file via OpsManager."""
        from ragfs import RagfsOpsManager

        ops = RagfsOpsManager(str(source_path), str(db_path))
        new_file = source_path / "new_file.txt"

        result = await ops.create_file(str(new_file), "Test content")
        assert result.success
        assert new_file.exists()
        assert new_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_mkdir(self, source_path: Path, db_path: Path):
        """Test creating a directory via OpsManager."""
        from ragfs import RagfsOpsManager

        ops = RagfsOpsManager(str(source_path), str(db_path))
        new_dir = source_path / "new_directory"

        result = await ops.mkdir(str(new_dir))
        assert result.success
        assert new_dir.is_dir()


@pytest.mark.requires_model
class TestSemanticManager:
    """Tests for RagfsSemanticManager (requires model)."""

    def test_semantic_manager_creation(self, source_path: Path, db_path: Path):
        """Test creating a SemanticManager."""
        from ragfs import RagfsSemanticManager

        semantic = RagfsSemanticManager(
            source_path=str(source_path),
            db_path=str(db_path),
        )
        assert semantic is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_semantic_manager_init(self, source_path: Path, db_path: Path):
        """Test initializing SemanticManager (loads model)."""
        from ragfs import RagfsSemanticManager

        semantic = RagfsSemanticManager(
            source_path=str(source_path),
            db_path=str(db_path),
        )
        await semantic.init()
        assert await semantic.is_available()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_list_pending_plans_empty(self, source_path: Path, db_path: Path):
        """Test listing pending plans when empty."""
        from ragfs import RagfsSemanticManager

        semantic = RagfsSemanticManager(
            source_path=str(source_path),
            db_path=str(db_path),
        )
        await semantic.init()

        plans = await semantic.list_pending_plans()
        assert plans == []
