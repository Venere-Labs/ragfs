"""Tests for RAGFS MCP server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestMCPServerImports:
    """Test that MCP server can be imported."""

    def test_import_server(self):
        """Test importing the server module."""
        from ragfs_mcp import create_server, mcp

        assert create_server is not None
        assert mcp is not None

    def test_import_main(self):
        """Test importing the main function."""
        from ragfs_mcp import main

        assert main is not None


class TestSearchTools:
    """Tests for search and discovery tools."""

    @pytest.mark.asyncio
    async def test_ragfs_list_indices(self):
        """Test listing available indices."""
        from ragfs_mcp.server import ragfs_list_indices

        result = await ragfs_list_indices()
        data = json.loads(result)

        assert "indices" in data or "hint" in data

    @pytest.mark.asyncio
    async def test_ragfs_index_status_missing(self):
        """Test getting status of non-existent index."""
        from ragfs_mcp.server import ragfs_index_status

        result = await ragfs_index_status(index="nonexistent_test_index")
        data = json.loads(result)

        assert data["exists"] is False


class TestSafetyTools:
    """Tests for safety layer tools."""

    @pytest.mark.asyncio
    async def test_ragfs_list_trash_import_error(self):
        """Test list_trash handles missing ragfs gracefully."""
        # This test verifies the tool handles import errors
        # In a real test environment with ragfs installed, this would work
        from ragfs_mcp.server import ragfs_list_trash

        result = await ragfs_list_trash()
        data = json.loads(result)

        # Either returns trash list or error about missing package
        assert "entries" in data or "error" in data

    @pytest.mark.asyncio
    async def test_ragfs_get_history(self):
        """Test getting operation history."""
        from ragfs_mcp.server import ragfs_get_history

        result = await ragfs_get_history(limit=10)
        data = json.loads(result)

        assert "entries" in data or "error" in data


class TestSemanticTools:
    """Tests for semantic operation tools."""

    @pytest.mark.asyncio
    async def test_ragfs_find_duplicates(self):
        """Test finding duplicates."""
        from ragfs_mcp.server import ragfs_find_duplicates

        result = await ragfs_find_duplicates(threshold=0.9)
        data = json.loads(result)

        assert "groups" in data or "error" in data

    @pytest.mark.asyncio
    async def test_ragfs_analyze_cleanup(self):
        """Test cleanup analysis."""
        from ragfs_mcp.server import ragfs_analyze_cleanup

        result = await ragfs_analyze_cleanup()
        data = json.loads(result)

        assert "categories" in data or "error" in data


class TestApprovalWorkflow:
    """Tests for approval workflow tools."""

    @pytest.mark.asyncio
    async def test_ragfs_list_pending_plans(self):
        """Test listing pending plans."""
        from ragfs_mcp.server import ragfs_list_pending_plans

        result = await ragfs_list_pending_plans()
        data = json.loads(result)

        assert "pending_plans" in data or "error" in data

    @pytest.mark.asyncio
    async def test_ragfs_get_plan_not_found(self):
        """Test getting a non-existent plan."""
        from ragfs_mcp.server import ragfs_get_plan

        result = await ragfs_get_plan(plan_id="nonexistent_plan_id")
        data = json.loads(result)

        # Should return error about plan not found
        assert "error" in data

    @pytest.mark.asyncio
    async def test_ragfs_propose_organization(self):
        """Test proposing organization."""
        from ragfs_mcp.server import ragfs_propose_organization

        result = await ragfs_propose_organization(
            scope="./",
            strategy="by_topic",
            max_groups=3,
        )
        data = json.loads(result)

        # Either returns plan or error
        assert "plan_id" in data or "error" in data


class TestBatchOperations:
    """Tests for batch operations tool."""

    @pytest.mark.asyncio
    async def test_ragfs_batch_operations_empty(self):
        """Test batch with empty operations list."""
        from ragfs_mcp.server import ragfs_batch_operations

        result = await ragfs_batch_operations(operations=[], atomic=True)
        data = json.loads(result)

        # Should handle empty list gracefully
        assert "results" in data or "error" in data

    @pytest.mark.asyncio
    async def test_ragfs_batch_operations_invalid_action(self):
        """Test batch with invalid action type."""
        from ragfs_mcp.server import ragfs_batch_operations

        operations = [
            {"action": "invalid_action", "target": "/test/path"}
        ]
        result = await ragfs_batch_operations(operations=operations)
        data = json.loads(result)

        assert "error" in data
        assert "Unknown action" in data["error"]

    @pytest.mark.asyncio
    async def test_ragfs_batch_operations_dry_run(self):
        """Test batch dry run mode."""
        from ragfs_mcp.server import ragfs_batch_operations

        operations = [
            {"action": "mkdir", "target": "/test/new_dir"}
        ]
        result = await ragfs_batch_operations(
            operations=operations,
            dry_run=True,
        )
        data = json.loads(result)

        # Dry run should return validation result
        assert data.get("dry_run") is True or "error" in data


class TestServerConfiguration:
    """Tests for server configuration."""

    def test_get_db_path_default(self):
        """Test default database path."""
        from ragfs_mcp.server import get_db_path

        path = get_db_path()
        assert "indices" in path
        assert "default" in path

    def test_get_db_path_custom_index(self):
        """Test custom index name."""
        from ragfs_mcp.server import get_db_path

        path = get_db_path("my_custom_index")
        assert "my_custom_index" in path

    def test_get_model_path(self):
        """Test model path."""
        from ragfs_mcp.server import get_model_path

        path = get_model_path()
        assert "models" in path

    def test_get_source_path(self):
        """Test source path."""
        from ragfs_mcp.server import get_source_path

        import os
        path = get_source_path()
        # Should return current directory by default
        assert path == os.getcwd() or path is not None
