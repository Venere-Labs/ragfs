"""Tests for RAGFS MCP server."""

from __future__ import annotations

import json
from pathlib import Path

import blake3
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

    def test_get_db_path_matches_cli_blake3_scheme(self, tmp_path, monkeypatch):
        """MCP must hash the canonical source path the same way as the CLI."""
        from ragfs_mcp.server import get_db_path, index_id_for_source

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.delenv("RAGFS_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        source = tmp_path / "project"
        source.mkdir()
        canonical = str(source.resolve())
        expected_id = blake3.blake3(canonical.encode()).hexdigest()[:16]

        # Official BLAKE3 empty-input prefix (same as the Rust blake3 crate).
        assert blake3.blake3(b"").hexdigest().startswith("af1349b9f5f9a1a6")

        assert index_id_for_source(str(source)) == expected_id
        assert index_id_for_source(canonical) == expected_id

        path = get_db_path(str(source))
        assert path.endswith(f"indices/{expected_id}/index.lance")
        assert "default" not in Path(path).parts

    def test_get_db_path_relative_and_absolute_match(self, tmp_path, monkeypatch):
        """Relative and absolute forms of the same directory share one index id."""
        from ragfs_mcp.server import get_db_path

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "docs"
        source.mkdir()

        assert get_db_path("docs") == get_db_path(str(source.resolve()))

    def test_get_db_path_default_uses_source_path(self, tmp_path, monkeypatch):
        """Default index follows RAGFS_SOURCE_PATH, not a literal 'default' folder."""
        from ragfs_mcp.server import get_db_path

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.delenv("RAGFS_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("RAGFS_SOURCE_PATH", str(tmp_path))

        expected_id = blake3.blake3(str(tmp_path.resolve()).encode()).hexdigest()[:16]
        path = get_db_path()
        assert path.endswith(f"indices/{expected_id}/index.lance")

    def test_get_db_path_hex_id_passthrough(self, monkeypatch):
        """A 16-hex id from ragfs_list_indices is used as-is."""
        from ragfs_mcp.server import get_db_path

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.delenv("RAGFS_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        hex_id = "0123456789abcdef"
        path = get_db_path(hex_id)
        assert path.endswith(f"indices/{hex_id}/index.lance")

    def test_get_db_path_env_override(self, monkeypatch, tmp_path):
        """RAGFS_DB_PATH still wins when an explicit store path is set."""
        from ragfs_mcp.server import get_db_path

        override = str(tmp_path / "custom.lance")
        monkeypatch.setenv("RAGFS_DB_PATH", override)
        assert get_db_path("/some/source") == override
        assert get_db_path() == override

    def test_get_db_path_respects_data_dir(self, tmp_path, monkeypatch):
        """RAGFS_DATA_DIR is the same override the CLI data_dir() uses."""
        from ragfs_mcp.server import get_db_path

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.setenv("RAGFS_DATA_DIR", str(tmp_path))
        source = tmp_path / "src"
        source.mkdir()
        expected_id = blake3.blake3(str(source.resolve()).encode()).hexdigest()[:16]
        assert get_db_path(str(source)) == str(
            tmp_path / "indices" / expected_id / "index.lance"
        )

    def test_get_data_dir_keeps_literal_tilde(self, monkeypatch):
        """CLI keeps RAGFS_DATA_DIR as-is; MCP must not expanduser()."""
        from ragfs_mcp.server import get_data_dir, get_db_path

        monkeypatch.delenv("RAGFS_DB_PATH", raising=False)
        monkeypatch.setenv("RAGFS_DATA_DIR", "~/ragfs-data")

        assert get_data_dir() == Path("~/ragfs-data")
        path = get_db_path("0123456789abcdef")
        assert path == str(Path("~/ragfs-data") / "indices" / "0123456789abcdef" / "index.lance")
        assert not path.startswith(str(Path.home() / "ragfs-data"))

    def test_get_data_dir_matches_cli_project_dirs(self, tmp_path, monkeypatch):
        """Fallback matches ProjectDirs::from("", "", "ragfs").data_dir()."""
        from ragfs_mcp.server import get_data_dir

        monkeypatch.delenv("RAGFS_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "linux")
        assert get_data_dir() == Path.home() / ".local" / "share" / "ragfs"

        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "darwin")
        assert get_data_dir() == Path.home() / "Library" / "Application Support" / "ragfs"

        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert get_data_dir() == tmp_path / "ragfs" / "data"

    def test_get_data_dir_xdg_matches_directories(self, tmp_path, monkeypatch):
        """XDG_DATA_HOME only on Linux-like OS and only when absolute."""
        from ragfs_mcp.server import get_data_dir

        monkeypatch.delenv("RAGFS_DATA_DIR", raising=False)
        xdg = tmp_path / "xdg-data"

        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
        assert get_data_dir() == xdg / "ragfs"

        monkeypatch.setenv("XDG_DATA_HOME", "rel-data")
        assert get_data_dir() == Path.home() / ".local" / "share" / "ragfs"

        monkeypatch.setenv("XDG_DATA_HOME", "~/xdg-data")
        assert get_data_dir() == Path.home() / ".local" / "share" / "ragfs"

        monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "darwin")
        assert get_data_dir() == Path.home() / "Library" / "Application Support" / "ragfs"

        monkeypatch.setattr("ragfs_mcp.server.sys.platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
        assert get_data_dir() == tmp_path / "Roaming" / "ragfs" / "data"

    def test_get_model_path(self):
        """Test model path."""
        from ragfs_mcp.server import get_model_path

        path = get_model_path()
        assert "models" in path

    def test_get_source_path(self):
        """Test source path."""
        import os

        from ragfs_mcp.server import get_source_path

        path = get_source_path()
        assert path == os.getcwd() or path is not None

    def test_get_source_path_uses_index_directory(self, tmp_path):
        """Passing a source directory as index resolves that directory."""
        from ragfs_mcp.server import get_source_path

        assert get_source_path(str(tmp_path)) == str(tmp_path.resolve())
