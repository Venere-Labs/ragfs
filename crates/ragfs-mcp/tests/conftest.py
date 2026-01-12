"""Pytest configuration and fixtures for MCP server tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp(prefix="ragfs_mcp_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir: Path) -> Path:
    """Create sample files for testing."""
    (temp_dir / "doc1.txt").write_text(
        "Machine learning is a subset of artificial intelligence."
    )
    (temp_dir / "doc2.txt").write_text(
        "Deep learning is based on neural networks."
    )
    return temp_dir


@pytest.fixture
def db_path(temp_dir: Path) -> Path:
    """Get a path for the test database."""
    return temp_dir / "test_db"


@pytest.fixture
def source_path(sample_files: Path) -> Path:
    """Get the source path with sample files."""
    return sample_files


@pytest.fixture
def env_vars(source_path: Path, db_path: Path):
    """Set environment variables for MCP server."""
    old_source = os.environ.get("RAGFS_SOURCE_PATH")
    old_db = os.environ.get("RAGFS_DB_PATH")

    os.environ["RAGFS_SOURCE_PATH"] = str(source_path)
    os.environ["RAGFS_DB_PATH"] = str(db_path)

    yield

    # Restore
    if old_source:
        os.environ["RAGFS_SOURCE_PATH"] = old_source
    elif "RAGFS_SOURCE_PATH" in os.environ:
        del os.environ["RAGFS_SOURCE_PATH"]

    if old_db:
        os.environ["RAGFS_DB_PATH"] = old_db
    elif "RAGFS_DB_PATH" in os.environ:
        del os.environ["RAGFS_DB_PATH"]


def pytest_configure(config):
    """Add custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
