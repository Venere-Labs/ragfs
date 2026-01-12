"""Pytest configuration and fixtures for RAGFS Python tests."""

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
    temp = tempfile.mkdtemp(prefix="ragfs_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir: Path) -> Path:
    """Create sample files for testing."""
    # Create some sample text files
    (temp_dir / "doc1.txt").write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data and improve from experience."
    )
    (temp_dir / "doc2.txt").write_text(
        "Deep learning is a type of machine learning based on neural networks. "
        "It can learn complex patterns from large amounts of data."
    )
    (temp_dir / "doc3.txt").write_text(
        "Natural language processing enables computers to understand human language. "
        "It's a key component of many AI applications."
    )
    (temp_dir / "readme.md").write_text(
        "# Test Project\n\n"
        "This is a sample project for testing RAGFS functionality.\n"
    )

    # Create a subdirectory with more files
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "code.py").write_text(
        "def hello():\n    print('Hello, world!')\n"
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


# Mark slow tests
def pytest_configure(config):
    """Add custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_model: requires embedding model")


# Skip tests if ragfs is not built
def pytest_collection_modifyitems(config, items):
    """Skip tests if ragfs module is not available."""
    try:
        import ragfs
    except ImportError:
        skip_marker = pytest.mark.skip(
            reason="ragfs module not built. Run: maturin develop"
        )
        for item in items:
            item.add_marker(skip_marker)
