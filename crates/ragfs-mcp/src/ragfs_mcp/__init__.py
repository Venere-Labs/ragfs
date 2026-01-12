"""RAGFS MCP Server - Semantic filesystem for AI assistants.

This package provides an MCP (Model Context Protocol) server that exposes
RAGFS semantic filesystem capabilities to AI assistants like Claude.

RAGFS is NOT just a vector database - it's a FILESYSTEM for AI agents with:
- Safety Layer: Soft delete, audit history, undo support
- Semantic Operations: AI-powered file organization, similarity, deduplication
- Approval Workflow: Propose-Review-Apply pattern for safe AI agent operations

Tools exposed:

**Search & Discovery:**
- ragfs_search: Semantic search in indexed files
- ragfs_index_status: Get indexing status
- ragfs_similar: Find similar files
- ragfs_list_indices: List available indices

**Safety Layer (Reversible Operations):**
- ragfs_delete_to_trash: Soft delete with undo
- ragfs_list_trash: View deleted files
- ragfs_restore_from_trash: Restore deleted files
- ragfs_get_history: Audit log
- ragfs_undo: Undo any reversible operation

**Semantic Operations (AI-Powered):**
- ragfs_find_duplicates: Find duplicate files
- ragfs_analyze_cleanup: Analyze for cleanup

**Approval Workflow (Propose-Review-Apply):**
- ragfs_propose_organization: Create organization plan
- ragfs_propose_cleanup: Create cleanup plan
- ragfs_list_pending_plans: List pending plans
- ragfs_get_plan: Get plan details
- ragfs_approve_plan: Execute plan
- ragfs_reject_plan: Discard plan

**Structured Operations:**
- ragfs_batch_operations: Atomic batch with rollback

Usage:
    # Start the server
    python -m ragfs_mcp

    # Or use as a module
    from ragfs_mcp import create_server
    server = create_server()

Configuration via environment variables:
    RAGFS_DB_PATH: Path to the vector database (default: ~/.local/share/ragfs/indices/default)
    RAGFS_MODEL_PATH: Path to embedding model (default: ~/.local/share/ragfs/models)
    RAGFS_SOURCE_PATH: Source directory for file operations (default: current directory)
"""

from .server import create_server, main, mcp

__all__ = ["create_server", "main", "mcp"]
__version__ = "1.0.0"


def run() -> None:
    """Synchronous entry point for the MCP server."""
    import asyncio
    asyncio.run(main())
