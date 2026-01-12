"""RAGFS MCP Server implementation.

This module provides the MCP server that exposes RAGFS semantic filesystem
capabilities to AI assistants like Claude.

RAGFS is NOT just a vector database - it's a FILESYSTEM for AI agents with:
- Safety Layer: Soft delete, audit history, undo support
- Semantic Operations: AI-powered file organization, similarity, deduplication
- Approval Workflow: Propose-Review-Apply pattern for safe AI agent operations

Tools exposed:

**Search & Discovery:**
- ragfs_search: Semantic search in indexed files
- ragfs_index_status: Get indexing status for a directory
- ragfs_similar: Find files similar to a given file
- ragfs_list_indices: List all available indices

**Safety Layer (Reversible Operations):**
- ragfs_delete_to_trash: Soft delete with undo capability
- ragfs_list_trash: View all files in trash
- ragfs_restore_from_trash: Restore deleted files
- ragfs_get_history: Get operation audit log
- ragfs_undo: Undo any reversible operation

**Semantic Operations (AI-Powered):**
- ragfs_find_duplicates: Find duplicate/near-duplicate files
- ragfs_analyze_cleanup: Analyze files for potential cleanup

**Approval Workflow (Propose-Review-Apply):**
- ragfs_propose_organization: Create organization plan (NOT executed until approved)
- ragfs_propose_cleanup: Create cleanup plan (NOT executed until approved)
- ragfs_list_pending_plans: List pending plans for review
- ragfs_get_plan: Get full plan details
- ragfs_approve_plan: Approve and execute a plan
- ragfs_reject_plan: Reject and discard a plan

**Structured Operations:**
- ragfs_batch_operations: Execute multiple operations atomically
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    "ragfs",
    description="Semantic search filesystem for AI assistants",
)

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "ragfs"
DEFAULT_INDICES_DIR = DEFAULT_DATA_DIR / "indices"
DEFAULT_MODEL_PATH = DEFAULT_DATA_DIR / "models"


def get_db_path(index_name: str = "default") -> str:
    """Get the database path for an index."""
    db_path = os.environ.get("RAGFS_DB_PATH")
    if db_path:
        return db_path
    return str(DEFAULT_INDICES_DIR / index_name)


def get_model_path() -> str:
    """Get the model path."""
    return os.environ.get("RAGFS_MODEL_PATH", str(DEFAULT_MODEL_PATH))


@mcp.tool()
async def ragfs_search(
    query: str,
    index: str = "default",
    limit: int = 10,
    hybrid: bool = True,
) -> str:
    """Search for documents using semantic search.

    Performs vector similarity search combined with full-text search (hybrid)
    to find relevant content in indexed files.

    Args:
        query: The search query (natural language).
        index: Name of the index to search (default: "default").
        limit: Maximum number of results to return (default: 10).
        hybrid: Enable hybrid search combining vector and full-text (default: True).

    Returns:
        JSON string with search results including file paths, content snippets, and scores.
    """
    try:
        from ragfs import RagfsRetriever
    except ImportError:
        return '{"error": "ragfs package not installed. Install with: pip install ragfs"}'

    import json

    db_path = get_db_path(index)

    if not Path(db_path).exists():
        return json.dumps({
            "error": f"Index '{index}' not found at {db_path}",
            "hint": "Index a directory first using the ragfs CLI: ragfs index /path/to/dir",
        })

    try:
        retriever = RagfsRetriever(
            db_path=db_path,
            model_path=get_model_path(),
            hybrid=hybrid,
            k=limit,
        )
        await retriever.init()

        results = await retriever.search(query, hybrid=hybrid, k=limit)

        output = []
        for result in results:
            output.append({
                "file_path": result.document.metadata.get("file_path", "unknown"),
                "content": result.document.page_content[:500],  # Truncate for readability
                "score": round(result.score, 4),
                "metadata": result.document.metadata,
            })

        return json.dumps({
            "query": query,
            "index": index,
            "count": len(output),
            "results": output,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_index_status(index: str = "default") -> str:
    """Get the status of an index.

    Returns information about the indexed files, chunk count, and last update time.

    Args:
        index: Name of the index to check (default: "default").

    Returns:
        JSON string with index statistics.
    """
    import json

    db_path = get_db_path(index)
    db_dir = Path(db_path)

    if not db_dir.exists():
        return json.dumps({
            "exists": False,
            "index": index,
            "path": str(db_path),
            "hint": "Index a directory using: ragfs index /path/to/dir",
        })

    # Check for LanceDB files
    lance_files = list(db_dir.glob("*.lance")) + list(db_dir.glob("**/data/*.lance"))
    manifest_file = db_dir / "_latest.manifest"

    status = {
        "exists": True,
        "index": index,
        "path": str(db_path),
        "has_data": len(lance_files) > 0 or manifest_file.exists(),
    }

    # Get directory stats
    if db_dir.is_dir():
        total_size = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())
        status["size_bytes"] = total_size
        status["size_mb"] = round(total_size / (1024 * 1024), 2)

        # Get modification time
        if lance_files:
            latest = max(f.stat().st_mtime for f in lance_files)
            from datetime import datetime
            status["last_modified"] = datetime.fromtimestamp(latest).isoformat()

    return json.dumps(status, indent=2)


@mcp.tool()
async def ragfs_similar(
    file_path: str,
    index: str = "default",
    limit: int = 5,
) -> str:
    """Find files similar to a given file.

    Uses the indexed embeddings to find semantically similar files.

    Args:
        file_path: Path to the source file to find similar files for.
        index: Name of the index to search (default: "default").
        limit: Maximum number of similar files to return (default: 5).

    Returns:
        JSON string with similar files and their similarity scores.
    """
    import json

    source = Path(file_path)
    if not source.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    try:
        from ragfs import RagfsRetriever, RagfsDocumentLoader
    except ImportError:
        return '{"error": "ragfs package not installed"}'

    db_path = get_db_path(index)

    if not Path(db_path).exists():
        return json.dumps({"error": f"Index '{index}' not found"})

    try:
        # Load and extract content from the source file
        loader = RagfsDocumentLoader()
        await loader.init()
        docs = await loader.load(str(source))

        if not docs:
            return json.dumps({"error": "Could not extract content from file"})

        # Use the content as query
        content = docs[0].page_content[:2000]  # Limit query size

        # Search for similar
        retriever = RagfsRetriever(
            db_path=db_path,
            model_path=get_model_path(),
            hybrid=False,  # Pure vector search for similarity
            k=limit + 1,  # +1 to exclude self
        )
        await retriever.init()

        results = await retriever.search(content, hybrid=False, k=limit + 1)

        # Filter out the source file and format results
        similar = []
        source_abs = source.resolve()
        for result in results:
            result_path = result.document.metadata.get("file_path", "")
            if Path(result_path).resolve() != source_abs:
                similar.append({
                    "file_path": result_path,
                    "similarity": round(result.score, 4),
                    "preview": result.document.page_content[:200],
                })
            if len(similar) >= limit:
                break

        return json.dumps({
            "source_file": str(source),
            "similar_files": similar,
            "count": len(similar),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_list_indices() -> str:
    """List all available RAGFS indices.

    Scans the default indices directory for available indices.

    Returns:
        JSON string with list of available indices.
    """
    import json
    from datetime import datetime

    indices_dir = DEFAULT_INDICES_DIR

    if not indices_dir.exists():
        return json.dumps({
            "indices": [],
            "hint": "No indices found. Create one with: ragfs index /path/to/dir",
        })

    indices = []
    for item in indices_dir.iterdir():
        if item.is_dir():
            lance_files = list(item.glob("*.lance")) + list(item.glob("**/data/*.lance"))
            if lance_files or (item / "_latest.manifest").exists():
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                latest_mtime = max(
                    (f.stat().st_mtime for f in item.rglob("*") if f.is_file()),
                    default=0,
                )
                indices.append({
                    "name": item.name,
                    "path": str(item),
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "last_modified": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime else None,
                })

    return json.dumps({
        "indices_dir": str(indices_dir),
        "count": len(indices),
        "indices": indices,
    }, indent=2)


# =============================================================================
# SAFETY LAYER TOOLS - Reversible Operations
# =============================================================================


def get_source_path(index: str = "default") -> str:
    """Get the source directory path for safety/semantic operations."""
    source = os.environ.get("RAGFS_SOURCE_PATH")
    if source:
        return source
    # Default to current directory if not specified
    return os.getcwd()


@mcp.tool()
async def ragfs_delete_to_trash(
    path: str,
    index: str = "default",
) -> str:
    """Safely delete a file to trash (can be undone).

    Unlike permanent deletion, this moves the file to RAGFS trash
    and returns an undo_id that can be used to restore it.

    This is a core RAGFS safety feature for AI agent operations.

    Args:
        path: File path to delete (absolute or relative to source).
        index: Name of the index (default: "default").

    Returns:
        JSON with undo_id for restoration, or error message.
    """
    try:
        from ragfs import RagfsSafetyManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        safety = RagfsSafetyManager(source_path, db_path)
        trash_entry = await safety.delete_to_trash(path)

        return json.dumps({
            "success": True,
            "message": f"File moved to trash: {path}",
            "undo_id": trash_entry.id,
            "original_path": trash_entry.original_path,
            "deleted_at": trash_entry.deleted_at,
            "hint": f"Use ragfs_restore_from_trash with undo_id='{trash_entry.id}' to restore",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_list_trash(index: str = "default") -> str:
    """List all files in trash that can be restored.

    Shows all soft-deleted files with their undo IDs.

    Args:
        index: Name of the index (default: "default").

    Returns:
        JSON with list of trash entries.
    """
    try:
        from ragfs import RagfsSafetyManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        safety = RagfsSafetyManager(source_path, db_path)
        entries = await safety.list_trash()

        trash_list = []
        for entry in entries:
            trash_list.append({
                "undo_id": entry.id,
                "original_path": entry.original_path,
                "deleted_at": entry.deleted_at,
                "size_bytes": entry.size_bytes,
            })

        return json.dumps({
            "count": len(trash_list),
            "entries": trash_list,
            "hint": "Use ragfs_restore_from_trash to restore any file",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_restore_from_trash(
    undo_id: str,
    index: str = "default",
) -> str:
    """Restore a file from trash using its undo_id.

    Undoes a previous ragfs_delete_to_trash operation.

    Args:
        undo_id: The undo_id returned by ragfs_delete_to_trash.
        index: Name of the index (default: "default").

    Returns:
        JSON with restored file path, or error message.
    """
    try:
        from ragfs import RagfsSafetyManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        safety = RagfsSafetyManager(source_path, db_path)
        restored_path = await safety.restore_from_trash(undo_id)

        return json.dumps({
            "success": True,
            "message": "File restored from trash",
            "restored_path": restored_path,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_get_history(
    limit: int = 50,
    path: Optional[str] = None,
    index: str = "default",
) -> str:
    """Get operation history for audit trail.

    Shows all tracked operations with their undo IDs (if reversible).

    Args:
        limit: Maximum number of entries to return (default: 50).
        path: Optional filter by file path.
        index: Name of the index (default: "default").

    Returns:
        JSON with operation history entries.
    """
    try:
        from ragfs import RagfsSafetyManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        safety = RagfsSafetyManager(source_path, db_path)

        if path:
            entries = await safety.get_file_history(path)
        else:
            entries = safety.get_history(limit=limit)

        history_list = []
        for entry in entries:
            history_list.append({
                "id": entry.id,
                "timestamp": entry.timestamp,
                "operation_type": entry.operation.operation_type,
                "path": entry.operation.path,
                "reversible": entry.reversible,
            })

        return json.dumps({
            "count": len(history_list),
            "entries": history_list,
            "hint": "Use ragfs_undo with the entry id to undo reversible operations",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_undo(
    undo_id: str,
    index: str = "default",
) -> str:
    """Undo a previous operation by its ID.

    Works for any reversible operation tracked in history.

    Args:
        undo_id: The operation ID from history.
        index: Name of the index (default: "default").

    Returns:
        JSON with undo result, or error message.
    """
    try:
        from ragfs import RagfsSafetyManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        safety = RagfsSafetyManager(source_path, db_path)

        # Check if operation can be undone
        if not safety.can_undo(undo_id):
            return json.dumps({
                "success": False,
                "error": "Operation cannot be undone (not reversible or already undone)",
            })

        result = await safety.undo(undo_id)

        return json.dumps({
            "success": True,
            "message": result,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# SEMANTIC OPERATIONS TOOLS - AI-Powered Analysis
# =============================================================================


@mcp.tool()
async def ragfs_find_duplicates(
    threshold: float = 0.95,
    index: str = "default",
) -> str:
    """Find duplicate or near-duplicate files.

    Uses semantic embeddings to identify files with very similar content,
    even if they have different names or minor text differences.

    Args:
        threshold: Similarity threshold (0.0-1.0). Default 0.95 for near-exact duplicates.
        index: Name of the index (default: "default").

    Returns:
        JSON with duplicate groups and potential savings.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
            duplicate_threshold=threshold,
        )
        await semantic.init()

        duplicates = await semantic.find_duplicates()

        groups = []
        for group in duplicates.groups:
            groups.append({
                "files": group.files,
                "similarity": round(group.similarity, 4),
                "size_bytes": group.size_bytes,
            })

        return json.dumps({
            "threshold": threshold,
            "group_count": len(groups),
            "groups": groups,
            "total_potential_savings_bytes": duplicates.total_potential_savings,
            "hint": "Use ragfs_propose_cleanup to create a plan for handling duplicates",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_analyze_cleanup(index: str = "default") -> str:
    """Analyze files for potential cleanup.

    Identifies:
    - Duplicate files
    - Stale/unused files
    - Temporary files
    - Generated files that can be regenerated
    - Empty files

    Args:
        index: Name of the index (default: "default").

    Returns:
        JSON with cleanup candidates and potential savings.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        analysis = await semantic.analyze_cleanup()

        return json.dumps({
            "total_files_analyzed": analysis.total_files,
            "cleanup_candidates": analysis.candidates,
            "categories": {
                "duplicates": analysis.duplicate_count,
                "stale": analysis.stale_count,
                "temporary": analysis.temporary_count,
                "generated": analysis.generated_count,
                "empty": analysis.empty_count,
            },
            "potential_savings_bytes": analysis.potential_savings,
            "potential_savings_mb": round(analysis.potential_savings / (1024 * 1024), 2),
            "hint": "Use ragfs_propose_cleanup to create a cleanup plan for review",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# APPROVAL WORKFLOW TOOLS - Propose-Review-Apply Pattern
# =============================================================================


@mcp.tool()
async def ragfs_propose_organization(
    scope: str,
    strategy: str = "by_topic",
    max_groups: int = 10,
    similarity_threshold: float = 0.7,
    index: str = "default",
) -> str:
    """Create an organization plan (NOT executed until approved).

    This is the "Propose" phase of the Propose-Review-Apply pattern.
    The plan shows what WOULD happen, but no files are moved until approved.

    Strategies:
    - "by_topic": Group files by semantic similarity (topics)
    - "by_type": Group by file type/extension
    - "by_project": Group by detected project structure

    Args:
        scope: Directory scope to organize (relative to source root).
        strategy: Organization strategy (default: "by_topic").
        max_groups: Maximum number of groups to create (default: 10).
        similarity_threshold: Minimum similarity for grouping (0.0-1.0).
        index: Name of the index (default: "default").

    Returns:
        JSON with plan_id and proposed actions for review.
    """
    try:
        from ragfs import RagfsSemanticManager, OrganizeStrategy, OrganizeRequest
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        # Map strategy string to OrganizeStrategy
        if strategy == "by_topic":
            strat = OrganizeStrategy.by_topic()
        elif strategy == "by_type":
            strat = OrganizeStrategy.by_type()
        elif strategy == "by_project":
            strat = OrganizeStrategy.by_project()
        else:
            strat = OrganizeStrategy.by_topic()

        request = OrganizeRequest(
            scope=scope,
            strategy=strat,
            max_groups=max_groups,
            similarity_threshold=similarity_threshold,
        )

        plan = await semantic.create_organize_plan(request)

        actions = []
        for action in plan.actions:
            actions.append({
                "action_type": action.action.action_type,
                "source": action.action.source,
                "target": action.action.target,
                "reason": action.reason,
                "confidence": round(action.confidence, 2),
            })

        return json.dumps({
            "plan_id": plan.id,
            "status": plan.status,
            "description": plan.description,
            "action_count": len(actions),
            "actions": actions,
            "created_at": plan.created_at,
            "hint": f"Review the actions above. Use ragfs_approve_plan(plan_id='{plan.id}') to execute, or ragfs_reject_plan to discard.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_propose_cleanup(index: str = "default") -> str:
    """Create a cleanup plan for redundant/stale files (NOT executed until approved).

    Analyzes the indexed files and proposes cleanup actions like:
    - Removing duplicates (keeping one copy)
    - Deleting temporary files
    - Archiving stale files

    Args:
        index: Name of the index (default: "default").

    Returns:
        JSON with plan_id and proposed cleanup actions.
    """
    try:
        from ragfs import RagfsSemanticManager, OrganizeStrategy, OrganizeRequest
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        # Create a cleanup-focused plan
        request = OrganizeRequest(
            scope="./",
            strategy=OrganizeStrategy.by_topic(),
            max_groups=1,
            similarity_threshold=0.0,
        )

        plan = await semantic.create_organize_plan(request)

        actions = []
        for action in plan.actions:
            actions.append({
                "action_type": action.action.action_type,
                "source": action.action.source,
                "target": action.action.target,
                "reason": action.reason,
                "confidence": round(action.confidence, 2),
            })

        return json.dumps({
            "plan_id": plan.id,
            "status": plan.status,
            "description": "Cleanup plan for redundant and stale files",
            "action_count": len(actions),
            "actions": actions,
            "hint": f"Review the actions. Use ragfs_approve_plan(plan_id='{plan.id}') to execute.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_list_pending_plans(index: str = "default") -> str:
    """List all pending plans awaiting approval.

    Shows all proposed plans that haven't been approved or rejected yet.

    Args:
        index: Name of the index (default: "default").

    Returns:
        JSON with list of pending plans.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        plans = await semantic.list_pending_plans()

        pending = []
        for plan in plans:
            pending.append({
                "plan_id": plan.id,
                "status": plan.status,
                "description": plan.description,
                "action_count": len(plan.actions),
                "created_at": plan.created_at,
            })

        return json.dumps({
            "count": len(pending),
            "pending_plans": pending,
            "hint": "Use ragfs_get_plan to see full details, ragfs_approve_plan to execute, or ragfs_reject_plan to discard",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_get_plan(
    plan_id: str,
    index: str = "default",
) -> str:
    """Get full details of a plan including all proposed actions.

    Args:
        plan_id: The plan ID.
        index: Name of the index (default: "default").

    Returns:
        JSON with complete plan details.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        plan = await semantic.get_plan(plan_id)

        if plan is None:
            return json.dumps({"error": f"Plan not found: {plan_id}"})

        actions = []
        for action in plan.actions:
            actions.append({
                "action_type": action.action.action_type,
                "source": action.action.source,
                "target": action.action.target,
                "reason": action.reason,
                "confidence": round(action.confidence, 2),
            })

        return json.dumps({
            "plan_id": plan.id,
            "status": plan.status,
            "description": plan.description,
            "action_count": len(actions),
            "actions": actions,
            "created_at": plan.created_at,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_approve_plan(
    plan_id: str,
    index: str = "default",
) -> str:
    """Approve and execute a plan.

    This is the "Apply" phase of the Propose-Review-Apply pattern.
    All proposed actions are executed and become reversible via
    the safety layer (each action gets an undo_id).

    Args:
        plan_id: The plan ID to approve.
        index: Name of the index (default: "default").

    Returns:
        JSON with execution result and undo information.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        result = await semantic.approve_plan(plan_id)

        return json.dumps({
            "success": True,
            "plan_id": result.id,
            "status": result.status,
            "message": "Plan executed successfully",
            "action_count": len(result.actions),
            "hint": "All actions are reversible. Use ragfs_get_history to see undo IDs.",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ragfs_reject_plan(
    plan_id: str,
    index: str = "default",
) -> str:
    """Reject and discard a plan (no changes made).

    The plan is marked as rejected and no actions are executed.

    Args:
        plan_id: The plan ID to reject.
        index: Name of the index (default: "default").

    Returns:
        JSON confirming rejection.
    """
    try:
        from ragfs import RagfsSemanticManager
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=get_model_path(),
        )
        await semantic.init()

        result = await semantic.reject_plan(plan_id)

        return json.dumps({
            "success": True,
            "plan_id": result.id,
            "status": result.status,
            "message": "Plan rejected and discarded",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# STRUCTURED OPERATIONS TOOLS - Batch with Rollback
# =============================================================================


@mcp.tool()
async def ragfs_batch_operations(
    operations: List[Dict[str, Any]],
    atomic: bool = True,
    dry_run: bool = False,
    index: str = "default",
) -> str:
    """Execute multiple file operations atomically.

    All operations succeed or all fail (atomic mode).
    Each operation returns an undo_id for individual rollback.

    Operations format:
    [
        {"action": "create", "target": "/path/file.txt", "content": "..."},
        {"action": "move", "source": "/old/path", "target": "/new/path"},
        {"action": "copy", "source": "/src", "target": "/dst"},
        {"action": "delete", "target": "/path/to/delete"},
        {"action": "mkdir", "target": "/new/directory"},
    ]

    Args:
        operations: List of operation objects with action, source, target, content.
        atomic: If True, rollback all on any failure (default: True).
        dry_run: If True, validate without executing (default: False).
        index: Name of the index (default: "default").

    Returns:
        JSON with batch result and undo IDs.
    """
    try:
        from ragfs import RagfsOpsManager, Operation
    except ImportError:
        return json.dumps({"error": "ragfs package not installed"})

    source_path = get_source_path(index)
    db_path = get_db_path(index)

    try:
        ops = RagfsOpsManager(source_path, db_path)

        # Convert dict operations to Operation objects
        op_list = []
        for op in operations:
            action = op.get("action", "")
            source = op.get("source")
            target = op.get("target", "")
            content = op.get("content")

            # Create Operation based on action type
            if action == "create":
                op_obj = Operation.create(target, content or "")
            elif action == "move":
                op_obj = Operation.move(source or "", target)
            elif action == "copy":
                op_obj = Operation.copy(source or "", target)
            elif action == "delete":
                op_obj = Operation.delete(target)
            elif action == "mkdir":
                op_obj = Operation.mkdir(target)
            elif action == "write":
                append = op.get("append", False)
                op_obj = Operation.write(target, content or "", append)
            elif action == "symlink":
                op_obj = Operation.symlink(source or "", target)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

            op_list.append(op_obj)

        if dry_run:
            validation = await ops.dry_run(op_list)
            return json.dumps({
                "dry_run": True,
                "valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }, indent=2)

        result = await ops.batch(op_list, atomic=atomic)

        results = []
        for r in result.results:
            results.append({
                "success": r.success,
                "operation": r.operation,
                "path": r.path,
                "undo_id": r.undo_id,
                "error": r.error,
            })

        return json.dumps({
            "success": result.success,
            "atomic": atomic,
            "total_operations": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
            "rollback_id": result.rollback_id,
            "hint": "Use ragfs_undo with rollback_id to undo the entire batch, or individual undo_ids for specific operations",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def create_server() -> FastMCP:
    """Create and return the MCP server instance.

    This is the entry point used by MCP clients.
    """
    return mcp


async def main() -> None:
    """Run the MCP server using stdio transport.

    This is used when running the server directly:
        python -m ragfs_mcp
    """
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
