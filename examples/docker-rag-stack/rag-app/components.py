"""
UI Components module for RAGFS Docker Stack

Provides reusable Chainlit UI components for file management.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import chainlit as cl

from file_manager import DocumentInfo, OperationResult
from organizer import (
    format_duplicate_groups,
    format_cleanup_analysis,
    format_plan,
    format_file_size,
)

# Try to import RAGFS types
try:
    from ragfs import TrashEntry, HistoryEntry, SemanticPlan, DuplicateGroups, CleanupAnalysis
except ImportError:
    TrashEntry = None
    HistoryEntry = None
    SemanticPlan = None
    DuplicateGroups = None
    CleanupAnalysis = None


async def render_file_browser(documents: List[DocumentInfo]) -> cl.Message:
    """Render a file browser showing indexed documents.

    Args:
        documents: List of DocumentInfo objects.

    Returns:
        Chainlit Message with file list and action buttons.
    """
    if not documents:
        return cl.Message(content="No documents found. Upload some files to get started!")

    # Build content
    lines = [f"**Documents** ({len(documents)} files)\n"]

    elements = []
    actions = []

    for i, doc in enumerate(documents):
        rel_path = doc.path
        try:
            docs_path = os.environ.get("DOCUMENTS_PATH", "/data/docs")
            rel_path = str(Path(doc.path).relative_to(docs_path))
        except ValueError:
            pass

        size = format_file_size(doc.size)
        modified = doc.modified.strftime("%Y-%m-%d %H:%M")

        lines.append(f"{i+1}. **{doc.name}** ({size})")
        lines.append(f"   `{rel_path}` - {modified}")

        # Add delete action for each file
        actions.append(cl.Action(
            name="delete_file",
            label=f"Delete {doc.name[:20]}",
            value=doc.path,
            description=f"Move {doc.name} to trash",
        ))

    content = "\n".join(lines)

    # Limit actions to first 10 files
    if len(actions) > 10:
        actions = actions[:10]
        content += f"\n\n*Showing delete buttons for first 10 files*"

    return cl.Message(content=content, actions=actions)


async def render_trash_list(entries: List) -> cl.Message:
    """Render the trash contents.

    Args:
        entries: List of TrashEntry objects.

    Returns:
        Chainlit Message with trash list and restore buttons.
    """
    if not entries:
        return cl.Message(content="Trash is empty.")

    lines = [f"**Trash** ({len(entries)} items)\n"]
    actions = []

    for entry in entries:
        entry_id = str(entry.id) if hasattr(entry, 'id') else str(entry)
        original = entry.original_path if hasattr(entry, 'original_path') else "Unknown"
        deleted_at = entry.deleted_at if hasattr(entry, 'deleted_at') else "Unknown"

        name = Path(original).name if original != "Unknown" else "Unknown"
        lines.append(f"- **{name}**")
        lines.append(f"  Original: `{original}`")
        lines.append(f"  Deleted: {deleted_at}")
        lines.append(f"  ID: `{entry_id}`")
        lines.append("")

        actions.append(cl.Action(
            name="restore_file",
            label=f"Restore {name[:15]}",
            value=entry_id,
            description=f"Restore {name} from trash",
        ))

    content = "\n".join(lines)

    # Limit actions
    if len(actions) > 10:
        actions = actions[:10]

    return cl.Message(content=content, actions=actions)


async def render_history(entries: List, file_manager=None) -> cl.Message:
    """Render operation history.

    Args:
        entries: List of HistoryEntry objects.
        file_manager: Optional FileManager for undo capability check.

    Returns:
        Chainlit Message with history and undo buttons.
    """
    if not entries:
        return cl.Message(content="No operation history.")

    lines = [f"**Operation History** (last {len(entries)})\n"]
    actions = []

    for entry in entries:
        entry_id = str(entry.id) if hasattr(entry, 'id') else str(entry)
        op_type = "Unknown"
        if hasattr(entry, 'operation'):
            op_type = entry.operation.operation_type if hasattr(entry.operation, 'operation_type') else str(entry.operation)
        timestamp = entry.timestamp if hasattr(entry, 'timestamp') else "Unknown"
        reversible = entry.reversible if hasattr(entry, 'reversible') else False

        status = "Reversible" if reversible else ""
        lines.append(f"- **{op_type}** {status}")
        lines.append(f"  Time: {timestamp}")
        lines.append(f"  ID: `{entry_id}`")
        lines.append("")

        if reversible:
            actions.append(cl.Action(
                name="undo_operation",
                label=f"Undo {op_type[:10]}",
                value=entry_id,
                description=f"Undo this {op_type} operation",
            ))

    content = "\n".join(lines)

    # Limit actions
    if len(actions) > 5:
        actions = actions[:5]

    return cl.Message(content=content, actions=actions)


async def render_duplicates(groups) -> cl.Message:
    """Render duplicate file groups.

    Args:
        groups: DuplicateGroups object.

    Returns:
        Chainlit Message with duplicate groups.
    """
    content = format_duplicate_groups(groups)
    return cl.Message(content=content)


async def render_cleanup_analysis(analysis) -> cl.Message:
    """Render cleanup analysis.

    Args:
        analysis: CleanupAnalysis object.

    Returns:
        Chainlit Message with cleanup candidates.
    """
    content = format_cleanup_analysis(analysis)
    return cl.Message(content=content)


async def render_organization_plan(plan) -> cl.Message:
    """Render an organization plan with approve/reject buttons.

    Args:
        plan: SemanticPlan object.

    Returns:
        Chainlit Message with plan details and action buttons.
    """
    content = format_plan(plan)

    plan_id = str(plan.id) if hasattr(plan, 'id') else ""

    actions = [
        cl.Action(
            name="approve_plan",
            label="Approve",
            value=plan_id,
            description="Execute this organization plan",
        ),
        cl.Action(
            name="reject_plan",
            label="Reject",
            value=plan_id,
            description="Discard this plan",
        ),
    ]

    return cl.Message(content=content, actions=actions)


async def render_pending_plans(plans: List) -> cl.Message:
    """Render list of pending plans.

    Args:
        plans: List of SemanticPlan objects.

    Returns:
        Chainlit Message with pending plans.
    """
    if not plans:
        return cl.Message(content="No pending organization plans.")

    lines = [f"**Pending Plans** ({len(plans)})\n"]
    actions = []

    for plan in plans:
        plan_id = str(plan.id) if hasattr(plan, 'id') else str(plan)
        action_count = len(plan.actions) if hasattr(plan, 'actions') else 0

        lines.append(f"- **Plan {plan_id[:8]}...**")
        lines.append(f"  Actions: {action_count}")
        lines.append("")

        actions.append(cl.Action(
            name="view_plan",
            label=f"View {plan_id[:8]}",
            value=plan_id,
            description="View plan details",
        ))

    content = "\n".join(lines)

    return cl.Message(content=content, actions=actions)


async def render_upload_result(result: OperationResult) -> cl.Message:
    """Render upload result.

    Args:
        result: OperationResult from upload.

    Returns:
        Chainlit Message with result.
    """
    if result.success:
        return cl.Message(content=f"**Upload successful**\n\n{result.message}")
    else:
        return cl.Message(content=f"**Upload failed**\n\n{result.message}\n\nError: {result.error}")


async def render_operation_result(result: OperationResult, operation: str = "Operation") -> cl.Message:
    """Render a generic operation result.

    Args:
        result: OperationResult object.
        operation: Name of the operation.

    Returns:
        Chainlit Message with result.
    """
    if result.success:
        content = f"**{operation} successful**\n\n{result.message}"
        if result.undo_id:
            content += f"\n\nUndo ID: `{result.undo_id}`"
        return cl.Message(content=content)
    else:
        return cl.Message(content=f"**{operation} failed**\n\n{result.message}\n\nError: {result.error}")


async def show_help() -> cl.Message:
    """Show help message with available commands.

    Returns:
        Chainlit Message with help content.
    """
    content = """# RAGFS File Manager Commands

## File Operations
- **Upload files** using the attachment button below the chat
- Type `/files` to list all indexed documents
- Type `/trash` to view deleted files
- Type `/history` to see operation history

## Semantic Operations
- Type `/duplicates` to find duplicate files
- Type `/cleanup` to analyze cleanup candidates
- Type `/organize` to create an AI organization plan
- Type `/pending` to view pending organization plans

## Search
- Just type your question to search documents semantically
- The AI will find relevant documents and answer your question

## Tips
- Deleted files are moved to trash and can be restored
- Organization plans require approval before execution
- New files are automatically indexed when uploaded
"""
    return cl.Message(content=content)
