"""
RAGFS Chainlit Application v2

A professional chat interface for semantic document search with full file management.

Features:
- Semantic search with RAG
- File upload and automatic indexing
- File browser
- Trash management with restore
- Operation history with undo
- AI-powered organization (duplicates, cleanup, organize)
"""

import os
import re
from typing import List, Optional

import chainlit as cl

from rag_chain import RAGChain, get_sources
from file_manager import FileManager, DocumentInfo, UploadResult
from organizer import SemanticOrganizer
from components import (
    render_file_browser,
    render_trash_list,
    render_history,
    render_duplicates,
    render_cleanup_analysis,
    render_organization_plan,
    render_pending_plans,
    render_operation_result,
    show_help,
)


# Configuration
DB_PATH = os.environ.get("RAGFS_DB_PATH", "/data/index")
DOCUMENTS_PATH = os.environ.get("DOCUMENTS_PATH", "/data/docs")
TRASH_PATH = os.environ.get("RAGFS_TRASH_PATH", "/data/trash")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# Slash command patterns
COMMAND_PATTERNS = {
    "files": re.compile(r"^/files?\s*$", re.IGNORECASE),
    "trash": re.compile(r"^/trash\s*$", re.IGNORECASE),
    "history": re.compile(r"^/history\s*$", re.IGNORECASE),
    "duplicates": re.compile(r"^/duplicates?\s*$", re.IGNORECASE),
    "cleanup": re.compile(r"^/cleanup\s*$", re.IGNORECASE),
    "organize": re.compile(r"^/organize\s*(.*)$", re.IGNORECASE),
    "pending": re.compile(r"^/pending\s*$", re.IGNORECASE),
    "help": re.compile(r"^/help\s*$", re.IGNORECASE),
}


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with all managers."""
    await cl.Message(content="Initializing RAGFS...").send()

    # Initialize RAG chain
    rag_chain = RAGChain(
        db_path=DB_PATH,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_model=OLLAMA_MODEL,
        k=4,
        hybrid=True,
    )

    # Initialize File Manager
    file_manager = FileManager(
        documents_path=DOCUMENTS_PATH,
        db_path=DB_PATH,
        trash_path=TRASH_PATH,
    )

    # Initialize Semantic Organizer
    organizer = SemanticOrganizer(
        documents_path=DOCUMENTS_PATH,
        db_path=DB_PATH,
    )

    try:
        await rag_chain.initialize()
        await file_manager.initialize()
        # Organizer initializes lazily on first use

        cl.user_session.set("rag_chain", rag_chain)
        cl.user_session.set("file_manager", file_manager)
        cl.user_session.set("organizer", organizer)

        welcome = f"""Ready! Using **{OLLAMA_MODEL}** for answers.

**Quick Commands:**
- `/files` - List documents
- `/trash` - View deleted files
- `/history` - Operation history
- `/duplicates` - Find duplicates
- `/organize` - AI organization
- `/help` - All commands

**Or just ask a question** to search your documents semantically.

**Upload files** using the attachment button below."""

        await cl.Message(content=welcome).send()

    except Exception as e:
        await cl.Message(
            content=f"Error initializing: {str(e)}\n\nPlease ensure documents have been indexed.",
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages and slash commands."""
    text = message.content.strip()

    # Check for slash commands
    for cmd, pattern in COMMAND_PATTERNS.items():
        match = pattern.match(text)
        if match:
            await handle_command(cmd, match)
            return

    # Handle as RAG query
    await handle_query(text)


async def handle_command(command: str, match: re.Match):
    """Handle slash commands."""
    file_manager: FileManager = cl.user_session.get("file_manager")
    organizer: SemanticOrganizer = cl.user_session.get("organizer")

    try:
        if command == "files":
            documents = await file_manager.list_documents()
            msg = await render_file_browser(documents)
            await msg.send()

        elif command == "trash":
            entries = await file_manager.list_trash()
            msg = await render_trash_list(entries)
            await msg.send()

        elif command == "history":
            entries = await file_manager.get_history(limit=20)
            msg = await render_history(entries, file_manager)
            await msg.send()

        elif command == "duplicates":
            await cl.Message(content="Analyzing for duplicates...").send()
            groups = await organizer.find_duplicates()
            msg = await render_duplicates(groups)
            await msg.send()

        elif command == "cleanup":
            await cl.Message(content="Analyzing for cleanup candidates...").send()
            analysis = await organizer.analyze_cleanup()
            msg = await render_cleanup_analysis(analysis)
            await msg.send()

        elif command == "organize":
            strategy = match.group(1).strip() if match.group(1) else "by_topic"
            await cl.Message(content=f"Creating organization plan (strategy: {strategy})...").send()
            result = await organizer.propose_organization(strategy=strategy)
            if result.success and result.plan_id:
                plan = await organizer.get_plan(result.plan_id)
                if plan:
                    msg = await render_organization_plan(plan)
                    await msg.send()
                else:
                    await cl.Message(content=result.message).send()
            else:
                await cl.Message(content=result.message).send()

        elif command == "pending":
            plans = await organizer.list_pending_plans()
            msg = await render_pending_plans(plans)
            await msg.send()

        elif command == "help":
            msg = await show_help()
            await msg.send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()


async def handle_query(question: str):
    """Handle a RAG query."""
    rag_chain: RAGChain = cl.user_session.get("rag_chain")

    if not rag_chain:
        await cl.Message(content="RAG chain not initialized. Please refresh the page.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Get documents for sources
        docs = await rag_chain.retrieve(question)
        sources = get_sources(docs)

        # Stream response
        full_response = ""
        async for chunk in rag_chain.stream(question):
            full_response += chunk
            await msg.stream_token(chunk)

        await msg.update()

        # Show sources
        if sources:
            source_lines = ["\n\n---\n**Sources:**"]
            for i, source in enumerate(sources, 1):
                path = source.get("path", "Unknown")
                source_lines.append(f"{i}. `{path}`")

            await cl.Message(content="\n".join(source_lines)).send()

    except Exception as e:
        await msg.update()
        await cl.Message(content=f"Error: {str(e)}").send()


# =============================================================================
# File Upload Handler
# =============================================================================

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handle audio chunks (not used but required by Chainlit)."""
    pass


# Note: Chainlit uses on_message with message.elements for file uploads
# We need to check for file attachments in on_message

@cl.on_message
async def on_message_with_files(message: cl.Message):
    """This is merged with on_message above - files come as message.elements."""
    # Files are handled in the main on_message via message.elements
    pass


# Override on_message to handle files
_original_on_message = on_message

async def on_message(message: cl.Message):
    """Handle messages including file uploads."""
    # Check for file attachments
    if message.elements:
        file_manager: FileManager = cl.user_session.get("file_manager")
        if file_manager:
            for element in message.elements:
                if hasattr(element, 'path') and element.path:
                    # Read file content
                    try:
                        with open(element.path, 'rb') as f:
                            content = f.read()

                        name = element.name if hasattr(element, 'name') else os.path.basename(element.path)
                        result = await file_manager.upload(name, content)

                        msg = await render_operation_result(result, "Upload")
                        await msg.send()
                    except Exception as e:
                        await cl.Message(content=f"Failed to upload: {e}").send()

        # If there's also text, process it
        if message.content.strip():
            await _original_on_message(message)
        return

    # No files, handle normally
    await _original_on_message(message)


# =============================================================================
# Action Callbacks
# =============================================================================

@cl.action_callback("delete_file")
async def on_delete_file(action: cl.Action):
    """Handle delete file action."""
    file_manager: FileManager = cl.user_session.get("file_manager")
    if not file_manager:
        await cl.Message(content="File manager not initialized").send()
        return

    result = await file_manager.delete(action.value)
    msg = await render_operation_result(result, "Delete")
    await msg.send()


@cl.action_callback("restore_file")
async def on_restore_file(action: cl.Action):
    """Handle restore file action."""
    file_manager: FileManager = cl.user_session.get("file_manager")
    if not file_manager:
        await cl.Message(content="File manager not initialized").send()
        return

    result = await file_manager.restore(action.value)
    msg = await render_operation_result(result, "Restore")
    await msg.send()


@cl.action_callback("undo_operation")
async def on_undo_operation(action: cl.Action):
    """Handle undo operation action."""
    file_manager: FileManager = cl.user_session.get("file_manager")
    if not file_manager:
        await cl.Message(content="File manager not initialized").send()
        return

    result = await file_manager.undo(action.value)
    msg = await render_operation_result(result, "Undo")
    await msg.send()


@cl.action_callback("approve_plan")
async def on_approve_plan(action: cl.Action):
    """Handle approve plan action."""
    organizer: SemanticOrganizer = cl.user_session.get("organizer")
    if not organizer:
        await cl.Message(content="Organizer not initialized").send()
        return

    await cl.Message(content="Executing plan...").send()
    result = await organizer.approve_plan(action.value)
    msg = await render_operation_result(result, "Plan Execution")
    await msg.send()


@cl.action_callback("reject_plan")
async def on_reject_plan(action: cl.Action):
    """Handle reject plan action."""
    organizer: SemanticOrganizer = cl.user_session.get("organizer")
    if not organizer:
        await cl.Message(content="Organizer not initialized").send()
        return

    result = await organizer.reject_plan(action.value)
    msg = await render_operation_result(result, "Plan Rejection")
    await msg.send()


@cl.action_callback("view_plan")
async def on_view_plan(action: cl.Action):
    """Handle view plan action."""
    organizer: SemanticOrganizer = cl.user_session.get("organizer")
    if not organizer:
        await cl.Message(content="Organizer not initialized").send()
        return

    plan = await organizer.get_plan(action.value)
    if plan:
        msg = await render_organization_plan(plan)
        await msg.send()
    else:
        await cl.Message(content="Plan not found").send()


# =============================================================================
# Health Check
# =============================================================================

@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat session ends."""
    pass
