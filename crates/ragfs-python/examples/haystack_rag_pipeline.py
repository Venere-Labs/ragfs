#!/usr/bin/env python3
"""
RAG Pipeline Example using ragfs and Haystack

This example demonstrates building a complete RAG (Retrieval Augmented Generation)
pipeline using ragfs components with Haystack's Pipeline API.

Features demonstrated:
- Multi-format document loading (40+ formats including PDF, images)
- Code-aware text chunking with tree-sitter
- Local embeddings (GTE-small, 384 dimensions, no API calls)
- Hybrid search (vector + full-text)
- Haystack Pipeline API for composable pipelines
- Configurable LLM providers (OpenAI, Anthropic, Ollama)
- Safety Layer (soft delete, undo, history)
- AI-Powered Organization (Propose-Review-Apply pattern)

Requirements:
    pip install ragfs[haystack] haystack-ai[openai]  # or [anthropic], [ollama]

Usage:
    # Index a directory of documents
    python haystack_rag_pipeline.py index ./docs --db ./my_db

    # Query with OpenAI (default)
    python haystack_rag_pipeline.py query "How does authentication work?" --db ./my_db

    # Query with Anthropic
    python haystack_rag_pipeline.py query "Explain the API" --provider anthropic

    # Query with local Ollama
    python haystack_rag_pipeline.py query "What is this project?" --provider ollama

    # Search only (no LLM)
    python haystack_rag_pipeline.py search "error handling" -k 5

    # Safety Layer commands
    python haystack_rag_pipeline.py delete ./docs/old_file.txt --db ./my_db
    python haystack_rag_pipeline.py trash --db ./my_db
    python haystack_rag_pipeline.py restore <undo_id> --db ./my_db
    python haystack_rag_pipeline.py history --db ./my_db

    # AI-Powered Organization (Propose-Review-Apply)
    python haystack_rag_pipeline.py organize ./docs --strategy by_topic
    python haystack_rag_pipeline.py pending --db ./my_db
    python haystack_rag_pipeline.py approve <plan_id> --db ./my_db
    python haystack_rag_pipeline.py reject <plan_id> --db ./my_db

Environment Variables:
    OPENAI_API_KEY      - Required for OpenAI provider
    ANTHROPIC_API_KEY   - Required for Anthropic provider
    (No key needed for Ollama - runs locally)
"""

from __future__ import annotations

import argparse
import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Load .env file if present (for API keys and defaults)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

# ragfs imports
from ragfs.haystack import (
    RagfsDocumentConverter,
    RagfsDocumentSplitter,
    RagfsDocumentEmbedder,
    RagfsDocumentStore,
    RagfsDocumentWriter,
    RagfsRetriever,
    RagfsTextEmbedder,
    RagfsSafeDocumentStore,
    RagfsOrganizer,
)

# Haystack imports
try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
except ImportError:
    raise ImportError(
        "haystack-ai is required. Install with: pip install ragfs[haystack]"
    )


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


def get_generator(provider: LLMProvider, model: Optional[str] = None):
    """Create a Haystack Generator based on provider.

    Args:
        provider: The LLM provider to use.
        model: Optional model name override.

    Returns:
        A Haystack Generator component.

    Raises:
        ImportError: If the provider's package is not installed.
        ValueError: If required API key is missing.
    """
    if provider == LLMProvider.OPENAI:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider"
            )
        try:
            from haystack.components.generators import OpenAIGenerator

            return OpenAIGenerator(model=model or "gpt-4o-mini")
        except ImportError:
            raise ImportError(
                "haystack-ai[openai] is required for OpenAI provider. "
                "Install with: pip install haystack-ai[openai]"
            )

    elif provider == LLMProvider.ANTHROPIC:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
        try:
            from haystack_integrations.components.generators.anthropic import (
                AnthropicGenerator,
            )

            return AnthropicGenerator(model=model or "claude-sonnet-4-5-20250929")
        except ImportError:
            raise ImportError(
                "haystack-anthropic is required for Anthropic provider. "
                "Install with: pip install anthropic-haystack"
            )

    elif provider == LLMProvider.OLLAMA:
        try:
            from haystack_integrations.components.generators.ollama import (
                OllamaGenerator,
            )

            return OllamaGenerator(model=model or "llama3.2")
        except ImportError:
            raise ImportError(
                "ollama-haystack is required for Ollama provider. "
                "Install with: pip install ollama-haystack"
            )

    raise ValueError(f"Unknown provider: {provider}")


def format_documents(documents: List[Document]) -> str:
    """Format retrieved documents as context string.

    Args:
        documents: List of Haystack Document objects.

    Returns:
        Formatted string with documents separated by dividers.
    """
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.meta.get("file_path", doc.meta.get("source", "unknown"))
        formatted.append(f"[{i}] Source: {source}\n{doc.content}")
    return "\n\n---\n\n".join(formatted)


def create_indexing_pipeline(
    db_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunker_type: str = "auto",
) -> Pipeline:
    """Create a Haystack pipeline for indexing documents.

    Args:
        db_path: Path to the LanceDB database.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks.
        chunker_type: Chunking strategy.

    Returns:
        Configured Haystack Pipeline.
    """
    # Create document store
    document_store = RagfsDocumentStore(db_path)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component("converter", RagfsDocumentConverter())
    pipeline.add_component(
        "splitter",
        RagfsDocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_by=chunker_type,
        ),
    )
    pipeline.add_component("embedder", RagfsDocumentEmbedder())
    pipeline.add_component("writer", RagfsDocumentWriter(document_store=document_store))

    # Connect components
    pipeline.connect("converter.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


def create_rag_pipeline(
    db_path: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    k: int = 4,
    hybrid: bool = True,
) -> Pipeline:
    """Create a Haystack RAG pipeline for querying.

    Args:
        db_path: Path to the LanceDB database.
        provider: LLM provider to use.
        model: Optional model name override.
        k: Number of documents to retrieve.
        hybrid: Use hybrid search.

    Returns:
        Configured Haystack RAG Pipeline.
    """
    # Create components
    retriever = RagfsRetriever(db_path, hybrid=hybrid, top_k=k)
    generator = get_generator(provider, model)

    # RAG prompt template
    prompt_template = """Answer the question based only on the following context.
If you cannot answer from the context, say so clearly.
Be concise but thorough.

Context:
{% for doc in documents %}
[{{ loop.index }}] Source: {{ doc.meta.get('file_path', 'unknown') }}
{{ doc.content }}
---
{% endfor %}

Question: {{ question }}

Answer:"""

    prompt_builder = PromptBuilder(template=prompt_template)

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    # Connect components
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")

    return pipeline


async def index_documents(
    source_path: str,
    db_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunker_type: str = "auto",
) -> None:
    """Load, chunk, and index documents from a directory.

    Args:
        source_path: Path to file or directory to index.
        db_path: Path to the LanceDB database.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        chunker_type: Chunking strategy: auto, fixed, code, or semantic.
    """
    print(f"Loading documents from: {source_path}")

    # Create and run indexing pipeline
    pipeline = create_indexing_pipeline(
        db_path=db_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker_type=chunker_type,
    )

    print("Running indexing pipeline (first run downloads ~45MB embedding model)...")
    result = pipeline.run({"converter": {"sources": [source_path]}})

    documents_written = result.get("writer", {}).get("documents_written", 0)
    print(f"Indexed {documents_written} document chunks")
    print(f"Index saved to: {db_path}")
    print("Done!")


async def query_rag(
    question: str,
    db_path: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    k: int = 4,
    hybrid: bool = True,
) -> str:
    """Run a RAG query using Haystack pipeline.

    Args:
        question: The question to ask.
        db_path: Path to the LanceDB database.
        provider: LLM provider to use.
        model: Optional model name override.
        k: Number of documents to retrieve.
        hybrid: Use hybrid search (vector + full-text).

    Returns:
        The answer from the LLM.
    """
    pipeline = create_rag_pipeline(
        db_path=db_path,
        provider=provider,
        model=model,
        k=k,
        hybrid=hybrid,
    )

    result = pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    # Extract answer from generator output
    replies = result.get("generator", {}).get("replies", [])
    if replies:
        return replies[0]
    return "No answer generated"


async def search_only(
    query: str,
    db_path: str,
    k: int = 4,
    hybrid: bool = True,
) -> None:
    """Search without LLM, showing retrieved documents.

    Args:
        query: Search query.
        db_path: Path to the LanceDB database.
        k: Number of results.
        hybrid: Use hybrid search.
    """
    retriever = RagfsRetriever(db_path, hybrid=hybrid, top_k=k)
    result = retriever.run(query=query)
    documents = result.get("documents", [])

    print(f"\nFound {len(documents)} results:\n")
    for i, doc in enumerate(documents, 1):
        source = doc.meta.get("file_path", doc.meta.get("source", "unknown"))
        score = doc.score or 0.0
        print(f"[{i}] Score: {score:.4f}")
        print(f"    Source: {source}")
        print(f"    Content: {doc.content[:200]}...")
        print()


# =============================================================================
# Safety Layer Commands
# =============================================================================


async def delete_document(
    file_path: str,
    db_path: str,
    source_path: Optional[str] = None,
) -> None:
    """Soft delete a document (can be undone).

    Args:
        file_path: Path to the file to delete.
        db_path: Path to the LanceDB database.
        source_path: Path to source directory for safety tracking.
    """
    store = RagfsSafeDocumentStore(
        db_path=db_path,
        safety_enabled=True,
        source_path=source_path or str(Path(db_path).parent),
    )

    result = store.safe_delete_documents([file_path])

    if result["soft_deleted"]:
        undo_ids = result["undo_ids"]
        print(f"Soft deleted: {file_path}")
        if undo_ids:
            print(f"Undo ID: {undo_ids.get(file_path)}")
            print("Use 'restore <undo_id>' to restore")
    else:
        print(f"Hard deleted: {file_path} (no undo available)")


async def show_trash(db_path: str, source_path: Optional[str] = None) -> None:
    """List all items in trash.

    Args:
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
    """
    store = RagfsSafeDocumentStore(
        db_path=db_path,
        safety_enabled=True,
        source_path=source_path or str(Path(db_path).parent),
    )

    trash = store.get_trash_contents()

    if not trash:
        print("Trash is empty.")
        return

    print(f"\nTrash contents ({len(trash)} items):\n")
    for entry in trash:
        print(f"  ID: {entry.id}")
        print(f"  Original: {entry.original_path}")
        print(f"  Deleted: {entry.deleted_at}")
        print()


async def restore_document(
    undo_id: str,
    db_path: str,
    source_path: Optional[str] = None,
) -> None:
    """Restore a soft-deleted document from trash.

    Args:
        undo_id: The undo ID from safe_delete_documents.
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
    """
    store = RagfsSafeDocumentStore(
        db_path=db_path,
        safety_enabled=True,
        source_path=source_path or str(Path(db_path).parent),
    )

    try:
        restored = store.restore_documents([undo_id])
        if restored:
            print(f"Restored: {restored[0]}")
        else:
            print("Nothing restored")
    except Exception as e:
        print(f"Failed to restore: {e}")


async def show_history(
    db_path: str,
    source_path: Optional[str] = None,
    limit: int = 10,
) -> None:
    """Show operation history.

    Args:
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
        limit: Maximum entries to show.
    """
    store = RagfsSafeDocumentStore(
        db_path=db_path,
        safety_enabled=True,
        source_path=source_path or str(Path(db_path).parent),
    )

    history = store.get_history(limit=limit)

    if not history:
        print("No history entries.")
        return

    print(f"\nOperation history (last {len(history)}):\n")
    for entry in history:
        can_undo = store.can_undo(entry.id)
        undo_status = "[UNDOABLE]" if can_undo else ""
        print(f"  ID: {entry.id} {undo_status}")
        print(f"  Operation: {entry.operation.operation_type}")
        print(f"  Timestamp: {entry.timestamp}")
        print()


# =============================================================================
# Organizer Commands (Propose-Review-Apply)
# =============================================================================


async def propose_organization(
    scope: str,
    db_path: str,
    source_path: Optional[str] = None,
    strategy: str = "by_topic",
    max_groups: int = 10,
) -> None:
    """Create an organization plan using Haystack pipeline.

    Args:
        scope: Directory scope to organize.
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
        strategy: Organization strategy.
        max_groups: Maximum number of groups.
    """
    src = source_path or str(Path(db_path).parent)
    organizer = RagfsOrganizer(src, db_path)

    print(f"Analyzing files in: {scope}")
    print(f"Strategy: {strategy}")

    # Use Haystack component interface
    result = organizer.run(
        operation="propose_organization",
        scope=scope,
        strategy=strategy,
        max_groups=max_groups,
    )

    plan = result.get("plan")
    if not plan:
        print("Failed to create plan")
        return

    print(f"\n=== Plan Created ===")
    print(f"Plan ID: {plan.id}")
    print(f"Status: {plan.status}")
    print(f"Actions: {len(plan.actions)}")
    print()

    for i, action in enumerate(plan.actions, 1):
        print(f"  [{i}] {action.action.action_type}")
        print(f"      Reason: {action.reason}")
        print()

    print("Use 'approve <plan_id>' to execute or 'reject <plan_id>' to discard")


async def list_pending_plans(
    db_path: str,
    source_path: Optional[str] = None,
) -> None:
    """List all pending plans.

    Args:
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
    """
    src = source_path or str(Path(db_path).parent)
    organizer = RagfsOrganizer(src, db_path)

    result = organizer.run(operation="list_pending")
    plans = result.get("pending_plans", [])

    if not plans:
        print("No pending plans.")
        return

    print(f"\nPending plans ({len(plans)}):\n")
    for plan in plans:
        print(f"  ID: {plan.id}")
        print(f"  Actions: {len(plan.actions)}")
        print(f"  Created: {plan.created_at}")
        print()


async def approve_plan(
    plan_id: str,
    db_path: str,
    source_path: Optional[str] = None,
) -> None:
    """Approve and execute a plan.

    Args:
        plan_id: The plan ID to approve.
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
    """
    src = source_path or str(Path(db_path).parent)
    organizer = RagfsOrganizer(src, db_path)

    print(f"Approving plan: {plan_id}")
    result = organizer.run(operation="approve", plan_id=plan_id)

    plan_result = result.get("result")
    if plan_result:
        print(f"Status: {plan_result.status}")
        if plan_result.status == "completed":
            print("All actions executed successfully!")
    else:
        print("Failed to approve plan")


async def reject_plan(
    plan_id: str,
    db_path: str,
    source_path: Optional[str] = None,
) -> None:
    """Reject and discard a plan.

    Args:
        plan_id: The plan ID to reject.
        db_path: Path to the LanceDB database.
        source_path: Path to source directory.
    """
    src = source_path or str(Path(db_path).parent)
    organizer = RagfsOrganizer(src, db_path)

    print(f"Rejecting plan: {plan_id}")
    result = organizer.run(operation="reject", plan_id=plan_id)

    plan_result = result.get("result")
    if plan_result:
        print(f"Status: {plan_result.status}")
        print("Plan discarded, no changes made.")
    else:
        print("Failed to reject plan")


def main():
    """CLI entry point."""
    # Get defaults from environment
    default_db = os.environ.get("RAGFS_DB_PATH", "./ragfs_db")
    default_provider = os.environ.get("RAGFS_DEFAULT_PROVIDER", "openai")
    default_model = os.environ.get("RAGFS_DEFAULT_MODEL") or None

    parser = argparse.ArgumentParser(
        description="RAG pipeline example using ragfs and Haystack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    db_arg = {"default": default_db, "help": f"Database path (default: {default_db})"}
    source_arg = {"default": None, "help": "Source directory for safety tracking"}

    # =========================================================================
    # Index command
    # =========================================================================
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("source", help="Path to file or directory to index")
    index_parser.add_argument("--db", **db_arg)
    index_parser.add_argument("--chunk-size", type=int, default=512)
    index_parser.add_argument("--chunk-overlap", type=int, default=64)
    index_parser.add_argument(
        "--chunker",
        default="auto",
        choices=["auto", "fixed", "code", "semantic"],
    )

    # =========================================================================
    # Query command
    # =========================================================================
    query_parser = subparsers.add_parser("query", help="Query with LLM")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--db", **db_arg)
    query_parser.add_argument(
        "--provider",
        default=default_provider,
        choices=["openai", "anthropic", "ollama"],
        help=f"LLM provider (default: {default_provider})",
    )
    query_parser.add_argument("--model", default=default_model, help="Model name override")
    query_parser.add_argument("-k", type=int, default=4, help="Documents to retrieve")
    query_parser.add_argument("--no-hybrid", action="store_true")

    # =========================================================================
    # Search command
    # =========================================================================
    search_parser = subparsers.add_parser("search", help="Search without LLM")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--db", **db_arg)
    search_parser.add_argument("-k", type=int, default=4)
    search_parser.add_argument("--no-hybrid", action="store_true")

    # =========================================================================
    # Safety Layer commands
    # =========================================================================
    delete_parser = subparsers.add_parser("delete", help="Soft delete a document")
    delete_parser.add_argument("file_path", help="File to delete")
    delete_parser.add_argument("--db", **db_arg)
    delete_parser.add_argument("--source", **source_arg)

    trash_parser = subparsers.add_parser("trash", help="List trash contents")
    trash_parser.add_argument("--db", **db_arg)
    trash_parser.add_argument("--source", **source_arg)

    restore_parser = subparsers.add_parser("restore", help="Restore from trash")
    restore_parser.add_argument("undo_id", help="Undo ID to restore")
    restore_parser.add_argument("--db", **db_arg)
    restore_parser.add_argument("--source", **source_arg)

    history_parser = subparsers.add_parser("history", help="Show operation history")
    history_parser.add_argument("--db", **db_arg)
    history_parser.add_argument("--source", **source_arg)
    history_parser.add_argument("--limit", type=int, default=10)

    # =========================================================================
    # Organizer commands (Propose-Review-Apply)
    # =========================================================================
    organize_parser = subparsers.add_parser(
        "organize", help="Propose file organization"
    )
    organize_parser.add_argument("scope", help="Directory scope to organize")
    organize_parser.add_argument("--db", **db_arg)
    organize_parser.add_argument("--source", **source_arg)
    organize_parser.add_argument(
        "--strategy",
        default="by_topic",
        choices=["by_topic", "by_type", "by_project"],
    )
    organize_parser.add_argument("--max-groups", type=int, default=10)

    pending_parser = subparsers.add_parser("pending", help="List pending plans")
    pending_parser.add_argument("--db", **db_arg)
    pending_parser.add_argument("--source", **source_arg)

    approve_parser = subparsers.add_parser("approve", help="Approve and execute a plan")
    approve_parser.add_argument("plan_id", help="Plan ID to approve")
    approve_parser.add_argument("--db", **db_arg)
    approve_parser.add_argument("--source", **source_arg)

    reject_parser = subparsers.add_parser("reject", help="Reject and discard a plan")
    reject_parser.add_argument("plan_id", help="Plan ID to reject")
    reject_parser.add_argument("--db", **db_arg)
    reject_parser.add_argument("--source", **source_arg)

    args = parser.parse_args()

    # =========================================================================
    # Command dispatch
    # =========================================================================
    if args.command == "index":
        asyncio.run(
            index_documents(
                source_path=args.source,
                db_path=args.db,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                chunker_type=args.chunker,
            )
        )

    elif args.command == "query":
        if not Path(args.db).exists():
            print(f"Error: Database not found at {args.db}")
            print("Run 'python haystack_rag_pipeline.py index <path>' first.")
            return

        provider = LLMProvider(args.provider)
        answer = asyncio.run(
            query_rag(
                question=args.question,
                db_path=args.db,
                provider=provider,
                model=args.model,
                k=args.k,
                hybrid=not args.no_hybrid,
            )
        )
        print(f"\nAnswer: {answer}")

    elif args.command == "search":
        if not Path(args.db).exists():
            print(f"Error: Database not found at {args.db}")
            return

        asyncio.run(
            search_only(
                query=args.query,
                db_path=args.db,
                k=args.k,
                hybrid=not args.no_hybrid,
            )
        )

    elif args.command == "delete":
        asyncio.run(
            delete_document(
                file_path=args.file_path,
                db_path=args.db,
                source_path=args.source,
            )
        )

    elif args.command == "trash":
        asyncio.run(show_trash(db_path=args.db, source_path=args.source))

    elif args.command == "restore":
        asyncio.run(
            restore_document(
                undo_id=args.undo_id,
                db_path=args.db,
                source_path=args.source,
            )
        )

    elif args.command == "history":
        asyncio.run(
            show_history(
                db_path=args.db,
                source_path=args.source,
                limit=args.limit,
            )
        )

    elif args.command == "organize":
        asyncio.run(
            propose_organization(
                scope=args.scope,
                db_path=args.db,
                source_path=args.source,
                strategy=args.strategy,
                max_groups=args.max_groups,
            )
        )

    elif args.command == "pending":
        asyncio.run(list_pending_plans(db_path=args.db, source_path=args.source))

    elif args.command == "approve":
        asyncio.run(
            approve_plan(
                plan_id=args.plan_id,
                db_path=args.db,
                source_path=args.source,
            )
        )

    elif args.command == "reject":
        asyncio.run(
            reject_plan(
                plan_id=args.plan_id,
                db_path=args.db,
                source_path=args.source,
            )
        )


if __name__ == "__main__":
    main()
