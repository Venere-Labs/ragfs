"""LlamaIndex adapters for RAGFS.

This module provides LlamaIndex-compatible wrappers for RAGFS components.

Example:
    from ragfs.llamaindex import RagfsEmbeddings, RagfsVectorStore, create_ragfs_index

    # Simple: Create index from documents
    index = await create_ragfs_index(
        db_path="./my_index",
        source_path="./docs",
    )
    query_engine = index.as_query_engine()
    response = query_engine.query("What is this about?")

    # Advanced: Use components individually
    embed_model = RagfsEmbeddings()
    vector_store = RagfsVectorStore("/path/to/db")
    await vector_store.ainit()

Safety Layer (soft delete, undo, history):
    from ragfs.llamaindex import RagfsSafeVectorStore

    store = RagfsSafeVectorStore("/path/to/db", safety_enabled=True)
    await store.ainit()

    # Soft delete with undo
    result = await store.safe_delete("document_id")
    await store.undo_delete(result["undo_id"])

AI-Powered Organization (Propose-Review-Apply pattern):
    from ragfs.llamaindex import RagfsOrganizer

    organizer = RagfsOrganizer("/path/to/source", "/path/to/db")
    await organizer.init()

    # Create organization plan (NOT executed until approved)
    plan = await organizer.propose_organization("./docs", strategy="by_topic")

    # Review and approve/reject
    for action in plan.actions:
        print(f"{action.action.action_type}: {action.reason}")
    await organizer.approve(plan.id)  # Execute
"""

from .embeddings import LlamaIndexRagfsEmbeddings as RagfsEmbeddings
from .retriever import LlamaIndexRagfsRetriever as RagfsRetriever
from .vectorstore import LlamaIndexRagfsVectorStore as RagfsVectorStore
from .node_parser import LlamaIndexRagfsNodeParser as RagfsNodeParser
from .reader import LlamaIndexRagfsReader as RagfsReader
from .index import create_ragfs_index, create_ragfs_index_sync
from .safe_vectorstore import LlamaIndexRagfsSafeVectorStore as RagfsSafeVectorStore
from .organizer import LlamaIndexRagfsOrganizer as RagfsOrganizer

__all__ = [
    # Embeddings
    "RagfsEmbeddings",
    # Vector Store
    "RagfsVectorStore",
    # Safe Vector Store (with safety layer)
    "RagfsSafeVectorStore",
    # Retriever
    "RagfsRetriever",
    # Node Parser
    "RagfsNodeParser",
    # Reader
    "RagfsReader",
    # Index Factory
    "create_ragfs_index",
    "create_ragfs_index_sync",
    # Semantic Organizer (Propose-Review-Apply)
    "RagfsOrganizer",
]
