"""RAGFS - Local semantic search and RAG pipeline.

This package provides Python bindings for RAGFS, a high-performance
semantic search and RAG (Retrieval Augmented Generation) pipeline
written in Rust.

Features:
- Local embeddings using GTE-small (384 dimensions, no API calls)
- Vector storage with LanceDB (hybrid search support)
- Multi-format document loading (40+ text formats, PDF, images)
- Code-aware text splitting with tree-sitter
- Framework adapters for LangChain, LlamaIndex, and Haystack
- **FUSE filesystem capabilities for AI agent operations**

Quick Start:
    from ragfs import RagfsRetriever

    # Initialize
    retriever = RagfsRetriever("/path/to/db")
    await retriever.init()

    # Search
    docs = await retriever.get_relevant_documents("my query")

Using with LangChain:
    from ragfs.langchain import RagfsEmbeddings, RagfsVectorStore

    embeddings = RagfsEmbeddings()
    await embeddings.init()

    vectorstore = RagfsVectorStore.from_embeddings(embeddings, "/path/to/db")

Safety Layer (soft delete, undo, history):
    from ragfs import RagfsSafetyManager

    safety = RagfsSafetyManager("/path/to/source")
    entry = await safety.delete_to_trash("/path/to/file.txt")
    await safety.restore_from_trash(entry.id)

AI-Powered Organization (Propose-Review-Apply pattern):
    from ragfs import RagfsSemanticManager, OrganizeRequest, OrganizeStrategy

    semantic = RagfsSemanticManager("/path/to/source", "/path/to/db")
    await semantic.init()

    # Create an organization plan (NOT executed until approved)
    request = OrganizeRequest("./docs", OrganizeStrategy.by_topic())
    plan = await semantic.create_organize_plan(request)

    # Review the plan
    for action in plan.actions:
        print(f"{action.action} - {action.reason}")

    # Approve or reject
    await semantic.approve_plan(plan.id)  # Execute
    # OR: await semantic.reject_plan(plan.id)  # Discard
"""

from ragfs._core import (
    # Core types
    Document,
    SearchResultPy as SearchResult,
    PyChunk,
    # Core components
    RagfsEmbeddings,
    RagfsVectorStore,
    RagfsDocumentLoader,
    RagfsTextSplitter,
    RagfsRetriever,
    # Safety layer (soft delete, undo, history)
    RagfsSafetyManager,
    TrashEntry,
    HistoryEntry,
    HistoryOperation,
    # Semantic operations (AI-powered file organization)
    RagfsSemanticManager,
    OrganizeStrategy,
    OrganizeRequest,
    SemanticPlan,
    PlanAction,
    PlanImpact,
    SimilarFile,
    SimilarFilesResult,
    DuplicateEntry,
    DuplicateGroup,
    DuplicateGroups,
    CleanupCandidate,
    CleanupAnalysis,
    # Operations manager (structured file ops with JSON feedback)
    RagfsOpsManager,
    Operation,
    OperationResult,
    BatchResult,
)

__all__ = [
    # Core types
    "Document",
    "SearchResult",
    "PyChunk",
    # Core components
    "RagfsEmbeddings",
    "RagfsVectorStore",
    "RagfsDocumentLoader",
    "RagfsTextSplitter",
    "RagfsRetriever",
    # Safety layer
    "RagfsSafetyManager",
    "TrashEntry",
    "HistoryEntry",
    "HistoryOperation",
    # Semantic operations
    "RagfsSemanticManager",
    "OrganizeStrategy",
    "OrganizeRequest",
    "SemanticPlan",
    "PlanAction",
    "PlanImpact",
    "SimilarFile",
    "SimilarFilesResult",
    "DuplicateEntry",
    "DuplicateGroup",
    "DuplicateGroups",
    "CleanupCandidate",
    "CleanupAnalysis",
    # Operations manager
    "RagfsOpsManager",
    "Operation",
    "OperationResult",
    "BatchResult",
]

__version__ = "0.2.0"
