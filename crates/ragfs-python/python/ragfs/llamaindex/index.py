"""LlamaIndex index factory for RAGFS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.schema import Document
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )

from .vectorstore import LlamaIndexRagfsVectorStore
from .embeddings import LlamaIndexRagfsEmbeddings
from .node_parser import LlamaIndexRagfsNodeParser
from .reader import LlamaIndexRagfsReader


async def create_ragfs_index(
    db_path: str,
    documents: Optional[List[Document]] = None,
    source_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunker_type: str = "auto",
    show_progress: bool = True,
    **kwargs: Any,
) -> VectorStoreIndex:
    """Create a LlamaIndex VectorStoreIndex using RAGFS components.

    This factory function creates a complete index with:
    - Local embeddings (GTE-small, 384 dimensions)
    - LanceDB vector storage
    - Code/semantic-aware chunking

    Example:
        from ragfs.llamaindex import create_ragfs_index
        from llama_index.core import Document

        # Create from documents
        index = await create_ragfs_index(
            db_path="./my_index",
            documents=[Document(text="Hello world")],
        )

        # Or load from directory
        index = await create_ragfs_index(
            db_path="./my_index",
            source_path="./docs",
        )

        # Query
        query_engine = index.as_query_engine()
        response = query_engine.query("What is this about?")

    Args:
        db_path: Path to the LanceDB database.
        documents: Optional list of documents to index.
        source_path: Optional path to load documents from.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        chunker_type: Chunking strategy (auto, fixed, code, semantic).
        show_progress: Whether to show progress bars.
        **kwargs: Additional arguments passed to VectorStoreIndex.

    Returns:
        Initialized VectorStoreIndex ready for querying.
    """
    # Initialize components
    embed_model = LlamaIndexRagfsEmbeddings()
    vector_store = LlamaIndexRagfsVectorStore(db_path=db_path)
    node_parser = LlamaIndexRagfsNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker_type=chunker_type,
    )

    # Initialize vector store
    await vector_store.ainit()

    # Load documents if source_path provided
    all_documents = list(documents) if documents else []

    if source_path:
        reader = LlamaIndexRagfsReader()
        loaded_docs = await reader.aload_data(source_path)
        all_documents.extend(loaded_docs)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    if all_documents:
        index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[node_parser],
            show_progress=show_progress,
            **kwargs,
        )
    else:
        # Create empty index from existing store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
            **kwargs,
        )

    return index


def create_ragfs_index_sync(
    db_path: str,
    documents: Optional[List[Document]] = None,
    source_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunker_type: str = "auto",
    show_progress: bool = True,
    **kwargs: Any,
) -> VectorStoreIndex:
    """Synchronous version of create_ragfs_index.

    See create_ragfs_index for documentation.
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        create_ragfs_index(
            db_path=db_path,
            documents=documents,
            source_path=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_type=chunker_type,
            show_progress=show_progress,
            **kwargs,
        )
    )
