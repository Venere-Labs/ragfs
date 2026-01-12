#!/usr/bin/env python3
"""
RAG Chain Example using ragfs and LangChain LCEL

This example demonstrates building a complete RAG (Retrieval Augmented Generation)
pipeline using ragfs components with LangChain Expression Language (LCEL).

Features demonstrated:
- Multi-format document loading (40+ formats including PDF, images)
- Code-aware text chunking with tree-sitter
- Local embeddings (GTE-small, 384 dimensions, no API calls)
- Hybrid search (vector + full-text)
- Configurable LLM providers (OpenAI, Anthropic, Ollama)
- Streaming responses

Requirements:
    pip install ragfs[langchain] langchain-openai  # or langchain-anthropic, langchain-ollama

Usage:
    # Index a directory of documents
    python langchain_rag_chain.py index ./docs --db ./my_db

    # Query with OpenAI (default)
    python langchain_rag_chain.py query "How does authentication work?" --db ./my_db

    # Query with Anthropic
    python langchain_rag_chain.py query "Explain the API" --provider anthropic

    # Query with local Ollama
    python langchain_rag_chain.py query "What is this project?" --provider ollama

    # Search only (no LLM)
    python langchain_rag_chain.py search "error handling" -k 5

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

# LangChain imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ragfs imports
from ragfs.langchain import (
    RagfsEmbeddings,
    RagfsLoader,
    RagfsRetriever,
    RagfsTextSplitter,
    RagfsVectorStore,
)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


def get_llm(provider: LLMProvider, model: Optional[str] = None):
    """Create an LLM instance based on provider.

    Args:
        provider: The LLM provider to use.
        model: Optional model name override.

    Returns:
        A LangChain chat model instance.

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
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install with: pip install langchain-openai"
            )

    elif provider == LLMProvider.ANTHROPIC:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model or "claude-sonnet-4-5-20250929", temperature=0)
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic provider. "
                "Install with: pip install langchain-anthropic"
            )

    elif provider == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model or "llama3.2")
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama provider. "
                "Install with: pip install langchain-ollama"
            )

    raise ValueError(f"Unknown provider: {provider}")


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents as context string.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        Formatted string with documents separated by dividers.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_path", doc.metadata.get("source", "unknown"))
        formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


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

    # Load documents (supports 40+ formats)
    loader = RagfsLoader(source_path)
    documents = await loader.aload()
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents found!")
        return

    # Split with code-aware chunking
    splitter = RagfsTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker_type=chunker_type,
    )
    chunks = await splitter.asplit_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Initialize embeddings
    print("Initializing embeddings (first run downloads ~45MB model)...")
    embeddings = RagfsEmbeddings()
    await embeddings.ainit()

    # Create vector store and add documents
    print(f"Indexing to: {db_path}")
    store = await RagfsVectorStore.afrom_documents(
        chunks,
        embeddings,
        db_path=db_path,
    )

    stats = await store._store.stats()
    print(f"Indexed {stats.get('total_chunks', 0)} chunks from {stats.get('total_files', 0)} files")
    print("Done!")


def create_rag_chain(retriever, llm):
    """Build a RAG chain using LCEL.

    Args:
        retriever: A LangChain retriever instance.
        llm: A LangChain chat model instance.

    Returns:
        A runnable LCEL chain.
    """
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context.
If you cannot answer from the context, say so clearly.
Be concise but thorough.

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


async def query_chain(
    question: str,
    db_path: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    k: int = 4,
    hybrid: bool = True,
    stream: bool = False,
) -> str:
    """Run a RAG query.

    Args:
        question: The question to ask.
        db_path: Path to the LanceDB database.
        provider: LLM provider to use.
        model: Optional model name override.
        k: Number of documents to retrieve.
        hybrid: Use hybrid search (vector + full-text).
        stream: Stream the response.

    Returns:
        The answer from the LLM.
    """
    # Initialize retriever
    retriever = RagfsRetriever(db_path, hybrid=hybrid, k=k)
    await retriever.ainit()

    # Get LLM
    llm = get_llm(provider, model)

    # Create chain
    chain = create_rag_chain(retriever, llm)

    # Run query
    if stream:
        print("\nAnswer: ", end="", flush=True)
        async for chunk in chain.astream(question):
            print(chunk, end="", flush=True)
        print("\n")
        return ""
    else:
        answer = await chain.ainvoke(question)
        return answer


async def search_only(
    query: str,
    db_path: str,
    k: int = 4,
    hybrid: bool = True,
) -> List[tuple]:
    """Search without LLM, showing retrieved documents.

    Args:
        query: Search query.
        db_path: Path to the LanceDB database.
        k: Number of results.
        hybrid: Use hybrid search.

    Returns:
        List of (document, score) tuples.
    """
    retriever = RagfsRetriever(db_path, hybrid=hybrid, k=k)
    await retriever.ainit()

    results = await retriever.asearch(query, k=k)
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG chain example using ragfs and LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents into the vector store",
    )
    index_parser.add_argument(
        "source",
        help="Path to file or directory to index",
    )
    index_parser.add_argument(
        "--db",
        default="./ragfs_db",
        help="Database path (default: ./ragfs_db)",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Chunk overlap in tokens (default: 64)",
    )
    index_parser.add_argument(
        "--chunker",
        default="auto",
        choices=["auto", "fixed", "code", "semantic"],
        help="Chunking strategy (default: auto)",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG chain with an LLM",
    )
    query_parser.add_argument(
        "question",
        help="Question to ask",
    )
    query_parser.add_argument(
        "--db",
        default="./ragfs_db",
        help="Database path (default: ./ragfs_db)",
    )
    query_parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider (default: openai)",
    )
    query_parser.add_argument(
        "--model",
        help="Model name override",
    )
    query_parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)",
    )
    query_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search (vector only)",
    )
    query_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search without LLM (show retrieved documents)",
    )
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--db",
        default="./ragfs_db",
        help="Database path (default: ./ragfs_db)",
    )
    search_parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of results (default: 4)",
    )
    search_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search",
    )

    args = parser.parse_args()

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
            print("Run 'python langchain_rag_chain.py index <path>' first.")
            return

        provider = LLMProvider(args.provider)
        answer = asyncio.run(
            query_chain(
                question=args.question,
                db_path=args.db,
                provider=provider,
                model=args.model,
                k=args.k,
                hybrid=not args.no_hybrid,
                stream=args.stream,
            )
        )
        if answer:
            print(f"\nAnswer: {answer}")

    elif args.command == "search":
        if not Path(args.db).exists():
            print(f"Error: Database not found at {args.db}")
            print("Run 'python langchain_rag_chain.py index <path>' first.")
            return

        results = asyncio.run(
            search_only(
                query=args.query,
                db_path=args.db,
                k=args.k,
                hybrid=not args.no_hybrid,
            )
        )

        print(f"\nFound {len(results)} results:\n")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("file_path", doc.metadata.get("source", "unknown"))
            print(f"[{i}] Score: {score:.4f}")
            print(f"    Source: {source}")
            print(f"    Content: {doc.page_content[:200]}...")
            print()


if __name__ == "__main__":
    main()
