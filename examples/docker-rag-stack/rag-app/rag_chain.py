"""
RAG Chain module for RAGFS Docker Stack

This module provides the RAG pipeline using RAGFS components with LangChain.
Adapted from the langchain_rag_chain.py example for Docker deployment.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ragfs.langchain import RagfsEmbeddings, RagfsRetriever


# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the provided context.
Answer the question based only on the following context. If you cannot answer
from the context, say so clearly. Be concise but thorough.

When citing sources, reference them by their source path.

Context:
{context}

Question: {question}

Answer:"""
)


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


def get_sources(docs: List[Document]) -> List[dict]:
    """Extract source information from documents.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        List of source dictionaries with path and preview.
    """
    sources = []
    for doc in docs:
        source = doc.metadata.get("file_path", doc.metadata.get("source", "unknown"))
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        sources.append({
            "path": source,
            "preview": preview,
            "metadata": doc.metadata
        })
    return sources


class RAGChain:
    """RAG Chain wrapper for RAGFS with LangChain."""

    def __init__(
        self,
        db_path: str | None = None,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        k: int = 4,
        hybrid: bool = True,
    ):
        """Initialize the RAG chain.

        Args:
            db_path: Path to the RAGFS database. Defaults to RAGFS_DB_PATH env var.
            ollama_base_url: Ollama server URL. Defaults to OLLAMA_BASE_URL env var.
            ollama_model: Ollama model name. Defaults to OLLAMA_MODEL env var.
            k: Number of documents to retrieve.
            hybrid: Use hybrid search (vector + full-text).
        """
        self.db_path = db_path or os.environ.get("RAGFS_DB_PATH", "/data/index")
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.k = k
        self.hybrid = hybrid

        self.retriever: RagfsRetriever | None = None
        self.llm = None
        self.chain = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG chain components."""
        if self._initialized:
            return

        # Initialize retriever
        self.retriever = RagfsRetriever(
            self.db_path,
            hybrid=self.hybrid,
            k=self.k
        )
        await self.retriever.ainit()

        # Initialize LLM
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

        # Build the chain
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        self._initialized = True

    async def query(self, question: str) -> str:
        """Run a RAG query and return the answer.

        Args:
            question: The question to ask.

        Returns:
            The LLM's answer.
        """
        if not self._initialized:
            await self.initialize()

        return await self.chain.ainvoke(question)

    async def stream(self, question: str) -> AsyncIterator[str]:
        """Stream a RAG query response.

        Args:
            question: The question to ask.

        Yields:
            Chunks of the LLM's response.
        """
        if not self._initialized:
            await self.initialize()

        async for chunk in self.chain.astream(question):
            yield chunk

    async def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents without LLM.

        Args:
            query: The search query.

        Returns:
            List of relevant documents.
        """
        if not self._initialized:
            await self.initialize()

        return await self.retriever.ainvoke(query)

    async def query_with_sources(self, question: str) -> tuple[str, List[dict]]:
        """Run a RAG query and return answer with sources.

        Args:
            question: The question to ask.

        Returns:
            Tuple of (answer, sources).
        """
        if not self._initialized:
            await self.initialize()

        # Get documents first
        docs = await self.retriever.ainvoke(question)
        sources = get_sources(docs)

        # Build context and get answer
        context = format_docs(docs)
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = await self.llm.ainvoke(prompt)

        return answer.content, sources
