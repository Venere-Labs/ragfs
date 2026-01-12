"""Haystack Embedder adapters for RAGFS."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ragfs import RagfsEmbeddings as CoreEmbeddings

try:
    from haystack import component, Document
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


@component
class HaystackRagfsTextEmbedder:
    """Haystack-compatible text embedder using RAGFS.

    Embeds a single string (typically a query).

    Example:
        from ragfs.haystack import RagfsTextEmbedder

        embedder = RagfsTextEmbedder()
        result = embedder.run(text="my query")
        embedding = result["embedding"]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """Initialize the text embedder.

        Args:
            model_path: Path to store/load model files.
            batch_size: Batch size for embedding.
            normalize: Whether to L2-normalize embeddings.
        """
        self._embedder = CoreEmbeddings(
            model_path=model_path,
            batch_size=batch_size,
            normalize=normalize,
        )
        self._initialized = False

    def warm_up(self) -> None:
        """Warm up the embedder (downloads model if needed)."""
        asyncio.get_event_loop().run_until_complete(self._embedder.init())
        self._initialized = True

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str) -> Dict[str, Any]:
        """Embed a text string.

        Args:
            text: The text to embed.

        Returns:
            Dictionary with 'embedding' (list of floats) and 'meta'.
        """
        if not self._initialized:
            self.warm_up()

        embedding = asyncio.get_event_loop().run_until_complete(
            self._embedder.embed_query(text)
        )

        return {
            "embedding": embedding,
            "meta": {
                "model": self._embedder.model_name,
                "dimension": self._embedder.dimension,
            },
        }


@component
class HaystackRagfsDocumentEmbedder:
    """Haystack-compatible document embedder using RAGFS.

    Embeds a list of documents.

    Example:
        from ragfs.haystack import RagfsDocumentEmbedder
        from haystack import Document

        embedder = RagfsDocumentEmbedder()
        docs = [Document(content="Hello world")]
        result = embedder.run(documents=docs)
        embedded_docs = result["documents"]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
    ):
        """Initialize the document embedder.

        Args:
            model_path: Path to store/load model files.
            batch_size: Batch size for embedding.
            normalize: Whether to L2-normalize embeddings.
            meta_fields_to_embed: Metadata fields to include in embedding.
        """
        self._embedder = CoreEmbeddings(
            model_path=model_path,
            batch_size=batch_size,
            normalize=normalize,
        )
        self._meta_fields = meta_fields_to_embed or []
        self._initialized = False

    def warm_up(self) -> None:
        """Warm up the embedder (downloads model if needed)."""
        asyncio.get_event_loop().run_until_complete(self._embedder.init())
        self._initialized = True

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """Embed documents.

        Args:
            documents: List of Haystack Documents to embed.

        Returns:
            Dictionary with 'documents' (with embeddings) and 'meta'.
        """
        if not self._initialized:
            self.warm_up()

        # Prepare texts
        texts = []
        for doc in documents:
            text_parts = [doc.content or ""]
            for field in self._meta_fields:
                if field in (doc.meta or {}):
                    text_parts.append(str(doc.meta[field]))
            texts.append(" ".join(text_parts))

        # Embed
        embeddings = asyncio.get_event_loop().run_until_complete(
            self._embedder.embed_documents(texts)
        )

        # Update documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return {
            "documents": documents,
            "meta": {
                "model": self._embedder.model_name,
                "dimension": self._embedder.dimension,
            },
        }
