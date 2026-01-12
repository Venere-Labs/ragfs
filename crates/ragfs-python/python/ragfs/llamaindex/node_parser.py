"""LlamaIndex NodeParser adapter for RAGFS."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ragfs import RagfsTextSplitter as CoreSplitter

try:
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import BaseNode, Document, TextNode
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.utils import get_tqdm_iterable
except ImportError:
    raise ImportError(
        "llama-index-core is required for LlamaIndex integration. "
        "Install with: pip install ragfs[llamaindex]"
    )


class LlamaIndexRagfsNodeParser(NodeParser):
    """LlamaIndex-compatible node parser using RAGFS text splitter.

    Supports multiple chunking strategies:
    - "auto": Automatically select based on content type
    - "fixed": Token-based chunking with overlap
    - "code": AST-aware chunking for source code
    - "semantic": Structure-aware chunking on headings/paragraphs

    Example:
        from ragfs.llamaindex import RagfsNodeParser
        from llama_index.core import Document

        parser = RagfsNodeParser(chunk_size=512, chunker_type="auto")
        nodes = parser.get_nodes_from_documents([Document(text="...")])
    """

    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks in tokens")
    chunker_type: str = Field(
        default="auto",
        description="Chunking strategy: auto, fixed, code, semantic"
    )

    _splitter: Any = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunker_type: str = "auto",
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        **kwargs: Any,
    ):
        """Initialize the node parser.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
            chunker_type: Chunking strategy (auto, fixed, code, semantic).
            include_metadata: Whether to include metadata in nodes.
            include_prev_next_rel: Whether to include prev/next relationships.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_type=chunker_type,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            **kwargs,
        )
        self._splitter = CoreSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_type=chunker_type,
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "RagfsNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes into smaller chunks synchronously.

        Args:
            nodes: Sequence of nodes to parse.
            show_progress: Whether to show progress bar.

        Returns:
            List of parsed nodes.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._aparse_nodes(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aparse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes into smaller chunks asynchronously.

        Args:
            nodes: Sequence of nodes to parse.
            show_progress: Whether to show progress bar.

        Returns:
            List of parsed nodes.
        """
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing nodes"
        )

        for node in nodes_with_progress:
            parsed_nodes = await self._split_node(node)
            all_nodes.extend(parsed_nodes)

        return all_nodes

    async def _split_node(self, node: BaseNode) -> List[BaseNode]:
        """Split a single node into chunks.

        Args:
            node: Node to split.

        Returns:
            List of split nodes.
        """
        text = node.get_content()
        if not text:
            return [node]

        # Detect language from metadata for code chunking
        language = None
        if node.metadata:
            language = node.metadata.get("language")
            # Also check file extension
            file_path = node.metadata.get("file_path", "")
            if not language and file_path:
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                ext_to_lang = {
                    "py": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "rs": "rust",
                    "go": "go",
                    "java": "java",
                    "c": "c",
                    "cpp": "cpp",
                    "h": "c",
                    "hpp": "cpp",
                    "rb": "ruby",
                    "php": "php",
                    "swift": "swift",
                    "kt": "kotlin",
                    "scala": "scala",
                    "cs": "csharp",
                }
                language = ext_to_lang.get(ext)

        # Split the text
        chunks = await self._splitter.split_text(text, language=language)

        if len(chunks) <= 1:
            return [node]

        # Create nodes for each chunk
        result_nodes = []
        for i, chunk_text in enumerate(chunks):
            # Create metadata with chunk info
            metadata = dict(node.metadata) if node.metadata else {}
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(chunks)

            # Create new node
            new_node = TextNode(
                text=chunk_text,
                metadata=metadata,
                excluded_embed_metadata_keys=node.excluded_embed_metadata_keys if hasattr(node, "excluded_embed_metadata_keys") else [],
                excluded_llm_metadata_keys=node.excluded_llm_metadata_keys if hasattr(node, "excluded_llm_metadata_keys") else [],
            )

            # Set relationships
            if self.include_prev_next_rel:
                if i > 0 and result_nodes:
                    new_node.relationships["previous"] = result_nodes[-1].as_related_node_info()
                    result_nodes[-1].relationships["next"] = new_node.as_related_node_info()

            result_nodes.append(new_node)

        return result_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Get nodes from documents synchronously.

        Args:
            documents: Sequence of documents to parse.
            show_progress: Whether to show progress bar.

        Returns:
            List of parsed nodes.
        """
        # Convert documents to nodes first
        nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
        return self._parse_nodes(nodes, show_progress=show_progress, **kwargs)

    async def aget_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Get nodes from documents asynchronously.

        Args:
            documents: Sequence of documents to parse.
            show_progress: Whether to show progress bar.

        Returns:
            List of parsed nodes.
        """
        # Convert documents to nodes first
        nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
        return await self._aparse_nodes(nodes, show_progress=show_progress, **kwargs)
