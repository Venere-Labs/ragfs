"""Haystack semantic organizer component for AI-powered file operations."""

from __future__ import annotations

from typing import Dict, List, Optional

from ragfs import (
    RagfsSemanticManager,
    OrganizeStrategy,
    OrganizeRequest,
    SemanticPlan,
    SimilarFile,
    SimilarFilesResult,
    DuplicateGroups,
    CleanupAnalysis,
)

try:
    from haystack import component
except ImportError:
    raise ImportError(
        "haystack-ai is required for Haystack integration. "
        "Install with: pip install ragfs[haystack]"
    )


@component
class HaystackRagfsOrganizer:
    """AI-powered file organization component with approval workflow.

    This component exposes RAGFS's semantic operations to Haystack pipelines,
    implementing the Propose-Review-Apply pattern for safe AI agent operations.

    **Key Pattern: Propose-Review-Apply**
    1. Agents propose operations (organization, cleanup, deduplication)
    2. Plans are created but NOT executed
    3. Users/agents review the proposed actions
    4. Approve to execute, or reject to discard

    Example:
        from haystack import Pipeline
        from ragfs.haystack import RagfsOrganizer

        # Create organizer component
        organizer = RagfsOrganizer("/path/to/source", "/path/to/db")

        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_component("organizer", organizer)

        # Find similar files
        result = pipeline.run({
            "organizer": {
                "operation": "find_similar",
                "file_path": "/path/to/file.txt",
                "k": 5
            }
        })

        # Propose organization (NOT executed yet!)
        result = pipeline.run({
            "organizer": {
                "operation": "propose_organization",
                "scope": "./docs",
                "strategy": "by_topic"
            }
        })
        plan = result["organizer"]["plan"]

        # Approve (executes the plan)
        result = pipeline.run({
            "organizer": {
                "operation": "approve",
                "plan_id": plan.id
            }
        })
    """

    def __init__(
        self,
        source_path: str,
        db_path: str,
        model_path: Optional[str] = None,
        duplicate_threshold: float = 0.95,
        similar_limit: int = 10,
    ):
        """Initialize the organizer.

        Args:
            source_path: Path to the source directory to manage.
            db_path: Path to the LanceDB database.
            model_path: Optional path for embedding models.
            duplicate_threshold: Similarity threshold for duplicates (0.0-1.0).
            similar_limit: Max results for similar file search.
        """
        self._source_path = source_path
        self._db_path = db_path
        self._model_path = model_path
        self._duplicate_threshold = duplicate_threshold
        self._similar_limit = similar_limit
        self._semantic: Optional[RagfsSemanticManager] = None
        self._initialized = False

    def warm_up(self) -> None:
        """Warm up the component by initializing the semantic manager."""
        if not self._initialized:
            import asyncio
            asyncio.get_event_loop().run_until_complete(self._init())

    async def _init(self) -> None:
        """Initialize the semantic manager."""
        if not self._initialized:
            self._semantic = RagfsSemanticManager(
                source_path=self._source_path,
                db_path=self._db_path,
                model_path=self._model_path,
                duplicate_threshold=self._duplicate_threshold,
                similar_limit=self._similar_limit,
            )
            await self._semantic.init()
            self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the component is initialized."""
        if not self._initialized:
            import asyncio
            asyncio.get_event_loop().run_until_complete(self._init())

    @component.output_types(
        similar_files=Optional[SimilarFilesResult],
        duplicates=Optional[DuplicateGroups],
        cleanup_analysis=Optional[CleanupAnalysis],
        plan=Optional[SemanticPlan],
        pending_plans=Optional[List[SemanticPlan]],
        result=Optional[SemanticPlan],
    )
    def run(
        self,
        operation: str,
        file_path: Optional[str] = None,
        k: Optional[int] = None,
        scope: Optional[str] = None,
        strategy: Optional[str] = None,
        max_groups: int = 10,
        similarity_threshold: float = 0.7,
        plan_id: Optional[str] = None,
    ) -> Dict:
        """Execute an organizer operation.

        Args:
            operation: Operation to perform. One of:
                - "find_similar": Find similar files
                - "find_duplicates": Find duplicate files
                - "analyze_cleanup": Analyze for cleanup
                - "propose_organization": Create organization plan
                - "propose_cleanup": Create cleanup plan
                - "list_pending": List pending plans
                - "get_plan": Get a specific plan
                - "approve": Approve and execute a plan
                - "reject": Reject a plan
            file_path: Path for find_similar operation.
            k: Number of results for find_similar.
            scope: Directory scope for propose_organization.
            strategy: Strategy for propose_organization ("by_topic", "by_type", "by_project").
            max_groups: Max groups for propose_organization.
            similarity_threshold: Threshold for propose_organization.
            plan_id: Plan ID for approve/reject/get_plan.

        Returns:
            Dictionary with operation-specific results.
        """
        import asyncio

        self._ensure_initialized()

        return asyncio.get_event_loop().run_until_complete(
            self._async_run(
                operation=operation,
                file_path=file_path,
                k=k,
                scope=scope,
                strategy=strategy,
                max_groups=max_groups,
                similarity_threshold=similarity_threshold,
                plan_id=plan_id,
            )
        )

    async def _async_run(
        self,
        operation: str,
        file_path: Optional[str],
        k: Optional[int],
        scope: Optional[str],
        strategy: Optional[str],
        max_groups: int,
        similarity_threshold: float,
        plan_id: Optional[str],
    ) -> Dict:
        """Async implementation of run."""
        result: Dict = {
            "similar_files": None,
            "duplicates": None,
            "cleanup_analysis": None,
            "plan": None,
            "pending_plans": None,
            "result": None,
        }

        if operation == "find_similar":
            if not file_path:
                raise ValueError("file_path required for find_similar")
            result["similar_files"] = await self._semantic.find_similar(file_path, k=k)

        elif operation == "find_duplicates":
            result["duplicates"] = await self._semantic.find_duplicates()

        elif operation == "analyze_cleanup":
            result["cleanup_analysis"] = await self._semantic.analyze_cleanup()

        elif operation == "propose_organization":
            if not scope:
                raise ValueError("scope required for propose_organization")

            # Map strategy string to OrganizeStrategy
            strat = self._map_strategy(strategy or "by_topic")
            request = OrganizeRequest(
                scope=scope,
                strategy=strat,
                max_groups=max_groups,
                similarity_threshold=similarity_threshold,
            )
            result["plan"] = await self._semantic.create_organize_plan(request)

        elif operation == "propose_cleanup":
            # Create cleanup-focused organization plan
            request = OrganizeRequest(
                scope="./",
                strategy=OrganizeStrategy.by_topic(),
                max_groups=1,
                similarity_threshold=0.0,
            )
            result["plan"] = await self._semantic.create_organize_plan(request)

        elif operation == "list_pending":
            result["pending_plans"] = await self._semantic.list_pending_plans()

        elif operation == "get_plan":
            if not plan_id:
                raise ValueError("plan_id required for get_plan")
            result["plan"] = await self._semantic.get_plan(plan_id)

        elif operation == "approve":
            if not plan_id:
                raise ValueError("plan_id required for approve")
            result["result"] = await self._semantic.approve_plan(plan_id)

        elif operation == "reject":
            if not plan_id:
                raise ValueError("plan_id required for reject")
            result["result"] = await self._semantic.reject_plan(plan_id)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return result

    def _map_strategy(self, strategy: str) -> OrganizeStrategy:
        """Map strategy string to OrganizeStrategy."""
        if strategy == "by_topic":
            return OrganizeStrategy.by_topic()
        elif strategy == "by_type":
            return OrganizeStrategy.by_type()
        elif strategy == "by_project":
            return OrganizeStrategy.by_project()
        else:
            return OrganizeStrategy.by_topic()  # Default

    # =========================================================================
    # Direct access methods (for use outside pipelines)
    # =========================================================================

    async def find_similar(
        self,
        file_path: str,
        k: Optional[int] = None,
    ) -> SimilarFilesResult:
        """Find files semantically similar to a given file."""
        self._ensure_initialized()
        return await self._semantic.find_similar(file_path, k=k)

    async def find_duplicates(self) -> DuplicateGroups:
        """Find duplicate or near-duplicate file groups."""
        self._ensure_initialized()
        return await self._semantic.find_duplicates()

    async def analyze_cleanup(self) -> CleanupAnalysis:
        """Analyze files for potential cleanup."""
        self._ensure_initialized()
        return await self._semantic.analyze_cleanup()

    async def propose_organization(
        self,
        scope: str,
        strategy: str = "by_topic",
        max_groups: int = 10,
        similarity_threshold: float = 0.7,
    ) -> SemanticPlan:
        """Create an organization plan (NOT executed until approved)."""
        self._ensure_initialized()
        strat = self._map_strategy(strategy)
        request = OrganizeRequest(
            scope=scope,
            strategy=strat,
            max_groups=max_groups,
            similarity_threshold=similarity_threshold,
        )
        return await self._semantic.create_organize_plan(request)

    async def list_pending_plans(self) -> List[SemanticPlan]:
        """List all pending plans awaiting approval."""
        self._ensure_initialized()
        return await self._semantic.list_pending_plans()

    async def get_plan(self, plan_id: str) -> Optional[SemanticPlan]:
        """Get a specific plan by ID."""
        self._ensure_initialized()
        return await self._semantic.get_plan(plan_id)

    async def approve(self, plan_id: str) -> SemanticPlan:
        """Approve and execute a plan."""
        self._ensure_initialized()
        return await self._semantic.approve_plan(plan_id)

    async def reject(self, plan_id: str) -> SemanticPlan:
        """Reject and discard a plan."""
        self._ensure_initialized()
        return await self._semantic.reject_plan(plan_id)

    @property
    def source_path(self) -> str:
        """Get the source directory path."""
        return self._source_path

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path


# Convenience alias
RagfsOrganizer = HaystackRagfsOrganizer
