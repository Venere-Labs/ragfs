"""LlamaIndex semantic organizer for AI-powered file operations."""

from __future__ import annotations

from typing import List, Optional

from ragfs import (
    RagfsSemanticManager,
    OrganizeStrategy,
    OrganizeRequest,
    SemanticPlan,
    PlanAction,
    SimilarFile,
    SimilarFilesResult,
    DuplicateGroups,
    CleanupAnalysis,
)


class LlamaIndexRagfsOrganizer:
    """AI-powered file organization with approval workflow.

    This component exposes RAGFS's semantic operations to LlamaIndex pipelines,
    implementing the Propose-Review-Apply pattern for safe AI agent operations.

    **Key Pattern: Propose-Review-Apply**
    1. Agents propose operations (organization, cleanup, deduplication)
    2. Plans are created but NOT executed
    3. Users/agents review the proposed actions
    4. Approve to execute, or reject to discard

    Example:
        from ragfs.llamaindex import RagfsOrganizer, create_ragfs_index

        # Create organizer
        organizer = RagfsOrganizer("/path/to/source", "/path/to/db")
        await organizer.init()

        # Find similar files
        similar = await organizer.find_similar("/path/to/file.txt", k=5)
        for f in similar.similar:
            print(f"{f.path}: {f.similarity:.2%}")

        # Create an organization plan (NOT executed yet!)
        plan = await organizer.propose_organization(
            scope="./docs",
            strategy="by_topic",
            max_groups=5
        )

        # Review the plan
        print(f"Plan {plan.id} proposes {len(plan.actions)} actions:")
        for action in plan.actions:
            print(f"  {action.action.action_type}: {action.reason}")

        # Approve and execute (or reject)
        if user_approves:
            result = await organizer.approve(plan.id)
            print(f"Executed: {result.status}")
        else:
            await organizer.reject(plan.id)
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
        self._semantic = RagfsSemanticManager(
            source_path=source_path,
            db_path=db_path,
            model_path=model_path,
            duplicate_threshold=duplicate_threshold,
            similar_limit=similar_limit,
        )
        self._initialized = False

    async def init(self) -> None:
        """Initialize the semantic manager (loads model and store)."""
        if not self._initialized:
            await self._semantic.init()
            self._initialized = True

    async def is_available(self) -> bool:
        """Check if the manager is initialized and available."""
        return await self._semantic.is_available()

    # =========================================================================
    # Similarity Operations
    # =========================================================================

    async def find_similar(
        self,
        file_path: str,
        k: Optional[int] = None,
    ) -> SimilarFilesResult:
        """Find files semantically similar to a given file.

        Args:
            file_path: Path to the source file.
            k: Number of similar files to return.

        Returns:
            SimilarFilesResult with list of similar files and scores.
        """
        await self.init()
        return await self._semantic.find_similar(file_path, k=k)

    # =========================================================================
    # Analysis Operations
    # =========================================================================

    async def find_duplicates(self) -> DuplicateGroups:
        """Find duplicate or near-duplicate file groups.

        Returns:
            DuplicateGroups with groups of similar files and potential savings.
        """
        await self.init()
        return await self._semantic.find_duplicates()

    async def analyze_cleanup(self) -> CleanupAnalysis:
        """Analyze files for potential cleanup.

        Identifies:
        - Duplicate files
        - Stale/unused files
        - Temporary files
        - Generated files
        - Empty files

        Returns:
            CleanupAnalysis with candidates and potential savings.
        """
        await self.init()
        return await self._semantic.analyze_cleanup()

    # =========================================================================
    # Propose-Review-Apply Pattern
    # =========================================================================

    async def propose_organization(
        self,
        scope: str,
        strategy: str = "by_topic",
        max_groups: int = 10,
        similarity_threshold: float = 0.7,
    ) -> SemanticPlan:
        """Create an organization plan (NOT executed until approved).

        This is the "Propose" phase of the Propose-Review-Apply pattern.

        Args:
            scope: Directory scope (relative to source root).
            strategy: Organization strategy. One of:
                - "by_topic": Group by semantic similarity
                - "by_type": Group by file type
                - "by_project": Group by project structure
                - "custom": Use custom categories
            max_groups: Maximum number of groups to create.
            similarity_threshold: Minimum similarity for grouping (0.0-1.0).

        Returns:
            SemanticPlan with proposed actions to review.
        """
        await self.init()

        # Map strategy string to OrganizeStrategy
        if strategy == "by_topic":
            strat = OrganizeStrategy.by_topic()
        elif strategy == "by_type":
            strat = OrganizeStrategy.by_type()
        elif strategy == "by_project":
            strat = OrganizeStrategy.by_project()
        else:
            strat = OrganizeStrategy.by_topic()  # Default

        request = OrganizeRequest(
            scope=scope,
            strategy=strat,
            max_groups=max_groups,
            similarity_threshold=similarity_threshold,
        )

        return await self._semantic.create_organize_plan(request)

    async def propose_cleanup(self) -> SemanticPlan:
        """Create a cleanup plan for redundant/stale files.

        This is the "Propose" phase for cleanup operations.

        Returns:
            SemanticPlan with proposed cleanup actions.
        """
        await self.init()

        # First analyze, then create a plan
        analysis = await self._semantic.analyze_cleanup()

        # Create an organization request targeting cleanup candidates
        # Use the cleanup analysis to build a cleanup-focused plan
        request = OrganizeRequest(
            scope="./",
            strategy=OrganizeStrategy.by_topic(),
            max_groups=1,  # All cleanup candidates in one group
            similarity_threshold=0.0,
        )

        return await self._semantic.create_organize_plan(request)

    async def list_pending_plans(self) -> List[SemanticPlan]:
        """List all pending plans awaiting approval.

        Returns:
            List of SemanticPlan objects with status "pending".
        """
        await self.init()
        return await self._semantic.list_pending_plans()

    async def get_plan(self, plan_id: str) -> Optional[SemanticPlan]:
        """Get a specific plan by ID.

        Args:
            plan_id: The plan ID.

        Returns:
            SemanticPlan or None if not found.
        """
        await self.init()
        return await self._semantic.get_plan(plan_id)

    async def approve(self, plan_id: str) -> SemanticPlan:
        """Approve and execute a plan.

        This is the "Apply" phase of the Propose-Review-Apply pattern.
        All proposed actions are executed and become reversible via
        the safety layer (undo support).

        Args:
            plan_id: The plan ID to approve.

        Returns:
            SemanticPlan with updated status (completed or failed).
        """
        await self.init()
        return await self._semantic.approve_plan(plan_id)

    async def reject(self, plan_id: str) -> SemanticPlan:
        """Reject and discard a plan (no changes made).

        Args:
            plan_id: The plan ID to reject.

        Returns:
            SemanticPlan with updated status (rejected).
        """
        await self.init()
        return await self._semantic.reject_plan(plan_id)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def source_path(self) -> str:
        """Get the source directory path."""
        return self._source_path

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    @property
    def semantic_manager(self) -> RagfsSemanticManager:
        """Get the underlying semantic manager."""
        return self._semantic


# Convenience alias
RagfsOrganizer = LlamaIndexRagfsOrganizer
