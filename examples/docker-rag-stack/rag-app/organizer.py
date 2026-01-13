"""
Semantic Organizer module for RAGFS Docker Stack

Provides AI-powered file organization using RAGFS semantic capabilities:
- Find duplicate files
- Analyze cleanup candidates
- Propose organization plans
- Approve/reject plans (Propose-Review-Apply pattern)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ragfs import (
    RagfsSemanticManager,
    OrganizeStrategy,
    OrganizeRequest,
    SemanticPlan,
    PlanAction,
    DuplicateGroup,
    DuplicateGroups,
    CleanupCandidate,
    CleanupAnalysis,
    SimilarFile,
    SimilarFilesResult,
)


@dataclass
class OrganizationResult:
    """Result of an organization operation."""
    success: bool
    message: str
    plan_id: Optional[str] = None
    error: Optional[str] = None


class SemanticOrganizer:
    """AI-powered file organization using RAGFS semantic operations."""

    def __init__(
        self,
        documents_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        """Initialize the semantic organizer.

        Args:
            documents_path: Path to documents directory.
            db_path: Path to RAGFS index database.
        """
        self.documents_path = Path(documents_path or os.environ.get("DOCUMENTS_PATH", "/data/docs"))
        self.db_path = Path(db_path or os.environ.get("RAGFS_DB_PATH", "/data/index"))

        self._semantic_manager: Optional[RagfsSemanticManager] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the semantic manager."""
        if self._initialized:
            return

        self._semantic_manager = RagfsSemanticManager(
            str(self.documents_path),
            str(self.db_path),
        )
        await self._semantic_manager.init()

        self._initialized = True

    async def find_similar(
        self,
        file_path: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> SimilarFilesResult:
        """Find files similar to the given file.

        Args:
            file_path: Path to the reference file.
            limit: Maximum number of similar files.
            threshold: Similarity threshold (0-1).

        Returns:
            SimilarFilesResult with list of similar files.
        """
        if not self._initialized:
            await self.initialize()

        return await self._semantic_manager.find_similar(
            file_path,
            limit=limit,
            threshold=threshold,
        )

    async def find_duplicates(
        self,
        scope: Optional[str] = None,
        threshold: float = 0.95,
    ) -> DuplicateGroups:
        """Find duplicate files in the index.

        Args:
            scope: Optional subdirectory to search.
            threshold: Similarity threshold for duplicates.

        Returns:
            DuplicateGroups with groups of duplicate files.
        """
        if not self._initialized:
            await self.initialize()

        search_path = scope or str(self.documents_path)
        return await self._semantic_manager.find_duplicates(
            search_path,
            threshold=threshold,
        )

    async def analyze_cleanup(
        self,
        scope: Optional[str] = None,
    ) -> CleanupAnalysis:
        """Analyze files for potential cleanup.

        Identifies:
        - Empty files
        - Very small files
        - Temporary files
        - Old/stale files

        Args:
            scope: Optional subdirectory to analyze.

        Returns:
            CleanupAnalysis with cleanup candidates.
        """
        if not self._initialized:
            await self.initialize()

        search_path = scope or str(self.documents_path)
        return await self._semantic_manager.analyze_cleanup(search_path)

    async def propose_organization(
        self,
        scope: Optional[str] = None,
        strategy: str = "by_topic",
        max_groups: int = 10,
    ) -> OrganizationResult:
        """Create an organization plan (NOT executed until approved).

        Args:
            scope: Directory scope to organize.
            strategy: Organization strategy (by_topic, by_type, by_date, by_project).
            max_groups: Maximum number of groups to create.

        Returns:
            OrganizationResult with plan ID.
        """
        if not self._initialized:
            await self.initialize()

        try:
            search_path = scope or str(self.documents_path)

            # Map strategy string to OrganizeStrategy
            strategy_map = {
                "by_topic": OrganizeStrategy.by_topic(),
                "by_type": OrganizeStrategy.by_type(),
                "by_date": OrganizeStrategy.by_date(),
                "by_project": OrganizeStrategy.by_project(),
            }

            org_strategy = strategy_map.get(strategy, OrganizeStrategy.by_topic())

            request = OrganizeRequest(
                scope=search_path,
                strategy=org_strategy,
                max_groups=max_groups,
            )

            plan = await self._semantic_manager.create_organize_plan(request)

            return OrganizationResult(
                success=True,
                message=f"Created organization plan with {len(plan.actions)} actions",
                plan_id=str(plan.id),
            )

        except Exception as e:
            return OrganizationResult(
                success=False,
                message=f"Failed to create plan: {e}",
                error=str(e),
            )

    async def get_plan(self, plan_id: str) -> Optional[SemanticPlan]:
        """Get a plan by ID.

        Args:
            plan_id: ID of the plan.

        Returns:
            SemanticPlan if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._semantic_manager.get_plan(plan_id)
        except Exception:
            return None

    async def list_pending_plans(self) -> List[SemanticPlan]:
        """List all pending (unapproved) plans.

        Returns:
            List of pending SemanticPlan objects.
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._semantic_manager.list_pending_plans()
        except Exception:
            return []

    async def approve_plan(self, plan_id: str) -> OrganizationResult:
        """Approve and execute a plan.

        Args:
            plan_id: ID of the plan to approve.

        Returns:
            OrganizationResult with status.
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._semantic_manager.approve_plan(plan_id)

            return OrganizationResult(
                success=True,
                message=f"Plan executed successfully. {result.succeeded} actions completed.",
                plan_id=plan_id,
            )

        except Exception as e:
            return OrganizationResult(
                success=False,
                message=f"Failed to execute plan: {e}",
                error=str(e),
            )

    async def reject_plan(self, plan_id: str) -> OrganizationResult:
        """Reject and discard a plan.

        Args:
            plan_id: ID of the plan to reject.

        Returns:
            OrganizationResult with status.
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self._semantic_manager.reject_plan(plan_id)

            return OrganizationResult(
                success=True,
                message="Plan rejected and discarded",
                plan_id=plan_id,
            )

        except Exception as e:
            return OrganizationResult(
                success=False,
                message=f"Failed to reject plan: {e}",
                error=str(e),
            )


def format_file_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_duplicate_groups(groups: DuplicateGroups) -> str:
    """Format duplicate groups for display."""
    if not groups.groups:
        return "No duplicates found."

    lines = [f"Found {len(groups.groups)} groups of duplicates:\n"]

    for i, group in enumerate(groups.groups, 1):
        lines.append(f"**Group {i}** (similarity: {group.similarity:.0%})")
        for entry in group.files:
            size = format_file_size(entry.size) if hasattr(entry, 'size') else ""
            lines.append(f"  - `{entry.path}` {size}")
        lines.append("")

    return "\n".join(lines)


def format_cleanup_analysis(analysis: CleanupAnalysis) -> str:
    """Format cleanup analysis for display."""
    if not analysis.candidates:
        return "No cleanup candidates found."

    total_size = sum(c.size for c in analysis.candidates if hasattr(c, 'size'))

    lines = [
        f"Found {len(analysis.candidates)} cleanup candidates",
        f"Total size: {format_file_size(total_size)}\n",
    ]

    # Group by reason
    by_reason: dict[str, list] = {}
    for candidate in analysis.candidates:
        reason = candidate.reason if hasattr(candidate, 'reason') else "Unknown"
        by_reason.setdefault(reason, []).append(candidate)

    for reason, candidates in by_reason.items():
        lines.append(f"**{reason}** ({len(candidates)} files)")
        for c in candidates[:5]:  # Show max 5 per category
            size = format_file_size(c.size) if hasattr(c, 'size') else ""
            lines.append(f"  - `{c.path}` {size}")
        if len(candidates) > 5:
            lines.append(f"  ... and {len(candidates) - 5} more")
        lines.append("")

    return "\n".join(lines)


def format_plan(plan: SemanticPlan) -> str:
    """Format a semantic plan for display."""
    lines = [
        f"**Plan ID:** `{plan.id}`",
        f"**Status:** {plan.status}",
        f"**Actions:** {len(plan.actions)}\n",
    ]

    for i, action in enumerate(plan.actions, 1):
        action_type = action.action.action_type if hasattr(action.action, 'action_type') else str(action.action)
        lines.append(f"{i}. **{action_type}**")
        lines.append(f"   Reason: {action.reason}")

    return "\n".join(lines)
