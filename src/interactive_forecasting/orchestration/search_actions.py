"""Narrow approved bridge to the existing deterministic Stage 3 search engine."""

from __future__ import annotations

import asyncio

from interactive_forecasting.agents.contracts import (
    AgentAction,
    AgentContext,
    ApprovedExecution,
    ExecuteApproved,
    ExecutionRequest,
    ModelDeveloperContext,
    ModelManagerContext,
    ToolResult,
    UnauthorizedAction,
)
from interactive_forecasting.domain.types import JobStatus
from interactive_forecasting.orchestration.state_machine import VersionConflict
from interactive_forecasting.services.optimization.engine import SearchEngine
from interactive_forecasting.storage.interfaces import ExperimentRepositoryPort


class SearchExecutionApprover:
    """Application-side preflight; issuing an approval is not a model decision."""

    def __init__(self, runs: ExperimentRepositoryPort):
        self.runs = runs

    def approve(self, request: ExecutionRequest, context: ModelManagerContext) -> ApprovedExecution:
        run = self.runs.get(request.run_id)
        if run is None or run.task_definition is None:
            raise UnauthorizedAction("requested search run does not exist")
        if run.task_definition.task_id != context.task.task_id or run.run_id != context.run_id:
            raise UnauthorizedAction("search run belongs to another task")
        if run.version != request.expected_run_version:
            raise VersionConflict("stale search execution request")
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise UnauthorizedAction("search run is not executable")
        if run.final_test_metrics is not None:
            raise UnauthorizedAction("final-evaluated run cannot be searched again")
        if (
            run.effective_space is None
            or request.operation == "run_fixed"
            and request.candidate is None
        ):
            raise UnauthorizedAction("search run lacks an executable effective space")
        if request.candidate is not None:
            run.effective_space.validate_values(request.candidate.family, request.candidate.values)
            if request.candidate.family not in run.templates:
                raise UnauthorizedAction("fixed candidate lacks a complete family template")
        return ApprovedExecution.model_validate(request.model_dump(mode="json"))


class SearchActionExecutor:
    """Model Developer can execute an approved request, never edit search policy."""

    def __init__(self, engine: SearchEngine, runs: ExperimentRepositoryPort):
        self.engine = engine
        self.runs = runs

    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult:
        if not isinstance(action, ExecuteApproved) or not isinstance(
            context, ModelDeveloperContext
        ):
            raise UnauthorizedAction("this service only executes approved search requests")
        approved = context.approved_request
        if action.approval_id != approved.approval_id:
            raise UnauthorizedAction("execution approval does not match developer context")
        run = self.runs.get(approved.run_id)
        if run is None or run.task_definition is None:
            raise UnauthorizedAction("approved search run does not exist")
        if run.task_definition.task_id != context.task.task_id:
            raise UnauthorizedAction("approved search run belongs to another task")
        if run.version != approved.expected_run_version:
            raise VersionConflict("approved search request is stale")
        if run.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise UnauthorizedAction("approved search run is no longer executable")
        if approved.operation == "run_fixed":
            assert approved.candidate is not None
            completed = await asyncio.to_thread(
                self.engine.run_fixed, approved.run_id, approved.candidate
            )
        else:
            completed = await asyncio.to_thread(self.engine.run_round, approved.run_id)
        return ToolResult(
            status="completed",
            data={
                "run_id": str(completed.run_id),
                "status": completed.status.value,
                "trial_count": len(completed.trials),
                "selected_trial_id": (
                    str(completed.selected_trial_id) if completed.selected_trial_id else None
                ),
            },
        )
