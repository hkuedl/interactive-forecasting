"""Persisted interactive optimization state; search results remain in ExperimentRun."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.forecasting import ResolvedCandidate
from interactive_forecasting.domain.search import CandidateRequest, GuidanceCommand, SearchSpace
from interactive_forecasting.domain.types import ModelFamily, OptimizationMode


class OptimizationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class OptimizationPhase(str, Enum):
    INITIAL_EXPLORATION = "initial_exploration"
    BOUNDARY = "boundary"
    WAITING_FOR_USER = "waiting_for_user"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecentTrial(OptimizationRecord):
    trial_id: UUID
    number: int
    round_number: int
    family: ModelFamily
    status: str
    objective: float | None = None


class OptimizationSummary(OptimizationRecord):
    run_id: UUID
    run_version: int
    current_round: int
    completed_trials: int
    failed_trials: int
    best_validation_objective: float | None = None
    best_candidate: CandidateRequest | None = None
    best_configuration: ResolvedCandidate | None = None
    recent_trials: tuple[RecentTrial, ...] = ()
    family_trial_counts: dict[ModelFamily, int] = Field(default_factory=dict)
    family_best: dict[ModelFamily, float] = Field(default_factory=dict)
    objective_trend: tuple[float, ...] = ()
    effective_space: SearchSpace
    previous_guidance: tuple[GuidanceCommand, ...] = ()
    remaining_budget: int
    stopping_status: str


class GuidanceDraft(OptimizationRecord):
    draft_id: UUID = Field(default_factory=uuid4)
    commands: tuple[GuidanceCommand, ...]
    original_text: str | None = None
    confirmed: bool = False

    @model_validator(mode="after")
    def nonempty(self) -> GuidanceDraft:
        if not self.commands:
            raise ValueError("guidance draft needs a typed command")
        return self


class ProposedRoundPlan(OptimizationRecord):
    """An advisory MM plan; it has no effect on the executable search run."""

    plan_id: UUID = Field(default_factory=uuid4)
    round_number: int = Field(ge=0)
    run_version: int = Field(ge=0)
    commands: tuple[GuidanceCommand, ...] = ()
    rationale: str = Field(min_length=1)
    user_text: str | None = None
    persistent_approval: Literal["none", "pending", "approved", "discarded"] = "none"

    @model_validator(mode="after")
    def approval_matches_commands(self) -> ProposedRoundPlan:
        from interactive_forecasting.domain.search import is_persistent_guidance

        has_restriction = any(is_persistent_guidance(item) for item in self.commands)
        if has_restriction != (self.persistent_approval != "none"):
            raise ValueError("restriction approval does not match the proposed plan")
        return self


class GuidanceEffect(OptimizationRecord):
    draft_id: UUID | None = None
    source: str
    round_number: int
    commands: tuple[GuidanceCommand, ...] = ()
    original_text: str | None = None
    rationale: str | None = None
    space_before: UUID
    guidance_count_before: int | None = Field(default=None, ge=0)
    space_after: UUID | None = None
    status: str
    error: str | None = None
    candidate_trial_numbers: tuple[int, ...] = ()


class OptimizationSession(OptimizationRecord):
    task_id: UUID
    run_id: UUID
    mode: OptimizationMode
    version: int = Field(default=0, ge=0)
    phase: OptimizationPhase = OptimizationPhase.INITIAL_EXPLORATION
    pause_at_boundary: bool = False
    initial_target: int = Field(ge=0)
    guidance_draft: GuidanceDraft | None = None
    proposed_plan: ProposedRoundPlan | None = None
    executing_plan: ProposedRoundPlan | None = None
    latest_execution_report: str | None = None
    guidance_effects: tuple[GuidanceEffect, ...] = ()
    summaries: tuple[OptimizationSummary, ...] = ()
    last_reconciled_round: int = Field(default=0, ge=0)
    last_error: str | None = None
