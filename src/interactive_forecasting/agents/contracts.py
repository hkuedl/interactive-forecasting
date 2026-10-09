"""Typed role contexts and action requests; no numerical or final-test data access."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.models import (
    AdjustmentProposal,
    ArtifactRef,
    Task,
    TaskDefinition,
)
from interactive_forecasting.domain.optimization import OptimizationSummary
from interactive_forecasting.domain.preparation import (
    ColumnMappingDraft,
    DataQualityReport,
    ForecastTaskDraft,
    PreparationPlan,
    PreparationWorkingState,
    SchemaInspection,
)
from interactive_forecasting.domain.search import CandidateRequest, GuidanceCommand, SearchSpace
from interactive_forecasting.domain.types import AGENT_ROLES, Actor


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationTrialSummary(Contract):
    trial_id: UUID
    family: str
    status: str
    validation_objective: float | None = None


class ToolResult(Contract):
    status: Literal["completed", "failed"]
    data: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> ToolResult:
        if any(
            forbidden in key.lower()
            for key in self.data
            for forbidden in ("final_test", "test_label", "raw_target", "target_values")
        ):
            raise ValueError("tool result cannot expose final-test or raw target data")
        if self.status == "failed" and not self.error:
            raise ValueError("failed deterministic result requires an error")
        if self.status == "completed" and self.error is not None:
            raise ValueError("completed deterministic result cannot contain an error")
        return self


class TaskManagerContext(Contract):
    task: Task
    user_text: str | None = None
    optimization_session_version: int | None = None
    latest_specialist_role: Actor | None = None
    latest_result: ToolResult | None = None


class PreparationContext(Contract):
    task: Task
    dataset_summary_ref: ArtifactRef | None = None
    data_quality_status: str | None = None
    task_definition: TaskDefinition | None = None
    preparation_version: int | None = None
    schema_inspection: SchemaInspection | None = None
    mapping_draft: ColumnMappingDraft | None = None
    quality_report: DataQualityReport | None = None
    plan_draft: PreparationPlan | None = None
    forecast_draft: ForecastTaskDraft | None = None
    working_state: PreparationWorkingState | None = None
    missing_fields: tuple[str, ...] = ()
    recent_dialogue: tuple[dict[str, str], ...] = ()


class ModelManagerContext(Contract):
    task: Task
    task_definition: TaskDefinition
    run_id: UUID
    effective_search_space: SearchSpace
    validation_history: tuple[ValidationTrialSummary, ...] = ()
    guidance_history: tuple[GuidanceCommand, ...] = ()
    optimization_summary: OptimizationSummary | None = None
    user_guidance: tuple[GuidanceCommand, ...] = ()
    user_guidance_text: str | None = None
    previous_execution_report: str | None = None
    proposed_guidance: tuple[GuidanceCommand, ...] = ()
    proposed_rationale: str | None = None
    pending_restriction_approval: bool = False
    proposed_restriction_status: Literal["none", "pending", "approved", "discarded"] | None = None
    recent_dialogue: tuple[dict[str, str], ...] = ()


class ExecutionRequest(Contract):
    run_id: UUID
    operation: Literal["run_round", "run_fixed"]
    candidate: CandidateRequest | None = None
    expected_run_version: int = Field(ge=0)

    @model_validator(mode="after")
    def candidate_for_fixed(self) -> ExecutionRequest:
        if (self.operation == "run_fixed") != (self.candidate is not None):
            raise ValueError("run_fixed requires exactly one fixed candidate")
        return self


class ApprovedExecution(ExecutionRequest):
    approval_id: UUID = Field(default_factory=uuid4)
    idempotency_key: UUID = Field(default_factory=uuid4)


class ModelDeveloperContext(Contract):
    task: Task
    approved_request: ApprovedExecution
    approved_plan_text: str | None = None
    approved_guidance: tuple[GuidanceCommand, ...] = ()
    execution_result: ToolResult | None = None
    round_trials: tuple[ValidationTrialSummary, ...] = ()


class DeploymentForecastPoint(Contract):
    timestamp: str
    values: tuple[float, ...]


class DeploymentReferenceSummary(Contract):
    d_minus_1: str
    d_minus_7: str
    d_minus_365: str
    weather_dates_and_distances: tuple[str, ...] = ()
    weather_unavailable_reason: str | None = None


class DeploymentContext(Contract):
    task: Task
    selected_model_artifact: ArtifactRef | None = None
    forecast_id: UUID | None = None
    session_id: UUID | None = None
    session_version: int | None = None
    current_version_id: UUID | None = None
    original_version_id: UUID | None = None
    sensitivity_variables: tuple[str, ...] = ()
    forecast_points: tuple[DeploymentForecastPoint, ...] = ()
    reference_summary: DeploymentReferenceSummary | None = None
    previous_adjustments: tuple[str, ...] = ()
    user_request_text: str | None = None


AgentContext = (
    TaskManagerContext
    | PreparationContext
    | ModelManagerContext
    | ModelDeveloperContext
    | DeploymentContext
)

_CONTEXT_BY_ROLE: dict[Actor, type[Contract]] = {
    Actor.TASK_MANAGER: TaskManagerContext,
    Actor.PREPARATION_ASSISTANT: PreparationContext,
    Actor.MODEL_MANAGER: ModelManagerContext,
    Actor.MODEL_DEVELOPER: ModelDeveloperContext,
    Actor.DEPLOYMENT_OPERATOR: DeploymentContext,
}


class RouteSpecialist(Contract):
    kind: Literal["route_specialist"] = "route_specialist"
    target_role: Actor
    instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def specialist_only(self) -> RouteSpecialist:
        if self.target_role not in AGENT_ROLES - {Actor.TASK_MANAGER}:
            raise ValueError("routing target must be a specialist")
        return self


class PresentToUser(Contract):
    kind: Literal["present_to_user"] = "present_to_user"
    text: str = Field(min_length=1)


class RequestPreparation(Contract):
    kind: Literal["request_preparation"] = "request_preparation"
    operation: Literal["inspect_summary", "validate_draft", "request_action"]
    task_id: UUID


class ProposeMappingDraft(Contract):
    kind: Literal["propose_mapping_draft"] = "propose_mapping_draft"
    task_id: UUID
    expected_preparation_version: int = Field(ge=0)
    draft: ColumnMappingDraft


class ProposePlanDraft(Contract):
    kind: Literal["propose_plan_draft"] = "propose_plan_draft"
    task_id: UUID
    expected_preparation_version: int = Field(ge=0)
    draft: PreparationPlan


class ProposeForecastTaskDraft(Contract):
    kind: Literal["propose_forecast_task_draft"] = "propose_forecast_task_draft"
    task_id: UUID
    expected_preparation_version: int = Field(ge=0)
    draft: ForecastTaskDraft


class ProposeGuidance(Contract):
    kind: Literal["propose_guidance"] = "propose_guidance"
    run_id: UUID
    command: GuidanceCommand


class ProposeGuidanceBatch(Contract):
    kind: Literal["propose_guidance_batch"] = "propose_guidance_batch"
    run_id: UUID
    commands: tuple[GuidanceCommand, ...] = Field(min_length=1)


class ProposeHumanGuidance(Contract):
    kind: Literal["propose_human_guidance"] = "propose_human_guidance"
    task_id: UUID
    expected_session_version: int = Field(ge=0)
    commands: tuple[GuidanceCommand, ...] = Field(min_length=1)


class RequestExecution(Contract):
    kind: Literal["request_execution"] = "request_execution"
    request: ExecutionRequest


class ExecuteApproved(Contract):
    kind: Literal["execute_approved"] = "execute_approved"
    approval_id: UUID


class ProposeAdjustment(Contract):
    kind: Literal["propose_adjustment"] = "propose_adjustment"
    task_id: UUID
    forecast_id: UUID
    parent_version_id: UUID
    expected_session_version: int = Field(ge=0)
    proposal: AdjustmentProposal


class RequestSensitivity(Contract):
    kind: Literal["request_sensitivity"] = "request_sensitivity"
    task_id: UUID
    forecast_id: UUID
    base_version_id: UUID
    expected_session_version: int = Field(ge=0)
    variable: str = Field(min_length=1)
    perturbation_type: Literal["absolute", "percent"]
    value: float = Field(allow_inf_nan=False)


class RequestDeployment(Contract):
    kind: Literal["request_deployment"] = "request_deployment"
    operation: Literal["inspect_status", "request_forecast"]
    task_id: UUID


AgentAction = Annotated[
    RouteSpecialist
    | PresentToUser
    | RequestPreparation
    | ProposeMappingDraft
    | ProposePlanDraft
    | ProposeForecastTaskDraft
    | ProposeGuidance
    | ProposeGuidanceBatch
    | ProposeHumanGuidance
    | RequestExecution
    | ExecuteApproved
    | RequestDeployment
    | RequestSensitivity
    | ProposeAdjustment,
    Field(discriminator="kind"),
]


class AgentDecision(Contract):
    explanation: str = Field(min_length=1)
    action: AgentAction | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


@dataclass(frozen=True)
class RoleDefinition:
    role: Actor
    user_facing: bool
    allowed_actions: frozenset[str]
    purpose: str
    prompt_version: str = "4a-v1"


ROLES: dict[Actor, RoleDefinition] = {
    Actor.TASK_MANAGER: RoleDefinition(
        Actor.TASK_MANAGER,
        True,
        frozenset({"route_specialist", "present_to_user", "propose_human_guidance"}),
        "Sole user-facing role; routes work and explains recorded results.",
    ),
    Actor.PREPARATION_ASSISTANT: RoleDefinition(
        Actor.PREPARATION_ASSISTANT,
        False,
        frozenset(
            {
                "request_preparation",
                "propose_mapping_draft",
                "propose_plan_draft",
                "propose_forecast_task_draft",
            }
        ),
        "Proposes task preparation actions; never calculates numerical results.",
    ),
    Actor.MODEL_MANAGER: RoleDefinition(
        Actor.MODEL_MANAGER,
        False,
        frozenset({"propose_guidance", "propose_guidance_batch", "request_execution"}),
        "Reasons over validation history and proposes search guidance or execution.",
    ),
    Actor.MODEL_DEVELOPER: RoleDefinition(
        Actor.MODEL_DEVELOPER,
        False,
        frozenset({"execute_approved"}),
        "Executes only an approved request through deterministic backend services.",
    ),
    Actor.DEPLOYMENT_OPERATOR: RoleDefinition(
        Actor.DEPLOYMENT_OPERATOR,
        False,
        frozenset({"request_deployment", "request_sensitivity", "propose_adjustment"}),
        "Interprets bounded deployment context and proposes typed drafts; never applies them.",
    ),
}


class UnauthorizedAction(ValueError):
    pass


def validate_context(role: Actor, context: AgentContext) -> None:
    expected = _CONTEXT_BY_ROLE.get(role)
    if expected is None or type(context) is not expected:
        raise ValueError(f"{role.value} requires its own typed context")


def authorize_decision(role: Actor, context: AgentContext, decision: AgentDecision) -> None:
    validate_context(role, context)
    action = decision.action
    if action is None:
        return
    if action.kind not in ROLES[role].allowed_actions:
        raise UnauthorizedAction(f"{role.value} cannot request {action.kind}")
    if isinstance(action, ExecuteApproved):
        assert isinstance(context, ModelDeveloperContext)
        if action.approval_id != context.approved_request.approval_id:
            raise UnauthorizedAction("developer execution does not match approved request")
    if isinstance(action, RequestExecution):
        assert isinstance(context, ModelManagerContext)
        if action.request.run_id != context.run_id:
            raise UnauthorizedAction("execution request belongs to another run")
    if isinstance(action, (ProposeGuidance, ProposeGuidanceBatch)):
        assert isinstance(context, ModelManagerContext)
        if action.run_id != context.run_id:
            raise UnauthorizedAction("guidance belongs to another run")
    if isinstance(action, ProposeHumanGuidance):
        assert isinstance(context, TaskManagerContext)
        if action.task_id != context.task.task_id:
            raise UnauthorizedAction("human guidance belongs to another task")
        if action.expected_session_version != context.optimization_session_version:
            raise UnauthorizedAction("human guidance proposal has stale session version")
    if isinstance(
        action,
        (
            RequestPreparation,
            RequestDeployment,
            ProposeMappingDraft,
            ProposePlanDraft,
            ProposeForecastTaskDraft,
        ),
    ):
        if action.task_id != context.task.task_id:
            raise UnauthorizedAction("action belongs to another task")
    if isinstance(action, RequestSensitivity):
        assert isinstance(context, DeploymentContext)
        if (
            action.task_id != context.task.task_id
            or action.forecast_id != context.forecast_id
            or action.base_version_id != context.original_version_id
            or action.expected_session_version != context.session_version
            or action.variable not in context.sensitivity_variables
            or not isfinite(action.value)
        ):
            raise UnauthorizedAction("sensitivity request does not match frozen forecast inputs")
    if isinstance(action, ProposeAdjustment):
        assert isinstance(context, DeploymentContext)
        if (
            action.task_id != context.task.task_id
            or action.forecast_id != context.forecast_id
            or action.parent_version_id != context.current_version_id
            or action.expected_session_version != context.session_version
        ):
            raise UnauthorizedAction("deployment proposal does not match current forecast version")
    if isinstance(action, (ProposeMappingDraft, ProposePlanDraft, ProposeForecastTaskDraft)):
        assert isinstance(context, PreparationContext)
        if action.expected_preparation_version != context.preparation_version:
            raise UnauthorizedAction("proposal has stale Preparation version")
