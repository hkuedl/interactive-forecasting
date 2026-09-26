"""Strict SDK output DTOs; application validation and authority remain authoritative."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypeVar
from uuid import UUID

from pydantic import Field

from interactive_forecasting.agents.contracts import (
    AgentDecision,
    Contract,
    ExecuteApproved,
    PresentToUser,
    ProposeAdjustment,
    ProposeMappingDraft,
    ProposePlanDraft,
    RequestDeployment,
    RequestPreparation,
    RequestSensitivity,
    RouteSpecialist,
)
from interactive_forecasting.domain.forecasting import AuxiliaryPolicy, OutputConfig
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.search import Scalar
from interactive_forecasting.domain.types import ModelFamily

Key = TypeVar("Key")
Value = TypeVar("Value")


def _unique(items: Iterable[tuple[Key, Value]]) -> dict[Key, Value]:
    result: dict[Key, Value] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate transport key: {key}")
        result[key] = value
    return result


class ParameterValue(Contract):
    name: str
    value: Scalar


class FamilyAllocation(Contract):
    family: ModelFamily
    count: int


class NamedAuxiliaryPolicy(Contract):
    name: str
    policy: AuxiliaryPolicy


class CandidateTransport(Contract):
    family: ModelFamily
    values: tuple[ParameterValue, ...]
    source: Literal["sampled", "enqueued", "guided", "local_refinement", "fixed_reference"]
    seed: int

    def application_payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "values": _unique((item.name, item.value) for item in self.values),
        }


class GuidanceTransport(Contract):
    operation: Literal[
        "prefer_family",
        "allocate_family_trials",
        "exclude_family",
        "restrict_families",
        "fix_parameter",
        "narrow_parameter",
        "restrict_choices",
        "force_feature",
        "disable_feature",
        "enqueue_candidate",
        "local_refinement",
    ]
    families: tuple[ModelFamily, ...] = ()
    allocation: tuple[FamilyAllocation, ...] = ()
    parameter: str | None = None
    value: Scalar = None
    low: float | None = None
    high: float | None = None
    choices: tuple[Scalar, ...] = ()
    candidate: CandidateTransport | None = None
    radius: float | None = None

    def application_payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "allocation": _unique((item.family, item.count) for item in self.allocation),
            "candidate": self.candidate.application_payload() if self.candidate else None,
        }


class ForecastTaskTransport(Contract):
    delta: int = 1
    horizon: int = 1
    time_unit: Literal["samples", "hours"] = "samples"
    output: OutputConfig = Field(default_factory=OutputConfig)
    objective_id: Literal["mae", "mape", "crps", "weighted_mae", "asymmetric_mae"] = "mae"
    metric_spec: MetricSpec | None = None
    temperature_policy: AuxiliaryPolicy | None = None
    other_policies: tuple[NamedAuxiliaryPolicy, ...] = ()
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    def application_payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "other_policies": _unique(
                (item.name, item.policy.model_dump(mode="json")) for item in self.other_policies
            ),
        }


class ProposeForecastTaskTransport(Contract):
    kind: Literal["propose_forecast_task_draft"] = "propose_forecast_task_draft"
    task_id: UUID
    expected_preparation_version: int
    draft: ForecastTaskTransport


class ProposeGuidanceTransport(Contract):
    kind: Literal["propose_guidance"] = "propose_guidance"
    run_id: UUID
    command: GuidanceTransport


class ProposeGuidanceBatchTransport(Contract):
    kind: Literal["propose_guidance_batch"] = "propose_guidance_batch"
    run_id: UUID
    commands: tuple[GuidanceTransport, ...]


class ProposeHumanGuidanceTransport(Contract):
    kind: Literal["propose_human_guidance"] = "propose_human_guidance"
    task_id: UUID
    expected_session_version: int
    commands: tuple[GuidanceTransport, ...]


class ExecutionTransport(Contract):
    run_id: UUID
    operation: Literal["run_round", "run_fixed"]
    candidate: CandidateTransport | None = None
    expected_run_version: int


class RequestExecutionTransport(Contract):
    kind: Literal["request_execution"] = "request_execution"
    request: ExecutionTransport


class AgentDecisionTransport(Contract):
    explanation: str
    # A plain union emits anyOf, supported by strict Structured Outputs.
    action: (
        RouteSpecialist
        | PresentToUser
        | RequestPreparation
        | ProposeMappingDraft
        | ProposePlanDraft
        | ProposeForecastTaskTransport
        | ProposeGuidanceTransport
        | ProposeGuidanceBatchTransport
        | ProposeHumanGuidanceTransport
        | RequestExecutionTransport
        | ExecuteApproved
        | RequestDeployment
        | RequestSensitivity
        | ProposeAdjustment
        | None
    ) = None
    confidence: float | None = None

    def to_application(self) -> AgentDecision:
        payload = self.model_dump(mode="json")
        action = self.action
        if isinstance(action, ProposeForecastTaskTransport):
            payload["action"]["draft"] = action.draft.application_payload()
        elif isinstance(action, ProposeGuidanceTransport):
            payload["action"]["command"] = action.command.application_payload()
        elif isinstance(action, (ProposeGuidanceBatchTransport, ProposeHumanGuidanceTransport)):
            payload["action"]["commands"] = [c.application_payload() for c in action.commands]
        elif isinstance(action, RequestExecutionTransport) and action.request.candidate:
            payload["action"]["request"]["candidate"] = (
                action.request.candidate.application_payload()
            )
        return AgentDecision.model_validate(payload)
