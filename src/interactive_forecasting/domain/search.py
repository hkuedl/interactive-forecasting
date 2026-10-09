"""Versioned, immutable search contracts; no optimizer or model-library imports."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.forecasting import ResolvedCandidate
from interactive_forecasting.domain.types import ModelFamily

Scalar = str | int | float | bool | None


class SearchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ParameterDomain(SearchRecord):
    name: str = Field(min_length=1)
    kind: Literal["fixed", "categorical", "integer", "float"]
    value: Scalar = None
    choices: tuple[Scalar, ...] = ()
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    condition_on: str | None = None
    condition_value: Scalar = None

    @model_validator(mode="after")
    def valid(self) -> ParameterDomain:
        if self.kind == "fixed":
            if self.choices or self.low is not None or self.high is not None:
                raise ValueError("fixed domain cannot have choices or bounds")
        elif self.kind == "categorical":
            if not self.choices or len(set(self.choices)) != len(self.choices):
                raise ValueError("categorical choices must be nonempty and unique")
            if self.low is not None or self.high is not None or self.step is not None or self.log:
                raise ValueError("categorical domain cannot have numeric settings")
        else:
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError("numeric domain requires ordered bounds")
            if self.kind == "integer" and (
                not float(self.low).is_integer()
                or not float(self.high).is_integer()
                or (self.step is not None and not float(self.step).is_integer())
            ):
                raise ValueError("integer domain requires integral bounds and step")
            if self.step is not None and (self.step <= 0 or self.log):
                raise ValueError("step must be positive and cannot combine with log")
            if self.log and self.low <= 0:
                raise ValueError("log domain requires positive lower bound")
            if self.choices:
                raise ValueError("numeric domain cannot have choices")
        return self

    def contains(self, value: Scalar) -> bool:
        if self.kind == "fixed":
            return value == self.value
        if self.kind == "categorical":
            return value in self.choices
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            return False
        if self.low is None or self.high is None or not self.low <= value <= self.high:
            return False
        if self.kind == "integer" and not isinstance(value, int):
            return False
        if self.step is not None:
            ratio = (value - self.low) / self.step
            return abs(ratio - round(ratio)) < 1e-9
        return True


class FamilySpace(SearchRecord):
    family: ModelFamily
    parameters: tuple[ParameterDomain, ...] = ()

    @model_validator(mode="after")
    def unique_names(self) -> FamilySpace:
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate family parameter")
        seen: set[str] = set()
        for item in self.parameters:
            if item.condition_on is not None and item.condition_on not in seen:
                raise ValueError("conditional family parameter must follow its parent")
            seen.add(item.name)
        return self


class SpaceChange(SearchRecord):
    operation: str
    parameter: str | None = None
    value: Scalar = None
    families: tuple[ModelFamily, ...] = ()
    low: float | None = None
    high: float | None = None
    choices: tuple[Scalar, ...] = ()
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchSpace(SearchRecord):
    space_id: UUID = Field(default_factory=uuid4)
    version: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    parent_space_id: UUID | None = None
    families: tuple[FamilySpace, ...]
    features: tuple[ParameterDomain, ...] = ()
    changes: tuple[SpaceChange, ...] = ()

    @model_validator(mode="after")
    def nonempty(self) -> SearchSpace:
        if not self.families or len({item.family for item in self.families}) != len(self.families):
            raise ValueError("search space needs unique eligible families")
        names = [item.name for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature parameter")
        seen: set[str] = set()
        for item in self.features:
            if item.condition_on is not None and item.condition_on not in seen:
                raise ValueError("conditional feature parameter must follow its parent")
            seen.add(item.name)
        if any(
            seen.intersection({param.name for param in family.parameters})
            for family in self.families
        ):
            raise ValueError("feature and family parameter names must not overlap")
        return self

    def parameters_for(self, family: ModelFamily) -> tuple[ParameterDomain, ...]:
        for item in self.families:
            if item.family == family:
                return item.parameters
        raise ValueError(f"{family.value} is outside the search space")

    def derive(self, change: SpaceChange) -> SearchSpace:
        families = list(self.families)
        features = list(self.features)
        if change.operation in {"fix_family", "restrict_families", "exclude_family"}:
            requested = set(change.families)
            if not requested:
                raise ValueError("family transformation needs families")
            current = {item.family for item in families}
            if change.operation != "exclude_family" and not requested <= current:
                raise ValueError("requested family is outside parent space")
            if change.operation == "fix_family" and len(requested) != 1:
                raise ValueError("fix_family requires one family")
            families = [
                item
                for item in families
                if (
                    item.family not in requested
                    if change.operation == "exclude_family"
                    else item.family in requested
                )
            ]
        elif change.operation in {
            "fix",
            "narrow",
            "restrict_choices",
            "force_feature",
            "disable_feature",
        }:
            if change.parameter is None:
                raise ValueError("parameter transformation needs a name")
            target = change.parameter
            if target.startswith("feature."):
                name = target.removeprefix("feature.")
                locations = [(features, i) for i, item in enumerate(features) if item.name == name]
            else:
                if "." not in target:
                    raise ValueError("family parameter must be Family.name")
                family_name, name = target.split(".", 1)
                locations = [
                    (list(item.parameters), i)
                    for item in families
                    if item.family.value == family_name
                    for i, param in enumerate(item.parameters)
                    if param.name == name
                ]
            if len(locations) != 1:
                raise ValueError("transformation parameter is not unique in space")
            items, index = locations[0]
            old = items[index]
            if change.operation in {"fix", "force_feature", "disable_feature"}:
                value = "none" if change.operation == "disable_feature" else change.value
                if not old.contains(value):
                    raise ValueError("fixed value is outside parent domain")
                new = ParameterDomain(
                    name=old.name,
                    kind="fixed",
                    value=value,
                    condition_on=old.condition_on,
                    condition_value=old.condition_value,
                )
            elif change.operation == "restrict_choices":
                if (
                    old.kind != "categorical"
                    or not change.choices
                    or not set(change.choices) <= set(old.choices)
                ):
                    raise ValueError("categorical restriction must be a nonempty subset")
                new = old.model_copy(update={"choices": change.choices})
            else:
                if (
                    old.kind not in {"integer", "float"}
                    or change.low is None
                    or change.high is None
                ):
                    raise ValueError("numeric narrowing requires bounds")
                if (
                    old.low is None
                    or old.high is None
                    or not old.low <= change.low <= change.high <= old.high
                ):
                    raise ValueError("narrowing must stay within parent bounds")
                if old.kind == "integer" and (
                    not float(change.low).is_integer() or not float(change.high).is_integer()
                ):
                    raise ValueError("integer bounds must remain integral")
                if old.step is not None and (
                    not old.contains(int(change.low) if old.kind == "integer" else change.low)
                    or not old.contains(int(change.high) if old.kind == "integer" else change.high)
                ):
                    raise ValueError("narrowed bounds must align with step")
                new = ParameterDomain.model_validate(
                    {**old.model_dump(), "low": change.low, "high": change.high}
                )
            items[index] = new
            if target.startswith("feature."):
                features = items
            else:
                families = [
                    item.model_copy(update={"parameters": tuple(items)})
                    if item.family.value == family_name
                    else item
                    for item in families
                ]
        else:
            raise ValueError("unsupported search-space transformation")
        if not families:
            raise ValueError("transformation would leave an empty family space")
        return SearchSpace(
            version=f"{self.version}.{len(self.changes) + 1}",
            provenance=self.provenance,
            parent_space_id=self.space_id,
            families=tuple(families),
            features=tuple(features),
            changes=(*self.changes, change),
        )

    def domains_for(self, family: ModelFamily) -> tuple[ParameterDomain, ...]:
        sequential = family in {ModelFamily.LSTM, ModelFamily.GRU, ModelFamily.CNN}
        feature_domains = tuple(
            domain
            for domain in self.features
            if (
                domain.name not in {"load_mode", "load_ratio"}
                if sequential
                else not domain.name.startswith("sequence_")
            )
        )
        return (*self.parameters_for(family), *feature_domains)

    def validate_values(self, family: ModelFamily, values: dict[str, Scalar]) -> None:
        domains = self.domains_for(family)
        expected = {
            d.name
            for d in domains
            if d.condition_on is None or values.get(d.condition_on) == d.condition_value
        }
        if set(values) != expected:
            raise ValueError(
                f"candidate parameters differ from active search space: {set(values) ^ expected}"
            )
        if any(not d.contains(values[d.name]) for d in domains if d.name in expected):
            raise ValueError("candidate parameter value is outside search space")

    def validate_partial_values(self, family: ModelFamily, values: dict[str, Scalar]) -> None:
        """Validate a bounded partial enqueue without inventing absent parent choices."""
        domains = {domain.name: domain for domain in self.domains_for(family)}
        if not set(values) <= set(domains):
            raise ValueError("partial candidate contains an unknown parameter")
        for name, value in values.items():
            domain = domains[name]
            if not domain.contains(value):
                raise ValueError("partial candidate value is outside search space")
            if domain.condition_on is not None and (
                domain.condition_on not in values
                or values[domain.condition_on] != domain.condition_value
            ):
                raise ValueError("partial conditional value requires its active parent")


class CandidateRequest(SearchRecord):
    family: ModelFamily
    values: dict[str, Scalar]
    source: Literal["sampled", "enqueued", "guided", "local_refinement", "fixed_reference"]
    seed: int = Field(ge=0, le=2**32 - 1)


class TrialCandidate(SearchRecord):
    candidate: ResolvedCandidate
    request: CandidateRequest
    space_id: UUID
    task_definition_id: UUID
    protocol_id: UUID


class GuidanceCommand(SearchRecord):
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
    allocation: dict[ModelFamily, int] = Field(default_factory=dict)
    parameter: str | None = None
    value: Scalar = None
    low: float | None = None
    high: float | None = None
    choices: tuple[Scalar, ...] = ()
    candidate: CandidateRequest | None = None
    radius: float | None = Field(default=None, gt=0, le=1)


def is_persistent_guidance(command: GuidanceCommand) -> bool:
    """Classify commands by their actual SearchEngine effect, not their wording."""
    return command.operation in {
        "exclude_family",
        "restrict_families",
        "fix_parameter",
        "narrow_parameter",
        "restrict_choices",
        "force_feature",
        "disable_feature",
        "local_refinement",
    }


class BackendConfig(SearchRecord):
    sampler: Literal["tpe"]
    seed: int = Field(ge=0, le=2**32 - 1)
    direction: Literal["minimize", "maximize"]
    n_startup_trials: int = Field(ge=0)
    n_ei_candidates: int = Field(gt=0)
    prior_weight: float = Field(gt=0)
    consider_magic_clip: bool
    consider_endpoints: bool
    gamma_rule: Literal["optuna_builtin"]
    weights_rule: Literal["optuna_builtin"]
    multivariate: bool
    group: bool
    warn_independent_sampling: bool
    constant_liar: bool
    constraints: Literal["none"]
    trial_budget: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_rounds: int = Field(gt=0)
    no_improvement_rounds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def sampler_consistency(self) -> BackendConfig:
        if self.group and not self.multivariate:
            raise ValueError("group sampling requires multivariate TPE")
        return self


class ValidationMetricConfig(SearchRecord):
    objective: Literal["mae", "mape", "crps", "weighted_mae", "asymmetric_mae"]
    mape_zero_policy: Literal["error", "omit", "epsilon"] | None = None
    mape_epsilon: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def mape_policy(self) -> ValidationMetricConfig:
        if self.objective == "mape" and self.mape_zero_policy is None:
            raise ValueError("MAPE objective requires an explicit zero policy")
        if self.mape_zero_policy == "epsilon" and self.mape_epsilon is None:
            raise ValueError("epsilon MAPE policy requires epsilon")
        return self


class TrialResult(SearchRecord):
    trial_id: UUID = Field(default_factory=uuid4)
    number: int = Field(ge=0)
    round_number: int = Field(ge=0)
    candidate: TrialCandidate | None = None
    request: CandidateRequest
    space_id: UUID
    status: Literal["completed", "failed"]
    objective: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_uri: str | None = None
    artifact_sha256: str | None = None
    diagnostic_uri: str | None = None
    diagnostic_sha256: str | None = None
    diagnostic_size_bytes: int | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    started_at: datetime
    ended_at: datetime

    @model_validator(mode="after")
    def result_consistency(self) -> TrialResult:
        if self.status == "completed" and (self.objective is None or self.candidate is None):
            raise ValueError("completed trial requires a resolved candidate and objective")
        if self.status == "failed" and not self.failure_type:
            raise ValueError("failed trial requires failure type")
        return self
