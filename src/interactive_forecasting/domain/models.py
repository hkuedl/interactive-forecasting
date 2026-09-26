"""Foundational metadata schemas. Numerical arrays are stored as artifacts, never here."""

from datetime import date, datetime, timezone
from math import isfinite
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.forecasting import (
    AuxiliaryPolicy,
    ClockDatetime,
    OriginSchedule,
    OutputConfig,
    PointForecast,
    QuantileForecast,
    ResolvedCandidate,
)
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.search import (
    BackendConfig,
    CandidateRequest,
    GuidanceCommand,
    SearchSpace,
    TrialResult,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import (
    Actor,
    JobStatus,
    MessageKind,
    ModelFamily,
    OptimizationMode,
    Stage,
    Topic,
    TrialStatus,
    WorkflowStatus,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRef(Record):
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class Task(Record):
    task_id: UUID = Field(default_factory=uuid4)
    stage: Stage = Stage.NEW
    workflow_status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    substate: str | None = None
    version: int = Field(default=0, ge=0)
    definition_id: UUID | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)


class TaskDefinition(Record):
    definition_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    dataset_id: UUID
    delta: int = Field(gt=0)
    horizon: int = Field(gt=0)
    time_unit: str = Field(min_length=1)
    timezone_name: str = Field(min_length=1)
    target_unit: str | None = None
    objective_id: str = Field(min_length=1)
    metric_spec: MetricSpec | None = None
    forecast_output: OutputConfig = Field(default_factory=OutputConfig)
    auxiliary_policies: dict[str, AuxiliaryPolicy] = Field(default_factory=dict)
    target_range_start: AwareDatetime | None = None
    target_range_end: AwareDatetime | None = None

    @model_validator(mode="after")
    def metric_matches_task(self) -> "TaskDefinition":
        if self.metric_spec is not None and (
            self.objective_id != self.metric_spec.objective_id
            or (
                self.metric_spec.kind == "weighted"
                and self.timezone_name != self.metric_spec.timezone_name
            )
        ):
            raise ValueError("task metric specification differs from objective or timezone")
        return self


class DatasetSource(Record):
    dataset_id: UUID = Field(default_factory=uuid4)
    adapter_kind: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: dict[str, Any] = Field(default_factory=dict)
    series_ids: list[str] = Field(default_factory=list)


class ColumnMapping(Record):
    dataset_id: UUID
    timestamp_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    auxiliary_roles: dict[str, str] = Field(default_factory=dict)
    availability: dict[str, AuxiliaryPolicy] = Field(default_factory=dict)
    confirmed: bool = False

    @model_validator(mode="after")
    def distinct_required_columns(self) -> "ColumnMapping":
        if self.timestamp_column == self.target_column:
            raise ValueError("timestamp and target columns must differ")
        return self


class DatasetSnapshot(Record):
    snapshot_id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    mapping_version: str = Field(min_length=1)
    cleaning_version: str = Field(min_length=1)
    artifact: ArtifactRef
    frequency: str = Field(min_length=1)
    timezone_name: str = Field(min_length=1)
    series_id: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class TimeWindow(Record):
    start: ClockDatetime
    end: ClockDatetime

    @model_validator(mode="after")
    def chronological(self) -> "TimeWindow":
        if (self.start.tzinfo is None) != (self.end.tzinfo is None):
            raise ValueError("time window cannot mix source-clock and zoned timestamps")
        if self.start >= self.end:
            raise ValueError("window start must precede end")
        return self


class EvaluationProtocol(Record):
    protocol_id: UUID = Field(default_factory=uuid4)
    train: TimeWindow
    validation: TimeWindow
    test: TimeWindow | None = None
    origin_schedule_ref: ArtifactRef | None = None
    seed: int
    metric_ids: list[str] = Field(min_length=1)
    metric_spec: MetricSpec | None = None
    protocol_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def disjoint_windows(self) -> "EvaluationProtocol":
        windows = (self.train, self.validation) + ((self.test,) if self.test else ())
        if len({window.start.tzinfo is None for window in windows}) != 1:
            raise ValueError("evaluation windows cannot mix source-clock and zoned timestamps")
        if self.train.end > self.validation.start:
            raise ValueError("training and validation windows overlap")
        if self.test is not None and self.validation.end > self.test.start:
            raise ValueError("validation and test windows overlap")
        if self.metric_spec is not None and self.metric_spec.objective_id not in self.metric_ids:
            raise ValueError("protocol metric specification is absent from metric IDs")
        return self


class FeatureRecipe(Record):
    version: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)


class ModelConfiguration(Record):
    configuration_id: UUID = Field(default_factory=uuid4)
    family: ModelFamily
    features: FeatureRecipe
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_space_version: str = Field(min_length=1)


class OptimizationRun(Record):
    run_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    snapshot_id: UUID
    protocol_id: UUID
    mode: OptimizationMode
    budget: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    search_method: str = Field(min_length=1)
    search_backend: str = Field(min_length=1)
    search_backend_version: str = Field(min_length=1)
    search_config: dict[str, Any] = Field(default_factory=dict)
    seed: int
    status: JobStatus = JobStatus.QUEUED
    current_batch: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def tpe_baseline(self) -> "OptimizationRun":
        if self.mode == OptimizationMode.VANILLA_BO and self.search_method.lower() != "tpe":
            raise ValueError("Vanilla BO baseline method must be TPE")
        return self


class Guidance(Record):
    guidance_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    source: Literal["user", "llm"]
    input_message_id: UUID | None = None
    interpretation: dict[str, Any]
    resulting_actions: dict[str, Any] = Field(default_factory=dict)
    effective_batch: int = Field(ge=0)
    accepted: bool
    rejection_reason: str | None = None


class Trial(Record):
    trial_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    number: int = Field(ge=0)
    batch: int = Field(ge=0)
    configuration: ModelConfiguration
    status: TrialStatus = TrialStatus.QUEUED
    objective: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_seconds: float | None = Field(default=None, ge=0)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    failure: str | None = None


class ForecastOrigin(Record):
    task_id: UUID
    selected_trial_id: UUID
    model_artifact: ArtifactRef
    latest_observed_at: AwareDatetime
    targets: tuple[AwareDatetime, ...] = Field(min_length=1)
    delta: int = Field(gt=0)
    horizon: int = Field(gt=0)
    offset_unit: Literal["samples", "hours"]
    anchor: Literal["start_at_delta", "end_at_delta"]
    timezone_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_targets(self) -> "ForecastOrigin":
        if len(self.targets) != self.horizon or any(
            target <= self.latest_observed_at for target in self.targets
        ):
            raise ValueError("forecast origin targets must be future and match horizon")
        return self


class Forecast(Record):
    forecast_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    optimization_run_id: UUID
    selected_trial_id: UUID
    deployment_session_id: UUID | None = None
    model_artifact: ArtifactRef
    origin: ForecastOrigin
    target_timestamps: tuple[AwareDatetime, ...] = Field(min_length=1)
    prediction_representation: Literal["point", "quantile"]
    prediction: PointForecast | QuantileForecast
    raw_upload: ArtifactRef
    context_artifact: ArtifactRef
    future_auxiliary_artifact: ArtifactRef
    prepared_snapshot: ArtifactRef
    created_at: AwareDatetime = Field(default_factory=utc_now)
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def aligned(self) -> "Forecast":
        if (
            self.task_id != self.origin.task_id
            or self.selected_trial_id != self.origin.selected_trial_id
            or self.model_artifact != self.origin.model_artifact
            or self.target_timestamps != self.origin.targets
            or self.prediction_representation
            != ("point" if isinstance(self.prediction, PointForecast) else "quantile")
            or tuple(key.target for key in self.prediction.keys) != self.target_timestamps
            or any(key.origin != self.origin.latest_observed_at for key in self.prediction.keys)
        ):
            raise ValueError("forecast provenance or target alignment mismatch")
        return self


class ProfilePoint(Record):
    timestamp: AwareDatetime
    value: float = Field(allow_inf_nan=False)


class ReferenceDay(Record):
    label: Literal["D-1", "D-7", "D-365"]
    date: date
    available: bool
    reason: str | None = None
    load_profile: tuple[ProfilePoint, ...] = ()
    weather_profile: tuple[ProfilePoint, ...] = ()


class WeatherAnalog(Record):
    date: date
    distance: float = Field(ge=0, allow_inf_nan=False)
    load_profile: tuple[ProfilePoint, ...]
    weather_profile: tuple[ProfilePoint, ...]


class ReferenceAnalysis(Record):
    analysis_id: UUID = Field(default_factory=uuid4)
    forecast_id: UUID
    target_date: date
    target_timestamps: tuple[AwareDatetime, ...]
    d_minus_1: ReferenceDay
    d_minus_7: ReferenceDay
    d_minus_365: ReferenceDay
    target_weather_profile: tuple[ProfilePoint, ...] = ()
    weather_analogs: tuple[WeatherAnalog, ...] = ()
    weather_available: bool
    weather_unavailable_reason: str | None = None
    top_k: int = Field(default=3, gt=0, le=20)
    provenance: dict[str, str] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    schema_version: int = Field(default=1, ge=1)


class ManualReplacement(Record):
    timestamp: AwareDatetime
    values: tuple[float, ...] = Field(min_length=1)


class AdjustmentProposal(Record):
    adjustment_type: Literal["manual_override", "time_scaling", "load_scaling", "external_scaling"]
    selected_timestamps: tuple[AwareDatetime, ...] = ()
    start_at: AwareDatetime | None = None
    end_at: AwareDatetime | None = None
    lambda_value: float | None = None
    threshold: float | None = None
    comparison: Literal["gt", "lt"] | None = None
    external_variable: str | None = None
    manual_replacements: tuple[ManualReplacement, ...] = ()


class Adjustment(Record):
    adjustment_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    forecast_id: UUID
    parent_version_id: UUID
    adjustment_type: Literal["manual_override", "time_scaling", "load_scaling", "external_scaling"]
    selected_timestamps: tuple[AwareDatetime, ...] = ()
    start_at: AwareDatetime | None = None
    end_at: AwareDatetime | None = None
    lambda_value: float | None = None
    threshold: float | None = None
    comparison: Literal["gt", "lt"] | None = None
    external_variable: str | None = None
    manual_replacements: tuple[ManualReplacement, ...] = ()
    source: Literal["user", "deployment_operator_proposal"]
    user_request_text: str | None = None
    status: Literal["draft", "confirmed", "rejected", "applied"] = "draft"
    confirmed_at: AwareDatetime | None = None
    confirmation_id: UUID | None = None
    applied_version_id: UUID | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_request(self) -> "Adjustment":
        if self.start_at is not None and self.end_at is not None and self.start_at > self.end_at:
            raise ValueError("adjustment interval start must not follow end")
        if self.selected_timestamps and len(set(self.selected_timestamps)) != len(
            self.selected_timestamps
        ):
            raise ValueError("selected timestamps must be unique")
        if self.adjustment_type == "manual_override":
            if (
                not self.manual_replacements
                or self.lambda_value is not None
                or self.threshold is not None
                or self.comparison is not None
                or self.external_variable is not None
                or self.start_at is not None
                or self.end_at is not None
            ):
                raise ValueError("manual override needs replacement rows only")
            if len({item.timestamp for item in self.manual_replacements}) != len(
                self.manual_replacements
            ):
                raise ValueError("manual replacement timestamps must be unique")
        else:
            if self.manual_replacements:
                raise ValueError("nonmanual adjustment cannot have replacements")
            if (
                self.lambda_value is None
                or not isfinite(self.lambda_value)
                or self.lambda_value <= -1
            ):
                raise ValueError("scaling requires finite lambda greater than -1")
            if self.adjustment_type == "time_scaling" and self.start_at is None:
                raise ValueError("time scaling requires interval start")
            if self.adjustment_type in {"load_scaling", "external_scaling"}:
                if (
                    self.threshold is None
                    or not isfinite(self.threshold)
                    or self.comparison is None
                ):
                    raise ValueError("conditional scaling requires finite threshold and comparison")
            elif self.threshold is not None or self.comparison is not None:
                raise ValueError("unconditional scaling cannot have threshold or comparison")
            if self.adjustment_type == "external_scaling" and not self.external_variable:
                raise ValueError("external scaling requires variable name")
            if self.adjustment_type != "external_scaling" and self.external_variable is not None:
                raise ValueError("external variable applies only to external scaling")
        if self.status == "applied" and (
            self.confirmed_at is None
            or self.confirmation_id is None
            or self.applied_version_id is None
        ):
            raise ValueError("applied adjustment requires confirmation and version")
        if self.status == "rejected" and self.applied_version_id is not None:
            raise ValueError("rejected adjustment cannot have an applied version")
        return self


class VersionValueChange(Record):
    timestamp: AwareDatetime
    before: tuple[float, ...]
    after: tuple[float, ...]


class ForecastVersion(Record):
    version_id: UUID = Field(default_factory=uuid4)
    forecast_id: UUID
    version_number: int = Field(ge=0)
    parent_version_id: UUID | None = None
    adjustment_id: UUID | None = None
    prediction_representation: Literal["point", "quantile"]
    prediction: PointForecast | QuantileForecast
    affected_timestamps: tuple[AwareDatetime, ...] = ()
    value_changes: tuple[VersionValueChange, ...] = ()
    provenance: dict[str, str] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def original_immutable(self) -> "ForecastVersion":
        if self.prediction_representation != (
            "point" if isinstance(self.prediction, PointForecast) else "quantile"
        ):
            raise ValueError("version prediction representation mismatch")
        if self.version_number == 0:
            if (
                self.parent_version_id is not None
                or self.adjustment_id is not None
                or self.value_changes
            ):
                raise ValueError(
                    "original forecast version cannot have parent, adjustment or changes"
                )
        elif self.parent_version_id is None or self.adjustment_id is None:
            raise ValueError("descendant forecast version requires parent and adjustment")
        if tuple(change.timestamp for change in self.value_changes) != self.affected_timestamps:
            raise ValueError("version affected timestamps and value changes disagree")
        return self


class SensitivityRequest(Record):
    base_version_id: UUID
    variable: str = Field(min_length=1)
    perturbation_type: Literal["absolute", "percent"]
    value: float = Field(allow_inf_nan=False)


class SensitivityResult(Record):
    forecast_id: UUID
    base_version_id: UUID
    variable: str
    perturbation_type: Literal["absolute", "percent"]
    value: float = Field(allow_inf_nan=False)
    affected_input_timestamps: tuple[AwareDatetime, ...] = Field(min_length=1)
    baseline_prediction: PointForecast | QuantileForecast
    perturbed_prediction: PointForecast | QuantileForecast
    deltas: tuple[tuple[float, ...], ...]
    provenance: dict[str, str]
    created_at: AwareDatetime = Field(default_factory=utc_now)
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def aligned(self) -> "SensitivityResult":
        baseline = self.baseline_prediction
        perturbed = self.perturbed_prediction
        if (
            type(baseline) is not type(perturbed)
            or baseline.keys != perturbed.keys
            or len(self.deltas) != len(baseline.keys)
        ):
            raise ValueError("sensitivity forecasts and deltas must align")
        width = len(baseline.levels) if isinstance(baseline, QuantileForecast) else 1
        if any(
            len(row) != width or any(not isfinite(value) for value in row) for row in self.deltas
        ):
            raise ValueError("sensitivity delta rows must be finite and match output width")
        return self


class HistoryDeficit(Record):
    column: str
    required_steps: int = Field(ge=0)
    available_steps: int = Field(ge=0)
    missing_timestamps: tuple[AwareDatetime, ...] = ()


class DeploymentValidation(Record):
    ready: bool
    origin: ForecastOrigin
    history_deficits: tuple[HistoryDeficit, ...] = ()
    missing_auxiliaries: tuple[str, ...] = ()


class FutureAuxiliaryValue(Record):
    column: str = Field(min_length=1)
    valid_at: AwareDatetime
    value: float = Field(allow_inf_nan=False)
    available_at: AwareDatetime | None = None
    source_ref: str | None = None
    protocol_id: str | None = None


class DeploymentSession(Record):
    session_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    run_id: UUID
    selected_trial_id: UUID
    model_artifact: ArtifactRef
    state: Literal[
        "READY_FOR_DEPLOYMENT",
        "WAITING_FOR_DEPLOYMENT_DATA",
        "VALIDATING_DEPLOYMENT_INPUT",
        "READY_TO_FORECAST",
        "FORECAST_GENERATED",
        "REFERENCE_ANALYSIS_AVAILABLE",
        "ADJUSTMENT_DRAFT_PENDING",
        "WAITING_FOR_USER_CONFIRMATION",
        "ADJUSTMENT_APPLIED",
        "FORECAST_VERSION_UPDATED",
        "COMPLETED",
    ] = "READY_FOR_DEPLOYMENT"
    version: int = Field(default=0, ge=0)
    raw_upload: ArtifactRef | None = None
    upload_extension: Literal[".csv", ".parquet"] | None = None
    future_auxiliaries: tuple[FutureAuxiliaryValue, ...] = ()
    context_artifact: ArtifactRef | None = None
    validation: DeploymentValidation | None = None
    forecast_id: UUID | None = None
    reference_analysis_id: UUID | None = None
    current_version_id: UUID | None = None
    pending_adjustment_id: UUID | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class DeploymentReadiness(Record):
    ready: bool
    task_id: UUID
    optimization_run_id: UUID
    selected_trial_id: UUID
    model_artifact: ArtifactRef
    session: DeploymentSession | None = None


class Job(Record):
    job_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    run_id: UUID | None = None
    job_type: str = Field(min_length=1)
    status: JobStatus = JobStatus.QUEUED
    progress_completed: int = Field(default=0, ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    error: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)


class Message(Record):
    message_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    run_id: UUID | None = None
    source_role: Actor
    target_role: Actor | None = None
    topic: Topic
    kind: MessageKind
    message_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    correlation_id: UUID = Field(default_factory=uuid4)
    parent_message_id: UUID | None = None
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def message_semantics(self) -> "Message":
        if self.kind == MessageKind.USER:
            if (
                self.source_role != Actor.USER
                or self.target_role != Actor.TASK_MANAGER
                or self.topic != Topic.CHAT
            ):
                raise ValueError("user messages must address the Task Manager on chat")
        if self.kind == MessageKind.RESULT and self.parent_message_id is None:
            raise ValueError("result messages require a parent request")
        if self.topic == Topic.CHAT and self.target_role == Actor.USER:
            if self.source_role != Actor.TASK_MANAGER:
                raise ValueError("only the Task Manager may reply to the user")
        return self


class Event(Message):
    kind: Literal[MessageKind.EVENT] = MessageKind.EVENT
    outcome: str | None = None


class LLMCall(Record):
    call_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    message_id: UUID | None = None
    correlation_id: UUID
    runtime_name: str
    runtime_version: str | None = None
    provider: str
    model: str | None = None
    agent_role: Actor
    request_id: str | None = None
    trace_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    status: str
    error: str | None = None
    prompt_version: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ExperimentRun(Record):
    run_id: UUID = Field(default_factory=uuid4)
    spec_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: int = Field(default=0, ge=0)
    status: JobStatus = JobStatus.QUEUED
    task_definition: TaskDefinition | None = None
    dataset_snapshot: DatasetSnapshot | None = None
    protocol: EvaluationProtocol | None = None
    origin_schedule: OriginSchedule | None = None
    templates: dict[ModelFamily, ResolvedCandidate] = Field(default_factory=dict)
    canonical_space: SearchSpace | None = None
    effective_space: SearchSpace | None = None
    backend_config: BackendConfig | None = None
    metric_config: ValidationMetricConfig | None = None
    experiment_seed: int | None = None
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
    trials: tuple[TrialResult, ...] = ()
    space_history: tuple[SearchSpace, ...] = ()
    guidance_history: tuple[GuidanceCommand, ...] = ()
    pending_requests: tuple[CandidateRequest, ...] = ()
    next_allocation: dict[ModelFamily, int] = Field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()
    selected_trial_id: UUID | None = None
    final_test_metrics: dict[str, float] | None = None
    completed_rounds: int = Field(default=0, ge=0)
    stale_rounds: int = Field(default=0, ge=0)
    stop_requested: bool = False
    started_at: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None


class ArtifactManifest(Record):
    spec_id: str = Field(min_length=1)
    run_id: UUID
    command: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=utc_now)
