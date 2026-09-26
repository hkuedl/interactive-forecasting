"""Typed, persisted Preparation drafts and deterministic outputs."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.forecasting import AuxiliaryPolicy, OutputConfig
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.models import (
    ArtifactRef,
    ColumnMapping,
    DatasetSnapshot,
    DatasetSource,
    EvaluationProtocol,
    TaskDefinition,
)
from interactive_forecasting.domain.search import SearchSpace


class PreparationStep(str, Enum):
    UPLOAD_DATASET = "UPLOAD_DATASET"
    INSPECT_SCHEMA = "INSPECT_SCHEMA"
    PROPOSE_COLUMN_MAPPING = "PROPOSE_COLUMN_MAPPING"
    CONFIRM_COLUMN_MAPPING = "CONFIRM_COLUMN_MAPPING"
    ANALYZE_DATA_QUALITY = "ANALYZE_DATA_QUALITY"
    PROPOSE_PREPARATION = "PROPOSE_PREPARATION"
    CONFIRM_PREPARATION = "CONFIRM_PREPARATION"
    APPLY_PREPARATION = "APPLY_PREPARATION"
    GENERATE_DATASET_OVERVIEW = "GENERATE_DATASET_OVERVIEW"
    CONFIGURE_FORECAST_TASK = "CONFIGURE_FORECAST_TASK"
    REVIEW_PREPARATION = "REVIEW_PREPARATION"
    CONFIRM_TASK = "CONFIRM_TASK"
    PREPARATION_READY = "PREPARATION_READY"


PREPARATION_PATH = tuple(PreparationStep)


class PreparationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ColumnProfile(PreparationModel):
    name: str
    physical_dtype: str
    missing_count: int
    unique_count: int
    sample_values: tuple[str, ...]
    datetime_parse_ratio: float
    numeric_parse_ratio: float
    binary: bool
    identifier_like: bool
    constant: bool
    semantic_hints: tuple[str, ...] = ()


class SchemaInspection(PreparationModel):
    row_count: int
    columns: tuple[ColumnProfile, ...]
    plausible_time_columns: tuple[str, ...]
    interval_candidates: tuple[str, ...] = ()
    time_ordered: bool | None = None
    timestamp_unique: bool | None = None
    missing_timestamp_count: int | None = None


class ColumnMappingDraft(PreparationModel):
    timestamp_column: str | None = None
    target_column: str | None = None
    temperature_column: str | None = None
    other_features: tuple[str, ...] = ()
    excluded_features: tuple[str, ...] = ()
    temperature_available: bool | None = None
    ambiguous_roles: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def distinct(self) -> ColumnMappingDraft:
        chosen = [
            x for x in (self.timestamp_column, self.target_column, self.temperature_column) if x
        ]
        if len(chosen) != len(set(chosen)):
            raise ValueError("semantic role columns must be distinct")
        if set(self.other_features) & set(chosen):
            raise ValueError("other features overlap semantic roles")
        if set(self.other_features) & set(self.excluded_features):
            raise ValueError("feature cannot be included and excluded")
        if self.temperature_available is False and self.temperature_column is not None:
            raise ValueError("temperature cannot be mapped when unavailable")
        return self


class DatasetCapabilities(PreparationModel):
    has_temperature: bool
    available_other_features: tuple[str, ...]
    timestamp_frequency: str
    target_available: bool = True
    known_ahead_candidates: tuple[str, ...] = ()
    supported_feature_groups: tuple[str, ...]


class DataQualityReport(PreparationModel):
    row_count: int
    timestamp_parse_failures: int
    ordered: bool
    duplicate_timestamps: int
    missing_timestamps: int
    frequency: str | None
    irregular_intervals: int = 0
    missing_values: dict[str, int]
    invalid_numeric: dict[str, int]
    thousands_separator_counts: dict[str, int] = Field(default_factory=dict)
    constant_columns: tuple[str, ...]
    outlier_counts: dict[str, int]
    usable_start: str | None
    usable_end: str | None
    warnings: tuple[str, ...] = ()


class PreparationPlan(PreparationModel):
    version: int = 1
    timezone_name: str = "UTC"
    duplicate_policy: Literal["reject", "first", "last", "mean"] = "reject"
    missing_timestamp_policy: Literal["reject", "interpolate", "forward_fill"] = "reject"
    missing_value_policy: Literal["reject", "interpolate", "forward_fill", "drop"] = "reject"
    invalid_row_policy: Literal["reject", "drop"] = "reject"
    sort_chronologically: bool = True
    normalize_thousands_separators: bool = True
    rationale: tuple[str, ...] = ()


class PreparedSnapshot(PreparationModel):
    dataset: DatasetSnapshot
    mapping: ColumnMapping
    plan: PreparationPlan
    row_count: int
    time_start: str
    time_end: str
    columns: tuple[str, ...]
    applied_transformations: tuple[str, ...]
    capabilities: DatasetCapabilities


class ChartSeries(PreparationModel):
    name: str
    points: tuple[tuple[str, float], ...]


class DatasetOverview(PreparationModel):
    row_count: int
    time_start: str
    time_end: str
    frequency: str
    mapped_columns: dict[str, str]
    missing_values: dict[str, int]
    duplicate_timestamps: int
    load_min: float
    load_max: float
    load_mean: float
    other_feature_stats: dict[str, dict[str, float]]
    primary: ChartSeries
    secondary: ChartSeries | None = None
    missing_timestamp_count: int = 0


class ForecastTaskDraft(PreparationModel):
    delta: int = Field(default=1, gt=0)
    horizon: int = Field(default=1, gt=0)
    time_unit: Literal["samples", "hours"] = "samples"
    output: OutputConfig = Field(default_factory=OutputConfig)
    objective_id: Literal["mae", "mape", "crps", "weighted_mae", "asymmetric_mae"] = "mae"
    metric_spec: MetricSpec | None = None
    temperature_policy: AuxiliaryPolicy | None = None
    other_policies: dict[str, AuxiliaryPolicy] = Field(default_factory=dict)
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def consistent(self) -> ForecastTaskDraft:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-8:
            raise ValueError("evaluation split fractions must sum to 1")
        if self.objective_id == "weighted_mae" and self.metric_spec is None:
            raise ValueError("weighted MAE requires an explicit MetricSpec")
        if self.objective_id == "asymmetric_mae" and self.metric_spec is None:
            raise ValueError("asymmetric MAE requires an explicit MetricSpec")
        if self.metric_spec is not None and self.metric_spec.objective_id != self.objective_id:
            raise ValueError("task draft metric specification differs from objective")
        if self.output.representation == "point" and self.objective_id == "crps":
            raise ValueError("CRPS requires quantile output")
        if self.output.representation == "quantile" and self.objective_id != "crps":
            raise ValueError("quantile output currently requires CRPS objective")
        return self


class WorkingField(str, Enum):
    DATASET_NAME = "dataset_name"
    TIMESTAMP_COLUMN = "timestamp_column"
    TARGET_COLUMN = "target_column"
    TEMPERATURE_COLUMN = "temperature_column"
    TEMPERATURE_AVAILABLE = "temperature_available"
    OTHER_FEATURES = "other_features"
    TIMEZONE_NAME = "timezone_name"
    DUPLICATE_POLICY = "duplicate_policy"
    MISSING_TIMESTAMP_POLICY = "missing_timestamp_policy"
    MISSING_VALUE_POLICY = "missing_value_policy"
    INVALID_ROW_POLICY = "invalid_row_policy"
    SORT_CHRONOLOGICALLY = "sort_chronologically"
    NORMALIZE_THOUSANDS_SEPARATORS = "normalize_thousands_separators"
    FORECAST_INTENT = "forecast_intent"
    DELTA = "delta"
    HORIZON = "horizon"
    TIME_UNIT = "time_unit"
    OUTPUT = "output"
    OBJECTIVE_ID = "objective_id"
    METRIC_SPEC = "metric_spec"
    TEMPERATURE_POLICY = "temperature_policy"
    OTHER_POLICIES = "other_policies"
    TRAIN_FRACTION = "train_fraction"
    VALIDATION_FRACTION = "validation_fraction"
    TEST_FRACTION = "test_fraction"
    SEED = "seed"


class WorkingEntry(PreparationModel):
    field: WorkingField
    value: str = Field(max_length=2000)
    status: Literal["inferred", "proposed", "user_provided", "confirmed"]
    source_message_id: UUID | None = None


class WorkingSuggestion(PreparationModel):
    field: WorkingField
    value: str = Field(max_length=2000)
    status: Literal["inferred", "user_provided"]


class PreparationWorkingState(PreparationModel):
    entries: tuple[WorkingEntry, ...] = ()
    conflicts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_fields(self) -> PreparationWorkingState:
        if len({entry.field for entry in self.entries}) != len(self.entries):
            raise ValueError("working state has duplicate fields")
        return self


class PreparationRecord(PreparationModel):
    task_id: UUID
    version: int = 0
    step: PreparationStep = PreparationStep.UPLOAD_DATASET
    source: DatasetSource | None = None
    source_artifact: ArtifactRef | None = None
    original_filename: str | None = None
    schema_inspection: SchemaInspection | None = None
    mapping_draft: ColumnMappingDraft | None = None
    confirmed_mapping: ColumnMapping | None = None
    capabilities: DatasetCapabilities | None = None
    quality: DataQualityReport | None = None
    plan_draft: PreparationPlan | None = None
    plan_confirmed: bool = False
    prepared: PreparedSnapshot | None = None
    overview: DatasetOverview | None = None
    task_draft: ForecastTaskDraft | None = None
    working_state: PreparationWorkingState = Field(default_factory=PreparationWorkingState)
    definition: TaskDefinition | None = None
    protocol: EvaluationProtocol | None = None
    effective_search_space: SearchSpace | None = None
    frozen: bool = False
