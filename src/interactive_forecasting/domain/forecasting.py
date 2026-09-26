"""Resolved forecasting contracts. Choices are explicit, never inferred from search policy."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NaiveDatetime, model_validator

from interactive_forecasting.domain.types import ModelFamily

ClockDatetime = AwareDatetime | NaiveDatetime


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AuxiliaryPolicy(FrozenModel):
    kind: Literal["observed", "known_ahead", "forecast", "perfect_forecast"]
    role: Literal["temperature", "other"] = "other"
    source_ref: str | None = None
    protocol_id: str | None = None

    @model_validator(mode="after")
    def provenance_required(self) -> AuxiliaryPolicy:
        if self.kind == "forecast" and not self.source_ref:
            raise ValueError("forecast auxiliary requires a product source reference")
        if self.kind == "perfect_forecast" and (self.role != "temperature" or not self.protocol_id):
            raise ValueError("perfect_forecast requires temperature role and protocol id")
        return self


class IndexSpec(FrozenModel):
    delta: int = Field(ge=0)
    horizon: int = Field(gt=0)
    frequency: str
    offset_unit: Literal["samples", "hours"]
    anchor: Literal["start_at_delta", "end_at_delta"]

    @model_validator(mode="after")
    def offsets_positive(self) -> IndexSpec:
        if self.anchor == "end_at_delta" and self.delta < self.horizon:
            raise ValueError("end_at_delta requires all targets after the origin")
        if self.anchor == "start_at_delta" and self.delta == 0:
            raise ValueError("start_at_delta requires all targets after the origin")
        return self


class GroupSelection(FrozenModel):
    mode: Literal["none", "correlation", "fixed"]
    ratio: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid_ratio(self) -> GroupSelection:
        if (self.mode == "correlation") != (self.ratio is not None):
            raise ValueError("ratio is required only for correlation selection")
        return self


class CoreFeatureRecipe(FrozenModel):
    version: str
    calendar: Literal["none", "numerical", "categorical", "trigonometric"] = "none"
    load_lags: tuple[int, ...] = ()
    load_selection: GroupSelection | None = None
    temperature_column: str | None = None
    temperature_lags: tuple[int, ...] = ()
    temperature_leads: tuple[int, ...] = ()
    temperature_daily_days: tuple[int, ...] = ()
    temperature_selection: GroupSelection | None = None
    auxiliary_lags: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    auxiliary_leads: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    other_selection: GroupSelection | None = None
    interactions: tuple[tuple[str, str], ...] = ()
    sequence_length: int | None = Field(default=None, gt=0)
    sequence_frequency: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_recipe(self) -> CoreFeatureRecipe:
        all_lags = (*self.load_lags, *self.temperature_lags)
        if any(value < 0 for value in all_lags) or any(
            value < 0 for lags in self.auxiliary_lags.values() for value in lags
        ):
            raise ValueError("lags must be nonnegative")
        if any(lead <= 0 for lead in self.temperature_leads) or any(
            lead <= 0 for leads in self.auxiliary_leads.values() for lead in leads
        ):
            raise ValueError("auxiliary leads must be positive")
        if any(day <= 0 for day in self.temperature_daily_days):
            raise ValueError("temperature daily averages require previous days")
        if (self.sequence_length is None) != (self.sequence_frequency is None):
            raise ValueError("sequence length and frequency must be set together")
        if (
            self.temperature_lags or self.temperature_leads or self.temperature_daily_days
        ) and not self.temperature_column:
            raise ValueError("temperature column required for temperature features")
        for name, selection in (
            ("temperature", self.temperature_selection),
            ("other", self.other_selection),
        ):
            if selection is not None and selection.mode == "fixed":
                raise ValueError(f"{name} does not support fixed selection")
        if self.temperature_selection is not None:
            if self.temperature_selection.mode == "none" and (
                self.temperature_lags or self.temperature_daily_days
            ):
                raise ValueError("none temperature selection cannot contain historical candidates")
            if self.temperature_selection.mode == "correlation" and not (
                self.temperature_lags or self.temperature_daily_days
            ):
                raise ValueError("temperature correlation needs candidates")
        if self.load_selection is not None:
            if self.load_selection.mode == "none" and self.load_lags:
                raise ValueError("none load selection cannot contain candidates")
            if self.load_selection.mode in {"correlation", "fixed"} and not self.load_lags:
                raise ValueError("load selection needs candidates")
        if self.other_selection is not None:
            if self.other_selection.mode == "none" and (
                self.auxiliary_lags or self.auxiliary_leads
            ):
                raise ValueError("none other selection cannot contain candidates")
            if self.other_selection.mode == "correlation" and not (
                self.auxiliary_lags or self.auxiliary_leads
            ):
                raise ValueError("other correlation needs candidates")
        return self


class PreprocessConfig(FrozenModel):
    scale: Literal["none", "standard"] = "none"
    selection: Literal["none", "pearson"] = "none"
    selection_ratio: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_selection(self) -> PreprocessConfig:
        if self.selection == "pearson" and self.selection_ratio is None:
            raise ValueError("Pearson selection requires explicit ratio")
        if self.selection == "none" and self.selection_ratio is not None:
            raise ValueError("selection ratio requires Pearson selection")
        return self


class TrainingConfig(FrozenModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0)
    patience: int | None = Field(default=None, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    loss: Literal["native", "mae", "mse", "pinball"] = "native"
    deterministic: bool = True
    shuffle: bool = True
    min_delta: float = Field(default=0.0, ge=0)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    sgd_momentum: float = Field(default=0.0, ge=0)
    gradient_clip_norm: float | None = Field(default=None, gt=0)
    validation_metric: Literal["training_loss", "mae"] = "training_loss"

    @model_validator(mode="after")
    def optimizer_parameters(self) -> TrainingConfig:
        if any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.optimizer == "adam" and self.sgd_momentum != 0:
            raise ValueError("SGD momentum requires SGD")
        if self.optimizer == "sgd" and (
            self.adam_betas != (0.9, 0.999) or self.adam_epsilon != 1e-8
        ):
            raise ValueError("Adam parameters require Adam")
        return self


class OutputConfig(FrozenModel):
    representation: Literal["point", "quantile"] = "point"
    quantile_levels: tuple[float, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def default_quantiles(cls, value: object) -> object:
        if (
            isinstance(value, dict)
            and value.get("representation") == "quantile"
            and "quantile_levels" not in value
        ):
            return {**value, "quantile_levels": (0.1, 0.5, 0.9)}
        return value

    @model_validator(mode="after")
    def validate_levels(self) -> OutputConfig:
        if self.representation == "quantile":
            if len(self.quantile_levels) < 2:
                raise ValueError("quantile output requires at least two levels")
            if any(not 0 < level < 1 for level in self.quantile_levels):
                raise ValueError("quantile levels must be in (0, 1)")
            if tuple(sorted(set(self.quantile_levels))) != self.quantile_levels:
                raise ValueError("quantile levels must be unique and sorted")
        elif self.quantile_levels:
            raise ValueError("point output cannot specify quantile levels")
        return self


class ResolvedCandidate(FrozenModel):
    family: ModelFamily
    auxiliary_policies: dict[str, AuxiliaryPolicy]
    hyperparameters: dict[str, int | float | str | bool | None]
    features: CoreFeatureRecipe
    preprocessing: PreprocessConfig
    training: TrainingConfig
    output: OutputConfig
    index: IndexSpec
    configuration_version: str


class OriginSchedule(FrozenModel):
    version: str
    origins: dict[str, tuple[ClockDatetime, ...]]

    @model_validator(mode="after")
    def chronological(self) -> OriginSchedule:
        if not self.origins:
            raise ValueError("origin schedule cannot be empty")
        for series_id, times in self.origins.items():
            if times and len({value.tzinfo is None for value in times}) != 1:
                raise ValueError("origin times cannot mix source-clock and zoned timestamps")
            if not series_id or not times or tuple(sorted(set(times))) != times:
                raise ValueError("origin times must be nonempty, unique and chronological")
        return self


class ForecastKey(FrozenModel):
    series_id: str
    origin: ClockDatetime
    target: ClockDatetime

    @model_validator(mode="after")
    def forward_only(self) -> ForecastKey:
        if (self.origin.tzinfo is None) != (self.target.tzinfo is None):
            raise ValueError("forecast key cannot mix source-clock and zoned timestamps")
        if self.target <= self.origin:
            raise ValueError("forecast target must follow origin")
        return self


class PointForecast(FrozenModel):
    keys: tuple[ForecastKey, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def aligned(self) -> PointForecast:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("point forecast keys and values must align")
        if len(set(self.keys)) != len(self.keys) or any(not isfinite(v) for v in self.values):
            raise ValueError("point forecast keys must be unique and values finite")
        return self


class QuantileForecast(FrozenModel):
    keys: tuple[ForecastKey, ...]
    levels: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def aligned(self) -> QuantileForecast:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("quantile forecast keys and values must align")
        if len(self.levels) < 2 or tuple(sorted(set(self.levels))) != self.levels:
            raise ValueError("quantile levels must be unique and sorted")
        if any(not 0 < level < 1 for level in self.levels):
            raise ValueError("quantile levels must be in (0, 1)")
        if any(len(row) != len(self.levels) for row in self.values):
            raise ValueError("quantile values must match levels")
        if any(tuple(sorted(row)) != row for row in self.values):
            raise ValueError("quantile values must be nondecreasing")
        if len(set(self.keys)) != len(self.keys) or any(
            not isfinite(value) for row in self.values for value in row
        ):
            raise ValueError("quantile keys must be unique and values finite")
        return self


class SampleForecast(FrozenModel):
    keys: tuple[ForecastKey, ...]
    samples: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def aligned(self) -> SampleForecast:
        if not self.keys or len(self.keys) != len(self.samples):
            raise ValueError("sample forecast keys and samples must align")
        if any(not row for row in self.samples):
            raise ValueError("each target needs at least one sample")
        if len(set(self.keys)) != len(self.keys) or any(
            not isfinite(value) for row in self.samples for value in row
        ):
            raise ValueError("sample keys must be unique and values finite")
        return self


PredictiveDistribution = QuantileForecast | SampleForecast


class LabelVector(FrozenModel):
    keys: tuple[ForecastKey, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def aligned(self) -> LabelVector:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("label keys and values must align")
        if len(set(self.keys)) != len(self.keys) or any(not isfinite(v) for v in self.values):
            raise ValueError("label keys must be unique and values finite")
        return self


class Partition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FeatureProvenance(FrozenModel):
    train_keys: tuple[ForecastKey, ...]
    input_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    group_candidates: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    group_selected: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    group_settings: dict[str, GroupSelection] = Field(default_factory=dict)
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    fit_timestamp: datetime | None = None
    structured: FeatureProvenance | None = None


class MetricPlan(FrozenModel):
    validation_objective: Literal["mae", "mape", "weighted_mae", "weighted_mse", "asymmetric"]
    reporting_metrics: tuple[Literal["mae", "mape"], ...]
    mape_zero_policy: Literal["error", "omit", "epsilon"] | None = None
    mape_epsilon: float | None = Field(default=None, gt=0)
    asymmetric_over: float | None = Field(default=None, ge=0)
    asymmetric_under: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def applicable_parameters(self) -> MetricPlan:
        needs_mape = self.validation_objective == "mape" or "mape" in self.reporting_metrics
        if needs_mape != (self.mape_zero_policy is not None):
            raise ValueError("MAPE requires an explicit zero-target policy")
        if self.mape_zero_policy == "epsilon" and self.mape_epsilon is None:
            raise ValueError("epsilon MAPE requires an explicit epsilon")
        if self.mape_zero_policy != "epsilon" and self.mape_epsilon is not None:
            raise ValueError("MAPE epsilon applies only to epsilon policy")
        needs_asymmetric = self.validation_objective == "asymmetric"
        if needs_asymmetric:
            if (
                self.asymmetric_over is None
                or self.asymmetric_under is None
                or self.asymmetric_over + self.asymmetric_under == 0
            ):
                raise ValueError("asymmetric objective requires positive total coefficients")
        elif self.asymmetric_over is not None or self.asymmetric_under is not None:
            raise ValueError("asymmetric coefficients require asymmetric objective")
        return self


class MetricReport(FrozenModel):
    objective_name: str
    objective_value: float
    reporting: dict[str, float]
