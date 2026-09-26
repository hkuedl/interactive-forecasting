"""Validated normalized series, canonical forecast indexing and chronological partitions."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from interactive_forecasting.domain.forecasting import (
    AuxiliaryPolicy,
    ForecastKey,
    IndexSpec,
    LabelVector,
    OriginSchedule,
    Partition,
)
from interactive_forecasting.domain.models import ArtifactRef, EvaluationProtocol
from interactive_forecasting.storage.artifacts import ArtifactStore

SOURCE_CLOCK = "source_clock"


@dataclass(frozen=True)
class NormalizedSeries:
    frame: pd.DataFrame
    frequency: str
    timezone_name: str
    auxiliary_availability: dict[str, AuxiliaryPolicy]

    @classmethod
    def from_csv_artifact(
        cls,
        store: ArtifactStore,
        reference: ArtifactRef,
        *,
        frequency: str,
        timezone_name: str,
        auxiliary_availability: dict[str, AuxiliaryPolicy],
    ) -> NormalizedSeries:
        frame = pd.read_csv(BytesIO(store.read_bytes(reference)))
        return cls.validate(frame, frequency, timezone_name, auxiliary_availability)

    @classmethod
    def validate(
        cls,
        frame: pd.DataFrame,
        frequency: str,
        timezone_name: str,
        auxiliary_availability: dict[str, AuxiliaryPolicy],
    ) -> NormalizedSeries:
        required = {"timestamp", "series_id", "target"}
        if not required.issubset(frame.columns):
            raise ValueError("normalized data requires timestamp, series_id and target")
        if frame.empty:
            raise ValueError("normalized data is empty")
        data = frame.copy()
        raw_times = [pd.Timestamp(value) for value in data["timestamp"]]
        if timezone_name == SOURCE_CLOCK:
            if any(value.tzinfo is not None for value in raw_times):
                raise ValueError("source-clock timestamps must not carry timezone offsets")
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
        else:
            zone = ZoneInfo(timezone_name)
            if any(value.tzinfo is None for value in raw_times):
                raise ValueError("timestamps must include explicit timezone offsets")
            if any(value.utcoffset() != value.tz_convert(zone).utcoffset() for value in raw_times):
                raise ValueError("timestamp offset conflicts with declared timezone")
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
        if data["timestamp"].isna().any():
            raise ValueError("timestamps cannot be missing")
        if not data["series_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError("series ids must be nonempty strings")
        if data["series_id"].isna().any() or data["target"].isna().any():
            raise ValueError("series id and target cannot be missing")
        if data.duplicated(["series_id", "timestamp"]).any():
            raise ValueError("duplicate timestamp within series")
        step = pd.Timedelta(frequency)
        if pd.isna(step) or step <= pd.Timedelta(0):
            raise ValueError("frequency must be positive and fixed")
        if any(
            not isinstance(policy, AuxiliaryPolicy) for policy in auxiliary_availability.values()
        ):
            raise ValueError("every auxiliary requires a typed availability policy")
        forecast_columns = {
            f"{column}__available_at"
            for column, policy in auxiliary_availability.items()
            if policy.kind == "forecast"
        }
        optional_availability_columns = {
            f"{column}__available_at"
            for column, policy in auxiliary_availability.items()
            if policy.kind in {"observed", "known_ahead"}
            and f"{column}__available_at" in data.columns
        }
        availability_columns = forecast_columns | optional_availability_columns
        auxiliaries = set(data.columns) - required - availability_columns
        if auxiliaries != set(auxiliary_availability) or not forecast_columns.issubset(
            data.columns
        ):
            raise ValueError("auxiliary columns/policies or forecast issue times are missing")
        for column in availability_columns:
            raw_availability = [pd.Timestamp(value) for value in data[column]]
            if timezone_name == SOURCE_CLOCK:
                if any(value.tzinfo is not None for value in raw_availability):
                    raise ValueError("source-clock availability times must not carry offsets")
                data[column] = pd.to_datetime(data[column], errors="raise")
            else:
                if any(value.tzinfo is None for value in raw_availability):
                    raise ValueError("auxiliary availability times must be timezone aware")
                data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
        for column in ("target", *sorted(auxiliaries)):
            data[column] = pd.to_numeric(data[column], errors="raise")
            if not np.isfinite(data[column].to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite values in {column}")
        data = data.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
        for _, group in data.groupby("series_id", sort=False):
            differences = group["timestamp"].diff().dropna()
            if not differences.eq(step).all():
                raise ValueError("missing timestamp or invalid frequency")
        return cls(data, frequency, timezone_name, dict(auxiliary_availability))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)

    def at(self, series_id: str, timestamp: pd.Timestamp, column: str) -> float:
        row = self.frame.loc[
            (self.frame.series_id == series_id) & (self.frame.timestamp == timestamp), column
        ]
        if len(row) != 1:
            raise ValueError(f"missing {column} for {series_id} at {timestamp}")
        return float(row.iloc[0])

    def has_availability_time(self, column: str) -> bool:
        return f"{column}__available_at" in self.frame.columns

    def available_at(self, series_id: str, timestamp: pd.Timestamp, column: str) -> pd.Timestamp:
        availability_column = f"{column}__available_at"
        row = self.frame.loc[
            (self.frame.series_id == series_id) & (self.frame.timestamp == timestamp),
            availability_column,
        ]
        if len(row) != 1:
            raise ValueError("missing auxiliary availability timestamp")
        return pd.Timestamp(row.iloc[0])

    def timestamps(self, series_id: str) -> tuple[pd.Timestamp, ...]:
        return tuple(self.frame.loc[self.frame.series_id == series_id, "timestamp"])


@dataclass(frozen=True)
class ForecastIndex:
    spec: IndexSpec

    def __post_init__(self) -> None:
        if pd.isna(self.step) or self.step <= pd.Timedelta(0):
            raise ValueError("frequency must be positive and fixed")
        self.offset(self.spec.delta)
        if self.spec.horizon > 1:
            self.offset(1)

    @property
    def step(self) -> pd.Timedelta:
        return pd.Timedelta(self.spec.frequency)

    def offset(self, count: int) -> pd.Timedelta:
        if self.spec.offset_unit == "samples":
            return self.step * count
        duration = pd.Timedelta(hours=count)
        if duration % self.step:
            raise ValueError("hour offsets must align with sampling frequency")
        return duration

    def targets(self, origin: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
        start = self.spec.delta
        if self.spec.anchor == "end_at_delta":
            start -= self.spec.horizon - 1
        values = tuple(origin + self.offset(start + i) for i in range(self.spec.horizon))
        if any(target <= origin for target in values):
            raise ValueError("targets must follow the forecast origin")
        return values

    def lag(self, origin: pd.Timestamp, steps: int) -> pd.Timestamp:
        if steps < 0:
            raise ValueError("negative lag would access future information")
        return origin - self.step * steps


@dataclass(frozen=True)
class ForecastExample:
    key: ForecastKey
    label: float | None
    partition: Partition | None = None


def split_examples(
    data: NormalizedSeries,
    index: ForecastIndex,
    protocol: EvaluationProtocol,
    schedule: OriginSchedule,
) -> dict[Partition, tuple[ForecastExample, ...]]:
    if data.frequency != index.spec.frequency:
        raise ValueError("index frequency differs from dataset frequency")
    windows = {
        Partition.TRAIN: protocol.train,
        Partition.VALIDATION: protocol.validation,
        Partition.TEST: protocol.test,
    }
    result: dict[Partition, list[ForecastExample]] = {part: [] for part in Partition}
    series_ids = set(str(value) for value in data.frame.series_id.unique())
    if set(schedule.origins) != series_ids:
        raise ValueError("origin schedule must name exactly the dataset series")
    for series_id, origin_times in schedule.origins.items():
        available = set(data.timestamps(series_id))
        for origin_time in origin_times:
            origin = pd.Timestamp(origin_time)
            if origin not in available:
                raise ValueError("scheduled origin is missing from dataset")
            for target in index.targets(origin):
                if target not in available:
                    continue
                for part, window in windows.items():
                    if window is not None and window.start <= target < window.end:
                        key = ForecastKey(
                            series_id=str(series_id),
                            origin=origin.to_pydatetime(),
                            target=target.to_pydatetime(),
                        )
                        result[part].append(ForecastExample(key, None, part))
                        break
    if not result[Partition.TRAIN] or not result[Partition.VALIDATION]:
        raise ValueError("train and validation partitions need forecast examples")
    return {part: tuple(items) for part, items in result.items()}


def labels_for(
    data: NormalizedSeries, examples: tuple[ForecastExample, ...], *, partition: Partition
) -> LabelVector:
    if any(example.partition != partition for example in examples):
        raise ValueError("example partition does not match requested label access")
    if partition == Partition.TEST:
        raise ValueError("test labels are sealed; use explicit final evaluation access")
    return _labels(data, examples)


def final_test_labels(data: NormalizedSeries, examples: tuple[ForecastExample, ...]) -> LabelVector:
    """Explicit final-evaluation boundary; never called during candidate selection."""
    if any(example.partition != Partition.TEST for example in examples):
        raise ValueError("final evaluation requires test examples")
    return _labels(data, examples)


def _labels(data: NormalizedSeries, examples: tuple[ForecastExample, ...]) -> LabelVector:
    return LabelVector(
        keys=tuple(example.key for example in examples),
        values=tuple(
            data.at(example.key.series_id, pd.Timestamp(example.key.target), "target")
            for example in examples
        ),
    )
