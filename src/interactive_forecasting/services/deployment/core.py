"""Deterministic 5A deployment input resolution; no fitting or agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd

from interactive_forecasting.domain.forecasting import CoreFeatureRecipe, ForecastKey
from interactive_forecasting.domain.models import (
    DeploymentValidation,
    ForecastOrigin,
    FutureAuxiliaryValue,
    HistoryDeficit,
)
from interactive_forecasting.domain.preparation import PreparedSnapshot
from interactive_forecasting.services.data.core import (
    SOURCE_CLOCK,
    ForecastExample,
    ForecastIndex,
    NormalizedSeries,
)
from interactive_forecasting.services.models.core import FittedCoreModel
from interactive_forecasting.services.preparation import _prepare_frame, read_tabular
from interactive_forecasting.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class DeploymentSeries(NormalizedSeries):
    """Historical validated rows plus explicitly supplied, policy-checked future auxiliaries."""

    future: dict[tuple[str, pd.Timestamp], FutureAuxiliaryValue]
    origin: pd.Timestamp

    def at(self, series_id: str, timestamp: pd.Timestamp, column: str) -> float:
        if timestamp > self.origin:
            if column == "target":
                raise ValueError("future target is unavailable at deployment origin")
            value = self.future.get((column, timestamp))
            if value is None:
                raise ValueError(f"missing future auxiliary {column} at {timestamp}")
            return value.value
        return super().at(series_id, timestamp, column)

    def available_at(self, series_id: str, timestamp: pd.Timestamp, column: str) -> pd.Timestamp:
        if timestamp > self.origin:
            value = self.future.get((column, timestamp))
            if value is None:
                raise ValueError(f"missing future auxiliary {column} at {timestamp}")
            return pd.Timestamp(value.available_at or self.origin)
        return super().available_at(series_id, timestamp, column)


@dataclass(frozen=True)
class PreparedDeploymentContext:
    data: DeploymentSeries | None
    historical: pd.DataFrame
    origin: ForecastOrigin
    validation: DeploymentValidation


def _clock_timestamp(value: object, timezone_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timezone_name == SOURCE_CLOCK:
        if timestamp.tzinfo is not None:
            raise ValueError("source-clock timestamps must not carry timezone offsets")
        return timestamp
    if timestamp.tzinfo is None:
        raise ValueError("zoned deployment timestamps must include timezone offsets")
    return timestamp.tz_convert("UTC")


def required_history(recipe: CoreFeatureRecipe, index: ForecastIndex) -> dict[str, tuple[int, ...]]:
    """Exact sampled lags read by FeatureBuilder, including day windows and sequence stride."""
    offsets: dict[str, set[int]] = {}
    if recipe.sequence_length is not None:
        assert recipe.sequence_frequency is not None
        offsets["target"] = {
            step * recipe.sequence_frequency for step in range(recipe.sequence_length)
        }
    else:
        if recipe.load_lags:
            offsets["target"] = set(recipe.load_lags)
        if recipe.temperature_column is not None:
            temp = set(recipe.temperature_lags)
            if recipe.temperature_daily_days:
                steps = pd.Timedelta(days=1) / index.step
                if int(steps) != steps:
                    raise ValueError("daily temperature requires frequency dividing one day")
                for day in recipe.temperature_daily_days:
                    temp.update((day - 1) * int(steps) + i for i in range(int(steps)))
            if temp:
                offsets[recipe.temperature_column] = temp
        for column, lags in recipe.auxiliary_lags.items():
            offsets[column] = set(lags)
    return {column: tuple(sorted(values)) for column, values in offsets.items()}


def required_future(
    recipe: CoreFeatureRecipe, index: ForecastIndex, origin: pd.Timestamp
) -> set[tuple[str, pd.Timestamp]]:
    needed: set[tuple[str, pd.Timestamp]] = set()
    if recipe.temperature_column is not None:
        needed.update(
            (recipe.temperature_column, origin + index.step * lead)
            for lead in recipe.temperature_leads
        )
    needed.update(
        (column, origin + index.step * lead)
        for column, leads in recipe.auxiliary_leads.items()
        for lead in leads
    )
    return needed


def prepare_context(
    store: ArtifactStore,
    prepared: PreparedSnapshot,
    fitted: FittedCoreModel,
    task_id: object,
    trial_id: object,
    model_artifact: object,
    raw_upload: bytes,
    extension: str,
    future_values: tuple[FutureAuxiliaryValue, ...],
) -> PreparedDeploymentContext:
    """Map new observations with frozen Preparation decisions and check dependencies."""
    from uuid import UUID

    from interactive_forecasting.domain.models import ArtifactRef

    task_uuid = UUID(str(task_id))
    trial_uuid = UUID(str(trial_id))
    if not isinstance(model_artifact, ArtifactRef):
        raise ValueError("selected model artifact reference required")
    candidate = fitted.candidate
    snapshot = prepared.dataset
    if snapshot.timezone_name != prepared.plan.timezone_name:
        raise ValueError("prepared snapshot and plan timezone disagree")
    source = read_tabular(raw_upload, extension)
    required_columns = {
        prepared.mapping.timestamp_column,
        prepared.mapping.target_column,
        *prepared.mapping.auxiliary_roles,
    }
    if not required_columns.issubset(source.columns):
        raise ValueError(
            "deployment upload missing frozen mapped columns: "
            f"{sorted(required_columns - set(source.columns))}"
        )
    new_frame, _ = _prepare_frame(
        source,
        prepared.mapping,
        prepared.plan,
        minimum_rows=1,
        frequency_override=snapshot.frequency,
    )
    stored = pd.read_csv(BytesIO(store.read_bytes(snapshot.artifact)))
    series_ids = set(stored["series_id"])
    if len(series_ids) != 1:
        raise ValueError("deployment requires one frozen series")
    series_id = snapshot.series_id or next(iter(series_ids))
    if series_ids != {series_id}:
        raise ValueError("prepared snapshot series ID differs from frozen metadata")
    new_frame["series_id"] = series_id
    stored["timestamp"] = pd.to_datetime(
        stored["timestamp"], utc=snapshot.timezone_name != SOURCE_CLOCK
    )
    new_frame["timestamp"] = pd.to_datetime(
        new_frame["timestamp"], utc=snapshot.timezone_name != SOURCE_CLOCK
    )
    if snapshot.timezone_name != SOURCE_CLOCK:
        # Keep the declared local offset for the existing normalized-series validation.
        stored["timestamp"] = stored["timestamp"].dt.tz_convert(snapshot.timezone_name)
        new_frame["timestamp"] = new_frame["timestamp"].dt.tz_convert(snapshot.timezone_name)
    if new_frame["timestamp"].min() <= stored["timestamp"].max():
        raise ValueError(
            "deployment observations must follow the prepared snapshot without overlap"
        )
    if tuple(new_frame.columns) != tuple(stored.columns):
        raise ValueError("deployment mapped context schema differs from frozen snapshot")
    merged = pd.concat((stored, new_frame), ignore_index=True)
    origin_time = pd.Timestamp(new_frame["timestamp"].max())
    index = ForecastIndex(candidate.index)
    if snapshot.frequency != candidate.index.frequency:
        raise ValueError("selected model frequency differs from prepared snapshot")
    origin = ForecastOrigin(
        task_id=task_uuid,
        selected_trial_id=trial_uuid,
        model_artifact=model_artifact,
        latest_observed_at=origin_time.to_pydatetime(),
        targets=tuple(value.to_pydatetime() for value in index.targets(origin_time)),
        delta=candidate.index.delta,
        horizon=candidate.index.horizon,
        offset_unit=candidate.index.offset_unit,
        anchor=candidate.index.anchor,
        timezone_name=snapshot.timezone_name,
    )
    history = required_history(candidate.features, index)
    deficits: list[HistoryDeficit] = []
    timestamps = set(pd.DatetimeIndex(merged["timestamp"]))
    for column, lags in history.items():
        missing = tuple(
            index.lag(origin_time, lag)
            for lag in range(max(lags) + 1)
            if index.lag(origin_time, lag) not in timestamps
        )
        # Contiguous available lookback is more informative than counting sparse selected lags.
        available = 0
        while index.lag(origin_time, available) in timestamps and available < max(lags, default=0):
            available += 1
        if missing:
            deficits.append(
                HistoryDeficit(
                    column=column,
                    required_steps=max(lags),
                    available_steps=max(0, available - 1),
                    missing_timestamps=tuple(t.to_pydatetime() for t in missing),
                )
            )
    missing_aux: list[str] = []
    for column, lags in history.items():
        if column == "target":
            continue
        policy = candidate.auxiliary_policies.get(column)
        if policy is None:
            raise ValueError(f"selected recipe auxiliary {column} lacks frozen policy")
        issue_column = f"{column}__available_at"
        if policy.kind == "forecast" and issue_column not in merged:
            raise ValueError(f"forecast product issue times missing for {column}")
        if issue_column in merged and policy.kind in {"observed", "known_ahead", "forecast"}:
            for lag in lags:
                when = index.lag(origin_time, lag)
                rows = merged.loc[merged["timestamp"] == when, issue_column]
                if not rows.empty and pd.Timestamp(rows.iloc[0]) > origin_time:
                    missing_aux.append(f"{column}@{when.isoformat()}: not yet available")
    future: dict[tuple[str, pd.Timestamp], FutureAuxiliaryValue] = {}
    for value in future_values:
        key = (value.column, _clock_timestamp(value.valid_at, snapshot.timezone_name))
        if value.available_at is not None:
            _clock_timestamp(value.available_at, snapshot.timezone_name)
        if key in future:
            raise ValueError("duplicate future auxiliary value")
        if key[1] <= origin_time:
            raise ValueError("future auxiliary value must follow the origin")
        future[key] = value
    for column, when in sorted(required_future(candidate.features, index, origin_time)):
        policy = candidate.auxiliary_policies.get(column)
        item = future.get((column, when))
        if policy is None:
            raise ValueError(f"selected recipe auxiliary {column} lacks frozen policy")
        if item is None:
            missing_aux.append(f"{column}@{when.isoformat()}")
            continue
        if policy.kind == "observed":
            missing_aux.append(f"{column}@{when.isoformat()}: observed-only")
        elif policy.kind == "forecast" and (
            item.source_ref != policy.source_ref
            or item.available_at is None
            or pd.Timestamp(item.available_at) > origin_time
        ):
            missing_aux.append(f"{column}@{when.isoformat()}: wrong source or issue time")
        elif policy.kind == "perfect_forecast" and (
            column != candidate.features.temperature_column
            or item.protocol_id != policy.protocol_id
        ):
            missing_aux.append(f"{column}@{when.isoformat()}: oracle protocol mismatch")
        elif policy.kind == "known_ahead" and (
            item.available_at is not None and pd.Timestamp(item.available_at) > origin_time
        ):
            missing_aux.append(f"{column}@{when.isoformat()}: not yet available")
    validation = DeploymentValidation(
        ready=not deficits and not missing_aux,
        origin=origin,
        history_deficits=tuple(deficits),
        missing_auxiliaries=tuple(missing_aux),
    )
    if not validation.ready:
        return PreparedDeploymentContext(None, merged, origin, validation)
    earliest = min(
        (index.lag(origin_time, max(lags)) for lags in history.values()),
        default=origin_time,
    )
    context_frame = merged.loc[merged["timestamp"] >= earliest].copy()
    normalized = NormalizedSeries.validate(
        context_frame,
        snapshot.frequency,
        snapshot.timezone_name,
        candidate.auxiliary_policies,
    )
    # The original model builder consumes the same accessor methods as training.
    data = DeploymentSeries(
        normalized.frame,
        normalized.frequency,
        normalized.timezone_name,
        normalized.auxiliary_availability,
        future,
        origin_time,
    )
    return PreparedDeploymentContext(data, context_frame, origin, validation)


def forecast_examples(
    origin: ForecastOrigin, series_id: str = "default"
) -> tuple[ForecastExample, ...]:
    return tuple(
        ForecastExample(
            ForecastKey(
                series_id=series_id,
                origin=origin.latest_observed_at,
                target=target,
            ),
            None,
        )
        for target in origin.targets
    )
