"""Pure, typed postprocessing over an immutable, explicitly selected parent version."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from interactive_forecasting.domain.forecasting import (
    AuxiliaryPolicy,
    PointForecast,
    QuantileForecast,
)
from interactive_forecasting.domain.models import (
    Adjustment,
    Forecast,
    ForecastVersion,
    FutureAuxiliaryValue,
    VersionValueChange,
)


def _condition(value: float, threshold: float, operator: str) -> bool:
    return value > threshold if operator == "gt" else value < threshold


def _external_at(
    forecast: Forecast,
    column: str,
    target: pd.Timestamp,
    supplied: tuple[FutureAuxiliaryValue, ...],
    policies: dict[str, AuxiliaryPolicy],
) -> float:
    policy = policies.get(column)
    if policy is None:
        raise ValueError("external variable is absent from frozen task policy")
    matches = [
        item for item in supplied if item.column == column and pd.Timestamp(item.valid_at) == target
    ]
    if len(matches) != 1:
        raise ValueError("external variable needs one supplied value at each selected timestamp")
    item = matches[0]
    origin = pd.Timestamp(forecast.origin.latest_observed_at)
    if policy.kind == "observed":
        if target > origin:
            raise ValueError("observed external variable is unavailable at forecast origin")
        if item.available_at is not None and pd.Timestamp(item.available_at) > origin:
            raise ValueError("external observation was unavailable at origin")
    elif policy.kind == "known_ahead":
        if item.available_at is not None and pd.Timestamp(item.available_at) > origin:
            raise ValueError("known-ahead external variable was unavailable at origin")
    elif policy.kind == "forecast":
        if (
            item.source_ref != policy.source_ref
            or item.available_at is None
            or pd.Timestamp(item.available_at) > origin
        ):
            raise ValueError("external forecast source or issue time differs from frozen policy")
    elif policy.kind == "perfect_forecast":
        if policy.role != "temperature" or item.protocol_id != policy.protocol_id:
            raise ValueError("external oracle protocol differs from frozen policy")
    return item.value


def apply_adjustment(
    forecast: Forecast,
    parent: ForecastVersion,
    draft: Adjustment,
    *,
    future_values: tuple[FutureAuxiliaryValue, ...] = (),
    auxiliary_policies: dict[str, AuxiliaryPolicy] | None = None,
) -> ForecastVersion:
    """Validate an adjustment and calculate a new version without touching its parent."""
    if (
        draft.forecast_id != forecast.forecast_id
        or draft.task_id != forecast.task_id
        or draft.parent_version_id != parent.version_id
        or parent.forecast_id != forecast.forecast_id
        or parent.prediction_representation != forecast.prediction_representation
        or draft.status not in {"draft", "confirmed"}
    ):
        raise ValueError("adjustment does not match an eligible parent forecast version")
    keys = parent.prediction.keys
    if tuple(key.target for key in keys) != forecast.target_timestamps:
        raise ValueError("parent prediction keys differ from original forecast targets")
    available_times = tuple(pd.Timestamp(key.target) for key in keys)
    if draft.selected_timestamps and not set(
        pd.Timestamp(v) for v in draft.selected_timestamps
    ) <= set(available_times):
        raise ValueError("selected timestamps are outside the forecast")
    if draft.start_at is not None and pd.Timestamp(draft.start_at) > available_times[-1]:
        raise ValueError("adjustment interval starts after forecast")
    if draft.end_at is not None and pd.Timestamp(draft.end_at) < available_times[0]:
        raise ValueError("adjustment interval ends before forecast")
    if draft.adjustment_type == "load_scaling" and isinstance(parent.prediction, QuantileForecast):
        raise ValueError("load-threshold semantics for quantile forecasts are unspecified")
    replacements = {pd.Timestamp(item.timestamp): item.values for item in draft.manual_replacements}
    if draft.adjustment_type == "manual_override":
        if draft.selected_timestamps and set(
            pd.Timestamp(v) for v in draft.selected_timestamps
        ) != set(replacements):
            raise ValueError("manual selected timestamps differ from replacement rows")
        if not set(replacements) <= set(available_times):
            raise ValueError("manual replacement is outside the forecast")
    previous = (
        tuple((float(v),) for v in parent.prediction.values)
        if isinstance(parent.prediction, PointForecast)
        else parent.prediction.values
    )
    changed: list[VersionValueChange] = []
    new_rows = list(previous)
    for index, target in enumerate(available_times):
        if draft.selected_timestamps and target not in {
            pd.Timestamp(value) for value in draft.selected_timestamps
        }:
            continue
        if draft.start_at is not None and target < pd.Timestamp(draft.start_at):
            continue
        if draft.end_at is not None and target > pd.Timestamp(draft.end_at):
            continue
        before = tuple(previous[index])
        if draft.adjustment_type == "manual_override":
            if target not in replacements:
                continue
            after = replacements[target]
            if len(after) != len(before):
                raise ValueError("manual replacement must provide every prediction component")
            if isinstance(parent.prediction, QuantileForecast) and tuple(sorted(after)) != after:
                raise ValueError("manual quantile replacements must remain ordered")
        else:
            assert draft.lambda_value is not None
            if draft.adjustment_type == "load_scaling":
                assert draft.threshold is not None and draft.comparison is not None
                if not _condition(before[0], draft.threshold, draft.comparison):
                    continue
            if draft.adjustment_type == "external_scaling":
                assert draft.threshold is not None and draft.comparison is not None
                assert draft.external_variable is not None
                external = _external_at(
                    forecast,
                    draft.external_variable,
                    target,
                    future_values,
                    auxiliary_policies or {},
                )
                if not _condition(external, draft.threshold, draft.comparison):
                    continue
            factor = 1 + draft.lambda_value
            after = tuple(value * factor for value in before)
        if any(not isfinite(value) for value in after):
            raise ValueError("adjustment produced non-finite prediction")
        new_rows[index] = after
        changed.append(
            VersionValueChange(timestamp=target.to_pydatetime(), before=before, after=after)
        )
    if not changed:
        raise ValueError("adjustment selects no forecast timestamp")
    prediction: PointForecast | QuantileForecast
    if isinstance(parent.prediction, PointForecast):
        prediction = PointForecast(keys=keys, values=tuple(row[0] for row in new_rows))
    else:
        prediction = QuantileForecast(
            keys=keys, levels=parent.prediction.levels, values=tuple(new_rows)
        )
    return ForecastVersion(
        forecast_id=forecast.forecast_id,
        version_number=parent.version_number + 1,
        parent_version_id=parent.version_id,
        adjustment_id=draft.adjustment_id,
        prediction_representation=parent.prediction_representation,
        prediction=prediction,
        affected_timestamps=tuple(item.timestamp for item in changed),
        value_changes=tuple(changed),
        provenance={
            "adjustment_type": draft.adjustment_type,
            "source": draft.source,
            "parent_version_id": str(parent.version_id),
        },
    )
