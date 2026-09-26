"""Read-only local perturbation through the saved model and frozen feature recipe."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from interactive_forecasting.domain.forecasting import PointForecast, QuantileForecast
from interactive_forecasting.domain.models import (
    Forecast,
    ForecastVersion,
    SensitivityRequest,
    SensitivityResult,
)
from interactive_forecasting.services.data.core import ForecastIndex
from interactive_forecasting.services.deployment.core import (
    DeploymentSeries,
    PreparedDeploymentContext,
    forecast_examples,
    required_future,
    required_history,
)
from interactive_forecasting.services.models.core import FittedCoreModel


def available_variables(fitted: FittedCoreModel) -> tuple[str, ...]:
    """Only numeric auxiliaries actually read by the frozen recipe are eligible."""
    candidate = fitted.candidate
    index = ForecastIndex(candidate.index)
    names = set(required_history(candidate.features, index)) - {"target"}
    names.update(
        column
        for column, _ in required_future(
            candidate.features, index, pd.Timestamp("2024-01-01", tz="UTC")
        )
    )
    return tuple(sorted(names & set(candidate.auxiliary_policies)))


def _perturb(value: float, kind: str, amount: float) -> float:
    changed = value + amount if kind == "absolute" else value * (1.0 + amount / 100.0)
    if not isfinite(changed):
        raise ValueError("sensitivity perturbation produces non-finite input")
    return changed


def run_sensitivity(
    forecast: Forecast,
    base: ForecastVersion,
    request: SensitivityRequest,
    context: PreparedDeploymentContext,
    fitted: FittedCoreModel,
) -> SensitivityResult:
    """Never writes input artifacts, fitted transforms, forecasts or versions."""
    if (
        base.forecast_id != forecast.forecast_id
        or base.version_number != 0
        or base.version_id != request.base_version_id
        or base.prediction != forecast.prediction
        or context.data is None
        or context.origin != forecast.origin
    ):
        raise ValueError("sensitivity requires the original unmodified forecast and valid context")
    if request.variable not in available_variables(fitted):
        raise ValueError("variable is not a numeric external input in the frozen feature recipe")
    index = ForecastIndex(fitted.candidate.index)
    origin = pd.Timestamp(forecast.origin.latest_observed_at)
    historical = tuple(
        index.lag(origin, lag)
        for lag in required_history(fitted.candidate.features, index).get(request.variable, ())
    )
    future = tuple(
        when
        for column, when in required_future(fitted.candidate.features, index, origin)
        if column == request.variable
    )
    affected = tuple(sorted(set((*historical, *future))))
    if not affected:
        raise ValueError("selected variable has no feature timestamps")
    frame = context.data.frame.copy(deep=True)
    future_values = dict(context.data.future)
    for when in historical:
        match = (frame["series_id"] == "default") & (frame["timestamp"] == when)
        if int(match.sum()) != 1:
            raise ValueError("historical sensitivity input is missing")
        previous = float(frame.loc[match, request.variable].iloc[0])
        frame.loc[match, request.variable] = _perturb(
            previous, request.perturbation_type, request.value
        )
    for when in future:
        key = (request.variable, when)
        future_item = future_values.get(key)
        if future_item is None:
            raise ValueError("future sensitivity input is missing")
        future_values[key] = future_item.model_copy(
            update={"value": _perturb(future_item.value, request.perturbation_type, request.value)}
        )
    temporary = DeploymentSeries(
        frame,
        context.data.frequency,
        context.data.timezone_name,
        context.data.auxiliary_availability,
        future_values,
        context.data.origin,
    )
    perturbed = fitted.predict(temporary, forecast_examples(forecast.origin))
    baseline = base.prediction
    if type(perturbed) is not type(baseline) or perturbed.keys != baseline.keys:
        raise ValueError("perturbed prediction does not align with original forecast")
    deltas: tuple[tuple[float, ...], ...]
    if isinstance(baseline, PointForecast):
        assert isinstance(perturbed, PointForecast)
        deltas = tuple(
            (changed - original,)
            for original, changed in zip(baseline.values, perturbed.values, strict=True)
        )
    else:
        assert isinstance(baseline, QuantileForecast)
        assert isinstance(perturbed, QuantileForecast)
        if baseline.levels != perturbed.levels:
            raise ValueError("perturbed quantile levels differ from original forecast")
        deltas = tuple(
            tuple(changed - original for original, changed in zip(before, after, strict=True))
            for before, after in zip(baseline.values, perturbed.values, strict=True)
        )
    return SensitivityResult(
        forecast_id=forecast.forecast_id,
        base_version_id=base.version_id,
        variable=request.variable,
        perturbation_type=request.perturbation_type,
        value=request.value,
        affected_input_timestamps=tuple(when.to_pydatetime() for when in affected),
        baseline_prediction=baseline,
        perturbed_prediction=perturbed,
        deltas=deltas,
        provenance={
            "model_artifact_sha256": forecast.model_artifact.sha256,
            "context_sha256": forecast.context_artifact.sha256,
            "future_auxiliary_sha256": forecast.future_auxiliary_artifact.sha256,
            "feature_recipe_version": fitted.candidate.features.version,
            "baseline": "original_v0",
        },
    )
