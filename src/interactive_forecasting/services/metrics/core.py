"""Timestamp-aligned point and empirical-distribution metrics."""

from __future__ import annotations

from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np

from interactive_forecasting.domain.forecasting import (
    LabelVector,
    MetricPlan,
    MetricReport,
    PointForecast,
    PredictiveDistribution,
    QuantileForecast,
    SampleForecast,
)
from interactive_forecasting.domain.metric_spec import MetricSpec


def _aligned(labels: LabelVector, forecast: PointForecast | PredictiveDistribution) -> np.ndarray:
    if labels.keys != forecast.keys:
        raise ValueError("label and forecast keys must align exactly")
    return np.asarray(labels.values, dtype=float)


def _require_point(forecast: PointForecast) -> None:
    if not isinstance(forecast, PointForecast):
        raise ValueError("point metric requires PointForecast")


def mae(labels: LabelVector, forecast: PointForecast) -> float:
    _require_point(forecast)
    truth = _aligned(labels, forecast)
    return float(np.mean(np.abs(truth - np.asarray(forecast.values, dtype=float))))


def weighted_mae(labels: LabelVector, forecast: PointForecast, spec: MetricSpec) -> float:
    """Weight aligned point errors by the frozen target-local half-open time range."""
    _require_point(forecast)
    truth = _aligned(labels, forecast)
    if spec.kind != "weighted" or spec.base_metric != "mae":
        raise ValueError("weighted MAE requires a weighted-MAE MetricSpec")
    assert spec.time_range is not None and spec.timezone_name is not None
    zone = ZoneInfo(spec.timezone_name)
    weights = np.ones(len(labels.keys), dtype=float)
    for index, key in enumerate(labels.keys):
        local_time = key.target.astimezone(zone).time()
        if spec.time_range.start_local <= local_time < spec.time_range.end_local:
            weights[index] = spec.time_range.weight
    prediction = np.asarray(forecast.values, dtype=float)
    return weighted_loss(prediction - truth, weights, base="absolute")


def asymmetric_mae(labels: LabelVector, forecast: PointForecast, spec: MetricSpec) -> float:
    """Mean direction-weighted absolute error; exact matches contribute zero."""
    _require_point(forecast)
    truth = _aligned(labels, forecast)
    if spec.kind != "asymmetric" or spec.base_metric != "mae":
        raise ValueError("asymmetric MAE requires an asymmetric-MAE MetricSpec")
    assert spec.over_weight is not None and spec.under_weight is not None
    prediction = np.asarray(forecast.values, dtype=float)
    return asymmetric_loss(truth, prediction, over=spec.over_weight, under=spec.under_weight)


def mape(
    labels: LabelVector,
    forecast: PointForecast,
    *,
    zero_policy: Literal["error", "omit", "epsilon"],
    epsilon: float | None = None,
) -> float:
    _require_point(forecast)
    truth = _aligned(labels, forecast)
    prediction = np.asarray(forecast.values, dtype=float)
    if zero_policy == "error":
        if np.any(truth == 0):
            raise ValueError("zero target under error MAPE policy")
        denominator = np.abs(truth)
        error = np.abs(truth - prediction)
    elif zero_policy == "omit":
        mask = truth != 0
        if not mask.any():
            raise ValueError("no nonzero targets remain for MAPE")
        denominator = np.abs(truth[mask])
        error = np.abs(truth[mask] - prediction[mask])
    elif zero_policy == "epsilon":
        if epsilon is None or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon MAPE policy requires positive epsilon")
        denominator = np.maximum(np.abs(truth), epsilon)
        error = np.abs(truth - prediction)
    else:
        raise ValueError("unknown MAPE zero policy")
    return float(np.mean(error / denominator) * 100)


def weighted_loss(
    errors: np.ndarray,
    weights: np.ndarray,
    *,
    base: Literal["absolute", "squared"],
) -> float:
    errors = np.asarray(errors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if errors.shape != weights.shape or errors.ndim != 1 or errors.size == 0:
        raise ValueError("errors and weights must be matching nonempty vectors")
    if not np.isfinite(errors).all() or not np.isfinite(weights).all():
        raise ValueError("errors and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    if base not in {"absolute", "squared"}:
        raise ValueError("unknown weighted loss base")
    losses = np.abs(errors) if base == "absolute" else errors**2
    return float(np.dot(weights, losses) / weights.sum())


def asymmetric_loss(
    labels: np.ndarray, predictions: np.ndarray, *, over: float, under: float
) -> float:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.shape != predictions.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and predictions must be matching vectors")
    if not np.isfinite([over, under]).all() or over < 0 or under < 0 or over + under == 0:
        raise ValueError("asymmetric coefficients must be nonnegative and nonzero")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("loss inputs must be finite")
    error = predictions - labels
    return float(np.mean(np.where(error >= 0, over * error, -under * error)))


def pinball_loss(labels: np.ndarray, predictions: np.ndarray, *, quantile: float) -> float:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if labels.shape != predictions.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and predictions must be matching vectors")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("loss inputs must be finite")
    error = labels - predictions
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def crps(labels: LabelVector, forecast: PredictiveDistribution | PointForecast) -> float:
    """CRPS: midpoint-cell quantile quadrature or exact empirical sample formula."""
    if isinstance(forecast, PointForecast):
        raise ValueError("CRPS requires a predictive distribution, not point values")
    truth = _aligned(labels, forecast)
    if isinstance(forecast, QuantileForecast):
        levels = np.asarray(forecast.levels, dtype=float)
        boundaries = np.concatenate(([0.0], (levels[:-1] + levels[1:]) / 2, [1.0]))
        weights = np.diff(boundaries)
        errors = truth[:, None] - np.asarray(forecast.values, dtype=float)
        pinball = np.maximum(levels * errors, (levels - 1) * errors)
        return float(np.mean(2 * np.sum(weights * pinball, axis=1)))
    assert isinstance(forecast, SampleForecast)
    per_target = []
    for y, samples in zip(truth, forecast.samples, strict=True):
        values = np.asarray(samples, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("samples must be finite")
        first = np.mean(np.abs(values - y))
        second = np.mean(np.abs(values[:, None] - values[None, :])) / 2
        per_target.append(first - second)
    return float(np.mean(per_target))


def evaluate_point(
    labels: LabelVector,
    forecast: PointForecast,
    plan: MetricPlan,
    *,
    weights: np.ndarray | None = None,
) -> MetricReport:
    _require_point(forecast)
    truth = _aligned(labels, forecast)
    prediction = np.asarray(forecast.values, dtype=float)
    errors = prediction - truth
    report: dict[str, float] = {}
    if "mae" in plan.reporting_metrics:
        report["mae"] = mae(labels, forecast)
    if "mape" in plan.reporting_metrics:
        assert plan.mape_zero_policy is not None
        report["mape"] = mape(
            labels, forecast, zero_policy=plan.mape_zero_policy, epsilon=plan.mape_epsilon
        )
    if plan.validation_objective == "mae":
        objective = mae(labels, forecast)
    elif plan.validation_objective == "mape":
        assert plan.mape_zero_policy is not None
        objective = mape(
            labels, forecast, zero_policy=plan.mape_zero_policy, epsilon=plan.mape_epsilon
        )
    elif plan.validation_objective in {"weighted_mae", "weighted_mse"}:
        if weights is None:
            raise ValueError("weighted objective requires explicit weights")
        objective = weighted_loss(
            errors,
            weights,
            base="absolute" if plan.validation_objective == "weighted_mae" else "squared",
        )
    else:
        assert plan.asymmetric_over is not None and plan.asymmetric_under is not None
        objective = asymmetric_loss(
            truth, prediction, over=plan.asymmetric_over, under=plan.asymmetric_under
        )
    return MetricReport(
        objective_name=plan.validation_objective,
        objective_value=objective,
        reporting=report,
    )
