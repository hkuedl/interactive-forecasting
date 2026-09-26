"""Validation-only trial execution and an explicit, separate final-test operation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
import pandas as pd

from interactive_forecasting.domain.forecasting import (
    IndexSpec,
    OriginSchedule,
    Partition,
    PointForecast,
    QuantileForecast,
)
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.models import (
    ArtifactRef,
    DatasetSnapshot,
    EvaluationProtocol,
    ExperimentRun,
    TaskDefinition,
)
from interactive_forecasting.domain.search import (
    TrialCandidate,
    TrialResult,
    ValidationMetricConfig,
)
from interactive_forecasting.services.data.core import (
    ForecastExample,
    ForecastIndex,
    NormalizedSeries,
    final_test_labels,
    labels_for,
    split_examples,
)
from interactive_forecasting.services.metrics.core import (
    asymmetric_mae,
    crps,
    mae,
    mape,
    weighted_mae,
)
from interactive_forecasting.services.models.core import FittedCoreModel, fit_candidate
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.interfaces import ExperimentRepositoryPort


def _score(
    labels: object,
    forecast: PointForecast | QuantileForecast,
    config: ValidationMetricConfig,
    metric_spec: MetricSpec | None = None,
) -> dict[str, float]:
    from interactive_forecasting.domain.forecasting import LabelVector

    if not isinstance(labels, LabelVector):
        raise TypeError("expected aligned labels")
    if metric_spec is not None and metric_spec.objective_id != config.objective:
        raise ValueError("frozen MetricSpec and validation objective disagree")
    if config.objective == "crps":
        if not isinstance(forecast, QuantileForecast):
            raise ValueError("CRPS objective requires quantile output")
        value = crps(labels, forecast)
    elif config.objective == "mae":
        if not isinstance(forecast, PointForecast):
            raise ValueError("MAE objective requires point output")
        value = mae(labels, forecast)
    elif config.objective == "weighted_mae":
        if not isinstance(forecast, PointForecast) or metric_spec is None:
            raise ValueError("weighted MAE requires point output and frozen MetricSpec")
        value = weighted_mae(labels, forecast, metric_spec)
    elif config.objective == "asymmetric_mae":
        if not isinstance(forecast, PointForecast) or metric_spec is None:
            raise ValueError("asymmetric MAE requires point output and frozen MetricSpec")
        value = asymmetric_mae(labels, forecast, metric_spec)
    else:
        if not isinstance(forecast, PointForecast):
            raise ValueError("MAPE objective requires point output")
        assert config.mape_zero_policy is not None
        value = mape(
            labels,
            forecast,
            zero_policy=config.mape_zero_policy,
            epsilon=config.mape_epsilon,
        )
    if not np.isfinite(value):
        raise ValueError("validation objective must be finite")
    return {config.objective: value}


@dataclass(frozen=True)
class ValidationEvaluator:
    """Contains only rows before validation end; no test examples or labels."""

    _data: NormalizedSeries
    _train: tuple[ForecastExample, ...]
    _validation: tuple[ForecastExample, ...]
    _protocol: EvaluationProtocol
    _snapshot_sha256: str | None = None
    _metric_spec: MetricSpec | None = None

    @property
    def snapshot_sha256(self) -> str | None:
        return self._snapshot_sha256

    @classmethod
    def from_snapshot(
        cls,
        store: ArtifactStore,
        snapshot: DatasetSnapshot,
        task: TaskDefinition,
        index: IndexSpec,
        protocol: EvaluationProtocol,
        schedule: OriginSchedule,
    ) -> ValidationEvaluator:
        if snapshot.dataset_id != task.dataset_id or snapshot.timezone_name != task.timezone_name:
            raise ValueError("snapshot and task disagree")
        if task.metric_spec != protocol.metric_spec:
            raise ValueError("task and protocol metric specifications disagree")
        data = NormalizedSeries.from_csv_artifact(
            store,
            snapshot.artifact,
            frequency=snapshot.frequency,
            timezone_name=snapshot.timezone_name,
            auxiliary_availability=task.auxiliary_policies,
        )
        return replace(
            cls.from_dataset(data, index, protocol, schedule),
            _snapshot_sha256=snapshot.artifact.sha256,
            _metric_spec=task.metric_spec,
        )

    @classmethod
    def from_dataset(
        cls,
        data: NormalizedSeries,
        index: IndexSpec,
        protocol: EvaluationProtocol,
        schedule: OriginSchedule,
    ) -> ValidationEvaluator:
        # Physically omit test/future rows before constructing the search evaluator.
        frame = data.frame.loc[data.frame["timestamp"] < protocol.validation.end].copy()
        clipped = NormalizedSeries.validate(
            frame, data.frequency, data.timezone_name, data.auxiliary_availability
        )
        validation_schedule = OriginSchedule(
            version=schedule.version,
            origins={
                series_id: tuple(
                    origin for origin in origins if pd.Timestamp(origin) < protocol.validation.end
                )
                for series_id, origins in schedule.origins.items()
            },
        )
        splits = split_examples(clipped, ForecastIndex(index), protocol, validation_schedule)
        if splits[Partition.TEST]:
            raise ValueError("validation evaluator must not contain test examples")
        return cls(clipped, splits[Partition.TRAIN], splits[Partition.VALIDATION], protocol)

    def validation_diagnostics(self, fitted: FittedCoreModel) -> dict[str, object]:
        """Validation-only, capped actual versus predicted points; never test labels."""
        forecast = fitted.predict(self._data, self._validation)
        labels = labels_for(self._data, self._validation, partition=Partition.VALIDATION)
        positions = np.linspace(0, len(labels.values) - 1, min(400, len(labels.values)), dtype=int)
        level = None
        if isinstance(forecast, QuantileForecast):
            level_index = min(
                range(len(forecast.levels)), key=lambda index: abs(forecast.levels[index] - 0.5)
            )
            level = forecast.levels[level_index]
            predictions = [row[level_index] for row in forecast.values]
        else:
            predictions = list(forecast.values)
        return {
            "partition": "validation",
            "representative_quantile": level,
            "points": [
                {
                    "target": labels.keys[index].target.isoformat(),
                    "series_id": labels.keys[index].series_id,
                    "truth": labels.values[index],
                    "prediction": predictions[index],
                }
                for index in positions
            ],
        }

    def fit_score(
        self, trial: TrialCandidate, metric: ValidationMetricConfig
    ) -> tuple[FittedCoreModel, dict[str, float]]:
        if trial.protocol_id != self._protocol.protocol_id:
            raise ValueError("trial protocol differs from validation evaluator")
        candidate = trial.candidate
        fitted = fit_candidate(
            candidate,
            self._data,
            self._train,
            labels_for(self._data, self._train, partition=Partition.TRAIN),
            self._validation,
            labels_for(self._data, self._validation, partition=Partition.VALIDATION),
        )
        forecast = fitted.predict(self._data, self._validation)
        labels = labels_for(self._data, self._validation, partition=Partition.VALIDATION)
        return fitted, _score(labels, forecast, metric, self._metric_spec)


class TrialExecutor:
    def __init__(self, evaluator: ValidationEvaluator, store: ArtifactStore):
        self._evaluator = evaluator
        self._store = store

    @property
    def snapshot_sha256(self) -> str | None:
        return self._evaluator.snapshot_sha256

    def execute(
        self,
        run_id: UUID,
        spec_id: str,
        trial: TrialCandidate,
        metric: ValidationMetricConfig,
        *,
        number: int,
        round_number: int,
    ) -> tuple[TrialResult, ArtifactRef | None]:
        started = datetime.now(timezone.utc)
        try:
            fitted, scores = self._evaluator.fit_score(trial, metric)
            self._store.reproduction_dir(spec_id, run_id)
            artifact = fitted.save(
                self._store,
                f"reproduction/{spec_id}/{run_id}/raw/trials/{number}.zip",
            )
            diagnostic = self._store.put_bytes(
                f"reproduction/{spec_id}/{run_id}/metrics/{number}.json",
                json.dumps(self._evaluator.validation_diagnostics(fitted), sort_keys=True).encode(),
            )
            result = TrialResult(
                number=number,
                round_number=round_number,
                candidate=trial,
                request=trial.request,
                space_id=trial.space_id,
                status="completed",
                objective=scores[metric.objective],
                metrics=scores,
                artifact_uri=artifact.uri,
                artifact_sha256=artifact.sha256,
                diagnostic_uri=diagnostic.uri,
                diagnostic_sha256=diagnostic.sha256,
                diagnostic_size_bytes=diagnostic.size_bytes,
                started_at=started,
                ended_at=datetime.now(timezone.utc),
            )
            return result, artifact
        except Exception as exc:
            result = TrialResult(
                number=number,
                round_number=round_number,
                candidate=trial,
                request=trial.request,
                space_id=trial.space_id,
                status="failed",
                failure_type=type(exc).__name__,
                failure_message=str(exc),
                started_at=started,
                ended_at=datetime.now(timezone.utc),
            )
            return result, None


class FinalTestEvaluator:
    """Has no backend reference. Only a completed, selected run may unlock test labels."""

    def __init__(self, repository: ExperimentRepositoryPort, store: ArtifactStore):
        self._repository = repository
        self._store = store

    def evaluate(self, run_id: UUID) -> ExperimentRun:
        run = self._repository.get(run_id)
        if run is None or run.status.value != "completed" or run.selected_trial_id is None:
            raise ValueError("final test requires a completed run with frozen selection")
        if run.final_test_metrics is not None:
            raise ValueError("final test was already evaluated")
        if (
            run.protocol is None
            or run.metric_config is None
            or run.dataset_snapshot is None
            or run.task_definition is None
            or run.origin_schedule is None
        ):
            raise ValueError("run lacks frozen evaluation protocol or snapshot")
        selected = next(
            (item for item in run.trials if item.trial_id == run.selected_trial_id), None
        )
        if selected is None or selected.candidate is None or selected.artifact_uri is None:
            raise ValueError("selected trial has no fitted artifact")
        reference = next(
            (ref for ref in run.artifact_refs if ref.uri == selected.artifact_uri), None
        )
        if reference is None or reference.sha256 != selected.artifact_sha256:
            raise ValueError("selected artifact reference mismatch")
        fitted = FittedCoreModel.load(self._store, reference)
        data = NormalizedSeries.from_csv_artifact(
            self._store,
            run.dataset_snapshot.artifact,
            frequency=run.dataset_snapshot.frequency,
            timezone_name=run.dataset_snapshot.timezone_name,
            auxiliary_availability=run.task_definition.auxiliary_policies,
        )
        splits = split_examples(
            data, ForecastIndex(fitted.candidate.index), run.protocol, run.origin_schedule
        )
        test = splits[Partition.TEST]
        if not test:
            raise ValueError("final test partition is empty")
        forecast = fitted.predict(data, test)
        scores = _score(
            final_test_labels(data, test),
            forecast,
            run.metric_config,
            run.task_definition.metric_spec,
        )
        updated = run.model_copy(update={"version": run.version + 1, "final_test_metrics": scores})
        return self._repository.save(updated, expected_version=run.version)
