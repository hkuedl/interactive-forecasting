"""Deterministic validation-only views from persisted trials and trusted artifacts."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from io import BytesIO
from math import isfinite
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from interactive_forecasting.domain.models import ArtifactRef, ExperimentRun
from interactive_forecasting.domain.optimization import GuidanceEffect, OptimizationSession
from interactive_forecasting.domain.search import TrialResult
from interactive_forecasting.services.optimization.summary import summarize_optimization
from interactive_forecasting.storage.artifacts import ArtifactStore


class VisualRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProgressPoint(VisualRecord):
    trial_number: int
    objective: float | None
    best_so_far: float | None
    status: str


class FamilyComparison(VisualRecord):
    family: str
    trials: int
    completed: int
    failed: int
    best_objective: float | None


class ChoicePerformance(VisualRecord):
    dimension: str
    choice: str
    trials: int
    mean_objective: float


class ParameterInfluence(VisualRecord):
    parameter: str
    method: str
    sample_size: int
    score: float


class TrialListing(VisualRecord):
    trial_id: UUID
    number: int
    family: str
    status: str
    objective: float | None


class OptimizationVisuals(VisualRecord):
    run_id: UUID
    run_version: int
    source: str = "experiment_run_validation_v1"
    mode: str
    phase: str
    current_round: int
    total_trials: int
    completed_trials: int
    best_objective: float | None
    best_family: str | None
    remaining_budget: int
    stopping_status: str
    effective_space_id: UUID
    effective_space_version: str
    progress: tuple[ProgressPoint, ...]
    trial_catalog: tuple[TrialListing, ...]
    families: tuple[FamilyComparison, ...]
    feature_performance: tuple[ChoicePerformance, ...] = ()
    feature_status: str = "insufficient_data"
    parameter_influence: tuple[ParameterInfluence, ...] = ()
    influence_status: str = "insufficient_data"
    guidance: tuple[GuidanceEffect, ...] = ()


class PredictionPoint(VisualRecord):
    target: str
    series_id: str
    truth: float
    prediction: float


class TrialDetail(VisualRecord):
    trial: TrialResult
    duration_seconds: float
    artifact_references: tuple[str, ...] = ()
    training_loss_curve: tuple[float, ...] | None = None
    validation_loss_curve: tuple[float, ...] | None = None
    prediction_points: tuple[PredictionPoint, ...] = ()
    prediction_status: str = "unavailable"
    representative_quantile: float | None = None


class VisualizationService:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def overview(self, run: ExperimentRun, session: OptimizationSession) -> OptimizationVisuals:
        if run.run_id != session.run_id or run.effective_space is None:
            raise ValueError("visualization run/session mismatch")
        summary = summarize_optimization(run)
        best: float | None = None
        progress: list[ProgressPoint] = []
        groups: dict[str, list[TrialResult]] = defaultdict(list)
        for trial in run.trials:
            groups[trial.request.family.value].append(trial)
            if trial.status == "completed" and trial.objective is not None:
                best = min(best, trial.objective) if best is not None else trial.objective
            progress.append(
                ProgressPoint(
                    trial_number=trial.number,
                    objective=trial.objective,
                    best_so_far=best,
                    status=trial.status,
                )
            )
        comparisons = tuple(
            FamilyComparison(
                family=family,
                trials=len(trials),
                completed=sum(trial.status == "completed" for trial in trials),
                failed=sum(trial.status == "failed" for trial in trials),
                best_objective=min(
                    (trial.objective for trial in trials if trial.objective is not None),
                    default=None,
                ),
            )
            for family, trials in sorted(groups.items())
        )
        feature, feature_status = self._feature_performance(run)
        influence, influence_status = self._parameter_influence(run)
        best_family = (
            summary.best_candidate.family.value if summary.best_candidate is not None else None
        )
        return OptimizationVisuals(
            run_id=run.run_id,
            run_version=run.version,
            mode=session.mode.value,
            phase=session.phase.value,
            current_round=run.completed_rounds,
            total_trials=len(run.trials),
            completed_trials=summary.completed_trials,
            best_objective=summary.best_validation_objective,
            best_family=best_family,
            remaining_budget=summary.remaining_budget,
            stopping_status=summary.stopping_status,
            effective_space_id=run.effective_space.space_id,
            effective_space_version=run.effective_space.version,
            progress=tuple(progress),
            trial_catalog=tuple(
                TrialListing(
                    trial_id=trial.trial_id,
                    number=trial.number,
                    family=trial.request.family.value,
                    status=trial.status,
                    objective=trial.objective,
                )
                for trial in run.trials
            ),
            families=comparisons,
            feature_performance=feature,
            feature_status=feature_status,
            parameter_influence=influence,
            influence_status=influence_status,
            guidance=session.guidance_effects,
        )

    @staticmethod
    def _feature_performance(run: ExperimentRun) -> tuple[tuple[ChoicePerformance, ...], str]:
        if run.effective_space is None:
            return (), "unavailable"
        names = {item.name for item in run.effective_space.features}
        scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for trial in run.trials:
            if trial.status != "completed" or trial.objective is None:
                continue
            for name in names & trial.request.values.keys():
                scores[(name, str(trial.request.values[name]))].append(trial.objective)
        eligible_dimensions = {
            dimension
            for dimension, _ in scores
            if sum(
                1 for (name, _), values in scores.items() if name == dimension and len(values) >= 2
            )
            >= 2
        }
        result = tuple(
            ChoicePerformance(
                dimension=name,
                choice=choice,
                trials=len(values),
                mean_objective=sum(values) / len(values),
            )
            for (name, choice), values in sorted(scores.items())
            if name in eligible_dimensions and len(values) >= 2
        )
        return result, "available" if result else "insufficient_data"

    @staticmethod
    def _parameter_influence(
        run: ExperimentRun,
    ) -> tuple[tuple[ParameterInfluence, ...], str]:
        completed = [
            trial
            for trial in run.trials
            if trial.status == "completed" and trial.objective is not None
        ]
        if len(completed) < 6:
            return (), "insufficient_data"
        by_family: dict[str, list[TrialResult]] = defaultdict(list)
        for trial in completed:
            by_family[trial.request.family.value].append(trial)
        result: list[ParameterInfluence] = []
        for family, trials in sorted(by_family.items()):
            if len(trials) < 6:
                continue
            names = set.intersection(*(set(trial.request.values) for trial in trials))
            for name in sorted(names):
                values = [trial.request.values[name] for trial in trials]
                if (
                    any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        for value in values
                    )
                    or len(set(values)) < 3
                ):
                    continue
                objectives = [trial.objective for trial in trials]
                x = rankdata(values)
                y = rankdata(objectives)
                x = x - x.mean()
                y = y - y.mean()
                denominator = float((sum(x * x) * sum(y * y)) ** 0.5)
                if denominator == 0 or not isfinite(denominator):
                    continue
                result.append(
                    ParameterInfluence(
                        parameter=f"{family}.{name}",
                        method="absolute_spearman_rank_correlation_v1",
                        sample_size=len(trials),
                        score=abs(float(sum(x * y) / denominator)),
                    )
                )
        result.sort(key=lambda item: (-item.score, item.parameter))
        return tuple(result), "available" if result else "insufficient_data"

    def trial_detail(self, run: ExperimentRun, trial_id: UUID) -> TrialDetail:
        trial = next((item for item in run.trials if item.trial_id == trial_id), None)
        if trial is None:
            raise ValueError("trial does not belong to run")
        curves: dict[str, object] = {}
        refs: list[str] = []
        if trial.artifact_uri is not None:
            model_ref = next(
                (item for item in run.artifact_refs if item.uri == trial.artifact_uri), None
            )
            if model_ref is None or model_ref.sha256 != trial.artifact_sha256:
                raise ValueError("trial model artifact reference mismatch")
            refs.append(model_ref.uri)
            with zipfile.ZipFile(BytesIO(self.store.read_bytes(model_ref))) as bundle:
                curves = json.loads(bundle.read("metadata.json"))["environment"]
        points: tuple[PredictionPoint, ...] = ()
        quantile = None
        status = "unavailable"
        if (
            trial.diagnostic_uri is not None
            and trial.diagnostic_sha256 is not None
            and trial.diagnostic_size_bytes is not None
        ):
            diagnostic_ref = ArtifactRef(
                uri=trial.diagnostic_uri,
                sha256=trial.diagnostic_sha256,
                size_bytes=trial.diagnostic_size_bytes,
            )
            payload = json.loads(self.store.read_bytes(diagnostic_ref))
            if payload.get("partition") != "validation":
                raise ValueError("trial diagnostic is not validation-only")
            points = tuple(PredictionPoint.model_validate(item) for item in payload["points"])
            quantile = payload.get("representative_quantile")
            refs.append(diagnostic_ref.uri)
            status = "available" if points else "unavailable"
        train = curves.get("training_loss_curve")
        valid = curves.get("validation_loss_curve")
        return TrialDetail(
            trial=trial,
            duration_seconds=(trial.ended_at - trial.started_at).total_seconds(),
            artifact_references=tuple(refs),
            training_loss_curve=tuple(train) if isinstance(train, list) else None,
            validation_loss_curve=tuple(valid) if isinstance(valid, list) else None,
            prediction_points=points,
            prediction_status=status,
            representative_quantile=quantile,
        )
