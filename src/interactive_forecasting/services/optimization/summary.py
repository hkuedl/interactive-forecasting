"""Bounded, deterministic validation-only optimization summaries."""

from __future__ import annotations

from interactive_forecasting.domain.models import ExperimentRun
from interactive_forecasting.domain.optimization import OptimizationSummary, RecentTrial
from interactive_forecasting.domain.types import JobStatus, ModelFamily


def summarize_optimization(run: ExperimentRun) -> OptimizationSummary:
    if run.effective_space is None or run.backend_config is None:
        raise ValueError("search run is incomplete")
    completed = [trial for trial in run.trials if trial.status == "completed"]
    failed = len(run.trials) - len(completed)
    scored = [trial for trial in completed if trial.objective is not None]
    best = (
        min(
            scored,
            key=lambda trial: trial.objective if trial.objective is not None else float("inf"),
        )
        if scored
        else None
    )
    counts: dict[ModelFamily, int] = {}
    family_best: dict[ModelFamily, float] = {}
    for trial in run.trials:
        family = trial.request.family
        counts[family] = counts.get(family, 0) + 1
        if trial.objective is not None and trial.status == "completed":
            family_best[family] = min(family_best.get(family, trial.objective), trial.objective)
    remaining = max(0, run.backend_config.trial_budget - len(run.trials))
    if run.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
        stopping = run.status.value
    elif run.stop_requested:
        stopping = "stop_requested"
    elif remaining == 0:
        stopping = "trial_budget"
    elif run.completed_rounds >= run.backend_config.max_rounds:
        stopping = "round_budget"
    else:
        stopping = "continuing"
    return OptimizationSummary(
        run_id=run.run_id,
        run_version=run.version,
        current_round=run.completed_rounds,
        completed_trials=len(completed),
        failed_trials=failed,
        best_validation_objective=best.objective if best else None,
        best_candidate=best.request if best else None,
        best_configuration=best.candidate.candidate if best and best.candidate else None,
        recent_trials=tuple(
            RecentTrial(
                trial_id=trial.trial_id,
                number=trial.number,
                round_number=trial.round_number,
                family=trial.request.family,
                status=trial.status,
                objective=trial.objective,
            )
            for trial in run.trials[-12:]
        ),
        family_trial_counts=counts,
        family_best=family_best,
        objective_trend=tuple(
            trial.objective for trial in scored[-12:] if trial.objective is not None
        ),
        effective_space=run.effective_space,
        previous_guidance=run.guidance_history[-12:],
        remaining_budget=remaining,
        stopping_status=stopping,
    )
