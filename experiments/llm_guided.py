"""Non-human guided search through the existing OptimizationWorkflow and Agents SDK."""

from __future__ import annotations

import asyncio
import os

from common import Context, make_workflow, selected_metrics

from interactive_forecasting.agents.runtime import OpenAIAgentsRuntime
from interactive_forecasting.domain.optimization import OptimizationPhase
from interactive_forecasting.domain.types import Actor
from interactive_forecasting.storage.sql import LLMCallRepository


async def _run(context: Context, runtime) -> dict:
    workflow, setup = make_workflow(context, runtime)
    failure: str | None = None
    session = workflow.create(context.task.task_id, setup)
    try:
        while session.phase not in {
            OptimizationPhase.COMPLETED,
            OptimizationPhase.FAILED,
            OptimizationPhase.CANCELLED,
        }:
            session = await workflow.advance(context.task.task_id, session.version)
    except Exception as exc:
        # Persisted trials/calls remain auditable; never silently continue as unguided search.
        failure = f"{type(exc).__name__}: {exc}"
        session = workflow.sessions.get(context.task.task_id)
        assert session is not None
    run = workflow.runs.get(session.run_id)
    assert run is not None
    selected = next((t for t in run.trials if t.trial_id == run.selected_trial_id), None)
    calls = LLMCallRepository(context.db).list_for_task(context.task.task_id)
    return {
        "experiment": "llm_guided",
        "dataset": context.dataset,
        "series": context.series,
        "status": "failed" if failure else run.status.value,
        "run_id": str(run.run_id),
        "seed": context.seed,
        "budget": context.backend.trial_budget,
        "source": context.source_metadata,
        "snapshot_sha256": context.snapshot.artifact.sha256,
        "backend": context.backend.model_dump(mode="json"),
        "trials": [trial.model_dump(mode="json") for trial in run.trials],
        "selected_trial_id": str(run.selected_trial_id) if run.selected_trial_id else None,
        "selected_trial_number": selected.number if selected else None,
        "selected_candidate": selected.candidate.candidate.model_dump(mode="json")
        if selected and selected.candidate
        else None,
        "final_metrics": selected_metrics(context, run) if selected and not failure else {},
        "reason": failure,
        "audit": {
            "session": session.model_dump(mode="json"),
            "llm_calls": [call.model_dump(mode="json") for call in calls],
            "provider_model": str(runtime.role_agents[Actor.MODEL_MANAGER].model)
            if isinstance(runtime, OpenAIAgentsRuntime)
            else "provider_fixture",
        },
    }


def run(context: Context, model: str | None, *, temperature: float = 0, runtime=None) -> dict:
    if runtime is None:
        if not model:
            raise ValueError("LLM provider/model must be explicitly configured")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI provider unavailable: OPENAI_API_KEY is not configured")
        runtime = OpenAIAgentsRuntime.for_model(model, temperature=temperature)
    return asyncio.run(_run(context, runtime))


if __name__ == "__main__":
    import sys

    from run_all import main

    main(["--experiment", "llm_guided", *sys.argv[1:]])
