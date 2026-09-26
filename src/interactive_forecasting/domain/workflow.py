"""Serializable UI projection built only from persisted application records."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from interactive_forecasting.domain.models import ArtifactRef, ExperimentRun, Task
from interactive_forecasting.domain.types import ResearchStage, Stage, WorkflowStatus


class View(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreparationView(View):
    dataset_summary_ref: ArtifactRef | None = None
    data_quality_status: str | None = None
    task_definition_status: str = "not_started"
    visualization_refs: tuple[ArtifactRef, ...] = ()


class OptimizationView(View):
    run_id: UUID | None = None
    run_status: str = "not_started"
    current_trial_id: UUID | None = None
    current_trial_status: str | None = None
    current_validation_objective: float | None = None
    best_trial_id: UUID | None = None
    best_validation_objective: float | None = None
    trial_count: int = 0
    trial_history_ids: tuple[UUID, ...] = ()
    family_trial_counts: dict[str, int] = Field(default_factory=dict)
    effective_search_space_id: str | None = None
    guidance_count: int = 0
    visualization_refs: tuple[ArtifactRef, ...] = ()


class VisualizationView(View):
    artifacts: tuple[ArtifactRef, ...] = ()


class UserInterventionView(View):
    required: bool = False
    prompt_message_id: UUID | None = None
    correlation_id: UUID | None = None


class DeploymentView(View):
    status: str = "not_started"
    forecast_id: UUID | None = None
    visualization_refs: tuple[ArtifactRef, ...] = ()


class WorkflowSnapshot(View):
    task_id: UUID
    task_version: int
    research_stage: ResearchStage | None
    status: WorkflowStatus
    substate: str | None
    allowed_actions: tuple[str, ...]
    preparation: PreparationView = Field(default_factory=PreparationView)
    optimization: OptimizationView = Field(default_factory=OptimizationView)
    visualization: VisualizationView = Field(default_factory=VisualizationView)
    intervention: UserInterventionView = Field(default_factory=UserInterventionView)
    deployment: DeploymentView = Field(default_factory=DeploymentView)


def project_workflow(
    task: Task,
    *,
    allowed_actions: frozenset[str],
    run: ExperimentRun | None = None,
    preparation: PreparationView | None = None,
    intervention: UserInterventionView | None = None,
) -> WorkflowSnapshot:
    """Project deterministic records, never LLM-authored metric text."""
    stage = {
        Stage.PREPARATION: ResearchStage.PREPARATION,
        Stage.OPTIMIZATION: ResearchStage.TRAINING_EVALUATION,
        Stage.DEPLOYMENT: ResearchStage.DEPLOYMENT,
    }.get(task.stage)
    optimization = OptimizationView()
    if run is not None:
        if run.task_definition is not None and run.task_definition.task_id != task.task_id:
            raise ValueError("run belongs to another task")
        current = run.trials[-1] if run.trials else None
        selected = next(
            (trial for trial in run.trials if trial.trial_id == run.selected_trial_id), None
        )
        family_counts: dict[str, int] = {}
        for trial in run.trials:
            name = trial.request.family.value
            family_counts[name] = family_counts.get(name, 0) + 1
        optimization = OptimizationView(
            run_id=run.run_id,
            run_status=run.status.value,
            current_trial_id=current.trial_id if current is not None else None,
            current_trial_status=current.status if current is not None else None,
            current_validation_objective=current.objective if current is not None else None,
            best_trial_id=run.selected_trial_id,
            best_validation_objective=selected.objective if selected is not None else None,
            trial_count=len(run.trials),
            trial_history_ids=tuple(trial.trial_id for trial in run.trials),
            family_trial_counts=family_counts,
            effective_search_space_id=(
                str(run.effective_space.space_id) if run.effective_space is not None else None
            ),
            guidance_count=len(run.guidance_history),
        )
    return WorkflowSnapshot(
        task_id=task.task_id,
        task_version=task.version,
        research_stage=stage,
        status=task.workflow_status,
        substate=task.substate,
        allowed_actions=tuple(sorted(allowed_actions)),
        preparation=preparation or PreparationView(),
        optimization=optimization,
        intervention=intervention or UserInterventionView(),
        deployment=DeploymentView(
            status=task.workflow_status.value
            if stage == ResearchStage.DEPLOYMENT
            else "not_started"
        ),
    )
