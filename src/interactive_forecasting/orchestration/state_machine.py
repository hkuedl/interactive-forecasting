"""Deterministic transition rules; no SDK or numerical-service dependency."""

from collections.abc import Mapping
from enum import Enum

from interactive_forecasting.domain.models import Task, utc_now
from interactive_forecasting.domain.preparation import PREPARATION_PATH, PreparationStep
from interactive_forecasting.domain.types import (
    DeploymentState,
    OptimizationState,
    PreparationState,
    Stage,
    WorkflowStatus,
)


class InvalidTransition(ValueError):
    pass


class VersionConflict(ValueError):
    pass


SUBSTATES: Mapping[Stage, type[Enum]] = {
    Stage.PREPARATION: PreparationState,
    Stage.OPTIMIZATION: OptimizationState,
    Stage.DEPLOYMENT: DeploymentState,
}

INITIAL_SUBSTATE = {
    Stage.PREPARATION: PreparationState.COLLECT_METADATA.value,
    Stage.OPTIMIZATION: OptimizationState.INITIALIZE_SEARCH.value,
    Stage.DEPLOYMENT: DeploymentState.SELECT_BEST_CONFIGURATION.value,
}

SUBSTATE_PATHS: dict[Stage, dict[str, set[str]]] = {}
for stage, enum_type in SUBSTATES.items():
    values = list(enum_type)
    paths: dict[str, set[str]] = {}
    for current, following in zip(values, values[1:], strict=False):
        paths[current.value] = {following.value}
    paths[values[-1].value] = set()
    SUBSTATE_PATHS[stage] = paths

# Milestone 4B path coexists with historical Preparation substates for persisted tasks.
SUBSTATE_PATHS[Stage.PREPARATION][PreparationState.COLLECT_METADATA.value].add(
    PreparationStep.INSPECT_SCHEMA.value
)
for current, following in zip(PREPARATION_PATH[1:], PREPARATION_PATH[2:], strict=False):
    SUBSTATE_PATHS[Stage.PREPARATION].setdefault(current.value, set()).add(following.value)
SUBSTATE_PATHS[Stage.PREPARATION][PreparationStep.CONFIRM_TASK.value] = {
    PreparationState.PREPARATION_READY.value
}

SUBSTATE_PATHS[Stage.PREPARATION][PreparationState.VALIDATE_METADATA.value].add(
    PreparationState.COLLECT_METADATA.value
)
SUBSTATE_PATHS[Stage.PREPARATION][PreparationState.CHECK_REQUIRED_COLUMNS.value].update(
    {PreparationState.CONFIRM_COLUMNS.value, PreparationState.LOAD_DATASET.value}
)
SUBSTATE_PATHS[Stage.OPTIMIZATION][OptimizationState.CHECK_STOPPING_CONDITION.value].add(
    OptimizationState.SUMMARIZE_HISTORY.value
)
SUBSTATE_PATHS[Stage.DEPLOYMENT][DeploymentState.WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT.value].add(
    DeploymentState.VALIDATE_POSTPROCESS_REQUEST.value
)
SUBSTATE_PATHS[Stage.DEPLOYMENT][DeploymentState.SAVE_DEPLOYMENT_RESULT.value].add(
    DeploymentState.WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT.value
)


STATUS_PATHS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.NOT_STARTED: frozenset(
        {WorkflowStatus.ACTIVE, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.ACTIVE: frozenset(
        {
            WorkflowStatus.WAITING_FOR_USER,
            WorkflowStatus.WAITING_FOR_AGENT,
            WorkflowStatus.WAITING_FOR_JOB,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.WAITING_FOR_USER: frozenset(
        {WorkflowStatus.ACTIVE, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.WAITING_FOR_AGENT: frozenset(
        {WorkflowStatus.ACTIVE, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.WAITING_FOR_JOB: frozenset(
        {WorkflowStatus.ACTIVE, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.COMPLETED: frozenset(),
    WorkflowStatus.FAILED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
}


class WorkflowStateMachine:
    """Transition validation. Business preconditions are added in later milestones."""

    def allowed_actions(self, task: Task) -> frozenset[str]:
        if task.stage in {Stage.COMPLETED, Stage.FAILED} or task.workflow_status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }:
            return frozenset()
        actions: set[str] = set()
        if task.stage == Stage.NEW and task.workflow_status == WorkflowStatus.NOT_STARTED:
            actions.add("start_preparation")
        if task.stage in SUBSTATE_PATHS and task.substate is not None:
            if task.workflow_status == WorkflowStatus.ACTIVE and SUBSTATE_PATHS[task.stage].get(
                task.substate
            ):
                actions.add("advance_substate")
        if (
            task.stage == Stage.PREPARATION
            and task.substate == PreparationState.PREPARATION_READY
            and task.workflow_status in {WorkflowStatus.ACTIVE, WorkflowStatus.COMPLETED}
        ):
            actions.add("start_optimization")
        if (
            task.stage == Stage.OPTIMIZATION
            and task.substate == OptimizationState.OPTIMIZATION_COMPLETE
            and task.workflow_status in {WorkflowStatus.ACTIVE, WorkflowStatus.COMPLETED}
        ):
            actions.add("start_deployment")
        if (
            task.stage == Stage.DEPLOYMENT
            and task.substate
            in {
                DeploymentState.WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT,
                DeploymentState.SAVE_DEPLOYMENT_RESULT,
            }
            and task.workflow_status in {WorkflowStatus.ACTIVE, WorkflowStatus.COMPLETED}
        ):
            actions.add("complete_task")
        if task.workflow_status != WorkflowStatus.COMPLETED:
            actions.add("fail_task")
        return frozenset(actions)

    def transition_stage(self, task: Task, target: Stage, *, expected_version: int) -> Task:
        self._check_version(task, expected_version)
        action_for_target = {
            Stage.PREPARATION: "start_preparation",
            Stage.OPTIMIZATION: "start_optimization",
            Stage.DEPLOYMENT: "start_deployment",
            Stage.COMPLETED: "complete_task",
            Stage.FAILED: "fail_task",
        }
        action = action_for_target.get(target)
        if action is None or action not in self.allowed_actions(task):
            raise InvalidTransition(f"{task.stage} cannot transition to {target}")
        substate = INITIAL_SUBSTATE.get(target)
        return task.model_copy(
            update={
                "stage": target,
                "workflow_status": (
                    WorkflowStatus.COMPLETED
                    if target == Stage.COMPLETED
                    else WorkflowStatus.FAILED
                    if target == Stage.FAILED
                    else WorkflowStatus.ACTIVE
                ),
                "substate": substate,
                "version": task.version + 1,
                "updated_at": utc_now(),
            }
        )

    def transition_substate(self, task: Task, target: str, *, expected_version: int) -> Task:
        self._check_version(task, expected_version)
        if task.workflow_status != WorkflowStatus.ACTIVE:
            raise InvalidTransition("substate progression requires active workflow status")
        permitted = SUBSTATE_PATHS.get(task.stage, {}).get(task.substate or "", set())
        if target not in permitted:
            raise InvalidTransition(f"{task.substate} cannot transition to {target}")
        return task.model_copy(
            update={"substate": target, "version": task.version + 1, "updated_at": utc_now()}
        )

    def transition_status(
        self, task: Task, target: WorkflowStatus, *, expected_version: int
    ) -> Task:
        self._check_version(task, expected_version)
        if target not in STATUS_PATHS[task.workflow_status]:
            raise InvalidTransition(f"{task.workflow_status} cannot transition to {target}")
        if task.stage in {Stage.COMPLETED, Stage.FAILED}:
            raise InvalidTransition("terminal task cannot change lifecycle status")
        if target == WorkflowStatus.COMPLETED and not (
            task.stage == Stage.PREPARATION
            and task.substate == PreparationState.PREPARATION_READY
            or task.stage == Stage.OPTIMIZATION
            and task.substate == OptimizationState.OPTIMIZATION_COMPLETE
            or task.stage == Stage.DEPLOYMENT
            and task.substate
            in {
                DeploymentState.WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT,
                DeploymentState.SAVE_DEPLOYMENT_RESULT,
            }
        ):
            raise InvalidTransition("stage cannot complete before its terminal substate")
        if task.stage == Stage.NEW and target not in {
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }:
            raise InvalidTransition("new task must enter preparation before becoming active")
        return task.model_copy(
            update={"workflow_status": target, "version": task.version + 1, "updated_at": utc_now()}
        )

    @staticmethod
    def _check_version(task: Task, expected_version: int) -> None:
        if task.version != expected_version:
            raise VersionConflict(
                f"expected task version {expected_version}, actual {task.version}"
            )
