"""Application-owned round boundaries above the existing Stage 3 SearchEngine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID, uuid4

from interactive_forecasting.agents.contracts import (
    AgentAction,
    AgentContext,
    ExecuteApproved,
    ExecutionRequest,
    ModelDeveloperContext,
    ModelManagerContext,
    ProposeGuidance,
    ProposeGuidanceBatch,
    ProposeHumanGuidance,
    TaskManagerContext,
    ToolResult,
    UnauthorizedAction,
    ValidationTrialSummary,
)
from interactive_forecasting.agents.model_manager_policy import MODEL_MANAGER_PROMPT_VERSION
from interactive_forecasting.agents.runtime import AgentRequest, AgentRuntime
from interactive_forecasting.domain.forecasting import OriginSchedule, ResolvedCandidate
from interactive_forecasting.domain.models import Event, ExperimentRun, Message, Task
from interactive_forecasting.domain.optimization import (
    GuidanceDraft,
    GuidanceEffect,
    OptimizationPhase,
    OptimizationSession,
)
from interactive_forecasting.domain.search import (
    BackendConfig,
    GuidanceCommand,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import (
    Actor,
    JobStatus,
    MessageKind,
    ModelFamily,
    OptimizationMode,
    OptimizationState,
    Stage,
    Topic,
    WorkflowStatus,
)
from interactive_forecasting.orchestration.agent_workflow import ActionExecutor, AgentCoordinator
from interactive_forecasting.orchestration.messages import MessageBus
from interactive_forecasting.orchestration.search_actions import (
    SearchActionExecutor,
    SearchExecutionApprover,
)
from interactive_forecasting.orchestration.state_machine import (
    VersionConflict,
    WorkflowStateMachine,
)
from interactive_forecasting.services.optimization.backend import OptunaBackend
from interactive_forecasting.services.optimization.engine import SearchEngine
from interactive_forecasting.services.optimization.execution import (
    TrialExecutor,
    ValidationEvaluator,
)
from interactive_forecasting.services.optimization.summary import summarize_optimization
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.preparation import PreparationRepository
from interactive_forecasting.storage.sql import (
    ExperimentRepository,
    LLMCallRepository,
    MessageRepository,
    OptimizationSessionRepository,
    TaskRepository,
)


@dataclass(frozen=True)
class RunSetup:
    mode: OptimizationMode
    spec_id: str
    backend: BackendConfig
    schedule: OriginSchedule
    templates: dict[ModelFamily, ResolvedCandidate]
    metric: ValidationMetricConfig
    experiment_seed: int
    pause_at_boundary: bool = False


def parse_guidance_text(text: str) -> tuple[GuidanceCommand, ...] | None:
    """Deterministic bounded chat grammar; ambiguous language needs Task Manager runtime."""
    normalized = text.strip().rstrip(".")
    name = r"(Linear|SVR|MLP|XGBoost|LSTM|GRU|CNN)"
    match = re.fullmatch(rf"(?:prefer|prioritize) {name}", normalized, re.IGNORECASE)
    if match:
        family = next(item for item in ModelFamily if item.value.lower() == match.group(1).lower())
        return (GuidanceCommand(operation="prefer_family", families=(family,)),)
    match = re.fullmatch(rf"exclude {name}", normalized, re.IGNORECASE)
    if match:
        family = next(item for item in ModelFamily if item.value.lower() == match.group(1).lower())
        return (GuidanceCommand(operation="exclude_family", families=(family,)),)
    match = re.fullmatch(rf"allocate {name} (\d+)", normalized, re.IGNORECASE)
    if match:
        family = next(item for item in ModelFamily if item.value.lower() == match.group(1).lower())
        return (
            GuidanceCommand(
                operation="allocate_family_trials", allocation={family: int(match.group(2))}
            ),
        )
    match = re.fullmatch(
        r"narrow ([\w.]+) to (-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return (
            GuidanceCommand(
                operation="narrow_parameter",
                parameter=match.group(1),
                low=float(match.group(2)),
                high=float(match.group(3)),
            ),
        )
    return None


class _GuidanceExecutor:
    def __init__(self, workflow: OptimizationWorkflow, task_id: UUID):
        self.workflow = workflow
        self.task_id = task_id

    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult:
        if not isinstance(context, ModelManagerContext):
            raise UnauthorizedAction("Model Manager context required")
        commands: tuple[GuidanceCommand, ...]
        if isinstance(action, ProposeGuidance):
            commands = (action.command,)
        elif isinstance(action, ProposeGuidanceBatch):
            commands = action.commands
        else:
            raise UnauthorizedAction("Model Manager may only propose typed guidance here")
        session = self.workflow._session(self.task_id)
        user_effect = self.workflow._current_human_effect(session)
        source = "combined" if user_effect is not None else "model_manager"
        updated = self.workflow._apply_guidance(
            session,
            commands,
            source=source,
            original_text=user_effect.original_text if user_effect else None,
            rationale="Model Manager structured guidance",
        )
        return ToolResult(
            status="completed",
            data={"session_version": updated.version, "guidance_count": len(commands)},
        )


class _HumanDraftExecutor:
    def __init__(self, workflow: OptimizationWorkflow, task_id: UUID):
        self.workflow = workflow
        self.task_id = task_id

    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult:
        if not isinstance(action, ProposeHumanGuidance) or not isinstance(
            context, TaskManagerContext
        ):
            raise UnauthorizedAction("Task Manager may only propose typed human guidance here")
        updated = self.workflow.save_draft(
            self.task_id,
            action.expected_session_version,
            action.commands,
            original_text=context.user_text,
        )
        return ToolResult(status="completed", data={"session_version": updated.version})


@dataclass
class OptimizationWorkflow:
    tasks: TaskRepository
    preparations: PreparationRepository
    runs: ExperimentRepository
    sessions: OptimizationSessionRepository
    messages: MessageRepository
    calls: LLMCallRepository
    store: ArtifactStore
    machine: WorkflowStateMachine
    runtime: AgentRuntime | None = None

    def _session(self, task_id: UUID) -> OptimizationSession:
        session = self.sessions.get(task_id)
        if session is None:
            raise ValueError("optimization session not found")
        return session

    def _run(self, session: OptimizationSession) -> ExperimentRun:
        run = self.runs.get(session.run_id)
        if run is None:
            raise ValueError("search run not found")
        return run

    def _task(self, task_id: UUID) -> Task:
        task = self.tasks.get(task_id)
        if task is None or task.stage != Stage.OPTIMIZATION:
            raise ValueError("task is not in Training & Evaluation")
        return task

    def _save(self, session: OptimizationSession, **changes: object) -> OptimizationSession:
        next_state = session.model_copy(update={"version": session.version + 1, **changes})
        return self.sessions.save(
            OptimizationSession.model_validate(next_state.model_dump()),
            expected_version=session.version,
        )

    def _transition_to(self, task_id: UUID, target: OptimizationState) -> None:
        order = list(OptimizationState)
        while True:
            task = self._task(task_id)
            current = OptimizationState(task.substate or OptimizationState.INITIALIZE_SEARCH)
            if current == target:
                return
            if current == OptimizationState.CHECK_STOPPING_CONDITION:
                next_state = (
                    OptimizationState.OPTIMIZATION_COMPLETE
                    if target == OptimizationState.OPTIMIZATION_COMPLETE
                    else OptimizationState.SUMMARIZE_HISTORY
                )
            elif order.index(current) > order.index(target):
                return
            else:
                next_state = order[order.index(current) + 1]
            changed = self.machine.transition_substate(
                task, next_state.value, expected_version=task.version
            )
            self.tasks.save_transition(
                changed,
                Event(
                    task_id=task_id,
                    source_role=Actor.SYSTEM,
                    topic=Topic.SYSTEM,
                    message_type="optimization_step",
                    payload={"from": current.value, "to": next_state.value},
                    outcome="advanced",
                ),
                expected_version=task.version,
            )

    def _status(self, task_id: UUID, target: WorkflowStatus) -> None:
        task = self._task(task_id)
        if task.workflow_status == target:
            return
        changed = self.machine.transition_status(task, target, expected_version=task.version)
        self.tasks.save_transition(
            changed,
            Event(
                task_id=task_id,
                source_role=Actor.SYSTEM,
                topic=Topic.SYSTEM,
                message_type="optimization_status",
                payload={"status": target.value},
                outcome="advanced",
            ),
            expected_version=task.version,
        )

    def _notify(self, task_id: UUID, text: str) -> None:
        MessageBus(self.messages, self.tasks).publish(
            Message(
                task_id=task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=Actor.USER,
                topic=Topic.CHAT,
                kind=MessageKind.EVENT,
                message_type="optimization.status",
                payload={"text": text},
            )
        )

    def _engine(self, run: ExperimentRun) -> SearchEngine:
        if (
            run.backend_config is None
            or run.dataset_snapshot is None
            or run.task_definition is None
            or run.protocol is None
            or run.origin_schedule is None
            or not run.templates
        ):
            raise ValueError("run lacks frozen search inputs")
        index = next(iter(run.templates.values())).index
        evaluator = ValidationEvaluator.from_snapshot(
            self.store,
            run.dataset_snapshot,
            run.task_definition,
            index,
            run.protocol,
            run.origin_schedule,
        )
        return SearchEngine(
            self.runs,
            OptunaBackend(run.backend_config),
            TrialExecutor(evaluator, self.store),
        )

    def create(self, task_id: UUID, setup: RunSetup) -> OptimizationSession:
        task = self._task(task_id)
        if task.substate != OptimizationState.INITIALIZE_SEARCH or self.sessions.get(task_id):
            raise ValueError("task already has an optimization session")
        preparation = self.preparations.get(task_id)
        if (
            preparation is None
            or not preparation.frozen
            or preparation.prepared is None
            or preparation.definition is None
            or preparation.protocol is None
            or preparation.effective_search_space is None
            or task.definition_id != preparation.definition.definition_id
        ):
            raise ValueError("frozen Preparation inputs are required")
        if setup.mode != OptimizationMode.VANILLA_BO and self.runtime is None:
            raise ValueError("guided modes require a configured AgentRuntime")
        if setup.pause_at_boundary and setup.mode != OptimizationMode.HUMAN_LLM_GUIDED:
            raise ValueError("human intervention is available only in human_llm_guided mode")
        if setup.experiment_seed < 0 or setup.experiment_seed > 2**32 - 1:
            raise ValueError("experiment seed is out of range")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", setup.spec_id) or setup.spec_id in {".", ".."}:
            raise ValueError("invalid experiment spec ID")
        if not setup.templates:
            raise ValueError("complete family templates are required")
        index = next(iter(setup.templates.values())).index
        if any(
            candidate.family != family
            or candidate.index != index
            or candidate.auxiliary_policies != preparation.definition.auxiliary_policies
            for family, candidate in setup.templates.items()
        ):
            raise ValueError("templates must share frozen index and auxiliary policies")
        if (
            index.delta != preparation.definition.delta
            or index.horizon != preparation.definition.horizon
            or index.offset_unit != preparation.definition.time_unit
            or index.frequency != preparation.prepared.dataset.frequency
        ):
            raise ValueError("template index differs from frozen task")
        evaluator = ValidationEvaluator.from_snapshot(
            self.store,
            preparation.prepared.dataset,
            preparation.definition,
            index,
            preparation.protocol,
            setup.schedule,
        )
        engine = SearchEngine(
            self.runs, OptunaBackend(setup.backend), TrialExecutor(evaluator, self.store)
        )
        run = engine.create_run(
            spec_id=setup.spec_id,
            task=preparation.definition,
            snapshot=preparation.prepared.dataset,
            protocol=preparation.protocol,
            schedule=setup.schedule,
            space=preparation.effective_search_space,
            templates=setup.templates,
            metric=setup.metric,
            experiment_seed=setup.experiment_seed,
        )
        initial_target = min(setup.backend.n_startup_trials, setup.backend.trial_budget)
        session = self.sessions.create(
            OptimizationSession(
                task_id=task_id,
                run_id=run.run_id,
                mode=setup.mode,
                initial_target=initial_target,
                phase=(
                    OptimizationPhase.INITIAL_EXPLORATION
                    if initial_target
                    else OptimizationPhase.BOUNDARY
                ),
                pause_at_boundary=setup.pause_at_boundary,
            )
        )
        self._transition_to(task_id, OptimizationState.INITIAL_RANDOM_TRIALS)
        if initial_target == 0:
            self._transition_to(task_id, OptimizationState.SUMMARIZE_HISTORY)
        self._notify(task_id, f"{setup.mode.value} search initialized; no trial has run yet.")
        return session

    def get(self, task_id: UUID) -> OptimizationSession:
        session = self._session(task_id)
        if self._run(session).status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return self._reconcile(session)
        return session

    def save_draft(
        self,
        task_id: UUID,
        expected_version: int,
        commands: tuple[GuidanceCommand, ...],
        *,
        original_text: str | None = None,
    ) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization guidance draft")
        if session.mode != OptimizationMode.HUMAN_LLM_GUIDED or session.phase not in {
            OptimizationPhase.BOUNDARY,
            OptimizationPhase.WAITING_FOR_USER,
        }:
            raise ValueError("human guidance is unavailable in this mode or phase")
        session = self._recover_pending_guidance(session)
        return self._save(
            session,
            guidance_draft=GuidanceDraft(commands=commands, original_text=original_text),
        )

    @staticmethod
    def _current_human_effect(session: OptimizationSession) -> GuidanceEffect | None:
        run_round = session.last_reconciled_round
        return next(
            (
                effect
                for effect in reversed(session.guidance_effects)
                if effect.round_number == run_round
                and effect.source == "human"
                and effect.status == "applied"
            ),
            None,
        )

    def _apply_guidance(
        self,
        session: OptimizationSession,
        commands: tuple[GuidanceCommand, ...],
        *,
        source: str,
        original_text: str | None,
        rationale: str | None,
        draft_id: UUID | None = None,
    ) -> OptimizationSession:
        run = self._run(session)
        if run.effective_space is None:
            raise ValueError("search space is unavailable")
        before = run.effective_space.space_id
        effect = GuidanceEffect(
            draft_id=draft_id,
            source=source,
            round_number=run.completed_rounds,
            commands=commands,
            original_text=original_text,
            rationale=rationale,
            space_before=before,
            guidance_count_before=len(run.guidance_history),
            status="pending",
        )
        pending = self._save(session, guidance_effects=(*session.guidance_effects, effect))
        try:
            changed = self._engine(run).apply_guidance(run.run_id, commands)
        except Exception as exc:
            rejected = effect.model_copy(
                update={"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}
            )
            self._save(
                pending,
                guidance_effects=(*pending.guidance_effects[:-1], rejected),
                last_error=rejected.error,
            )
            raise
        applied = effect.model_copy(
            update={
                "status": "applied",
                "space_after": changed.effective_space.space_id
                if changed.effective_space
                else before,
            }
        )
        return self._save(
            pending,
            guidance_effects=(*pending.guidance_effects[:-1], applied),
            last_error=None,
        )

    def _consume_applied_draft(self, session: OptimizationSession) -> OptimizationSession:
        draft = session.guidance_draft
        if draft is not None and any(
            effect.draft_id == draft.draft_id and effect.status == "applied"
            for effect in session.guidance_effects
        ):
            return self._save(session, guidance_draft=None)
        return session

    def _recover_pending_guidance(self, session: OptimizationSession) -> OptimizationSession:
        if not session.guidance_effects or session.guidance_effects[-1].status != "pending":
            return self._consume_applied_draft(session)
        effect = session.guidance_effects[-1]
        run = self._run(session)
        applied = (
            effect.guidance_count_before is not None
            and len(run.guidance_history) == effect.guidance_count_before + len(effect.commands)
            and run.guidance_history[effect.guidance_count_before :] == effect.commands
        )
        revised = effect.model_copy(
            update={
                "status": "applied" if applied else "rejected",
                "space_after": run.effective_space.space_id
                if applied and run.effective_space is not None
                else None,
                "error": None if applied else "guidance interrupted before search commit",
            }
        )
        return self._save(
            session,
            guidance_effects=(*session.guidance_effects[:-1], revised),
            guidance_draft=None
            if applied
            and effect.source == "human"
            and session.guidance_draft is not None
            and effect.draft_id == session.guidance_draft.draft_id
            else session.guidance_draft,
            phase=OptimizationPhase.EXECUTING
            if applied and effect.source in {"model_manager", "combined"}
            else session.phase,
            last_error=revised.error,
        )

    def confirm_draft(self, task_id: UUID, expected_version: int) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization guidance draft")
        if session.mode != OptimizationMode.HUMAN_LLM_GUIDED or session.guidance_draft is None:
            raise ValueError("no human guidance draft to confirm")
        session = self._recover_pending_guidance(session)
        if session.guidance_draft is None:
            return session
        draft = session.guidance_draft
        updated = self._apply_guidance(
            session,
            draft.commands,
            source="human",
            original_text=draft.original_text,
            rationale="User-confirmed structured guidance",
            draft_id=draft.draft_id,
        )
        updated = self._save(updated, guidance_draft=None)
        self._notify(task_id, "Your structured guidance was validated and applied.")
        return updated

    def _coordinator(self, executor: ActionExecutor) -> AgentCoordinator:
        if self.runtime is None:
            raise ValueError("agent runtime is not configured")
        return AgentCoordinator(
            self.runtime,
            MessageBus(self.messages, self.tasks),
            self.tasks,
            self.messages,
            self.calls,
            executor,
        )

    async def chat(self, task_id: UUID, expected_version: int, text: str) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization guidance draft")
        if session.mode != OptimizationMode.HUMAN_LLM_GUIDED:
            raise ValueError("search guidance chat is unavailable outside human_llm_guided")
        user = Message(
            task_id=task_id,
            source_role=Actor.USER,
            target_role=Actor.TASK_MANAGER,
            topic=Topic.CHAT,
            kind=MessageKind.USER,
            message_type="optimization.guidance_input",
            payload={"text": text[:1000]},
        )
        bus = MessageBus(self.messages, self.tasks)
        bus.publish(user)
        parsed = parse_guidance_text(text)
        if parsed is not None:
            updated = self.save_draft(task_id, expected_version, parsed, original_text=text[:1000])
        elif self.runtime is not None:
            task = self._task(task_id)
            turn = await self._coordinator(_HumanDraftExecutor(self, task_id)).invoke(
                AgentRequest(
                    role=Actor.TASK_MANAGER,
                    task_id=task_id,
                    correlation_id=user.correlation_id,
                    prompt=(
                        "Interpret user search guidance as a typed, unconfirmed draft. "
                        "Ask for clarification if ambiguous."
                    ),
                    context=TaskManagerContext(
                        task=task,
                        user_text=text[:1000],
                        optimization_session_version=session.version,
                    ),
                    expected_task_version=task.version,
                    prompt_version="4c-task-manager-guidance-v1",
                ),
                parent_message_id=user.message_id,
            )
            if turn.action_status != "completed":
                raise ValueError("Task Manager did not produce a typed guidance draft")
            updated = self._session(task_id)
        else:
            self._notify(
                task_id, "Please use a supported guidance command or the structured controls."
            )
            return session
        self._notify(task_id, "Guidance draft saved. Review and confirm it before the next round.")
        return updated

    async def _plan(self, session: OptimizationSession) -> OptimizationSession:
        if session.mode == OptimizationMode.VANILLA_BO:
            return session
        if self.runtime is None:
            raise ValueError("guided mode requires AgentRuntime")
        run = self._run(session)
        if run.effective_space is None or run.task_definition is None:
            raise ValueError("search run is incomplete")
        self._transition_to(session.task_id, OptimizationState.MODEL_MANAGER_PLAN)
        task = self._task(session.task_id)
        correlation = uuid4()
        bus = MessageBus(self.messages, self.tasks)
        routed = Message(
            task_id=session.task_id,
            run_id=run.run_id,
            source_role=Actor.TASK_MANAGER,
            target_role=Actor.MODEL_MANAGER,
            topic=Topic.OPTIMIZE,
            kind=MessageKind.COMMAND,
            message_type="agent.route",
            payload={"instruction": "Plan the next search round from validation-only summary"},
            correlation_id=correlation,
        )
        bus.publish(routed)
        human = self._current_human_effect(session)
        context = ModelManagerContext(
            task=task,
            task_definition=run.task_definition,
            run_id=run.run_id,
            effective_search_space=run.effective_space,
            validation_history=tuple(
                ValidationTrialSummary(
                    trial_id=trial.trial_id,
                    family=trial.request.family.value,
                    status=trial.status,
                    validation_objective=trial.objective,
                )
                for trial in run.trials[-12:]
            ),
            guidance_history=run.guidance_history[-12:],
            optimization_summary=summarize_optimization(run),
            user_guidance=human.commands if human else (),
            user_guidance_text=human.original_text if human else None,
        )
        try:
            turn = await self._coordinator(_GuidanceExecutor(self, session.task_id)).invoke(
                AgentRequest(
                    role=Actor.MODEL_MANAGER,
                    task_id=session.task_id,
                    correlation_id=correlation,
                    prompt=(
                        "Analyze the summary, then emit supported typed guidance or no action. "
                        "Give a concise decision summary."
                    ),
                    context=context,
                    expected_task_version=task.version,
                    prompt_version=MODEL_MANAGER_PROMPT_VERSION,
                ),
                parent_message_id=routed.message_id,
            )
            if turn.action_status == "failed":
                raise ValueError("Model Manager guidance was rejected")
        except Exception as exc:
            current = self._session(session.task_id)
            self._save(current, phase=OptimizationPhase.BOUNDARY, last_error=str(exc))
            raise
        if turn.decision.action is not None:
            self._notify(session.task_id, f"Model Manager: {turn.decision.explanation[:400]}")
        return self._session(session.task_id)

    async def _execute(self, session: OptimizationSession) -> OptimizationSession:
        run = self._run(session)
        if run.task_definition is None or run.effective_space is None:
            raise ValueError("search run is incomplete")
        self._transition_to(session.task_id, OptimizationState.GENERATE_TRIAL_BATCH)
        self._transition_to(session.task_id, OptimizationState.EXECUTE_BATCH)
        task = self._task(session.task_id)
        context = ModelManagerContext(
            task=task,
            task_definition=run.task_definition,
            run_id=run.run_id,
            effective_search_space=run.effective_space,
        )
        approved = SearchExecutionApprover(self.runs).approve(
            ExecutionRequest(
                run_id=run.run_id,
                operation="run_round",
                expected_run_version=run.version,
            ),
            context,
        )
        session = self._save(session, phase=OptimizationPhase.EXECUTING, last_error=None)
        bus = MessageBus(self.messages, self.tasks)
        correlation = uuid4()
        approval_message = Message(
            task_id=session.task_id,
            run_id=run.run_id,
            source_role=Actor.SYSTEM,
            target_role=Actor.MODEL_DEVELOPER,
            topic=Topic.TRAIN,
            kind=MessageKind.COMMAND,
            message_type="execution.approved",
            payload={"approved_request": approved.model_dump(mode="json")},
            correlation_id=correlation,
        )
        bus.publish(approval_message)
        developer_context = ModelDeveloperContext(task=task, approved_request=approved)
        executor = SearchActionExecutor(self._engine(run), self.runs)
        try:
            if self.runtime is not None and session.mode != OptimizationMode.VANILLA_BO:
                turn = await self._coordinator(executor).invoke(
                    AgentRequest(
                        role=Actor.MODEL_DEVELOPER,
                        task_id=session.task_id,
                        correlation_id=correlation,
                        prompt="Execute only the approved request and report its ID.",
                        context=developer_context,
                        expected_task_version=task.version,
                        prompt_version="4c-model-developer-v1",
                    ),
                    parent_message_id=approval_message.message_id,
                )
                if turn.action_status != "completed":
                    raise ValueError("Model Developer execution failed")
            else:
                result = await executor.execute(
                    ExecuteApproved(approval_id=approved.approval_id), developer_context
                )
                bus.publish(
                    Message(
                        task_id=session.task_id,
                        run_id=run.run_id,
                        source_role=Actor.SERVICE,
                        target_role=Actor.TASK_MANAGER,
                        topic=Topic.TRAIN,
                        kind=MessageKind.RESULT,
                        message_type="action.result",
                        payload=result.model_dump(mode="json"),
                        correlation_id=correlation,
                        parent_message_id=approval_message.message_id,
                    )
                )
        except Exception as exc:
            current = self._session(session.task_id)
            self._save(current, last_error=f"{type(exc).__name__}: {exc}")
            raise
        return self._reconcile(self._session(session.task_id))

    def _reconcile(self, session: OptimizationSession) -> OptimizationSession:
        run = self._run(session)
        if run.completed_rounds <= session.last_reconciled_round and run.status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return session
        summary = summarize_optimization(run)
        summaries = session.summaries
        if not summaries or summaries[-1].run_version != run.version:
            summaries = (*summaries, summary)
        if run.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            phase = {
                JobStatus.COMPLETED: OptimizationPhase.COMPLETED,
                JobStatus.FAILED: OptimizationPhase.FAILED,
                JobStatus.CANCELLED: OptimizationPhase.CANCELLED,
            }[run.status]
        elif len(run.trials) < session.initial_target:
            phase = OptimizationPhase.INITIAL_EXPLORATION
        else:
            phase = OptimizationPhase.BOUNDARY
        effects = tuple(
            effect.model_copy(
                update={
                    "candidate_trial_numbers": tuple(
                        trial.number
                        for trial in run.trials
                        if trial.round_number == effect.round_number
                        and trial.space_id == (effect.space_after or effect.space_before)
                    )
                }
            )
            if effect.status == "applied" and not effect.candidate_trial_numbers
            else effect
            for effect in session.guidance_effects
        )
        for round_number in range(session.last_reconciled_round, run.completed_rounds):
            if any(
                effect.round_number == round_number and effect.status == "applied"
                for effect in effects
            ):
                continue
            trials = tuple(trial for trial in run.trials if trial.round_number == round_number)
            space_id = (
                trials[0].space_id
                if trials
                else run.effective_space.space_id
                if run.effective_space is not None
                else run.space_history[0].space_id
            )
            effects = (
                *effects,
                GuidanceEffect(
                    source="backend/unguided",
                    round_number=round_number,
                    rationale="No structured guidance was applied for this round.",
                    space_before=space_id,
                    space_after=space_id,
                    status="applied",
                    candidate_trial_numbers=tuple(trial.number for trial in trials),
                ),
            )
        changes = dict(
            phase=phase,
            summaries=summaries,
            guidance_effects=effects,
            last_reconciled_round=run.completed_rounds,
            last_error=None,
        )
        updated = (
            self._save(session, **changes)
            if any(getattr(session, name) != value for name, value in changes.items())
            else session
        )
        if phase in {
            OptimizationPhase.COMPLETED,
            OptimizationPhase.FAILED,
            OptimizationPhase.CANCELLED,
        }:
            self._transition_to(session.task_id, OptimizationState.OPTIMIZATION_COMPLETE)
            terminal_status = {
                OptimizationPhase.COMPLETED: WorkflowStatus.COMPLETED,
                OptimizationPhase.FAILED: WorkflowStatus.FAILED,
                OptimizationPhase.CANCELLED: WorkflowStatus.CANCELLED,
            }[phase]
            already_terminal = self._task(session.task_id).workflow_status == terminal_status
            self._status(session.task_id, terminal_status)
            if not already_terminal:
                self._notify(
                    session.task_id,
                    f"Optimization {phase.value}: {summary.completed_trials} completed trials; "
                    f"best validation objective {summary.best_validation_objective}.",
                )
        else:
            self._transition_to(session.task_id, OptimizationState.UPDATE_EXPERIMENT_STORE)
            self._transition_to(session.task_id, OptimizationState.UPDATE_VISUALIZATIONS)
            self._transition_to(session.task_id, OptimizationState.CHECK_STOPPING_CONDITION)
            self._transition_to(session.task_id, OptimizationState.SUMMARIZE_HISTORY)
            if phase == OptimizationPhase.BOUNDARY and session.pause_at_boundary:
                updated = self._save(updated, phase=OptimizationPhase.WAITING_FOR_USER)
                self._transition_to(
                    session.task_id, OptimizationState.WAIT_FOR_OPTIONAL_USER_GUIDANCE
                )
                self._status(session.task_id, WorkflowStatus.WAITING_FOR_USER)
            self._notify(
                session.task_id,
                f"Round {run.completed_rounds} recorded; {summary.remaining_budget} trials remain.",
            )
        return updated

    async def advance(self, task_id: UUID, expected_version: int) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization session")
        session = self._recover_pending_guidance(session)
        run = self._run(session)
        if run.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return self._reconcile(session)
        if session.phase == OptimizationPhase.EXECUTING:
            if run.completed_rounds > session.last_reconciled_round or run.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return self._reconcile(session)
            return await self._execute(session)
        if session.phase == OptimizationPhase.WAITING_FOR_USER:
            raise ValueError("resume or cancel the user-intervention boundary")
        if session.phase == OptimizationPhase.INITIAL_EXPLORATION:
            return await self._execute(session)
        if session.phase != OptimizationPhase.BOUNDARY:
            raise ValueError("optimization session is terminal")
        if session.guidance_draft is not None:
            raise ValueError("confirm or clear the guidance draft before continuing")
        if session.mode == OptimizationMode.HUMAN_LLM_GUIDED and session.pause_at_boundary:
            session = self._save(session, phase=OptimizationPhase.WAITING_FOR_USER)
            self._transition_to(task_id, OptimizationState.WAIT_FOR_OPTIONAL_USER_GUIDANCE)
            self._status(task_id, WorkflowStatus.WAITING_FOR_USER)
            return session
        return await self._plan_then_execute(session)

    async def _plan_then_execute(self, session: OptimizationSession) -> OptimizationSession:
        self._transition_to(session.task_id, OptimizationState.WAIT_FOR_OPTIONAL_USER_GUIDANCE)
        if session.mode != OptimizationMode.VANILLA_BO:
            session = await self._plan(session)
        self._transition_to(session.task_id, OptimizationState.VALIDATE_GUIDANCE)
        return await self._execute(self._session(session.task_id))

    async def resume(self, task_id: UUID, expected_version: int) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization session")
        if session.phase != OptimizationPhase.WAITING_FOR_USER:
            raise ValueError("optimization is not waiting for user")
        session = self._recover_pending_guidance(session)
        if session.guidance_draft is not None:
            raise ValueError("confirm or clear guidance before resuming")
        self._status(task_id, WorkflowStatus.ACTIVE)
        session = self._save(session, phase=OptimizationPhase.BOUNDARY)
        return await self._plan_then_execute(session)

    def clear_draft(self, task_id: UUID, expected_version: int) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization session")
        session = self._recover_pending_guidance(session)
        return (
            self._save(session, guidance_draft=None)
            if session.guidance_draft is not None
            else session
        )

    def cancel(self, task_id: UUID, expected_version: int) -> OptimizationSession:
        session = self._session(task_id)
        if session.version != expected_version:
            raise VersionConflict("stale optimization session")
        run = self._run(session)
        self._engine(run).cancel(run.run_id)
        if self._task(task_id).workflow_status == WorkflowStatus.WAITING_FOR_USER:
            self._status(task_id, WorkflowStatus.ACTIVE)
        return self._reconcile(session)
