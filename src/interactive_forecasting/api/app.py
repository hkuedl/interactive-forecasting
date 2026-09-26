"""FastAPI boundary for tasks and the versioned Preparation workspace."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from interactive_forecasting.agents.runtime import (
    AgentRuntime,
    AgentRuntimeError,
    OpenAIAgentsRuntime,
)
from interactive_forecasting.config import Settings
from interactive_forecasting.domain.forecasting import OriginSchedule, ResolvedCandidate
from interactive_forecasting.domain.models import (
    Adjustment,
    AdjustmentProposal,
    DeploymentReadiness,
    DeploymentSession,
    Forecast,
    ForecastVersion,
    FutureAuxiliaryValue,
    Message,
    ReferenceAnalysis,
    SensitivityRequest,
    SensitivityResult,
    Task,
)
from interactive_forecasting.domain.optimization import (
    OptimizationPhase,
    OptimizationSession,
    OptimizationSummary,
)
from interactive_forecasting.domain.preparation import (
    ColumnMappingDraft,
    ForecastTaskDraft,
    PreparationPlan,
    PreparationRecord,
)
from interactive_forecasting.domain.search import (
    BackendConfig,
    GuidanceCommand,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import (
    Actor,
    MessageKind,
    ModelFamily,
    OptimizationMode,
    Stage,
    Topic,
)
from interactive_forecasting.domain.workflow import (
    UserInterventionView,
    WorkflowSnapshot,
    project_workflow,
)
from interactive_forecasting.orchestration.deployment import DeploymentWorkflow
from interactive_forecasting.orchestration.deployment_agents import (
    DeploymentAgentBridge,
    DeploymentChatResult,
)
from interactive_forecasting.orchestration.optimization import OptimizationWorkflow, RunSetup
from interactive_forecasting.orchestration.preparation import PreparationWorkflow
from interactive_forecasting.orchestration.preparation_agents import PreparationAgentBridge
from interactive_forecasting.orchestration.state_machine import (
    InvalidTransition,
    VersionConflict,
    WorkflowStateMachine,
)
from interactive_forecasting.services.optimization.summary import summarize_optimization
from interactive_forecasting.services.visualization import (
    OptimizationVisuals,
    TrialDetail,
    VisualizationService,
)
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.preparation import PreparationRepository
from interactive_forecasting.storage.sql import (
    Database,
    DeploymentRepository,
    ExperimentRepository,
    LLMCallRepository,
    MessageRepository,
    OptimizationSessionRepository,
    TaskRepository,
)


@dataclass
class Container:
    settings: Settings
    database: Database
    tasks: TaskRepository
    messages: MessageRepository
    artifacts: ArtifactStore
    state_machine: WorkflowStateMachine
    preparation: PreparationWorkflow
    preparation_agent: PreparationAgentBridge | None = None
    optimization: OptimizationWorkflow | None = None
    visuals: VisualizationService | None = None
    deployment: DeploymentWorkflow | None = None
    deployment_agent: DeploymentAgentBridge | None = None


class TaskSnapshot(BaseModel):
    task: Task
    allowed_actions: list[str]
    workflow: WorkflowSnapshot


class VersionedAction(BaseModel):
    expected_version: int
    action_id: UUID | None = None


class AssistantProposalInput(VersionedAction):
    user_text: str | None = Field(default=None, max_length=1000)


class MappingEdit(VersionedAction):
    draft: ColumnMappingDraft


class PlanEdit(VersionedAction):
    draft: PreparationPlan


class ForecastEdit(VersionedAction):
    draft: ForecastTaskDraft


class ChatInput(VersionedAction):
    text: str


class RunSetupInput(BaseModel):
    mode: OptimizationMode
    spec_id: str
    backend: BackendConfig
    schedule: OriginSchedule
    templates: dict[ModelFamily, ResolvedCandidate]
    metric: ValidationMetricConfig
    experiment_seed: int = Field(ge=0, le=2**32 - 1)
    pause_at_boundary: bool = False


class GuidanceDraftInput(BaseModel):
    expected_version: int
    commands: tuple[GuidanceCommand, ...] = Field(min_length=1)


class DeploymentAction(VersionedAction):
    session_id: UUID


class DeploymentChatInput(DeploymentAction):
    text: str


class DeploymentValidationInput(DeploymentAction):
    future_auxiliaries: tuple[FutureAuxiliaryValue, ...] = ()


class AdjustmentDraftInput(DeploymentAction):
    proposal: AdjustmentProposal
    user_text: str | None = Field(default=None, max_length=1000)


class AdjustmentConfirmInput(DeploymentAction):
    confirmation_id: UUID


class ForecastResponse(BaseModel):
    forecast: Forecast
    original: ForecastVersion


class OptimizationStateResponse(BaseModel):
    session: OptimizationSession
    summary: OptimizationSummary
    visualizations: OptimizationVisuals


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    agent_runtime: AgentRuntime | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    db = database or Database(resolved.resolved_database_url)
    tasks = TaskRepository(db)
    messages = MessageRepository(db)
    machine = WorkflowStateMachine()
    preparation = PreparationWorkflow(
        tasks,
        PreparationRepository(db),
        messages,
        ArtifactStore(resolved.runtime_data_root),
        machine,
    )
    if agent_runtime is None and resolved.openai_model and os.getenv("OPENAI_API_KEY"):
        agent_runtime = OpenAIAgentsRuntime.for_model(resolved.openai_model)
    agent_bridge = (
        PreparationAgentBridge(agent_runtime, preparation, tasks, messages, LLMCallRepository(db))
        if agent_runtime is not None
        else None
    )
    runtime_store = ArtifactStore(resolved.runtime_data_root)
    optimization = OptimizationWorkflow(
        tasks=tasks,
        preparations=preparation.records,
        runs=ExperimentRepository(db),
        sessions=OptimizationSessionRepository(db),
        messages=messages,
        calls=LLMCallRepository(db),
        store=runtime_store,
        machine=machine,
        runtime=agent_runtime,
    )
    container = Container(
        settings=resolved,
        database=db,
        tasks=tasks,
        messages=messages,
        artifacts=ArtifactStore(resolved.artifact_root),
        state_machine=machine,
        preparation=preparation,
        preparation_agent=agent_bridge,
        optimization=optimization,
        visuals=VisualizationService(runtime_store),
        deployment=DeploymentWorkflow(
            tasks,
            preparation.records,
            optimization.sessions,
            optimization.runs,
            DeploymentRepository(db),
            runtime_store,
            machine,
            messages,
        ),
    )
    if agent_runtime is not None:
        assert container.deployment is not None
        container.deployment_agent = DeploymentAgentBridge(
            agent_runtime, container.deployment, tasks, messages, LLMCallRepository(db)
        )
    app = FastAPI(title="Interactive Forecasting", version="0.1.0")
    app.state.container = container

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tasks", response_model=Task, status_code=201)
    def create_task() -> Task:
        task = container.tasks.create(Task())
        container.preparation.records.create(PreparationRecord(task_id=task.task_id))
        return task

    def get_task_or_404(task_id: UUID) -> Task:
        task = container.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.get("/tasks/{task_id}", response_model=Task)
    def get_task(task_id: UUID) -> Task:
        return get_task_or_404(task_id)

    @app.get("/tasks/{task_id}/snapshot", response_model=TaskSnapshot)
    def snapshot(task_id: UUID) -> TaskSnapshot:
        task = get_task_or_404(task_id)
        actions = container.state_machine.allowed_actions(task)
        run = None
        intervention = None
        if task.stage == Stage.OPTIMIZATION and container.optimization is not None:
            session = container.optimization.sessions.get(task_id)
            if session is not None:
                run = container.optimization.runs.get(session.run_id)
                intervention = UserInterventionView(
                    required=session.phase == OptimizationPhase.WAITING_FOR_USER
                )
        return TaskSnapshot(
            task=task,
            allowed_actions=sorted(actions),
            workflow=project_workflow(
                task, allowed_actions=actions, run=run, intervention=intervention
            ),
        )

    @app.get("/tasks/{task_id}/allowed-actions", response_model=list[str])
    def allowed_actions(task_id: UUID) -> list[str]:
        return sorted(container.state_machine.allowed_actions(get_task_or_404(task_id)))

    @app.get("/tasks/{task_id}/messages", response_model=list[Message])
    def messages_for_task(task_id: UUID) -> list[Message]:
        get_task_or_404(task_id)
        return container.messages.list_for_task(task_id)

    @app.get("/tasks/{task_id}/preparation", response_model=PreparationRecord)
    def get_preparation(task_id: UUID) -> PreparationRecord:
        get_task_or_404(task_id)
        return container.preparation.get(task_id)

    @app.get("/tasks/{task_id}/preparation/assistant/status")
    def assistant_status(task_id: UUID) -> dict[str, bool]:
        get_task_or_404(task_id)
        return {"available": container.preparation_agent is not None}

    @app.post("/tasks/{task_id}/preparation/upload", response_model=PreparationRecord)
    async def upload_dataset(
        task_id: UUID,
        expected_version: Annotated[int, Form()],
        action_id: Annotated[UUID, Form()],
        file: Annotated[UploadFile, File()],
    ) -> PreparationRecord:
        content = await file.read(32 * 1024 * 1024 + 1)
        return container.preparation.upload(
            task_id, expected_version, file.filename or "", content, action_id
        )

    @app.post(
        "/tasks/{task_id}/preparation/actions/{action}", response_model=PreparationRecord | Task
    )
    def preparation_action(
        task_id: UUID, action: str, request: VersionedAction
    ) -> PreparationRecord | Task:
        workflow = container.preparation
        actions = {
            "inspect": workflow.inspect,
            "propose-mapping": workflow.propose_mapping,
            "confirm-mapping": workflow.confirm_mapping,
            "analyze-quality": workflow.analyze_quality,
            "propose-plan": workflow.propose_plan,
            "confirm-plan": workflow.confirm_plan,
            "generate-overview": workflow.generate_overview,
            "review": workflow.review,
            "validate-review": workflow.validate_review,
            "confirm-task": workflow.confirm_task,
        }
        if action == "apply":
            if request.action_id is None:
                raise HTTPException(status_code=422, detail="action_id required")
            return workflow.apply(task_id, request.expected_version, request.action_id)
        if action == "continue":
            if workflow.get(task_id).version != request.expected_version:
                raise VersionConflict("stale Preparation draft")
            return workflow.continue_to_training(task_id)
        handler = actions.get(action)
        if handler is None:
            raise HTTPException(status_code=404, detail="unknown Preparation action")
        return handler(task_id, request.expected_version)

    @app.post(
        "/tasks/{task_id}/preparation/assistant/propose-{kind}",
        response_model=PreparationRecord,
    )
    async def assistant_propose(
        task_id: UUID,
        kind: Literal["mapping", "plan", "task"],
        request: AssistantProposalInput,
    ) -> PreparationRecord:
        if container.preparation_agent is None:
            raise HTTPException(
                status_code=503, detail="Preparation Assistant runtime is not configured"
            )
        if container.preparation.get(task_id).version != request.expected_version:
            raise VersionConflict("stale Preparation draft")
        return await container.preparation_agent.propose(task_id, kind, user_text=request.user_text)

    @app.put("/tasks/{task_id}/preparation/mapping", response_model=PreparationRecord)
    def edit_mapping(task_id: UUID, request: MappingEdit) -> PreparationRecord:
        return container.preparation.update_mapping(
            task_id, request.expected_version, request.draft
        )

    @app.put("/tasks/{task_id}/preparation/plan", response_model=PreparationRecord)
    def edit_plan(task_id: UUID, request: PlanEdit) -> PreparationRecord:
        return container.preparation.update_plan(task_id, request.expected_version, request.draft)

    @app.put("/tasks/{task_id}/preparation/task", response_model=PreparationRecord)
    def edit_forecast(task_id: UUID, request: ForecastEdit) -> PreparationRecord:
        return container.preparation.update_task_draft(
            task_id, request.expected_version, request.draft
        )

    @app.post("/tasks/{task_id}/preparation/chat", response_model=PreparationRecord)
    async def chat(task_id: UUID, request: ChatInput) -> PreparationRecord:
        task = get_task_or_404(task_id)
        if task.stage not in {Stage.NEW, Stage.PREPARATION}:
            raise InvalidTransition("Preparation chat is not active in the current stage")
        record = container.preparation.get(task_id)
        if record.version != request.expected_version:
            raise VersionConflict("stale Preparation draft")
        if container.preparation_agent is None:
            raise HTTPException(
                status_code=503, detail="Preparation agent runtime is not configured"
            )
        user = Message(
            task_id=task_id,
            source_role=Actor.USER,
            target_role=Actor.TASK_MANAGER,
            topic=Topic.CHAT,
            kind=MessageKind.USER,
            message_type="preparation_chat",
            payload={"text": request.text[:1000]},
        )
        container.messages.append(user)
        from interactive_forecasting.orchestration.preparation_conversation import (
            PreparationConversation,
        )

        try:
            return await PreparationConversation(container.preparation_agent).run(user, record)
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=503, detail="Preparation provider call failed") from exc

    def optimization_state(task_id: UUID) -> OptimizationStateResponse:
        get_task_or_404(task_id)
        assert container.optimization is not None and container.visuals is not None
        session = container.optimization.get(task_id)
        run = container.optimization.runs.get(session.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="search run not found")
        return OptimizationStateResponse(
            session=session,
            summary=summarize_optimization(run),
            visualizations=container.visuals.overview(run, session),
        )

    @app.get("/tasks/{task_id}/optimization", response_model=OptimizationStateResponse)
    def get_optimization(task_id: UUID) -> OptimizationStateResponse:
        return optimization_state(task_id)

    @app.get("/tasks/{task_id}/optimization/setup")
    def optimization_setup(task_id: UUID) -> dict[str, object]:
        get_task_or_404(task_id)
        record = container.preparation.get(task_id)
        if not record.frozen:
            raise ValueError("frozen Preparation is required")
        return {
            "definition": record.definition,
            "protocol": record.protocol,
            "snapshot": record.prepared.dataset if record.prepared else None,
            "effective_search_space": record.effective_search_space,
            "required_explicit_inputs": [
                "mode",
                "spec_id",
                "backend",
                "schedule",
                "templates",
                "metric",
                "experiment_seed",
            ],
        }

    @app.post("/tasks/{task_id}/optimization/runs", response_model=OptimizationStateResponse)
    def start_optimization(task_id: UUID, request: RunSetupInput) -> OptimizationStateResponse:
        assert container.optimization is not None
        container.optimization.create(
            task_id,
            RunSetup(
                mode=request.mode,
                spec_id=request.spec_id,
                backend=request.backend,
                schedule=request.schedule,
                templates=request.templates,
                metric=request.metric,
                experiment_seed=request.experiment_seed,
                pause_at_boundary=request.pause_at_boundary,
            ),
        )
        return optimization_state(task_id)

    @app.post("/tasks/{task_id}/optimization/advance", response_model=OptimizationStateResponse)
    async def advance_optimization(
        task_id: UUID, request: VersionedAction
    ) -> OptimizationStateResponse:
        assert container.optimization is not None
        await container.optimization.advance(task_id, request.expected_version)
        return optimization_state(task_id)

    @app.post("/tasks/{task_id}/optimization/resume", response_model=OptimizationStateResponse)
    async def resume_optimization(
        task_id: UUID, request: VersionedAction
    ) -> OptimizationStateResponse:
        assert container.optimization is not None
        await container.optimization.resume(task_id, request.expected_version)
        return optimization_state(task_id)

    @app.post("/tasks/{task_id}/optimization/cancel", response_model=OptimizationStateResponse)
    def cancel_optimization(task_id: UUID, request: VersionedAction) -> OptimizationStateResponse:
        assert container.optimization is not None
        container.optimization.cancel(task_id, request.expected_version)
        return optimization_state(task_id)

    @app.put("/tasks/{task_id}/optimization/guidance", response_model=OptimizationStateResponse)
    def save_optimization_guidance(
        task_id: UUID, request: GuidanceDraftInput
    ) -> OptimizationStateResponse:
        assert container.optimization is not None
        container.optimization.save_draft(task_id, request.expected_version, request.commands)
        return optimization_state(task_id)

    @app.post(
        "/tasks/{task_id}/optimization/guidance/confirm", response_model=OptimizationStateResponse
    )
    def confirm_optimization_guidance(
        task_id: UUID, request: VersionedAction
    ) -> OptimizationStateResponse:
        assert container.optimization is not None
        container.optimization.confirm_draft(task_id, request.expected_version)
        return optimization_state(task_id)

    @app.post(
        "/tasks/{task_id}/optimization/guidance/clear", response_model=OptimizationStateResponse
    )
    def clear_optimization_guidance(
        task_id: UUID, request: VersionedAction
    ) -> OptimizationStateResponse:
        assert container.optimization is not None
        container.optimization.clear_draft(task_id, request.expected_version)
        return optimization_state(task_id)

    @app.post("/tasks/{task_id}/optimization/chat", response_model=OptimizationStateResponse)
    async def optimization_chat(task_id: UUID, request: ChatInput) -> OptimizationStateResponse:
        assert container.optimization is not None
        await container.optimization.chat(task_id, request.expected_version, request.text)
        return optimization_state(task_id)

    @app.get("/tasks/{task_id}/optimization/trials/{trial_id}", response_model=TrialDetail)
    def optimization_trial(task_id: UUID, trial_id: UUID) -> TrialDetail:
        get_task_or_404(task_id)
        assert container.optimization is not None and container.visuals is not None
        session = container.optimization.get(task_id)
        run = container.optimization.runs.get(session.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="search run not found")
        return container.visuals.trial_detail(run, trial_id)

    @app.get("/tasks/{task_id}/deployment/readiness", response_model=DeploymentReadiness)
    def deployment_readiness(task_id: UUID) -> DeploymentReadiness:
        assert container.deployment is not None
        return container.deployment.readiness(task_id)

    @app.post("/tasks/{task_id}/deployment/start", response_model=DeploymentSession)
    def start_deployment(
        task_id: UUID, new_session: bool = False, request: DeploymentAction | None = None
    ) -> DeploymentSession:
        assert container.deployment is not None
        return container.deployment.start(
            task_id,
            new_session=new_session,
            session_id=request.session_id if request else None,
            expected_version=request.expected_version if request else None,
        )

    @app.post("/tasks/{task_id}/deployment/upload", response_model=DeploymentSession)
    async def upload_deployment(
        task_id: UUID,
        session_id: Annotated[UUID, Form()],
        expected_version: Annotated[int, Form()],
        file: Annotated[UploadFile, File()],
    ) -> DeploymentSession:
        assert container.deployment is not None
        content = await file.read(32 * 1024 * 1024 + 1)
        return container.deployment.upload(
            task_id,
            expected_version,
            content,
            Path(file.filename or "").suffix.lower(),
            session_id=session_id,
        )

    @app.post("/tasks/{task_id}/deployment/validate", response_model=DeploymentSession)
    def validate_deployment(task_id: UUID, request: DeploymentValidationInput) -> DeploymentSession:
        assert container.deployment is not None
        return container.deployment.validate(
            task_id,
            request.expected_version,
            request.future_auxiliaries,
            session_id=request.session_id,
        )

    @app.post("/tasks/{task_id}/deployment/generate", response_model=ForecastResponse)
    def generate_deployment(task_id: UUID, request: DeploymentAction) -> ForecastResponse:
        assert container.deployment is not None
        forecast, original = container.deployment.generate(
            task_id, request.expected_version, session_id=request.session_id
        )
        return ForecastResponse(forecast=forecast, original=original)

    @app.get("/tasks/{task_id}/deployment/forecasts/{forecast_id}", response_model=ForecastResponse)
    def get_deployment_forecast(task_id: UUID, forecast_id: UUID) -> ForecastResponse:
        assert container.deployment is not None
        forecast, original = container.deployment.get_forecast(task_id, forecast_id)
        return ForecastResponse(forecast=forecast, original=original)

    @app.get("/tasks/{task_id}/deployment/state", response_model=DeploymentSession)
    def deployment_state(task_id: UUID) -> DeploymentSession:
        assert container.deployment is not None
        session = container.deployment.deployments.get(task_id)
        if session is None:
            raise HTTPException(status_code=404, detail="deployment session not found")
        return session

    @app.get("/tasks/{task_id}/deployment/sessions", response_model=list[DeploymentSession])
    def deployment_sessions(task_id: UUID) -> list[DeploymentSession]:
        assert container.deployment is not None
        return container.deployment.list_sessions(task_id)

    @app.get("/tasks/{task_id}/deployment/sessions/{session_id}", response_model=DeploymentSession)
    def deployment_session(task_id: UUID, session_id: UUID) -> DeploymentSession:
        assert container.deployment is not None
        session = container.deployment.deployments.get_session(session_id)
        if session is None or session.task_id != task_id:
            raise HTTPException(status_code=404, detail="deployment session not found")
        return session

    @app.get("/tasks/{task_id}/deployment/forecasts", response_model=list[Forecast])
    def deployment_forecasts(task_id: UUID) -> list[Forecast]:
        assert container.deployment is not None
        return container.deployment.list_forecasts(task_id)

    @app.post(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/references/analyze",
        response_model=ReferenceAnalysis,
    )
    def analyze_deployment(
        task_id: UUID, forecast_id: UUID, request: DeploymentAction, top_k: int = 3
    ) -> ReferenceAnalysis:
        assert container.deployment is not None
        return container.deployment.analyze(
            task_id,
            forecast_id,
            top_k=top_k,
            session_id=request.session_id,
            expected_version=request.expected_version,
        )

    @app.get(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/references",
        response_model=ReferenceAnalysis,
    )
    def deployment_references(task_id: UUID, forecast_id: UUID) -> ReferenceAnalysis:
        assert container.deployment is not None
        return container.deployment.reference(task_id, forecast_id)

    @app.get(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/versions",
        response_model=list[ForecastVersion],
    )
    def deployment_versions(task_id: UUID, forecast_id: UUID) -> list[ForecastVersion]:
        assert container.deployment is not None
        return container.deployment.versions(task_id, forecast_id)

    @app.get(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/versions/{version_id}",
        response_model=ForecastVersion,
    )
    def deployment_version(task_id: UUID, forecast_id: UUID, version_id: UUID) -> ForecastVersion:
        assert container.deployment is not None
        return container.deployment.version(task_id, forecast_id, version_id)

    @app.get(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/adjustments",
        response_model=list[Adjustment],
    )
    def deployment_adjustments(task_id: UUID, forecast_id: UUID) -> list[Adjustment]:
        assert container.deployment is not None
        return container.deployment.adjustments(task_id, forecast_id)

    @app.post(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/adjustments",
        response_model=Adjustment,
    )
    def create_adjustment(
        task_id: UUID, forecast_id: UUID, request: AdjustmentDraftInput
    ) -> Adjustment:
        assert container.deployment is not None
        return container.deployment.create_draft(
            task_id,
            forecast_id,
            request.expected_version,
            request.proposal,
            session_id=request.session_id,
            source="user",
            user_text=request.user_text,
        )

    @app.post(
        "/tasks/{task_id}/deployment/adjustments/{adjustment_id}/validate",
        response_model=Adjustment,
    )
    def validate_adjustment(
        task_id: UUID, adjustment_id: UUID, request: DeploymentAction
    ) -> Adjustment:
        assert container.deployment is not None
        return container.deployment.validate_draft(
            task_id, adjustment_id, request.expected_version, session_id=request.session_id
        )

    @app.post(
        "/tasks/{task_id}/deployment/adjustments/{adjustment_id}/reject", response_model=Adjustment
    )
    def reject_adjustment(
        task_id: UUID, adjustment_id: UUID, request: DeploymentAction
    ) -> Adjustment:
        assert container.deployment is not None
        return container.deployment.reject_draft(
            task_id, adjustment_id, request.expected_version, session_id=request.session_id
        )

    @app.post(
        "/tasks/{task_id}/deployment/adjustments/{adjustment_id}/confirm",
        response_model=ForecastVersion,
    )
    def confirm_adjustment(
        task_id: UUID, adjustment_id: UUID, request: AdjustmentConfirmInput
    ) -> ForecastVersion:
        assert container.deployment is not None
        return container.deployment.confirm_draft(
            task_id,
            adjustment_id,
            request.expected_version,
            request.confirmation_id,
            session_id=request.session_id,
        )

    @app.post("/tasks/{task_id}/deployment/complete", response_model=DeploymentSession)
    def complete_deployment(task_id: UUID, request: DeploymentAction) -> DeploymentSession:
        assert container.deployment is not None
        return container.deployment.complete(
            task_id, request.expected_version, session_id=request.session_id
        )

    @app.post("/tasks/{task_id}/deployment/chat", response_model=DeploymentChatResult)
    async def deployment_chat(task_id: UUID, request: DeploymentChatInput) -> DeploymentChatResult:
        if container.deployment_agent is None:
            raise HTTPException(status_code=503, detail="deployment agent runtime unavailable")
        return await container.deployment_agent.chat(
            task_id, request.expected_version, request.text, session_id=request.session_id
        )

    @app.get(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/sensitivity/variables",
        response_model=list[str],
    )
    def deployment_sensitivity_variables(task_id: UUID, forecast_id: UUID) -> list[str]:
        assert container.deployment is not None
        return list(container.deployment.sensitivity_variables(task_id, forecast_id))

    @app.post(
        "/tasks/{task_id}/deployment/forecasts/{forecast_id}/sensitivity",
        response_model=SensitivityResult,
    )
    def deployment_sensitivity(
        task_id: UUID, forecast_id: UUID, request: SensitivityRequest
    ) -> SensitivityResult:
        assert container.deployment is not None
        return container.deployment.sensitivity(task_id, forecast_id, request)

    @app.exception_handler(VersionConflict)
    def version_error(_request: object, exc: VersionConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidTransition)
    def transition_error(_request: object, exc: InvalidTransition) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def validation_error(_request: object, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    web_root = Path(__file__).resolve().parents[3] / "web"
    app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/")
    def workspace() -> FileResponse:
        return FileResponse(web_root / "index.html")

    return app


app = create_app()
