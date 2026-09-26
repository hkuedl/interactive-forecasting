"""Task Manager bridge to bounded Deployment Operator turns and draft-only actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from interactive_forecasting.agents.contracts import (
    AgentAction,
    AgentContext,
    DeploymentContext,
    DeploymentForecastPoint,
    DeploymentReferenceSummary,
    PresentToUser,
    ProposeAdjustment,
    RequestDeployment,
    RequestSensitivity,
    RouteSpecialist,
    TaskManagerContext,
    ToolResult,
)
from interactive_forecasting.agents.deployment_policy import DEPLOYMENT_PROMPT_VERSION
from interactive_forecasting.agents.runtime import AgentRequest, AgentRuntime
from interactive_forecasting.domain.forecasting import PointForecast, QuantileForecast
from interactive_forecasting.domain.models import (
    Adjustment,
    Message,
    SensitivityRequest,
    SensitivityResult,
)
from interactive_forecasting.domain.types import Actor, MessageKind, Stage, Topic
from interactive_forecasting.orchestration.agent_workflow import AgentCoordinator
from interactive_forecasting.orchestration.deployment import DeploymentWorkflow
from interactive_forecasting.orchestration.messages import MessageBus
from interactive_forecasting.orchestration.state_machine import VersionConflict
from interactive_forecasting.storage.sql import (
    LLMCallRepository,
    MessageRepository,
    TaskRepository,
)


@dataclass(frozen=True)
class DeploymentChatResult:
    text: str
    draft: Adjustment | None
    session_version: int
    sensitivity: SensitivityResult | None = None


class DeploymentActionExecutor:
    def __init__(self, workflow: DeploymentWorkflow):
        self.workflow = workflow

    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult:
        if isinstance(action, ProposeAdjustment) and isinstance(context, DeploymentContext):
            if context.session_id is None:
                raise ValueError("deployment context requires a session identity")
            draft = self.workflow.create_draft(
                action.task_id,
                action.forecast_id,
                action.expected_session_version,
                action.proposal,
                session_id=context.session_id,
                source="deployment_operator_proposal",
                user_text=context.user_request_text,
            )
            session = self.workflow.deployments.get(action.task_id)
            assert session is not None
            return ToolResult(
                status="completed",
                data={
                    "adjustment_id": str(draft.adjustment_id),
                    "session_version": session.version,
                },
            )
        if isinstance(action, RequestSensitivity) and isinstance(context, DeploymentContext):
            result = self.workflow.sensitivity(
                action.task_id,
                action.forecast_id,
                SensitivityRequest(
                    base_version_id=action.base_version_id,
                    variable=action.variable,
                    perturbation_type=action.perturbation_type,
                    value=action.value,
                ),
            )
            return ToolResult(
                status="completed",
                data={"sensitivity": json.dumps(result.model_dump(mode="json"))},
            )
        if isinstance(action, RequestDeployment) and isinstance(context, DeploymentContext):
            return ToolResult(status="completed", data={"forecast_id": str(context.forecast_id)})
        raise ValueError("deployment agent cannot execute this action")


class DeploymentAgentBridge:
    def __init__(
        self,
        runtime: AgentRuntime,
        workflow: DeploymentWorkflow,
        tasks: TaskRepository,
        messages: MessageRepository,
        calls: LLMCallRepository,
    ) -> None:
        self.runtime = runtime
        self.workflow = workflow
        self.tasks = tasks
        self.messages = messages
        self.calls = calls

    def _coordinator(self) -> AgentCoordinator:
        return AgentCoordinator(
            self.runtime,
            MessageBus(self.messages, self.tasks),
            self.tasks,
            self.messages,
            self.calls,
            DeploymentActionExecutor(self.workflow),
        )

    def _context(self, task_id: UUID, user_text: str) -> DeploymentContext:
        session = self.workflow.deployments.get(task_id)
        task = self.tasks.get(task_id)
        if (
            task is None
            or task.stage != Stage.DEPLOYMENT
            or session is None
            or session.forecast_id is None
        ):
            raise ValueError("deployment forecast is unavailable")
        forecast, original = self.workflow.get_forecast(task_id, session.forecast_id)
        current = (
            self.workflow.deployments.version(session.current_version_id)
            if session.current_version_id
            else original
        )
        if current is None:
            raise ValueError("current forecast version is missing")
        analysis = self.workflow.deployments.reference(forecast.forecast_id)
        reference = (
            DeploymentReferenceSummary(
                d_minus_1=(
                    f"{analysis.d_minus_1.date}: {analysis.d_minus_1.reason or 'available'}"
                ),
                d_minus_7=(
                    f"{analysis.d_minus_7.date}: {analysis.d_minus_7.reason or 'available'}"
                ),
                d_minus_365=(
                    f"{analysis.d_minus_365.date}: {analysis.d_minus_365.reason or 'available'}"
                ),
                weather_dates_and_distances=tuple(
                    f"{item.date}: {item.distance:.6g}" for item in analysis.weather_analogs[:3]
                ),
                weather_unavailable_reason=analysis.weather_unavailable_reason,
            )
            if analysis
            else None
        )
        prediction = current.prediction
        values: tuple[tuple[float, ...], ...]
        if isinstance(prediction, PointForecast):
            values = tuple((value,) for value in prediction.values)
        else:
            assert isinstance(prediction, QuantileForecast)
            values = prediction.values
        points = tuple(
            DeploymentForecastPoint(timestamp=key.target.isoformat(), values=tuple(row))
            for key, row in zip(current.prediction.keys[:48], values[:48], strict=True)
        )
        prior = tuple(
            f"{item.adjustment_type}: {item.adjustment_id}"
            for item in self.workflow.deployments.adjustments(forecast.forecast_id)
            if item.status == "applied"
        )[-12:]
        return DeploymentContext(
            task=task,
            selected_model_artifact=forecast.model_artifact,
            forecast_id=forecast.forecast_id,
            session_id=session.session_id,
            session_version=session.version,
            current_version_id=current.version_id,
            original_version_id=original.version_id,
            sensitivity_variables=self.workflow.sensitivity_variables(
                task_id, forecast.forecast_id
            ),
            forecast_points=points,
            reference_summary=reference,
            previous_adjustments=prior,
            user_request_text=user_text[:1000],
        )

    async def chat(
        self, task_id: UUID, expected_version: int, text: str, *, session_id: UUID
    ) -> DeploymentChatResult:
        if not text.strip() or len(text) > 1000:
            raise ValueError("deployment chat text must contain 1–1000 characters")
        session = self.workflow.deployments.get(task_id)
        task = self.tasks.get(task_id)
        if session is None or task is None or task.stage != Stage.DEPLOYMENT:
            raise ValueError("deployment session is unavailable")
        if session.session_id != session_id or session.version != expected_version:
            raise VersionConflict("deployment chat session identity or version mismatch")
        if session.forecast_id is None:
            raise ValueError("generate a forecast before deployment conversation")
        context = self._context(task_id, text)
        correlation = uuid4()
        bus = MessageBus(self.messages, self.tasks)
        user = Message(
            task_id=task_id,
            source_role=Actor.USER,
            target_role=Actor.TASK_MANAGER,
            topic=Topic.CHAT,
            kind=MessageKind.USER,
            message_type="deployment.chat",
            payload={"text": text},
            correlation_id=correlation,
        )
        bus.publish(user)
        coordinator = self._coordinator()
        manager = await coordinator.invoke(
            AgentRequest(
                role=Actor.TASK_MANAGER,
                task_id=task_id,
                correlation_id=correlation,
                prompt="Handle the user's deployment question or route it to Deployment Operator.",
                context=TaskManagerContext(task=task, user_text=text),
                expected_task_version=task.version,
                prompt_version="5b-task-manager-v1",
            ),
            parent_message_id=user.message_id,
        )
        if isinstance(manager.decision.action, PresentToUser):
            latest = self.workflow.deployments.get(task_id)
            assert latest is not None
            return DeploymentChatResult(manager.decision.action.text, None, latest.version)
        if not isinstance(manager.decision.action, RouteSpecialist) or (
            manager.decision.action.target_role != Actor.DEPLOYMENT_OPERATOR
        ):
            raise ValueError("Task Manager did not route the deployment request")
        assert manager.action_message_id is not None
        operator = await coordinator.invoke(
            AgentRequest(
                role=Actor.DEPLOYMENT_OPERATOR,
                task_id=task_id,
                correlation_id=correlation,
                prompt=manager.decision.action.instruction,
                context=context,
                expected_task_version=task.version,
                prompt_version=DEPLOYMENT_PROMPT_VERSION,
            ),
            parent_message_id=manager.action_message_id,
        )
        draft = None
        sensitivity = None
        if isinstance(operator.decision.action, RequestSensitivity):
            if operator.action_status != "completed" or operator.result_message_id is None:
                raise ValueError("Deployment Operator sensitivity request failed")
            result_message = self.messages.get(operator.result_message_id)
            if result_message is None:
                raise ValueError("Deployment Operator sensitivity result is missing")
            encoded = result_message.payload.get("data", {}).get("sensitivity")
            if not isinstance(encoded, str):
                raise ValueError("Deployment Operator sensitivity result is invalid")
            sensitivity = SensitivityResult.model_validate_json(encoded)
            reply = (
                f"Read-only sensitivity for {sensitivity.variable}: "
                f"{sensitivity.perturbation_type} {sensitivity.value:g}. "
                "The numerical comparison is attached; no forecast version changed."
            )
        elif isinstance(operator.decision.action, ProposeAdjustment):
            if operator.action_status != "completed":
                raise ValueError("Deployment Operator proposal was not accepted")
            latest_session = self.workflow.deployments.get(task_id)
            if latest_session is None or latest_session.pending_adjustment_id is None:
                raise ValueError("Deployment Operator draft was not persisted")
            draft = self.workflow.deployments.adjustment(latest_session.pending_adjustment_id)
            assert draft is not None
            reply = (
                f"Draft {draft.adjustment_id} ({draft.adjustment_type}) is ready for review. "
                "No forecast value has changed. Confirm or reject it explicitly."
            )
        else:
            reply = operator.decision.explanation[:1000]
        bus.publish(
            Message(
                task_id=task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=Actor.USER,
                topic=Topic.CHAT,
                kind=MessageKind.EVENT,
                message_type="deployment.chat.reply",
                payload={"text": reply, "draft_id": str(draft.adjustment_id) if draft else None},
                correlation_id=correlation,
                parent_message_id=operator.response_message_id,
            )
        )
        latest = self.workflow.deployments.get(task_id)
        assert latest is not None
        return DeploymentChatResult(reply, draft, latest.version, sensitivity)
