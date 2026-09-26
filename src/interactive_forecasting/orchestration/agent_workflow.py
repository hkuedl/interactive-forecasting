"""Application-owned bounded agent turns, authorization, routing and audit records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from interactive_forecasting.agents.contracts import (
    AgentAction,
    AgentContext,
    AgentDecision,
    ApprovedExecution,
    ExecutionRequest,
    ModelDeveloperContext,
    ModelManagerContext,
    PresentToUser,
    RequestDeployment,
    RequestExecution,
    RequestPreparation,
    RouteSpecialist,
    ToolResult,
    UnauthorizedAction,
    authorize_decision,
    validate_context,
)
from interactive_forecasting.agents.runtime import (
    AgentRequest,
    AgentRuntime,
    AgentRuntimeError,
    RuntimeMetadata,
)
from interactive_forecasting.domain.models import Event, LLMCall, Message, Task
from interactive_forecasting.domain.types import Actor, MessageKind, Topic
from interactive_forecasting.orchestration.messages import MessageBus, role_eligible
from interactive_forecasting.orchestration.state_machine import VersionConflict
from interactive_forecasting.storage.interfaces import (
    LLMCallRepositoryPort,
    MessageRepositoryPort,
    TaskRepositoryPort,
)


class ActionExecutor(Protocol):
    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult: ...


class ExecutionApprover(Protocol):
    def approve(
        self, request: ExecutionRequest, context: ModelManagerContext
    ) -> ApprovedExecution: ...


_ROLE_TOPIC: dict[Actor, Topic] = {
    Actor.TASK_MANAGER: Topic.CHAT,
    Actor.PREPARATION_ASSISTANT: Topic.PREPARE,
    Actor.MODEL_MANAGER: Topic.OPTIMIZE,
    Actor.MODEL_DEVELOPER: Topic.TRAIN,
    Actor.DEPLOYMENT_OPERATOR: Topic.DEPLOY,
}


@dataclass(frozen=True)
class AgentTurn:
    decision: AgentDecision
    invocation_message_id: UUID
    response_message_id: UUID
    action_message_id: UUID | None
    result_message_id: UUID | None
    llm_call_id: UUID
    action_status: str | None


class AgentCoordinator:
    """One invocation at a time; SDK sessions never drive workflow progression."""

    def __init__(
        self,
        runtime: AgentRuntime,
        bus: MessageBus,
        tasks: TaskRepositoryPort,
        messages: MessageRepositoryPort,
        calls: LLMCallRepositoryPort,
        executor: ActionExecutor,
        *,
        approver: ExecutionApprover | None = None,
    ):
        self.runtime = runtime
        self.bus = bus
        self.tasks = tasks
        self.messages = messages
        self.calls = calls
        self.executor = executor
        self.approver = approver

    async def invoke(
        self, request: AgentRequest, *, parent_message_id: UUID | None = None
    ) -> AgentTurn:
        task = self.tasks.get(request.task_id)
        if task is None:
            raise ValueError("agent task does not exist")
        try:
            self._authorize_invocation(task, request, parent_message_id)
        except (ValueError, VersionConflict) as exc:
            self._failure_event(
                task.task_id,
                request.correlation_id,
                None,
                "invocation_rejected",
                str(exc),
            )
            raise
        topic = _ROLE_TOPIC[request.role]
        invocation = Message(
            task_id=task.task_id,
            source_role=Actor.SYSTEM,
            target_role=request.role,
            topic=topic,
            kind=MessageKind.COMMAND,
            message_type="agent.invoke",
            payload={
                "role": request.role.value,
                "prompt_version": request.prompt_version,
                "context_type": type(request.context).__name__,
            },
            correlation_id=request.correlation_id,
            parent_message_id=parent_message_id,
        )
        self.bus.publish(invocation)
        try:
            runtime_result = await self.runtime.run(request, AgentDecision)
            decision = AgentDecision.model_validate(runtime_result.output)
        except Exception as exc:
            meta = (
                exc.metadata
                if isinstance(exc, AgentRuntimeError)
                else RuntimeMetadata(
                    runtime_name=type(self.runtime).__name__,
                    runtime_version=None,
                    provider="unknown",
                    model=None,
                    status="invalid_output" if isinstance(exc, ValidationError) else "failed",
                )
            )
            self._record_call(request, invocation.message_id, meta, error=str(exc))
            self._failure_event(
                task.task_id,
                request.correlation_id,
                invocation.message_id,
                "agent_failed",
                type(exc).__name__,
            )
            raise
        call = self._record_call(request, invocation.message_id, runtime_result.metadata)
        try:
            authorize_decision(request.role, cast(AgentContext, request.context), decision)
        except (UnauthorizedAction, ValueError) as exc:
            self._failure_event(
                task.task_id,
                request.correlation_id,
                invocation.message_id,
                "action_rejected",
                str(exc),
            )
            raise
        response = Message(
            task_id=task.task_id,
            source_role=request.role,
            target_role=(
                Actor.MODEL_MANAGER
                if request.role == Actor.MODEL_DEVELOPER
                else Actor.TASK_MANAGER
                if request.role != Actor.TASK_MANAGER
                else Actor.SYSTEM
            ),
            topic=topic,
            kind=MessageKind.RESULT,
            message_type="agent.decision",
            payload=decision.model_dump(mode="json"),
            correlation_id=request.correlation_id,
            parent_message_id=invocation.message_id,
        )
        self.bus.publish(response)
        action = decision.action
        if action is None:
            return AgentTurn(
                decision, invocation.message_id, response.message_id, None, None, call.call_id, None
            )
        return await self._dispatch_action(
            task, request, decision, invocation.message_id, response, call.call_id
        )

    def _authorize_invocation(
        self, task: Task, request: AgentRequest, parent_message_id: UUID | None
    ) -> None:
        if (
            request.expected_task_version is not None
            and task.version != request.expected_task_version
        ):
            raise VersionConflict("stale agent workflow-state request")
        if not role_eligible(request.role, task.stage):
            raise UnauthorizedAction("role is not eligible in current stage")
        if not isinstance(request.context, tuple(_CONTEXT_CLASSES)):
            raise ValueError("agent invocation requires a typed role context")
        context = cast(AgentContext, request.context)
        validate_context(request.role, context)
        if context.task.task_id != task.task_id:
            raise ValueError("agent context belongs to another task")
        if context.task.version != task.version:
            raise VersionConflict("agent context has stale task version")
        if parent_message_id is None:
            if request.role != Actor.TASK_MANAGER:
                raise UnauthorizedAction("specialists require an orchestrated parent command")
            return
        parent = self.messages.get(parent_message_id)
        if (
            parent is None
            or parent.task_id != task.task_id
            or parent.correlation_id != request.correlation_id
            or parent.target_role != request.role
            or parent.kind not in {MessageKind.COMMAND, MessageKind.USER}
        ):
            raise UnauthorizedAction("invalid specialist invocation parent")
        if request.role == Actor.MODEL_DEVELOPER:
            assert isinstance(context, ModelDeveloperContext)
            if parent.source_role != Actor.SYSTEM or parent.message_type != "execution.approved":
                raise UnauthorizedAction(
                    "developer requires orchestrator-issued execution approval"
                )
            if parent.payload.get("approved_request") != context.approved_request.model_dump(
                mode="json"
            ):
                raise UnauthorizedAction("developer context differs from approved execution")
        elif request.role != Actor.TASK_MANAGER and parent.source_role != Actor.TASK_MANAGER:
            raise UnauthorizedAction("specialists require Task Manager routing")

    async def _dispatch_action(
        self,
        task: Task,
        request: AgentRequest,
        decision: AgentDecision,
        invocation_id: UUID,
        response: Message,
        call_id: UUID,
    ) -> AgentTurn:
        action = decision.action
        assert action is not None
        if isinstance(action, PresentToUser):
            visible = Message(
                task_id=task.task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=Actor.USER,
                topic=Topic.CHAT,
                kind=MessageKind.EVENT,
                message_type="chat.reply",
                payload={"text": action.text},
                correlation_id=request.correlation_id,
                parent_message_id=response.message_id,
            )
            self.bus.publish(visible)
            return AgentTurn(
                decision,
                invocation_id,
                response.message_id,
                visible.message_id,
                None,
                call_id,
                "completed",
            )
        if isinstance(action, RouteSpecialist):
            target = action.target_role
            if not role_eligible(target, task.stage):
                self._failure_event(
                    task.task_id,
                    request.correlation_id,
                    response.message_id,
                    "action_rejected",
                    "specialist is unavailable in current stage",
                )
                raise UnauthorizedAction("specialist is unavailable in current stage")
            routed = Message(
                task_id=task.task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=target,
                topic=_ROLE_TOPIC[target],
                kind=MessageKind.COMMAND,
                message_type="agent.route",
                payload={"instruction": action.instruction},
                correlation_id=request.correlation_id,
                parent_message_id=response.message_id,
            )
            self.bus.publish(routed)
            return AgentTurn(
                decision,
                invocation_id,
                response.message_id,
                routed.message_id,
                None,
                call_id,
                "pending",
            )
        if isinstance(action, RequestExecution):
            if self.approver is None:
                self._failure_event(
                    task.task_id,
                    request.correlation_id,
                    response.message_id,
                    "action_rejected",
                    "execution approval service is not configured",
                )
                raise UnauthorizedAction("execution approval service is not configured")
            assert isinstance(request.context, ModelManagerContext)
            try:
                approved = self.approver.approve(action.request, request.context)
            except Exception as exc:
                self._failure_event(
                    task.task_id,
                    request.correlation_id,
                    response.message_id,
                    "action_rejected",
                    f"{type(exc).__name__}: {exc}",
                )
                raise
            if any(
                getattr(approved, key) != getattr(action.request, key)
                for key in ("run_id", "operation", "candidate", "expected_run_version")
            ):
                self._failure_event(
                    task.task_id,
                    request.correlation_id,
                    response.message_id,
                    "action_rejected",
                    "approval changed the requested execution",
                )
                raise UnauthorizedAction("approval changed the requested execution")
            routed = Message(
                task_id=task.task_id,
                run_id=approved.run_id,
                source_role=Actor.SYSTEM,
                target_role=Actor.MODEL_DEVELOPER,
                topic=Topic.TRAIN,
                kind=MessageKind.COMMAND,
                message_type="execution.approved",
                payload={"approved_request": approved.model_dump(mode="json")},
                correlation_id=request.correlation_id,
                parent_message_id=response.message_id,
            )
            self.bus.publish(routed)
            return AgentTurn(
                decision,
                invocation_id,
                response.message_id,
                routed.message_id,
                None,
                call_id,
                "pending",
            )
        command = Message(
            task_id=task.task_id,
            source_role=request.role,
            target_role=Actor.SERVICE,
            topic=_ROLE_TOPIC[request.role],
            kind=MessageKind.COMMAND,
            message_type=f"action.{action.kind}",
            payload=action.model_dump(mode="json"),
            correlation_id=request.correlation_id,
            parent_message_id=response.message_id,
        )
        self.bus.publish(command)
        try:
            result = await self.executor.execute(action, cast(AgentContext, request.context))
            result = ToolResult.model_validate(result)
        except Exception as exc:
            result = ToolResult(status="failed", error=f"{type(exc).__name__}: {exc}")
        result_message = Message(
            task_id=task.task_id,
            source_role=Actor.SERVICE,
            target_role=request.role,
            topic=_ROLE_TOPIC[request.role],
            kind=MessageKind.RESULT,
            message_type="action.result",
            payload=result.model_dump(mode="json"),
            correlation_id=request.correlation_id,
            parent_message_id=command.message_id,
        )
        self.bus.publish(result_message)
        return AgentTurn(
            decision,
            invocation_id,
            response.message_id,
            command.message_id,
            result_message.message_id,
            call_id,
            result.status,
        )

    async def retry_read_only_action(
        self,
        command_id: UUID,
        context: AgentContext,
        *,
        expected_task_version: int,
    ) -> ToolResult:
        """Retry only a previously failed read-only service command.

        Mutating search/deployment actions require a new orchestrator decision instead.
        """
        command = self.messages.get(command_id)
        if (
            command is None
            or command.kind != MessageKind.COMMAND
            or command.target_role != Actor.SERVICE
            or command.source_role not in _ROLE_TOPIC
        ):
            raise UnauthorizedAction("retry requires a persisted service command")
        task = self.tasks.get(command.task_id)
        if task is None or task.version != expected_task_version:
            raise VersionConflict("stale retry request")
        if not role_eligible(command.source_role, task.stage):
            raise UnauthorizedAction("action role is no longer eligible in current stage")
        validate_context(command.source_role, context)
        if context.task.task_id != task.task_id or context.task.version != task.version:
            raise VersionConflict("retry context does not match current task")
        action: AgentAction = TypeAdapter(AgentAction).validate_python(command.payload)
        allowed = (
            isinstance(action, RequestPreparation)
            and action.operation == "inspect_summary"
            or isinstance(action, RequestDeployment)
            and action.operation == "inspect_status"
        )
        if not allowed:
            raise UnauthorizedAction("only failed read-only actions can be retried")
        authorize_decision(
            command.source_role,
            context,
            AgentDecision(explanation="read-only retry", action=action),
        )
        previous = [
            message
            for message in self.messages.list_for_task(task.task_id)
            if message.parent_message_id == command_id
            and message.message_type.startswith("action.result")
        ]
        if not previous or previous[-1].payload.get("status") != "failed":
            raise UnauthorizedAction("retry requires a previously failed result")
        try:
            result = ToolResult.model_validate(await self.executor.execute(action, context))
        except Exception as exc:
            result = ToolResult(status="failed", error=f"{type(exc).__name__}: {exc}")
        self.bus.publish(
            Message(
                task_id=task.task_id,
                source_role=Actor.SERVICE,
                target_role=command.source_role,
                topic=command.topic,
                kind=MessageKind.RESULT,
                message_type="action.result.retry",
                payload=result.model_dump(mode="json"),
                correlation_id=command.correlation_id,
                parent_message_id=command_id,
            )
        )
        return result

    def _record_call(
        self,
        request: AgentRequest,
        message_id: UUID,
        meta: RuntimeMetadata,
        *,
        error: str | None = None,
    ) -> LLMCall:
        call = LLMCall(
            task_id=request.task_id,
            message_id=message_id,
            correlation_id=request.correlation_id,
            runtime_name=meta.runtime_name,
            runtime_version=meta.runtime_version,
            provider=meta.provider,
            model=meta.model,
            agent_role=request.role,
            request_id=meta.request_id,
            trace_id=meta.trace_id,
            input_tokens=meta.input_tokens,
            output_tokens=meta.output_tokens,
            cached_input_tokens=meta.cached_input_tokens,
            latency_ms=meta.latency_ms,
            status=meta.status,
            error=error,
            prompt_version=request.prompt_version,
        )
        return self.calls.create(call)

    def _failure_event(
        self,
        task_id: UUID,
        correlation_id: UUID,
        parent_message_id: UUID | None,
        code: str,
        detail: str,
    ) -> None:
        self.bus.publish(
            Event(
                task_id=task_id,
                source_role=Actor.SYSTEM,
                topic=Topic.SYSTEM,
                message_type=code,
                payload={"detail": detail},
                outcome="failed",
                correlation_id=correlation_id,
                parent_message_id=parent_message_id,
            )
        )


# A role's context is deliberately an exact schema, not a loose mapping.
from interactive_forecasting.agents.contracts import (  # noqa: E402
    DeploymentContext,
    PreparationContext,
    TaskManagerContext,
)

_CONTEXT_CLASSES = (
    TaskManagerContext,
    PreparationContext,
    ModelManagerContext,
    ModelDeveloperContext,
    DeploymentContext,
)
