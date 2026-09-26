"""Bounded Preparation Assistant proposals through the 4A coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.agents.contracts import (
    AgentAction,
    AgentContext,
    PreparationContext,
    ProposeForecastTaskDraft,
    ProposeMappingDraft,
    ProposePlanDraft,
    ToolResult,
    UnauthorizedAction,
)
from interactive_forecasting.agents.preparation_policy import PREPARATION_PROMPT_VERSION
from interactive_forecasting.agents.runtime import AgentRequest, AgentRuntime
from interactive_forecasting.domain.models import LLMCall, Message
from interactive_forecasting.domain.preparation import (
    PreparationRecord,
    PreparationStep,
    WorkingSuggestion,
)
from interactive_forecasting.domain.types import Actor, MessageKind, Topic
from interactive_forecasting.orchestration.agent_workflow import AgentCoordinator
from interactive_forecasting.orchestration.messages import MessageBus
from interactive_forecasting.orchestration.preparation import PreparationWorkflow
from interactive_forecasting.orchestration.state_machine import VersionConflict
from interactive_forecasting.services.preparation import confirm_mapping
from interactive_forecasting.services.preparation_working import projection
from interactive_forecasting.storage.sql import LLMCallRepository, MessageRepository, TaskRepository

ProposalKind = Literal["mapping", "plan", "task"]


class PreparationChatReply(BaseModel):
    """Internal specialist answer with an optional validated draft-command suggestion."""

    model_config = ConfigDict(extra="forbid")
    intent: Literal["question", "information", "correction", "confirmation", "other"]
    text: str = Field(min_length=1, max_length=2000)

    command: str | None
    updates: tuple[WorkingSuggestion, ...]

    @model_validator(mode="after")
    def command_is_correction_only(self) -> PreparationChatReply:
        if self.command is not None and self.intent != "correction":
            raise ValueError("only an explicit correction can suggest a draft command")
        if self.command is not None and not self.command.strip():
            raise ValueError("draft command cannot be empty")

        return self


def _chat_context(record: PreparationRecord) -> dict[str, object]:
    """Project recorded decisions and diagnostics, never raw rows or target labels."""
    schema = record.schema_inspection
    return {
        "step": record.step.value,
        "working": projection(record),
        "working_entries": record.working_state.model_dump(mode="json"),
        "schema": None
        if schema is None
        else {
            "row_count": schema.row_count,
            "columns": [
                {
                    "name": col.name,
                    "type": col.physical_dtype,
                    "missing_count": col.missing_count,
                    "datetime_parse_ratio": col.datetime_parse_ratio,
                    "numeric_parse_ratio": col.numeric_parse_ratio,
                    "semantic_hints": col.semantic_hints,
                }
                for col in schema.columns[:40]
            ],
            "plausible_time_columns": schema.plausible_time_columns,
            "interval_candidates": schema.interval_candidates,
        },
        "mapping_draft": record.mapping_draft.model_dump(mode="json")
        if record.mapping_draft
        else None,
        "confirmed_mapping": record.confirmed_mapping.model_dump(mode="json")
        if record.confirmed_mapping
        else None,
        "quality": record.quality.model_dump(mode="json") if record.quality else None,
        "plan_draft": record.plan_draft.model_dump(mode="json") if record.plan_draft else None,
        "plan_confirmed": record.plan_confirmed,
        "capabilities": record.capabilities.model_dump(mode="json")
        if record.capabilities
        else None,
        "task_draft": record.task_draft.model_dump(mode="json") if record.task_draft else None,
    }


def _recommendation_text(record: PreparationRecord, kind: ProposalKind) -> str:
    if kind == "mapping" and record.mapping_draft:
        mapping = record.mapping_draft
        parts = [f"{mapping.timestamp_column} for time", f"{mapping.target_column} for load"]
        if mapping.temperature_column:
            parts.append(f"{mapping.temperature_column} for temperature")
        return f"I suggest {', '.join(parts)}. Please check the mapping before confirming it."
    if kind == "plan" and record.plan_draft:
        plan = record.plan_draft
        reasons = " ".join(plan.rationale[:2])
        return (
            f"I suggest the preparation choices shown in the workspace. {reasons} "
            "Please review them before approving."
        )
    if kind == "task" and record.task_draft:
        task_draft = record.task_draft
        return (
            f"I suggest a {task_draft.output.representation} forecast with lead {task_draft.delta} "
            f"{task_draft.time_unit}, horizon {task_draft.horizon}, "
            f"and {task_draft.objective_id.upper()} "
            "for evaluation. Please review the settings before freezing the task."
        )
    return "I have a recommendation ready in the workspace. Please review it before confirming."


class PreparationActionExecutor:
    """Save typed suggestions as drafts, never as approvals or file operations."""

    def __init__(self, workflow: PreparationWorkflow):
        self.workflow = workflow

    async def execute(self, action: AgentAction, context: AgentContext) -> ToolResult:
        if not isinstance(
            action, (ProposeMappingDraft, ProposePlanDraft, ProposeForecastTaskDraft)
        ) or not isinstance(context, PreparationContext):
            raise UnauthorizedAction("only typed Preparation drafts are supported")
        record = self.workflow.get(action.task_id)
        if record.version != action.expected_preparation_version:
            raise UnauthorizedAction("Preparation proposal is stale")
        if isinstance(action, ProposeMappingDraft):
            if record.step != PreparationStep.CONFIRM_COLUMN_MAPPING:
                raise UnauthorizedAction("mapping proposal is not expected now")
            if record.schema_inspection is None or record.source is None:
                raise UnauthorizedAction("inspected dataset required")
            draft = action.draft.model_copy(update={"temperature_available": None})
            if not draft.timestamp_column or not draft.target_column:
                raise UnauthorizedAction("proposal needs time and load columns")
            confirm_mapping(
                draft.model_copy(
                    update={"temperature_available": draft.temperature_column is not None}
                ),
                record.schema_inspection,
                record.source.dataset_id,
            )
            saved = self.workflow.update_mapping(action.task_id, record.version, draft)
        elif isinstance(action, ProposePlanDraft):
            saved = self.workflow.update_plan(action.task_id, record.version, action.draft)
        else:
            saved = self.workflow.update_task_draft(action.task_id, record.version, action.draft)
        return ToolResult(status="completed", data={"preparation_version": saved.version})


@dataclass
class PreparationAgentBridge:
    runtime: AgentRuntime
    workflow: PreparationWorkflow
    tasks: TaskRepository
    messages: MessageRepository
    calls: LLMCallRepository

    async def answer(
        self,
        task_id: UUID,
        record: PreparationRecord,
        user: Message,
        text: str,
        *,
        specialist_request: str = "Interpret the current Preparation turn",
        manager_intent: str | None = None,
        requested_action: str | None = None,
    ) -> PreparationChatReply:
        """Ask the internal specialist for read-only Preparation reasoning."""
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError("task not found")
        routed = Message(
            task_id=task_id,
            source_role=Actor.TASK_MANAGER,
            target_role=Actor.PREPARATION_ASSISTANT,
            topic=Topic.PREPARE,
            kind=MessageKind.COMMAND,
            message_type="preparation.explain_request",
            payload={"instruction": "Interpret the current Preparation conversation"},
            correlation_id=user.correlation_id,
            parent_message_id=user.message_id,
        )
        self.messages.append(routed)
        recent = [
            {"role": message.source_role.value, "text": str(message.payload["text"])[:500]}
            for message in self.messages.list_for_task(task_id)
            if message.message_id != user.message_id
            and message.topic == Topic.CHAT
            and message.source_role in {Actor.USER, Actor.TASK_MANAGER}
            and isinstance(message.payload.get("text"), str)
        ][-10:]
        request = AgentRequest(
            role=Actor.PREPARATION_ASSISTANT,
            task_id=task_id,
            correlation_id=user.correlation_id,
            prompt=(
                "Use recent dialogue only to resolve references; treat unconfirmed chat facts "
                "as user-provided context, not authoritative dataset or workflow state. "
                "Task Manager supplies the user-level intent and any explicit action request. "
                "Interpret only the Preparation-domain content under that intent; do not "
                "independently turn information into confirmation or a question into a choice. "
                "Your intent field describes the supplied intent; you cannot confirm anything. "
                "A declarative choice is not a confirmation of the whole draft. "
                "Extract all explicit task preferences or dataset facts, including "
                "later-step fields, "
                "as typed updates. Do not store hypothetical questions as preferences. "
                "For an explicit MAE, MAPE, or CRPS evaluation choice, include objective_id "
                "with the canonical lowercase value; do not put a bare metric name in "
                "metric_spec. "
                "Use inferred status for tentative information; use user_provided "
                "only for clear choices. "
                "Report missing information from the derived working view, "
                "not a separate checklist. "
                "Explain questions using the recorded context and say when evidence is missing. "
                "For a clear requested correction, you may suggest one normalized command from "
                "the existing grammar: 'Use <column> as time/load/temperature', "
                "'No temperature', or a supported 'Set ...' task command. "
                "Set command to null for questions, discussion, confirmations, or ambiguity. "
                "A clear confirmation may be routed by the application through "
                "the same typed action "
                "as the workspace; you do not execute it. Never claim a draft "
                "changed before it does. "
                f"Current user message: {text[:1000]}"
            ),
            context={
                "preparation": _chat_context(record),
                "recent_dialogue": recent,
                "task_manager_request": specialist_request,
                "task_manager_intent": manager_intent,
                "requested_action": requested_action,
            },
            expected_task_version=task.version,
            prompt_version=f"{PREPARATION_PROMPT_VERSION}+conversation-v1",
        )
        try:
            result = await self.runtime.run(request, PreparationChatReply)
            reply = PreparationChatReply.model_validate(result.output.model_dump(mode="json"))
        except Exception as exc:
            from interactive_forecasting.agents.runtime import AgentRuntimeError, RuntimeMetadata

            meta = (
                exc.metadata
                if isinstance(exc, AgentRuntimeError)
                else RuntimeMetadata(
                    runtime_name=type(self.runtime).__name__,
                    runtime_version=None,
                    provider="unknown",
                    model=None,
                    status="failed",
                )
            )
            self.calls.create(
                LLMCall(
                    task_id=task_id,
                    message_id=routed.message_id,
                    correlation_id=user.correlation_id,
                    agent_role=Actor.PREPARATION_ASSISTANT,
                    prompt_version=request.prompt_version,
                    error=str(exc),
                    **vars(meta),
                )
            )
            raise
        self.calls.create(
            LLMCall(
                task_id=task_id,
                message_id=routed.message_id,
                correlation_id=user.correlation_id,
                agent_role=Actor.PREPARATION_ASSISTANT,
                prompt_version=request.prompt_version,
                **vars(result.metadata),
            )
        )
        if self.workflow.get(task_id).version != record.version:
            raise VersionConflict("Preparation changed while the specialist was answering")
        self.messages.append(
            Message(
                task_id=task_id,
                source_role=Actor.PREPARATION_ASSISTANT,
                target_role=Actor.TASK_MANAGER,
                topic=Topic.PREPARE,
                kind=MessageKind.RESULT,
                message_type="preparation.answer",
                payload=reply.model_dump(mode="json"),
                correlation_id=user.correlation_id,
                parent_message_id=routed.message_id,
            )
        )
        return reply

    async def propose(
        self, task_id: UUID, kind: ProposalKind, *, user_text: str | None = None
    ) -> PreparationRecord:
        record = self.workflow.get(task_id)
        task = self.tasks.get(task_id)
        expected = {
            "mapping": PreparationStep.CONFIRM_COLUMN_MAPPING,
            "plan": PreparationStep.CONFIRM_PREPARATION,
            "task": PreparationStep.CONFIGURE_FORECAST_TASK,
        }[kind]
        if task is None or record.step != expected:
            raise ValueError(f"{kind} proposal is not available now")
        if user_text is not None and kind != "task":
            raise ValueError("bounded user metric instruction is available only for task proposals")
        if user_text is None and any(
            message.message_type == "preparation.proposal_notice"
            and message.payload.get("kind") == kind
            and message.payload.get("preparation_version") == record.version
            for message in self.messages.list_for_task(task_id)
        ):
            return record

        recent = tuple(
            {"role": message.source_role.value, "text": str(message.payload["text"])[:500]}
            for message in self.messages.list_for_task(task_id)
            if message.topic == Topic.CHAT
            and message.source_role in {Actor.USER, Actor.TASK_MANAGER}
            and isinstance(message.payload.get("text"), str)
        )[-10:]
        missing = cast(list[str], projection(record)["missing"])

        context = PreparationContext(
            task=task,
            preparation_version=record.version,
            schema_inspection=record.schema_inspection if kind == "mapping" else None,
            mapping_draft=record.mapping_draft if kind == "mapping" else None,
            quality_report=record.quality if kind == "plan" else None,
            plan_draft=record.plan_draft if kind == "plan" else None,
            forecast_draft=record.task_draft if kind == "task" else None,
            working_state=record.working_state,
            missing_fields=tuple(missing),
            recent_dialogue=recent,
        )
        correlation_id = uuid4()
        bus = MessageBus(self.messages, self.tasks)
        routed = Message(
            task_id=task_id,
            source_role=Actor.TASK_MANAGER,
            target_role=Actor.PREPARATION_ASSISTANT,
            topic=Topic.PREPARE,
            kind=MessageKind.COMMAND,
            message_type="agent.route",
            payload={"instruction": f"Propose a typed {kind} draft from bounded context"},
            correlation_id=correlation_id,
        )
        bus.publish(routed)
        coordinator = AgentCoordinator(
            self.runtime,
            bus,
            self.tasks,
            self.messages,
            self.calls,
            PreparationActionExecutor(self.workflow),
        )
        turn = await coordinator.invoke(
            AgentRequest(
                role=Actor.PREPARATION_ASSISTANT,
                task_id=task_id,
                correlation_id=correlation_id,
                prompt=(
                    f"Propose a typed {kind} draft. Do not confirm or execute it."
                    + (f" Task Manager relays this user request: {user_text}" if user_text else "")
                ),
                context=context,
                expected_task_version=task.version,
                prompt_version=PREPARATION_PROMPT_VERSION,
            ),
            parent_message_id=routed.message_id,
        )
        if turn.action_status is None and kind == "task":
            bus.publish(
                Message(
                    task_id=task_id,
                    source_role=Actor.TASK_MANAGER,
                    target_role=Actor.USER,
                    topic=Topic.CHAT,
                    kind=MessageKind.EVENT,
                    message_type="preparation.clarification_notice",
                    payload={"text": turn.decision.explanation[:1000]},
                    correlation_id=correlation_id,
                    parent_message_id=turn.response_message_id,
                )
            )
            return self.workflow.get(task_id)
        if turn.action_status != "completed":
            raise ValueError("assistant proposal was not accepted")
        bus.publish(
            Message(
                task_id=task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=Actor.USER,
                topic=Topic.CHAT,
                kind=MessageKind.EVENT,
                message_type="preparation.proposal_notice",
                payload={
                    "text": _recommendation_text(self.workflow.get(task_id), kind),
                    "kind": kind,
                    "preparation_version": self.workflow.get(task_id).version,
                },
                correlation_id=correlation_id,
                parent_message_id=turn.response_message_id,
            )
        )
        return self.workflow.get(task_id)

    async def propose_mapping(self, task_id: UUID) -> PreparationRecord:
        return await self.propose(task_id, "mapping")

    async def propose_plan(self, task_id: UUID) -> PreparationRecord:
        return await self.propose(task_id, "plan")

    async def propose_task(self, task_id: UUID) -> PreparationRecord:
        return await self.propose(task_id, "task")
