"""Bounded live TM / PA turns; only existing workflow methods may mutate state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from interactive_forecasting.agents.preparation_manager import PreparationResponse, PreparationRoute
from interactive_forecasting.agents.runtime import (
    AgentRequest,
    AgentRuntimeError,
    InvalidAgentOutput,
    RuntimeMetadata,
)
from interactive_forecasting.domain.models import LLMCall, Message
from interactive_forecasting.domain.preparation import PreparationRecord
from interactive_forecasting.domain.types import Actor, MessageKind, Topic
from interactive_forecasting.orchestration.preparation_agents import PreparationAgentBridge
from interactive_forecasting.orchestration.preparation_chat import (
    is_pure_hypothetical,
    parse_mapping_command,
    parse_task_command,
)
from interactive_forecasting.orchestration.state_machine import VersionConflict
from interactive_forecasting.services.preparation_working import projection

Output = TypeVar("Output", bound=BaseModel)


@dataclass
class PreparationConversation:
    bridge: PreparationAgentBridge

    def message(
        self, user: Message, source: Actor, target: Actor, kind: str, payload: dict[str, Any]
    ) -> Message:
        message = Message(
            task_id=user.task_id,
            source_role=source,
            target_role=target,
            topic=Topic.PREPARE,
            kind=MessageKind.RESULT,
            message_type=kind,
            payload=payload,
            correlation_id=user.correlation_id,
            parent_message_id=user.message_id,
        )
        self.bridge.messages.append(message)
        return message

    def context(self, user: Message, record: PreparationRecord) -> dict[str, Any]:
        view = projection(record)
        return {
            "step": record.step.value,
            "preparation_version": record.version,
            "mapping_confirmed": record.confirmed_mapping is not None,
            "plan_confirmed": record.plan_confirmed,
            "task_frozen": record.frozen,
            "has_mapping_draft": record.mapping_draft is not None,
            "has_plan_draft": record.plan_draft is not None,
            "has_task_draft": record.task_draft is not None,
            "collected_fields": [
                {"field": entry["field"], "status": entry["status"]}
                for entry in cast(list[dict[str, Any]], view["collected"])
            ],
            "missing_fields": view["missing"],
            "recent_dialogue": [
                {"role": m.source_role.value, "text": str(m.payload["text"])[:500]}
                for m in self.bridge.messages.list_for_task(user.task_id)
                if m.message_id != user.message_id
                and m.topic == Topic.CHAT
                and m.source_role in {Actor.USER, Actor.TASK_MANAGER}
                and isinstance(m.payload.get("text"), str)
            ][-10:],
        }

    async def manager(
        self,
        user: Message,
        record: PreparationRecord,
        schema: type[Output],
        result: dict[str, Any] | None = None,
        retry_note: str | None = None,
    ) -> Output:
        context = self.context(user, record)
        if result is not None:
            context["execution_result"] = result
        if retry_note is not None:
            context["retry_note"] = retry_note
        routed = self.message(
            user, Actor.SYSTEM, Actor.TASK_MANAGER, "preparation.manager_request", context
        )
        request = AgentRequest(
            role=Actor.TASK_MANAGER,
            task_id=user.task_id,
            correlation_id=user.correlation_id,
            prompt=str(user.payload["text"]),
            context=context,
            prompt_version=(
                "preparation-manager-v1/repair-1"
                if retry_note is not None
                else "preparation-manager-v1"
            ),
            timeout_seconds=60,
        )
        try:
            response = await self.bridge.runtime.run(request, schema)
            output = schema.model_validate(response.output.model_dump(mode="json"))
            if isinstance(output, PreparationResponse) and result is not None:
                if output.outcome != result["outcome"]:
                    raise ValueError("Task Manager outcome disagrees with application result")
                if result.get("working_updated") and not result.get("confirmation_completed"):
                    if re.search(
                        r"\b(?:(?:has|have|is|was|were)\s+(?:now\s+)?(?:been\s+)?"
                        r"confirmed|confirmed[.!]?\s*$)",
                        output.text,
                        re.I,
                    ):
                        raise ValueError("Working information was described as confirmed")
        except Exception as exc:
            metadata = (
                exc.metadata
                if isinstance(exc, AgentRuntimeError)
                else RuntimeMetadata(
                    runtime_name=type(self.bridge.runtime).__name__,
                    runtime_version=None,
                    provider="unknown",
                    model=None,
                    status="invalid_output",
                )
            )
            self.bridge.calls.create(
                LLMCall(
                    task_id=user.task_id,
                    message_id=routed.message_id,
                    correlation_id=user.correlation_id,
                    agent_role=Actor.TASK_MANAGER,
                    prompt_version=request.prompt_version,
                    error=type(exc).__name__,
                    **vars(metadata),
                )
            )
            raise
        self.bridge.calls.create(
            LLMCall(
                task_id=user.task_id,
                message_id=routed.message_id,
                correlation_id=user.correlation_id,
                agent_role=Actor.TASK_MANAGER,
                prompt_version=request.prompt_version,
                **vars(response.metadata),
            )
        )
        if self.bridge.workflow.get(user.task_id).version != record.version:
            raise VersionConflict("Preparation changed during the Task Manager turn")
        self.message(
            user,
            Actor.TASK_MANAGER,
            Actor.SYSTEM,
            "preparation.manager_result",
            output.model_dump(mode="json"),
        )
        return output

    async def run(self, user: Message, record: PreparationRecord) -> PreparationRecord:
        try:
            decision = await self.manager(user, record, PreparationRoute)
        except VersionConflict:
            raise
        except (InvalidAgentOutput, ValueError):
            decision = await self.manager(
                user,
                record,
                PreparationRoute,
                retry_note=(
                    "The previous route failed validation. Use a plain-language reply "
                    "for route=respond, preserve explicit user intent, and keep "
                    "specialist extraction separate from confirmation."
                ),
            )
        text = decision.text
        if decision.route != "respond":
            result: dict[str, Any] = {"outcome": "read_only"}
            try:
                if decision.route == "specialist":
                    self.message(
                        user,
                        Actor.TASK_MANAGER,
                        Actor.PREPARATION_ASSISTANT,
                        "preparation.specialist_request",
                        {"request": decision.text},
                    )
                    answer = await self.bridge.answer(
                        user.task_id,
                        record,
                        user,
                        str(user.payload["text"]),
                        specialist_request=decision.text,
                        manager_intent=decision.intent,
                        requested_action=decision.action,
                    )
                    result["specialist"] = {
                        "text": answer.text if decision.intent == "question" else None,
                        "suggested_fields": [update.field.value for update in answer.updates],
                    }
                    if decision.intent in {"information", "correction", "confirmation"} and (
                        not is_pure_hypothetical(str(user.payload["text"]))
                    ):
                        if answer.updates or (decision.intent == "correction" and answer.command):
                            self.message(
                                user,
                                Actor.TASK_MANAGER,
                                Actor.SYSTEM,
                                "preparation.action_request",
                                answer.model_dump(mode="json"),
                            )
                            if answer.updates:
                                record, conflicts = self.bridge.workflow.record_working(
                                    user.task_id,
                                    record.version,
                                    answer.updates,
                                    user.message_id,
                                )
                                result["conflicts"] = conflicts
                                result["working_updated"] = True
                            elif answer.command:
                                # Parse a specialist proposal, never the user's routing intent.
                                mapping = parse_mapping_command(record, answer.command)
                                draft = parse_task_command(record, answer.command)
                                if mapping is not None:
                                    record = self.bridge.workflow.update_mapping(
                                        user.task_id, record.version, mapping
                                    )
                                elif draft is not None:
                                    record = self.bridge.workflow.update_task_draft(
                                        user.task_id, record.version, draft
                                    )
                                else:
                                    raise ValueError(
                                        "No editable draft matches the requested change"
                                    )
                                result["working_updated"] = True
                            result["outcome"] = "completed"
                if decision.action is not None:
                    if result.get("conflicts"):
                        raise ValueError("Conflicting Preparation choices require clarification")
                    self.message(
                        user,
                        Actor.TASK_MANAGER,
                        Actor.SYSTEM,
                        "preparation.action_request",
                        {"action": decision.action},
                    )
                    handler = getattr(self.bridge.workflow, decision.action)
                    if decision.action == "apply":
                        record = handler(user.task_id, record.version, user.message_id)
                    else:
                        record = handler(user.task_id, record.version)
                    result["outcome"] = "completed"
                    result["completed_action"] = decision.action
                    result["confirmation_completed"] = decision.action in {
                        "confirm_mapping",
                        "confirm_plan",
                        "confirm_task",
                    }
            except VersionConflict:
                raise
            except Exception as exc:
                result = {
                    "outcome": "failed",
                    "error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                }
                record = self.bridge.workflow.get(user.task_id)
            result["preparation_version"] = record.version
            result["authoritative_status"] = {
                "mapping_confirmed": record.confirmed_mapping is not None,
                "plan_confirmed": record.plan_confirmed,
                "task_frozen": record.frozen,
                "step": record.step.value,
            }
            self.message(
                user, Actor.SYSTEM, Actor.TASK_MANAGER, "preparation.execution_result", result
            )
            try:
                final = await self.manager(user, record, PreparationResponse, result)
            except VersionConflict:
                raise
            except (InvalidAgentOutput, ValueError):
                final = await self.manager(
                    user,
                    record,
                    PreparationResponse,
                    result,
                    retry_note=(
                        "The previous final response failed validation. Reply in natural "
                        "user-facing prose, without JSON or internal fields. Copy "
                        "execution_result.outcome exactly and describe only the actual result."
                    ),
                )
            text = final.text
        self.bridge.messages.append(
            Message(
                task_id=user.task_id,
                source_role=Actor.TASK_MANAGER,
                target_role=Actor.USER,
                topic=Topic.CHAT,
                kind=MessageKind.EVENT,
                message_type="preparation_reply",
                payload={"text": text, "intent": decision.intent},
                correlation_id=user.correlation_id,
                parent_message_id=user.message_id,
            )
        )
        return record
