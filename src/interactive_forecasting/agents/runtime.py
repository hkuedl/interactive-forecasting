"""Bounded Agents SDK adapter behind application-owned typed runtime contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from math import isfinite
from time import perf_counter
from typing import Any, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel, ValidationError

from interactive_forecasting.agents.contracts import ROLES, AgentContext, AgentDecision
from interactive_forecasting.agents.deployment_policy import (
    DEPLOYMENT_CORE_PROMPT,
    deployment_instructions,
)
from interactive_forecasting.agents.model_manager_policy import (
    MODEL_DEVELOPER_CORE_PROMPT,
    MODEL_MANAGER_CORE_PROMPT,
    model_manager_instructions,
)
from interactive_forecasting.agents.preparation_policy import (
    PREPARATION_CORE_PROMPT,
    preparation_instructions,
)
from interactive_forecasting.agents.transport import AgentDecisionTransport
from interactive_forecasting.domain.types import AGENT_ROLES, Actor

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class AgentRequest:
    role: Actor
    task_id: UUID
    correlation_id: UUID
    prompt: str
    context: AgentContext | Mapping[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    timeout_seconds: float = 30.0
    expected_task_version: int | None = None
    prompt_version: str = "4a-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.role, Actor) or self.role not in AGENT_ROLES:
            raise ValueError("AgentRuntime accepts only the five semantic agent roles")
        if not isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("agent timeout must be in (0, 300] seconds")
        if self.expected_task_version is not None and self.expected_task_version < 0:
            raise ValueError("expected task version must be nonnegative")
        if self.allowed_tools:
            raise ValueError(
                "SDK tools are unavailable; request typed actions through orchestration"
            )


@dataclass(frozen=True)
class RuntimeMetadata:
    runtime_name: str
    runtime_version: str | None
    provider: str
    model: str | None
    request_id: str | None = None
    trace_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    latency_ms: float | None = None
    status: str = "completed"


@dataclass(frozen=True)
class AgentResult:
    output: BaseModel
    metadata: RuntimeMetadata


class AgentRuntimeError(RuntimeError):
    def __init__(self, message: str, metadata: RuntimeMetadata):
        super().__init__(message)
        self.metadata = metadata


class AgentTimeout(AgentRuntimeError):
    pass


class AgentFailure(AgentRuntimeError):
    pass


class InvalidAgentOutput(AgentRuntimeError):
    pass


class AgentRuntime(Protocol):
    async def run(self, request: AgentRequest, output_schema: type[OutputT]) -> AgentResult: ...


class StubAgentRuntime:
    """Test adapter: explicit results, no SDK session or API calls."""

    def __init__(self, responses: Mapping[Actor, BaseModel | dict[str, Any]]):
        self.responses = responses

    async def run(self, request: AgentRequest, output_schema: type[OutputT]) -> AgentResult:
        if request.role not in self.responses:
            raise ValueError(f"no stub response for {request.role}")
        output = output_schema.model_validate(self.responses[request.role])
        return AgentResult(
            output=output,
            metadata=RuntimeMetadata(
                runtime_name="stub",
                runtime_version=None,
                provider="local",
                model=None,
            ),
        )


class OpenAIAgentsRuntime:
    """SDK objects cannot route, call services, or advance application state."""

    def __init__(self, role_agents: Mapping[Actor, Any] | None = None):
        self.role_agents = dict(role_agents or {})

    @classmethod
    def for_model(cls, model: str, *, temperature: float | None = None) -> OpenAIAgentsRuntime:
        """Create five bounded role profiles; the caller selects the model explicitly."""
        if not model:
            raise ValueError("OpenAI model must be configured explicitly")
        from agents import Agent, ModelSettings

        return cls(
            {
                role: Agent(
                    name=role.value,
                    instructions=(
                        PREPARATION_CORE_PROMPT
                        if role == Actor.PREPARATION_ASSISTANT
                        else MODEL_MANAGER_CORE_PROMPT
                        if role == Actor.MODEL_MANAGER
                        else MODEL_DEVELOPER_CORE_PROMPT
                        if role == Actor.MODEL_DEVELOPER
                        else DEPLOYMENT_CORE_PROMPT
                        if role == Actor.DEPLOYMENT_OPERATOR
                        else f"{definition.purpose} Return only a structured decision. "
                        "Do not invent metrics, execute services, access final-test labels, "
                        "or claim an action succeeded before a service result is returned."
                    ),
                    model=model,
                    model_settings=ModelSettings(temperature=temperature),
                    output_type=AgentDecisionTransport,
                )
                for role, definition in ROLES.items()
            }
        )

    async def run(self, request: AgentRequest, output_schema: type[OutputT]) -> AgentResult:
        agent = self.role_agents.get(request.role)
        if agent is None:
            raise RuntimeError(f"no SDK agent configured for {request.role.value}")
        if (
            getattr(agent, "tools", None)
            or getattr(agent, "handoffs", None)
            or getattr(agent, "mcp_servers", None)
        ):
            raise ValueError("application-managed turns prohibit SDK tools, MCP and handoffs")
        from agents import ModelBehaviorError, Runner

        transport_schema = (
            AgentDecisionTransport if output_schema is AgentDecision else output_schema
        )
        configured = replace(agent, output_type=transport_schema)
        from interactive_forecasting.agents.preparation_manager import (
            PREPARATION_MANAGER_PROMPT,
            PreparationResponse,
            PreparationRoute,
        )

        if request.role == Actor.TASK_MANAGER and output_schema in (
            PreparationRoute,
            PreparationResponse,
        ):
            configured = replace(configured, instructions=PREPARATION_MANAGER_PROMPT)
        if request.role == Actor.PREPARATION_ASSISTANT:
            configured = replace(configured, instructions=preparation_instructions())
        elif request.role == Actor.MODEL_MANAGER:
            configured = replace(configured, instructions=model_manager_instructions())
        elif request.role == Actor.DEPLOYMENT_OPERATOR:
            configured = replace(configured, instructions=deployment_instructions())
        context = (
            request.context.model_dump(mode="json")
            if isinstance(request.context, BaseModel)
            else dict(request.context)
        )
        payload = json.dumps({"task": request.prompt, "context": context}, sort_keys=True)
        model = str(agent.model) if agent.model is not None else None
        base: dict[str, Any] = dict(
            runtime_name="openai-agents",
            runtime_version=version("openai-agents"),
            provider="openai",
            model=model,
        )
        started = perf_counter()
        try:
            result = await asyncio.wait_for(
                Runner.run(configured, payload, max_turns=1),
                timeout=request.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeout(
                "agent invocation timed out",
                RuntimeMetadata(
                    **base, latency_ms=(perf_counter() - started) * 1000, status="timeout"
                ),
            ) from exc
        except ModelBehaviorError as exc:
            raise InvalidAgentOutput(
                "agent returned invalid structured output",
                RuntimeMetadata(
                    **base, latency_ms=(perf_counter() - started) * 1000, status="invalid_output"
                ),
            ) from exc
        except Exception as exc:
            raise AgentFailure(
                f"agent runtime failed: {type(exc).__name__}",
                RuntimeMetadata(
                    **base, latency_ms=(perf_counter() - started) * 1000, status="failed"
                ),
            ) from exc
        try:
            raw = result.final_output
            if isinstance(raw, BaseModel):
                raw = raw.model_dump(mode="json")
            if output_schema is AgentDecision:
                transport = (
                    AgentDecisionTransport.model_validate_json(raw)
                    if isinstance(raw, str)
                    else AgentDecisionTransport.model_validate(raw)
                )
                raw = transport.to_application()
            output = (
                output_schema.model_validate_json(raw)
                if isinstance(raw, str)
                else output_schema.model_validate(raw)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise InvalidAgentOutput(
                "agent returned invalid structured output",
                RuntimeMetadata(
                    **base, latency_ms=(perf_counter() - started) * 1000, status="invalid_output"
                ),
            ) from exc
        responses = getattr(result, "raw_responses", ())
        input_tokens = (
            sum(response.usage.input_tokens for response in responses) if responses else None
        )
        output_tokens = (
            sum(response.usage.output_tokens for response in responses) if responses else None
        )
        cached_tokens = (
            sum(response.usage.input_tokens_details.cached_tokens for response in responses)
            if responses
            else None
        )
        last = responses[-1] if responses else None
        return AgentResult(
            output=output,
            metadata=RuntimeMetadata(
                **base,
                request_id=last.request_id if last is not None else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                latency_ms=(perf_counter() - started) * 1000,
            ),
        )
