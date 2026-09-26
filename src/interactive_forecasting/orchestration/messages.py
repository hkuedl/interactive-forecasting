"""Local topic router over a persistent application message store."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from interactive_forecasting.domain.models import Message, Task
from interactive_forecasting.domain.types import (
    Actor,
    MessageKind,
    Stage,
    Topic,
)


class MessageStore(Protocol):
    def append(self, message: Message) -> bool: ...
    def get(self, message_id: object) -> Message | None: ...
    def list_for_task(self, task_id: object) -> list[Message]: ...
    def record_delivery(self, message_id: object, handler_name: str, status: str) -> None: ...
    def was_delivered(self, message_id: object, handler_name: str) -> bool: ...


class TaskLookup(Protocol):
    def get(self, task_id: object) -> Task | None: ...


Handler = Callable[[Message], None]


@dataclass(frozen=True)
class Subscription:
    name: str
    topic: Topic
    handler: Handler
    role: Actor | None = None


@dataclass(frozen=True)
class DeliveryOutcome:
    handler_name: str
    status: str


_STAGE_ROLES: dict[Stage, set[Actor]] = {
    Stage.PREPARATION: {Actor.PREPARATION_ASSISTANT},
    Stage.OPTIMIZATION: {Actor.MODEL_MANAGER, Actor.MODEL_DEVELOPER},
    Stage.DEPLOYMENT: {Actor.DEPLOYMENT_OPERATOR},
}

_TOPIC_STAGE: dict[Topic, Stage] = {
    Topic.PREPARE: Stage.PREPARATION,
    Topic.OPTIMIZE: Stage.OPTIMIZATION,
    Topic.TRAIN: Stage.OPTIMIZATION,
    Topic.DEPLOY: Stage.DEPLOYMENT,
}


def role_eligible(actor: Actor, stage: Stage) -> bool:
    if actor in {Actor.USER, Actor.SYSTEM, Actor.SERVICE, Actor.TASK_MANAGER}:
        return True
    return actor in _STAGE_ROLES.get(stage, set())


class MessageBus:
    """Persist before delivery; a replay reads application records, never SDK state."""

    def __init__(self, store: MessageStore, tasks: TaskLookup):
        self.store = store
        self.tasks = tasks
        self._subscriptions: dict[Topic, list[Subscription]] = defaultdict(list)

    def subscribe(
        self, topic: Topic, name: str, handler: Handler, *, role: Actor | None = None
    ) -> None:
        if any(s.name == name for s in self._subscriptions[topic]):
            raise ValueError(f"duplicate subscription {name} for {topic}")
        self._subscriptions[topic].append(Subscription(name, topic, handler, role))

    def publish(self, message: Message) -> list[DeliveryOutcome]:
        task = self.tasks.get(message.task_id)
        if task is None:
            raise ValueError("message task does not exist")
        if message.topic in _TOPIC_STAGE and task.stage != _TOPIC_STAGE[message.topic]:
            raise ValueError("message topic is not active in this workflow stage")
        if not role_eligible(message.source_role, task.stage):
            raise ValueError("source role is not eligible in this workflow stage")
        if message.target_role is not None and not role_eligible(message.target_role, task.stage):
            raise ValueError("target role is not eligible in this workflow stage")
        if message.source_role == Actor.USER and message.kind != MessageKind.USER:
            raise ValueError("user messages must have USER kind")
        if message.topic == Topic.CHAT and message.target_role == Actor.USER:
            if message.source_role != Actor.TASK_MANAGER:
                raise ValueError("only the Task Manager may address the user")
        if message.parent_message_id is not None:
            ancestor = self.store.get(message.parent_message_id)
            if (
                ancestor is None
                or ancestor.task_id != message.task_id
                or ancestor.correlation_id != message.correlation_id
            ):
                raise ValueError("message parent must share task and correlation")
        if message.kind == MessageKind.RESULT:
            parent = self.store.get(message.parent_message_id)
            if parent is None or parent.kind != MessageKind.COMMAND:
                raise ValueError("result requires a persisted command parent")
            if parent.task_id != message.task_id or parent.correlation_id != message.correlation_id:
                raise ValueError("result must match parent task and correlation")
        if not self.store.append(message):
            return []
        return self.dispatch(message)

    def dispatch(self, message: Message) -> list[DeliveryOutcome]:
        if self.store.get(message.message_id) is None:
            raise ValueError("cannot dispatch an unpersisted message")
        task = self.tasks.get(message.task_id)
        if task is None:
            raise ValueError("message task does not exist")
        outcomes: list[DeliveryOutcome] = []
        for subscription in self._subscriptions[message.topic]:
            if message.target_role is not None and subscription.role != message.target_role:
                continue
            if subscription.role is not None and not role_eligible(subscription.role, task.stage):
                continue
            if self.store.was_delivered(message.message_id, subscription.name):
                continue
            try:
                subscription.handler(message)
                status = "delivered"
            except Exception:
                status = "failed"
            self.store.record_delivery(message.message_id, subscription.name, status)
            outcomes.append(DeliveryOutcome(subscription.name, status))
        return outcomes
