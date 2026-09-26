"""Optimistically versioned Preparation aggregate with atomic task/event transitions."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import update

from interactive_forecasting.domain.models import Message, Task
from interactive_forecasting.domain.preparation import PreparationRecord
from interactive_forecasting.domain.types import Actor, MessageKind, Topic
from interactive_forecasting.orchestration.state_machine import (
    InvalidTransition,
    VersionConflict,
    WorkflowStateMachine,
)
from interactive_forecasting.storage.sql import (
    Database,
    PreparationRow,
    TaskRepository,
    TaskRow,
    _message_row,
)


class PreparationRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, record: PreparationRecord) -> PreparationRecord:
        with self.database.session() as session:
            if session.get(TaskRow, str(record.task_id)) is None:
                raise ValueError("task does not exist")
            session.add(
                PreparationRow(
                    task_id=str(record.task_id),
                    version=record.version,
                    document=record.model_dump(mode="json"),
                )
            )
            session.commit()
        return record

    def get(self, task_id: UUID) -> PreparationRecord | None:
        with self.database.session() as session:
            row = session.get(PreparationRow, str(task_id))
            return PreparationRecord.model_validate(row.document) if row else None

    def save(
        self,
        record: PreparationRecord,
        *,
        expected_version: int,
        task: Task | None = None,
        event: Message | None = None,
    ) -> PreparationRecord:
        if (task is None) != (event is None):
            raise ValueError("task transition and event must be committed together")
        if event is not None and (
            event.kind != MessageKind.EVENT
            or event.topic != Topic.SYSTEM
            or event.source_role != Actor.SYSTEM
        ):
            raise ValueError("Preparation transition needs a system event")
        if record.version != expected_version + 1:
            raise VersionConflict("preparation version must increment by one")
        with self.database.session() as session:
            if task is not None and event is not None:
                previous = TaskRepository._previous(session, task.task_id)
                if task.task_id != record.task_id or event.task_id != record.task_id:
                    raise ValueError("transition task mismatch")
                machine = WorkflowStateMachine()
                if task.stage != previous.stage:
                    valid = machine.transition_stage(
                        previous, task.stage, expected_version=previous.version
                    )
                elif task.substate != previous.substate and task.substate is not None:
                    valid = machine.transition_substate(
                        previous, task.substate, expected_version=previous.version
                    )
                else:
                    raise InvalidTransition("Preparation save must advance task state")
                if (valid.stage, valid.substate, valid.version) != (
                    task.stage,
                    task.substate,
                    task.version,
                ):
                    raise InvalidTransition("invalid Preparation task transition")
                TaskRepository._update(session, task, previous.version)
                session.add(_message_row(event))
            result = session.execute(
                update(PreparationRow)
                .where(
                    PreparationRow.task_id == str(record.task_id),
                    PreparationRow.version == expected_version,
                )
                .values(version=record.version, document=record.model_dump(mode="json"))
            )
            if getattr(result, "rowcount", 0) != 1:
                raise VersionConflict("Preparation draft has been modified")
            session.commit()
        return record
