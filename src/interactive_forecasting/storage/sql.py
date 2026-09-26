"""SQLite/SQLAlchemy metadata repositories. Scientific arrays stay in artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from interactive_forecasting.domain.models import (
    Adjustment,
    DeploymentSession,
    Event,
    ExperimentRun,
    Forecast,
    ForecastVersion,
    Job,
    LLMCall,
    Message,
    ReferenceAnalysis,
    Task,
)
from interactive_forecasting.domain.optimization import OptimizationSession
from interactive_forecasting.domain.preparation import PreparationRecord
from interactive_forecasting.domain.types import Actor, MessageKind, Stage, Topic
from interactive_forecasting.orchestration.state_machine import (
    InvalidTransition,
    VersionConflict,
    WorkflowStateMachine,
)


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    substate: Mapped[str | None] = mapped_column(String(80))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PreparationRow(Base):
    __tablename__ = "preparations"

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class MessageRow(Base):
    __tablename__ = "messages"

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    correlation_id: Mapped[str] = mapped_column(String(36), index=True)
    parent_message_id: Mapped[str | None] = mapped_column(String(36))
    topic: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DeliveryRow(Base):
    __tablename__ = "message_deliveries"

    delivery_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.message_id"), index=True)
    handler_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobRow(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExperimentRunRow(Base):
    __tablename__ = "experiment_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    spec_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class OptimizationSessionRow(Base):
    __tablename__ = "optimization_sessions"

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.run_id"), unique=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class DeploymentSessionRow(Base):
    __tablename__ = "deployment_sessions"

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class ForecastRow(Base):
    __tablename__ = "forecasts"

    forecast_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(36), index=True, unique=True)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class ForecastVersionRow(Base):
    __tablename__ = "forecast_versions"
    __table_args__ = (UniqueConstraint("forecast_id", "version_number"),)

    version_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.forecast_id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class ReferenceAnalysisRow(Base):
    __tablename__ = "reference_analyses"

    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.forecast_id"), primary_key=True)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class AdjustmentRow(Base):
    __tablename__ = "adjustments"

    adjustment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.forecast_id"), index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class LLMCallRow(Base):
    __tablename__ = "llm_calls"

    call_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    document: Mapped[dict] = mapped_column(JSON, nullable=False)


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        """Test/bootstrap helper; deployed databases use Alembic migrations."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.sessions() as session:
            yield session


def _document(model: Task | Message | Job | ExperimentRun | LLMCall) -> dict:
    return model.model_dump(mode="json")


class TaskRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, task: Task) -> Task:
        with self.database.session() as session:
            session.add(
                TaskRow(
                    task_id=str(task.task_id),
                    stage=task.stage.value,
                    substate=task.substate,
                    version=task.version,
                    document=_document(task),
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
            )
            session.commit()
        return task

    def get(self, task_id: UUID | object) -> Task | None:
        with self.database.session() as session:
            row = session.get(TaskRow, str(task_id))
            return Task.model_validate(row.document) if row is not None else None

    def save(self, task: Task, *, expected_version: int) -> Task:
        with self.database.session() as session:
            previous = self._previous(session, task.task_id)
            if (task.stage, task.substate, task.workflow_status) != (
                previous.stage,
                previous.substate,
                previous.workflow_status,
            ):
                raise InvalidTransition("workflow state changes require save_transition")
            self._update(session, task, expected_version)
            session.commit()
        return task

    def save_transition(self, task: Task, event: Message, *, expected_version: int) -> Task:
        if event.task_id != task.task_id:
            raise ValueError("transition event belongs to another task")
        if (
            event.kind != MessageKind.EVENT
            or event.topic != Topic.SYSTEM
            or event.source_role != Actor.SYSTEM
        ):
            raise ValueError("transition record must be a system-authored event")
        with self.database.session() as session:
            previous = self._previous(session, task.task_id)
            machine = WorkflowStateMachine()
            if previous.stage == Stage.PREPARATION and task.stage == Stage.OPTIMIZATION:
                preparation_row = session.get(PreparationRow, str(task.task_id))
                if preparation_row is not None:
                    preparation = PreparationRecord.model_validate(preparation_row.document)
                    if (
                        not preparation.frozen
                        or preparation.prepared is None
                        or preparation.definition is None
                        or preparation.protocol is None
                        or task.definition_id != preparation.definition.definition_id
                    ):
                        raise InvalidTransition(
                            "frozen prepared snapshot, task definition and protocol required"
                        )
            if task.stage != previous.stage:
                valid = machine.transition_stage(
                    previous, task.stage, expected_version=expected_version
                )
            elif task.substate != previous.substate and task.substate is not None:
                valid = machine.transition_substate(
                    previous, task.substate, expected_version=expected_version
                )
            elif task.workflow_status != previous.workflow_status:
                valid = machine.transition_status(
                    previous, task.workflow_status, expected_version=expected_version
                )
            else:
                raise InvalidTransition("transition must change workflow state")
            if (task.stage, task.substate, task.workflow_status, task.version) != (
                valid.stage,
                valid.substate,
                valid.workflow_status,
                valid.version,
            ):
                raise InvalidTransition("task does not match the state machine transition")
            self._update(session, task, expected_version)
            session.add(_message_row(event))
            session.commit()
        return task

    @staticmethod
    def _previous(session: Session, task_id: UUID) -> Task:
        row = session.get(TaskRow, str(task_id))
        if row is None:
            raise VersionConflict("task does not exist")
        return Task.model_validate(row.document)

    @staticmethod
    def _update(session: Session, task: Task, expected_version: int) -> None:
        if task.version != expected_version + 1:
            raise VersionConflict("new task version must increment exactly once")
        result = session.execute(
            update(TaskRow)
            .where(TaskRow.task_id == str(task.task_id), TaskRow.version == expected_version)
            .values(
                stage=task.stage.value,
                substate=task.substate,
                version=task.version,
                document=_document(task),
                updated_at=task.updated_at,
            )
        )
        if getattr(result, "rowcount", 0) != 1:
            raise VersionConflict("task was modified or does not exist")


def _message_row(message: Message) -> MessageRow:
    return MessageRow(
        message_id=str(message.message_id),
        task_id=str(message.task_id),
        correlation_id=str(message.correlation_id),
        parent_message_id=str(message.parent_message_id) if message.parent_message_id else None,
        topic=message.topic.value,
        kind=message.kind.value,
        document=_document(message),
        timestamp=message.timestamp,
    )


def _read_message(document: dict) -> Message:
    if document.get("kind") == MessageKind.EVENT.value and "outcome" in document:
        return Event.model_validate(document)
    return Message.model_validate(document)


class MessageRepository:
    def __init__(self, database: Database):
        self.database = database

    def append(self, message: Message) -> bool:
        with self.database.session() as session:
            session.add(_message_row(message))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                if self.get(message.message_id) is not None:
                    return False
                raise
        return True

    def get(self, message_id: UUID | object | None) -> Message | None:
        if message_id is None:
            return None
        with self.database.session() as session:
            row = session.scalar(select(MessageRow).where(MessageRow.message_id == str(message_id)))
            return _read_message(row.document) if row is not None else None

    def list_for_task(self, task_id: UUID | object) -> list[Message]:
        with self.database.session() as session:
            rows = session.scalars(
                select(MessageRow)
                .where(MessageRow.task_id == str(task_id))
                .order_by(MessageRow.sequence)
            ).all()
            return [_read_message(row.document) for row in rows]

    def record_delivery(self, message_id: UUID | object, handler_name: str, status: str) -> None:
        with self.database.session() as session:
            session.add(
                DeliveryRow(
                    message_id=str(message_id),
                    handler_name=handler_name,
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def was_delivered(self, message_id: UUID | object, handler_name: str) -> bool:
        with self.database.session() as session:
            return (
                session.scalar(
                    select(DeliveryRow.delivery_id).where(
                        DeliveryRow.message_id == str(message_id),
                        DeliveryRow.handler_name == handler_name,
                        DeliveryRow.status == "delivered",
                    )
                )
                is not None
            )

    def delivery_statuses(self, message_id: UUID | object) -> list[str]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(DeliveryRow.status)
                    .where(DeliveryRow.message_id == str(message_id))
                    .order_by(DeliveryRow.delivery_id)
                )
            )


class JobRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, job: Job) -> Job:
        with self.database.session() as session:
            session.add(
                JobRow(
                    job_id=str(job.job_id),
                    task_id=str(job.task_id),
                    status=job.status.value,
                    document=_document(job),
                    updated_at=job.updated_at,
                )
            )
            session.commit()
        return job

    def get(self, job_id: UUID) -> Job | None:
        with self.database.session() as session:
            row = session.get(JobRow, str(job_id))
            return Job.model_validate(row.document) if row is not None else None

    def save(self, job: Job) -> Job:
        with self.database.session() as session:
            row = session.get(JobRow, str(job.job_id))
            if row is None:
                raise KeyError(job.job_id)
            row.status = job.status.value
            row.document = _document(job)
            row.updated_at = job.updated_at
            session.commit()
        return job


class ExperimentRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, run: ExperimentRun) -> ExperimentRun:
        with self.database.session() as session:
            session.add(
                ExperimentRunRow(
                    run_id=str(run.run_id),
                    spec_id=run.spec_id,
                    status=run.status.value,
                    version=run.version,
                    document=_document(run),
                )
            )
            session.commit()
        return run

    def get(self, run_id: UUID) -> ExperimentRun | None:
        with self.database.session() as session:
            row = session.get(ExperimentRunRow, str(run_id))
            return ExperimentRun.model_validate(row.document) if row is not None else None

    def save(self, run: ExperimentRun, *, expected_version: int) -> ExperimentRun:
        if run.version != expected_version + 1:
            raise VersionConflict("experiment version must increment exactly once")
        with self.database.session() as session:
            result = session.execute(
                update(ExperimentRunRow)
                .where(
                    ExperimentRunRow.run_id == str(run.run_id),
                    ExperimentRunRow.version == expected_version,
                )
                .values(status=run.status.value, version=run.version, document=_document(run))
            )
            if getattr(result, "rowcount", 0) != 1:
                raise VersionConflict("experiment was modified or does not exist")
            session.commit()
        return run


class OptimizationSessionRepository:
    """One versioned interactive optimization session per prepared task."""

    def __init__(self, database: Database):
        self.database = database

    def create(self, record: OptimizationSession) -> OptimizationSession:
        with self.database.session() as session:
            if session.get(TaskRow, str(record.task_id)) is None:
                raise ValueError("task does not exist")
            if session.get(ExperimentRunRow, str(record.run_id)) is None:
                raise ValueError("search run does not exist")
            session.add(
                OptimizationSessionRow(
                    task_id=str(record.task_id),
                    run_id=str(record.run_id),
                    version=record.version,
                    document=record.model_dump(mode="json"),
                )
            )
            session.commit()
        return record

    def get(self, task_id: UUID) -> OptimizationSession | None:
        with self.database.session() as session:
            row = session.get(OptimizationSessionRow, str(task_id))
            return OptimizationSession.model_validate(row.document) if row else None

    def save(self, record: OptimizationSession, *, expected_version: int) -> OptimizationSession:
        if record.version != expected_version + 1:
            raise VersionConflict("optimization session version must increment by one")
        with self.database.session() as session:
            result = session.execute(
                update(OptimizationSessionRow)
                .where(
                    OptimizationSessionRow.task_id == str(record.task_id),
                    OptimizationSessionRow.version == expected_version,
                )
                .values(version=record.version, document=record.model_dump(mode="json"))
            )
            if getattr(result, "rowcount", 0) != 1:
                raise VersionConflict("optimization session was modified")
            session.commit()
        return record


class LLMCallRepository:
    """Own audit records; SDK tracing is supplementary and never replay authority."""

    def __init__(self, database: Database):
        self.database = database

    def create(self, call: LLMCall) -> LLMCall:
        with self.database.session() as session:
            session.add(
                LLMCallRow(
                    call_id=str(call.call_id),
                    task_id=str(call.task_id),
                    document=_document(call),
                )
            )
            session.commit()
        return call

    def get(self, call_id: UUID) -> LLMCall | None:
        with self.database.session() as session:
            row = session.get(LLMCallRow, str(call_id))
            return LLMCall.model_validate(row.document) if row is not None else None

    def list_for_task(self, task_id: UUID) -> list[LLMCall]:
        with self.database.session() as session:
            rows = session.scalars(
                select(LLMCallRow).where(LLMCallRow.task_id == str(task_id))
            ).all()
            return [LLMCall.model_validate(row.document) for row in rows]


class DeploymentRepository:
    """Single-writer sessions; immutable forecasts, reference results and version chain."""

    def __init__(self, database: Database):
        self.database = database

    def create(self, record: DeploymentSession) -> DeploymentSession:
        with self.database.session() as session:
            session.add(
                DeploymentSessionRow(
                    session_id=str(record.session_id),
                    task_id=str(record.task_id),
                    version=record.version,
                    document=record.model_dump(mode="json"),
                )
            )
            session.commit()
        return record

    def get(self, task_id: UUID) -> DeploymentSession | None:
        with self.database.session() as session:
            row = session.scalar(
                select(DeploymentSessionRow)
                .where(DeploymentSessionRow.task_id == str(task_id))
                .order_by(DeploymentSessionRow.sequence.desc())
            )
            return DeploymentSession.model_validate(row.document) if row else None

    def get_session(self, session_id: UUID) -> DeploymentSession | None:
        with self.database.session() as session:
            row = session.scalar(
                select(DeploymentSessionRow).where(
                    DeploymentSessionRow.session_id == str(session_id)
                )
            )
            return DeploymentSession.model_validate(row.document) if row else None

    def list_sessions(self, task_id: UUID) -> list[DeploymentSession]:
        with self.database.session() as session:
            rows = session.scalars(
                select(DeploymentSessionRow)
                .where(DeploymentSessionRow.task_id == str(task_id))
                .order_by(DeploymentSessionRow.sequence)
            ).all()
            return [DeploymentSession.model_validate(row.document) for row in rows]

    @staticmethod
    def _update_session(session: Session, record: DeploymentSession, expected_version: int) -> None:
        if record.version != expected_version + 1:
            raise VersionConflict("deployment session version must increment by one")
        result = session.execute(
            update(DeploymentSessionRow)
            .where(
                DeploymentSessionRow.session_id == str(record.session_id),
                DeploymentSessionRow.version == expected_version,
            )
            .values(version=record.version, document=record.model_dump(mode="json"))
        )
        if getattr(result, "rowcount", 0) != 1:
            raise VersionConflict("deployment session was modified")

    def save(self, record: DeploymentSession, *, expected_version: int) -> DeploymentSession:
        with self.database.session() as session:
            self._update_session(session, record, expected_version)
            session.commit()
        return record

    def commit_forecast(
        self,
        record: DeploymentSession,
        forecast: Forecast,
        original: ForecastVersion,
        *,
        expected_version: int,
    ) -> None:
        if (
            record.task_id != forecast.task_id
            or record.session_id != forecast.deployment_session_id
            or record.forecast_id != forecast.forecast_id
            or original.forecast_id != forecast.forecast_id
            or original.version_number != 0
            or original.prediction != forecast.prediction
            or original.prediction_representation != forecast.prediction_representation
        ):
            raise ValueError("forecast, original version and session must agree")
        with self.database.session() as session:
            prior = session.scalar(
                select(DeploymentSessionRow).where(
                    DeploymentSessionRow.session_id == str(record.session_id)
                )
            )
            existing = session.scalar(
                select(ForecastRow.forecast_id).where(
                    ForecastRow.session_id == str(record.session_id)
                )
            )
            if (
                prior is None
                or prior.document.get("forecast_id") is not None
                or existing is not None
            ):
                raise VersionConflict("deployment session already owns an original forecast")
            self._update_session(session, record, expected_version)
            session.add(
                ForecastRow(
                    forecast_id=str(forecast.forecast_id),
                    task_id=str(forecast.task_id),
                    session_id=str(record.session_id),
                    document=forecast.model_dump(mode="json"),
                )
            )
            session.add(
                ForecastVersionRow(
                    version_id=str(original.version_id),
                    forecast_id=str(original.forecast_id),
                    version_number=0,
                    document=original.model_dump(mode="json"),
                )
            )
            session.commit()

    def forecast(self, forecast_id: UUID) -> Forecast | None:
        with self.database.session() as session:
            row = session.get(ForecastRow, str(forecast_id))
            return Forecast.model_validate(row.document) if row else None

    def list_forecasts(self, task_id: UUID) -> list[Forecast]:
        with self.database.session() as session:
            rows = session.scalars(
                select(ForecastRow).where(ForecastRow.task_id == str(task_id))
            ).all()
            return [Forecast.model_validate(row.document) for row in rows]

    def original(self, forecast_id: UUID) -> ForecastVersion | None:
        with self.database.session() as session:
            row = session.scalar(
                select(ForecastVersionRow).where(
                    ForecastVersionRow.forecast_id == str(forecast_id),
                    ForecastVersionRow.version_number == 0,
                )
            )
            return ForecastVersion.model_validate(row.document) if row else None

    def version(self, version_id: UUID) -> ForecastVersion | None:
        with self.database.session() as session:
            row = session.get(ForecastVersionRow, str(version_id))
            return ForecastVersion.model_validate(row.document) if row else None

    def versions(self, forecast_id: UUID) -> list[ForecastVersion]:
        with self.database.session() as session:
            rows = session.scalars(
                select(ForecastVersionRow)
                .where(ForecastVersionRow.forecast_id == str(forecast_id))
                .order_by(ForecastVersionRow.version_number)
            ).all()
            return [ForecastVersion.model_validate(row.document) for row in rows]

    def reference(self, forecast_id: UUID) -> ReferenceAnalysis | None:
        with self.database.session() as session:
            row = session.get(ReferenceAnalysisRow, str(forecast_id))
            return ReferenceAnalysis.model_validate(row.document) if row else None

    def save_reference(
        self, record: DeploymentSession, analysis: ReferenceAnalysis, *, expected_version: int
    ) -> ReferenceAnalysis:
        if record.forecast_id != analysis.forecast_id:
            raise ValueError("reference analysis must match session forecast")
        with self.database.session() as session:
            self._update_session(session, record, expected_version)
            session.add(
                ReferenceAnalysisRow(
                    forecast_id=str(analysis.forecast_id),
                    document=analysis.model_dump(mode="json"),
                )
            )
            session.commit()
        return analysis

    def adjustment(self, adjustment_id: UUID) -> Adjustment | None:
        with self.database.session() as session:
            row = session.get(AdjustmentRow, str(adjustment_id))
            return Adjustment.model_validate(row.document) if row else None

    def adjustments(self, forecast_id: UUID) -> list[Adjustment]:
        with self.database.session() as session:
            rows = session.scalars(
                select(AdjustmentRow).where(AdjustmentRow.forecast_id == str(forecast_id))
            ).all()
            return [Adjustment.model_validate(row.document) for row in rows]

    def save_draft(
        self, record: DeploymentSession, draft: Adjustment, *, expected_version: int
    ) -> Adjustment:
        if (
            record.forecast_id != draft.forecast_id
            or record.pending_adjustment_id != draft.adjustment_id
        ):
            raise ValueError("draft must match pending session adjustment")
        with self.database.session() as session:
            self._update_session(session, record, expected_version)
            session.add(
                AdjustmentRow(
                    adjustment_id=str(draft.adjustment_id),
                    forecast_id=str(draft.forecast_id),
                    status=draft.status,
                    document=draft.model_dump(mode="json"),
                )
            )
            session.commit()
        return draft

    def update_draft(
        self, record: DeploymentSession, draft: Adjustment, *, expected_version: int
    ) -> Adjustment:
        with self.database.session() as session:
            self._update_session(session, record, expected_version)
            row = session.get(AdjustmentRow, str(draft.adjustment_id))
            if row is None or row.status != "draft":
                raise VersionConflict("draft was changed or is no longer pending")
            row.status = draft.status
            row.document = draft.model_dump(mode="json")
            session.commit()
        return draft

    def commit_adjustment(
        self,
        record: DeploymentSession,
        applied: Adjustment,
        version: ForecastVersion,
        *,
        expected_version: int,
    ) -> None:
        if (
            applied.status != "applied"
            or applied.applied_version_id != version.version_id
            or applied.forecast_id != version.forecast_id
            or applied.parent_version_id != version.parent_version_id
            or version.adjustment_id != applied.adjustment_id
            or record.pending_adjustment_id is not None
            or record.current_version_id != version.version_id
            or record.forecast_id != version.forecast_id
        ):
            raise ValueError("applied adjustment, version and session disagree")
        with self.database.session() as session:
            self._update_session(session, record, expected_version)
            row = session.get(AdjustmentRow, str(applied.adjustment_id))
            if row is None or row.status != "draft":
                raise VersionConflict("adjustment is no longer a pending draft")
            row.status = "applied"
            row.document = applied.model_dump(mode="json")
            session.add(
                ForecastVersionRow(
                    version_id=str(version.version_id),
                    forecast_id=str(version.forecast_id),
                    version_number=version.version_number,
                    document=version.model_dump(mode="json"),
                )
            )
            session.commit()
