"""Application-owned Preparation workflow. Agents may propose; only these methods mutate it."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid5

import pandas as pd
from sqlalchemy.exc import IntegrityError

from interactive_forecasting.domain.forecasting import AuxiliaryPolicy
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.models import (
    EvaluationProtocol,
    Event,
    Task,
    TaskDefinition,
    TimeWindow,
)
from interactive_forecasting.domain.preparation import (
    PREPARATION_PATH,
    ColumnMappingDraft,
    ForecastTaskDraft,
    PreparationPlan,
    PreparationRecord,
    PreparationStep,
    WorkingField,
    WorkingSuggestion,
)
from interactive_forecasting.domain.types import Actor, Stage, Topic
from interactive_forecasting.orchestration.state_machine import (
    InvalidTransition,
    VersionConflict,
    WorkflowStateMachine,
)
from interactive_forecasting.services import preparation as calculations
from interactive_forecasting.services.data.core import SOURCE_CLOCK
from interactive_forecasting.services.optimization.space import (
    canonical_research_space,
    space_for_capabilities,
    space_for_output,
)
from interactive_forecasting.services.preparation_working import (
    collect,
    mapping_from_working,
    plan_from_working,
    sync_confirmed,
    task_from_working,
)
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.preparation import PreparationRepository
from interactive_forecasting.storage.sql import MessageRepository, TaskRepository


@dataclass
class PreparationWorkflow:
    tasks: TaskRepository
    records: PreparationRepository
    messages: MessageRepository
    store: ArtifactStore
    machine: WorkflowStateMachine

    def get(self, task_id: UUID) -> PreparationRecord:
        record = self.records.get(task_id)
        if record is None:
            task = self.tasks.get(task_id)
            if task is not None and task.stage == Stage.NEW:
                try:
                    return self.records.create(PreparationRecord(task_id=task_id))
                except IntegrityError:
                    record = self.records.get(task_id)
            if record is None:
                raise ValueError("Preparation record not found")
        return record

    def _task(self, task_id: UUID) -> Task:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError("task not found")
        return task

    def _source_frame(self, record: PreparationRecord) -> pd.DataFrame:
        if record.source_artifact is None:
            raise ValueError("no uploaded dataset")
        extension = Path(record.source_artifact.uri).suffix.lower()
        return calculations.read_tabular(self.store.read_bytes(record.source_artifact), extension)

    def _save(
        self,
        record: PreparationRecord,
        expected_version: int,
        *,
        step: PreparationStep | None = None,
        **updates: object,
    ) -> PreparationRecord:
        if record.version != expected_version:
            raise VersionConflict("stale Preparation draft")
        task = self._task(record.task_id)
        transition: Task | None = None
        event: Event | None = None
        if step is not None:
            if task.stage != Stage.PREPARATION:
                raise InvalidTransition("task is not in Preparation")
            path = PREPARATION_PATH
            if path.index(step) != path.index(record.step) + 1:
                raise InvalidTransition(f"{record.step.value} cannot advance to {step.value}")
            target = step.value
            transition = self.machine.transition_substate(
                task, target, expected_version=task.version
            )
            event = Event(
                task_id=task.task_id,
                source_role=Actor.SYSTEM,
                topic=Topic.SYSTEM,
                message_type="preparation_step",
                payload={"from": record.step.value, "to": target},
                outcome="advanced",
            )
        saved = record.model_copy(
            update={"version": record.version + 1, "step": step or record.step, **updates}
        )
        if any(key in updates for key in ("confirmed_mapping", "plan_confirmed", "frozen")):
            saved = saved.model_copy(update={"working_state": sync_confirmed(saved)})
        return self.records.save(
            PreparationRecord.model_validate(saved.model_dump()),
            expected_version=expected_version,
            task=transition,
            event=event,
        )

    def record_working(
        self,
        task_id: UUID,
        expected_version: int,
        suggestions: tuple[WorkingSuggestion, ...],
        source_message_id: UUID,
    ) -> tuple[PreparationRecord, tuple[str, ...]]:
        """Collect PA information; only explicit user fields edit an eligible typed draft."""
        record = self.get(task_id)
        if record.version != expected_version:
            raise VersionConflict("stale Preparation draft")
        if self._task(task_id).stage not in {Stage.NEW, Stage.PREPARATION} or record.frozen:
            raise InvalidTransition("Preparation working information is frozen")
        working, conflicts = collect(record, suggestions, source_message_id)
        notes = list(conflicts)
        updates: dict[str, object] = {"working_state": working}
        staged = record.model_copy(update=updates)
        explicit = {
            suggestion.field
            for suggestion in suggestions
            if suggestion.status == "user_provided"
            and not any(suggestion.field.value in conflict for conflict in conflicts)
        }
        if record.step == PreparationStep.CONFIRM_COLUMN_MAPPING and record.mapping_draft:
            fields = explicit & {
                WorkingField.TIMESTAMP_COLUMN,
                WorkingField.TARGET_COLUMN,
                WorkingField.TEMPERATURE_COLUMN,
                WorkingField.TEMPERATURE_AVAILABLE,
                WorkingField.OTHER_FEATURES,
            }
            if fields:
                try:
                    updates["mapping_draft"] = mapping_from_working(
                        staged, record.mapping_draft, fields
                    )
                except ValueError as exc:
                    notes.append(f"I've kept that choice for later: {exc}")
        elif record.step == PreparationStep.CONFIRM_PREPARATION and record.plan_draft:
            fields = explicit & {
                WorkingField.TIMEZONE_NAME,
                WorkingField.DUPLICATE_POLICY,
                WorkingField.MISSING_TIMESTAMP_POLICY,
                WorkingField.MISSING_VALUE_POLICY,
                WorkingField.INVALID_ROW_POLICY,
                WorkingField.SORT_CHRONOLOGICALLY,
                WorkingField.NORMALIZE_THOUSANDS_SEPARATORS,
            }
            if fields:
                updates["plan_draft"] = plan_from_working(staged, record.plan_draft, fields)
        elif record.step == PreparationStep.CONFIGURE_FORECAST_TASK and record.task_draft:
            fields = explicit & {
                WorkingField.DELTA,
                WorkingField.HORIZON,
                WorkingField.TIME_UNIT,
                WorkingField.OUTPUT,
                WorkingField.OBJECTIVE_ID,
                WorkingField.METRIC_SPEC,
                WorkingField.TEMPERATURE_POLICY,
                WorkingField.OTHER_POLICIES,
                WorkingField.TRAIN_FRACTION,
                WorkingField.VALIDATION_FRACTION,
                WorkingField.TEST_FRACTION,
                WorkingField.SEED,
            }
            if fields:
                try:
                    draft = task_from_working(staged, record.task_draft, fields)
                    self._validate_task_draft(record, draft)
                except ValueError as exc:
                    notes.append(f"I've kept that choice for later: {exc}")
                else:
                    updates["task_draft"] = draft
        if all(getattr(record, key) == value for key, value in updates.items()):
            return record, tuple(notes)
        return self._save(
            record,
            expected_version,
            working_state=working,
            mapping_draft=updates.get("mapping_draft", record.mapping_draft),
            plan_draft=updates.get("plan_draft", record.plan_draft),
            task_draft=updates.get("task_draft", record.task_draft),
        ), tuple(notes)

    def upload(
        self,
        task_id: UUID,
        expected_version: int,
        filename: str,
        content: bytes,
        action_id: UUID,
    ) -> PreparationRecord:
        record = self.get(task_id)
        dataset_id = uuid5(task_id, str(action_id))
        if record.source is not None and record.source.dataset_id == dataset_id:
            if (
                record.source_artifact
                and record.source_artifact.sha256
                == __import__("hashlib").sha256(content).hexdigest()
            ):
                return record
            raise ValueError("action ID reused for different upload content")
        if record.step != PreparationStep.UPLOAD_DATASET:
            raise InvalidTransition("upload is only available before schema inspection")
        if record.version != expected_version:
            raise VersionConflict("stale Preparation draft")
        extension = Path(filename).suffix.lower()
        calculations.read_tabular(content, extension)
        reference = self.store.put_bytes(
            f"uploads/{task_id}/{dataset_id}/original{extension}", content
        )
        source = calculations.source_for_upload(
            dataset_id, filename, reference.uri, reference.sha256
        )
        task = self._task(task_id)
        if task.stage == Stage.NEW:
            started = self.machine.transition_stage(
                task, Stage.PREPARATION, expected_version=task.version
            )
            self.tasks.save_transition(
                started,
                Event(
                    task_id=task_id,
                    source_role=Actor.SYSTEM,
                    topic=Topic.SYSTEM,
                    message_type="start_preparation",
                    outcome="started",
                ),
                expected_version=task.version,
            )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.INSPECT_SCHEMA,
            source=source,
            source_artifact=reference,
            original_filename=Path(filename).name,
        )

    def inspect(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.INSPECT_SCHEMA:
            raise InvalidTransition("schema inspection is not next")
        schema = calculations.inspect_schema(self._source_frame(record))
        return self._save(
            record,
            expected_version,
            step=PreparationStep.PROPOSE_COLUMN_MAPPING,
            schema_inspection=schema,
        )

    def propose_mapping(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.PROPOSE_COLUMN_MAPPING
            or record.schema_inspection is None
        ):
            raise InvalidTransition("mapping proposal is not next")
        draft = calculations.propose_mapping(record.schema_inspection)
        draft = mapping_from_working(record, draft)
        return self._save(
            record,
            expected_version,
            step=PreparationStep.CONFIRM_COLUMN_MAPPING,
            mapping_draft=draft,
        )

    def update_mapping(
        self, task_id: UUID, expected_version: int, draft: ColumnMappingDraft
    ) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.CONFIRM_COLUMN_MAPPING:
            raise InvalidTransition("mapping is not editable now")
        return self._save(record, expected_version, mapping_draft=draft)

    def confirm_mapping(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.CONFIRM_COLUMN_MAPPING
            or record.source is None
            or record.schema_inspection is None
            or record.mapping_draft is None
        ):
            raise InvalidTransition("mapping proposal is not ready")
        mapping = calculations.confirm_mapping(
            record.mapping_draft, record.schema_inspection, record.source.dataset_id
        )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.ANALYZE_DATA_QUALITY,
            confirmed_mapping=mapping,
        )

    def analyze_quality(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.ANALYZE_DATA_QUALITY
            or record.confirmed_mapping is None
            or record.schema_inspection is None
        ):
            raise InvalidTransition("confirmed mapping required")
        quality = calculations.analyze_quality(
            self._source_frame(record), record.confirmed_mapping, record.schema_inspection
        )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.PROPOSE_PREPARATION,
            quality=quality,
        )

    def propose_plan(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.PROPOSE_PREPARATION or record.quality is None:
            raise InvalidTransition("quality report required")
        plan = calculations.propose_plan(record.quality)
        plan = plan_from_working(record, plan)
        return self._save(
            record,
            expected_version,
            step=PreparationStep.CONFIRM_PREPARATION,
            plan_draft=plan,
        )

    def update_plan(
        self, task_id: UUID, expected_version: int, plan: PreparationPlan
    ) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.CONFIRM_PREPARATION:
            raise InvalidTransition("plan is not editable now")
        return self._save(record, expected_version, plan_draft=plan, plan_confirmed=False)

    def confirm_plan(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.CONFIRM_PREPARATION or record.plan_draft is None:
            raise InvalidTransition("plan proposal required")
        # Preflight before approval; requires user to explicitly choose every needed policy.
        if record.confirmed_mapping is None:
            raise InvalidTransition("confirmed mapping required")
        calculations._prepare_frame(
            self._source_frame(record), record.confirmed_mapping, record.plan_draft
        )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.APPLY_PREPARATION,
            plan_confirmed=True,
        )

    def apply(self, task_id: UUID, expected_version: int, action_id: UUID) -> PreparationRecord:
        record = self.get(task_id)
        if record.prepared and record.prepared.dataset.snapshot_id == uuid5(
            task_id, str(action_id)
        ):
            return record
        if (
            record.step != PreparationStep.APPLY_PREPARATION
            or not record.plan_confirmed
            or record.confirmed_mapping is None
            or record.plan_draft is None
        ):
            raise InvalidTransition("approved plan required")
        if record.version != expected_version:
            raise VersionConflict("stale Preparation draft")
        prepared = calculations.apply_plan(
            self._source_frame(record),
            record.confirmed_mapping,
            record.plan_draft,
            self.store,
            task_id,
            uuid5(task_id, str(action_id)),
        )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.GENERATE_DATASET_OVERVIEW,
            prepared=prepared,
            capabilities=prepared.capabilities,
        )

    def generate_overview(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.GENERATE_DATASET_OVERVIEW
            or record.prepared is None
            or record.quality is None
        ):
            raise InvalidTransition("prepared dataset required")
        summary = calculations.overview(record.prepared, self.store, record.quality)
        policies = {
            name: AuxiliaryPolicy(kind="observed", role="other")
            for name in record.prepared.capabilities.available_other_features
        }
        temp = (
            next(
                (
                    name
                    for name, role in record.confirmed_mapping.auxiliary_roles.items()
                    if role == "temperature"
                ),
                None,
            )
            if record.confirmed_mapping
            else None
        )
        draft = ForecastTaskDraft(
            temperature_policy=AuxiliaryPolicy(kind="observed", role="temperature")
            if temp
            else None,
            other_policies=policies,
        )
        draft = task_from_working(record, draft)
        self._validate_task_draft(record, draft)
        return self._save(
            record,
            expected_version,
            step=PreparationStep.CONFIGURE_FORECAST_TASK,
            overview=summary,
            task_draft=draft,
        )

    def update_task_draft(
        self, task_id: UUID, expected_version: int, draft: ForecastTaskDraft
    ) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.CONFIGURE_FORECAST_TASK:
            raise InvalidTransition("forecast task is not editable now")
        self._validate_task_draft(record, draft)
        return self._save(record, expected_version, task_draft=draft)

    @staticmethod
    def _validate_task_draft(record: PreparationRecord, draft: ForecastTaskDraft) -> None:
        if (
            draft.metric_spec is not None
            and draft.metric_spec.kind == "weighted"
            and record.prepared is not None
            and draft.metric_spec.timezone_name != record.prepared.dataset.timezone_name
        ):
            raise ValueError("weighted metric timezone must match the frozen dataset timezone")
        if record.capabilities is None:
            raise ValueError("dataset capabilities unavailable")
        if not record.capabilities.has_temperature and draft.temperature_policy is not None:
            raise ValueError("dataset has no temperature; temperature policy is forbidden")
        if record.capabilities.has_temperature and (
            draft.temperature_policy is None or draft.temperature_policy.role != "temperature"
        ):
            raise ValueError("mapped temperature requires an explicit availability policy")
        if draft.temperature_policy is not None and draft.temperature_policy.kind == "forecast":
            temperature_column = (
                next(
                    (
                        name
                        for name, role in record.confirmed_mapping.auxiliary_roles.items()
                        if role == "temperature"
                    ),
                    None,
                )
                if record.confirmed_mapping
                else None
            )
            issue_column = f"{temperature_column}__available_at"
            if record.prepared is None or issue_column not in record.prepared.columns:
                raise ValueError("forecast-product temperature needs a prepared issue-time column")
        for name, policy in draft.other_policies.items():
            if policy.kind == "forecast" and (
                record.prepared is None or f"{name}__available_at" not in record.prepared.columns
            ):
                raise ValueError(
                    f"forecast-product auxiliary {name} needs a prepared issue-time column"
                )
        if set(draft.other_policies) != set(record.capabilities.available_other_features):
            raise ValueError("other-feature policies must match confirmed mapping")
        if any(policy.role != "other" for policy in draft.other_policies.values()):
            raise ValueError("other-feature policy role mismatch")
        if draft.horizon != 1:
            raise ValueError("current forecasting adapters support only H=1")
        if draft.time_unit == "hours" and record.prepared is not None:
            freq = pd.Timedelta(record.prepared.dataset.frequency)
            if pd.Timedelta(hours=draft.delta) % freq:
                raise ValueError("hour lead must align to sampling grid")

    def review(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if record.step != PreparationStep.CONFIGURE_FORECAST_TASK or record.task_draft is None:
            raise InvalidTransition("task draft required")
        self._validate_task_draft(record, record.task_draft)
        return self._save(record, expected_version, step=PreparationStep.REVIEW_PREPARATION)

    def validate_review(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.REVIEW_PREPARATION
            or record.prepared is None
            or record.overview is None
            or record.task_draft is None
            or record.confirmed_mapping is None
        ):
            raise InvalidTransition("Preparation review is incomplete")
        self._validate_task_draft(record, record.task_draft)
        return self._save(record, expected_version, step=PreparationStep.CONFIRM_TASK)

    def confirm_task(self, task_id: UUID, expected_version: int) -> PreparationRecord:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.CONFIRM_TASK
            or record.prepared is None
            or record.task_draft is None
            or record.confirmed_mapping is None
        ):
            raise InvalidTransition("final review is not ready")
        self._validate_task_draft(record, record.task_draft)
        draft = record.task_draft
        if draft.metric_spec is None:
            if draft.objective_id == "weighted_mae":
                raise ValueError("weighted MAE requires an explicit MetricSpec")
            if draft.objective_id == "asymmetric_mae":
                raise ValueError("asymmetric MAE requires an explicit MetricSpec")
            assert draft.objective_id in ("mae", "mape", "crps")
            metric_spec = MetricSpec(kind="standard", base_metric=draft.objective_id)
        else:
            metric_spec = draft.metric_spec
        times = pd.to_datetime(
            pd.read_csv(BytesIO(self.store.read_bytes(record.prepared.dataset.artifact)))[
                "timestamp"
            ],
            utc=record.prepared.dataset.timezone_name != SOURCE_CLOCK,
        )
        n = len(times)
        lead_steps = (
            draft.delta
            if draft.time_unit == "samples"
            else int(
                pd.Timedelta(hours=draft.delta) / pd.Timedelta(record.prepared.dataset.frequency)
            )
        )
        train_end = int(n * draft.train_fraction)
        val_end = int(n * (draft.train_fraction + draft.validation_fraction))
        val_start = train_end + lead_steps
        test_start = val_end + lead_steps
        if train_end <= lead_steps + 2 or val_start + 2 >= val_end or test_start + 2 >= n:
            raise ValueError("dataset is too short for the declared split, lead and horizon")
        step = pd.Timedelta(record.prepared.dataset.frequency)
        protocol = EvaluationProtocol(
            train=TimeWindow(
                start=times.iloc[lead_steps].to_pydatetime(),
                end=times.iloc[train_end].to_pydatetime(),
            ),
            validation=TimeWindow(
                start=times.iloc[val_start].to_pydatetime(), end=times.iloc[val_end].to_pydatetime()
            ),
            test=TimeWindow(
                start=times.iloc[test_start].to_pydatetime(),
                end=(times.iloc[-1] + step).to_pydatetime(),
            ),
            seed=draft.seed,
            metric_ids=[draft.objective_id],
            metric_spec=metric_spec,
            protocol_version="preparation-4b-v1",
        )
        policies = dict(draft.other_policies)
        temperature_column = next(
            (
                name
                for name, role in record.confirmed_mapping.auxiliary_roles.items()
                if role == "temperature"
            ),
            None,
        )
        if temperature_column and draft.temperature_policy:
            policies[temperature_column] = draft.temperature_policy
        mapping = record.confirmed_mapping.model_copy(update={"availability": policies})
        capability = record.prepared.capabilities
        groups = list(capability.supported_feature_groups)
        if (
            temperature_column
            and draft.temperature_policy
            and draft.temperature_policy.kind != "observed"
        ):
            groups.append("future_temperature")
        if any(
            policy.kind in {"known_ahead", "forecast"}
            for name, policy in draft.other_policies.items()
        ):
            groups.append("future_other_features")
        final_capabilities = capability.model_copy(
            update={
                "known_ahead_candidates": tuple(
                    name for name, policy in policies.items() if policy.kind == "known_ahead"
                ),
                "supported_feature_groups": tuple(groups),
            }
        )
        definition = TaskDefinition(
            task_id=task_id,
            dataset_id=record.prepared.dataset.dataset_id,
            delta=draft.delta,
            horizon=draft.horizon,
            time_unit=draft.time_unit,
            timezone_name=record.prepared.dataset.timezone_name,
            objective_id=draft.objective_id,
            metric_spec=metric_spec,
            forecast_output=draft.output,
            auxiliary_policies=policies,
            target_range_start=times.iloc[lead_steps].to_pydatetime(),
            target_range_end=times.iloc[-1].to_pydatetime(),
        )
        return self._save(
            record,
            expected_version,
            step=PreparationStep.PREPARATION_READY,
            confirmed_mapping=mapping,
            capabilities=final_capabilities,
            definition=definition,
            protocol=protocol,
            effective_search_space=space_for_output(
                space_for_capabilities(canonical_research_space(), record.prepared.capabilities),
                draft.output,
            ),
            frozen=True,
        )

    def continue_to_training(self, task_id: UUID) -> Task:
        record = self.get(task_id)
        if (
            record.step != PreparationStep.PREPARATION_READY
            or not record.frozen
            or record.definition is None
            or record.protocol is None
            or record.prepared is None
        ):
            raise InvalidTransition("complete frozen Preparation is required")
        task = self._task(task_id)
        advanced = self.machine.transition_stage(
            task, Stage.OPTIMIZATION, expected_version=task.version
        ).model_copy(update={"definition_id": record.definition.definition_id})
        self.tasks.save_transition(
            advanced,
            Event(
                task_id=task_id,
                source_role=Actor.SYSTEM,
                topic=Topic.SYSTEM,
                message_type="start_training_evaluation",
                outcome="started",
                payload={
                    "snapshot_id": str(record.prepared.dataset.snapshot_id),
                    "definition_id": str(record.definition.definition_id),
                },
            ),
            expected_version=task.version,
        )
        return advanced
