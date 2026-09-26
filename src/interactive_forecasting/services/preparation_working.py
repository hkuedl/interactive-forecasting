"""Task-scoped Preparation working slots; authoritative drafts remain in PreparationWorkflow."""

from __future__ import annotations

import json
from uuid import UUID

from pydantic import TypeAdapter

from interactive_forecasting.domain.forecasting import AuxiliaryPolicy
from interactive_forecasting.domain.metric_spec import MetricSpec
from interactive_forecasting.domain.preparation import (
    ColumnMappingDraft,
    ForecastTaskDraft,
    PreparationPlan,
    PreparationRecord,
    PreparationWorkingState,
    WorkingEntry,
    WorkingField,
    WorkingSuggestion,
)

_MAPPING = {
    WorkingField.TIMESTAMP_COLUMN,
    WorkingField.TARGET_COLUMN,
    WorkingField.TEMPERATURE_COLUMN,
    WorkingField.TEMPERATURE_AVAILABLE,
    WorkingField.OTHER_FEATURES,
}
_PLAN = {
    WorkingField.TIMEZONE_NAME,
    WorkingField.DUPLICATE_POLICY,
    WorkingField.MISSING_TIMESTAMP_POLICY,
    WorkingField.MISSING_VALUE_POLICY,
    WorkingField.INVALID_ROW_POLICY,
    WorkingField.SORT_CHRONOLOGICALLY,
    WorkingField.NORMALIZE_THOUSANDS_SEPARATORS,
}
_TASK = {
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
_REQUIRED = (
    WorkingField.TIMESTAMP_COLUMN,
    WorkingField.TARGET_COLUMN,
    WorkingField.TEMPERATURE_AVAILABLE,
    WorkingField.TIMEZONE_NAME,
    WorkingField.DUPLICATE_POLICY,
    WorkingField.MISSING_TIMESTAMP_POLICY,
    WorkingField.MISSING_VALUE_POLICY,
    WorkingField.INVALID_ROW_POLICY,
    WorkingField.SORT_CHRONOLOGICALLY,
    WorkingField.NORMALIZE_THOUSANDS_SEPARATORS,
    WorkingField.DELTA,
    WorkingField.HORIZON,
    WorkingField.TIME_UNIT,
    WorkingField.OUTPUT,
    WorkingField.OBJECTIVE_ID,
    WorkingField.TRAIN_FRACTION,
    WorkingField.VALIDATION_FRACTION,
    WorkingField.TEST_FRACTION,
    WorkingField.SEED,
)
_CHOICES: dict[WorkingField, set[str]] = {
    WorkingField.DUPLICATE_POLICY: {"reject", "first", "last", "mean"},
    WorkingField.MISSING_TIMESTAMP_POLICY: {"reject", "interpolate", "forward_fill"},
    WorkingField.MISSING_VALUE_POLICY: {"reject", "interpolate", "forward_fill", "drop"},
    WorkingField.INVALID_ROW_POLICY: {"reject", "drop"},
    WorkingField.TIME_UNIT: {"samples", "hours"},
    WorkingField.OUTPUT: {"point", "quantile"},
    WorkingField.OBJECTIVE_ID: {"mae", "mape", "crps", "weighted_mae", "asymmetric_mae"},
}


def normalize(record: PreparationRecord, field: WorkingField, value: str) -> str:
    """Validate one advisory value with the existing domain choices."""
    value = value.strip()
    if not value:
        raise ValueError(f"{field.value} cannot be empty")
    if field in {
        WorkingField.TIMESTAMP_COLUMN,
        WorkingField.TARGET_COLUMN,
        WorkingField.TEMPERATURE_COLUMN,
    }:
        if value.lower() == "none" and field == WorkingField.TEMPERATURE_COLUMN:
            return "none"
        schema = record.schema_inspection
        if schema is not None and value not in {col.name for col in schema.columns}:
            raise ValueError(f"{value} is not an inspected column")
        return value
    if field in _CHOICES:
        lowered = value.lower()
        if lowered not in _CHOICES[field]:
            raise ValueError(f"unsupported {field.value}: {value}")
        return lowered
    if field in {
        WorkingField.TEMPERATURE_AVAILABLE,
        WorkingField.SORT_CHRONOLOGICALLY,
        WorkingField.NORMALIZE_THOUSANDS_SEPARATORS,
    }:
        if value.lower() not in {"true", "false"}:
            raise ValueError(f"{field.value} needs true or false")
        return value.lower()
    if field in {WorkingField.DELTA, WorkingField.HORIZON, WorkingField.SEED}:
        number = int(value)
        if number < (0 if field == WorkingField.SEED else 1) or number > 2**32 - 1:
            raise ValueError(f"{field.value} is out of range")
        return str(number)
    if field in {
        WorkingField.TRAIN_FRACTION,
        WorkingField.VALIDATION_FRACTION,
        WorkingField.TEST_FRACTION,
    }:
        fraction = float(value)
        if not 0 < fraction < 1:
            raise ValueError(f"{field.value} must be between zero and one")
        return str(fraction)
    if field == WorkingField.METRIC_SPEC:
        return json.dumps(
            MetricSpec.model_validate_json(value).model_dump(mode="json"), sort_keys=True
        )
    if field == WorkingField.TEMPERATURE_POLICY:
        policy = AuxiliaryPolicy.model_validate_json(value)
        if policy.role != "temperature":
            raise ValueError("temperature policy has wrong role")
        return json.dumps(policy.model_dump(mode="json"), sort_keys=True)
    if field == WorkingField.OTHER_POLICIES:
        policies = TypeAdapter(dict[str, AuxiliaryPolicy]).validate_json(value)
        if any(policy.role != "other" for policy in policies.values()):
            raise ValueError("other-feature policy has wrong role")
        return json.dumps(
            {name: policy.model_dump(mode="json") for name, policy in policies.items()},
            sort_keys=True,
        )
    if field == WorkingField.OTHER_FEATURES:
        names = TypeAdapter(list[str]).validate_json(value)
        if len(names) != len(set(names)):
            raise ValueError("other-feature names must be unique")
        if record.schema_inspection is not None:
            known = {col.name for col in record.schema_inspection.columns}
            if not set(names) <= known:
                raise ValueError("other feature is not an inspected column")
        return json.dumps(names)
    return value[:2000]


def _confirmed(record: PreparationRecord) -> dict[WorkingField, str]:
    result: dict[WorkingField, str] = {}
    mapping = record.confirmed_mapping
    if mapping is not None:
        temperature = next(
            (name for name, role in mapping.auxiliary_roles.items() if role == "temperature"),
            None,
        )
        result.update(
            {
                WorkingField.TIMESTAMP_COLUMN: mapping.timestamp_column,
                WorkingField.TARGET_COLUMN: mapping.target_column,
                WorkingField.TEMPERATURE_COLUMN: temperature or "none",
                WorkingField.TEMPERATURE_AVAILABLE: str(temperature is not None).lower(),
                WorkingField.OTHER_FEATURES: json.dumps(
                    [name for name, role in mapping.auxiliary_roles.items() if role == "other"]
                ),
            }
        )
    if record.plan_confirmed and record.plan_draft:
        plan = record.plan_draft
        for field in _PLAN:
            value = getattr(plan, field.value)
            result[field] = str(value).lower() if isinstance(value, bool) else str(value)
    if record.frozen and record.task_draft:
        task = record.task_draft
        for field in _TASK:
            value = getattr(task, field.value if field != WorkingField.OUTPUT else "output")
            if field == WorkingField.OUTPUT:
                result[field] = task.output.representation
            elif value is not None and hasattr(value, "model_dump"):
                result[field] = json.dumps(value.model_dump(mode="json"), sort_keys=True)
            elif isinstance(value, dict):
                result[field] = json.dumps(
                    {name: policy.model_dump(mode="json") for name, policy in value.items()},
                    sort_keys=True,
                )
            elif value is not None:
                result[field] = str(value).lower()
    return result


def collect(
    record: PreparationRecord,
    suggestions: tuple[WorkingSuggestion, ...],
    source_message_id: UUID,
) -> tuple[PreparationWorkingState, tuple[str, ...]]:
    """Merge PA suggestions without ever replacing confirmed authoritative values."""
    current = {entry.field: entry for entry in record.working_state.entries}
    confirmed = _confirmed(record)
    conflicts: list[str] = []
    seen: set[WorkingField] = set()
    for suggestion in suggestions:
        field = suggestion.field
        if field in seen:
            raise ValueError(f"duplicate working field: {field.value}")
        seen.add(field)
        value = normalize(record, field, suggestion.value)
        if field in confirmed and confirmed[field] != value:
            conflicts.append(
                f"{field.value} is already confirmed as {confirmed[field]}; "
                "the current workflow cannot change that confirmed choice."
            )
            continue
        if field in confirmed:
            current[field] = WorkingEntry(field=field, value=confirmed[field], status="confirmed")
            continue
        current[field] = WorkingEntry(
            field=field,
            value=value,
            status=suggestion.status,
            source_message_id=source_message_id,
        )
    return PreparationWorkingState(
        entries=tuple(current.values()),
        conflicts=tuple((list(record.working_state.conflicts) + conflicts)[-10:]),
    ), tuple(conflicts)


def sync_confirmed(record: PreparationRecord) -> PreparationWorkingState:
    entries = {entry.field: entry for entry in record.working_state.entries}
    for field, value in _confirmed(record).items():
        entries[field] = WorkingEntry(field=field, value=value, status="confirmed")
    return PreparationWorkingState(
        entries=tuple(entries.values()), conflicts=record.working_state.conflicts
    )


def projection(record: PreparationRecord) -> dict[str, object]:
    """Authoritative confirmation > current draft > advisory working entry."""
    resolved = {entry.field: entry for entry in record.working_state.entries}
    if record.mapping_draft:
        draft = record.mapping_draft
        for field in _MAPPING:
            value = getattr(draft, field.value)
            if value is not None:
                if isinstance(value, tuple):
                    value = json.dumps(value)
                resolved[field] = WorkingEntry(
                    field=field,
                    value=str(value).lower() if isinstance(value, bool) else str(value),
                    status="proposed",
                )
    if record.plan_draft:
        for field in _PLAN:
            resolved[field] = WorkingEntry(
                field=field,
                value=str(getattr(record.plan_draft, field.value)).lower(),
                status="proposed",
            )
    if record.task_draft:
        for field in _TASK:
            value = getattr(
                record.task_draft, field.value if field != WorkingField.OUTPUT else "output"
            )
            if field == WorkingField.OUTPUT:
                value = record.task_draft.output.representation
            if value is not None:
                if hasattr(value, "model_dump"):
                    value = json.dumps(value.model_dump(mode="json"), sort_keys=True)
                elif isinstance(value, dict):
                    value = json.dumps(
                        {name: policy.model_dump(mode="json") for name, policy in value.items()},
                        sort_keys=True,
                    )
                resolved[field] = WorkingEntry(
                    field=field,
                    value=str(value).lower() if isinstance(value, bool) else str(value),
                    status="proposed",
                )
    for field, value in _confirmed(record).items():
        resolved[field] = WorkingEntry(field=field, value=value, status="confirmed")
    required = list(_REQUIRED)
    temperature = resolved.get(WorkingField.TEMPERATURE_AVAILABLE)
    if temperature is not None and temperature.value == "true":
        required.append(WorkingField.TEMPERATURE_POLICY)
    metric = resolved.get(WorkingField.OBJECTIVE_ID)
    if metric is not None and metric.value in {"weighted_mae", "asymmetric_mae"}:
        required.append(WorkingField.METRIC_SPEC)
    other = resolved.get(WorkingField.OTHER_FEATURES)
    if other is not None and json.loads(other.value):
        required.append(WorkingField.OTHER_POLICIES)
    missing = [field.value for field in required if field not in resolved]
    awaiting = [
        field.value
        for field in required
        if field in resolved and resolved[field].status != "confirmed"
    ]
    return {
        "collected": [entry.model_dump(mode="json") for entry in resolved.values()],
        "missing": missing,
        "awaiting_confirmation": awaiting,
        "conflicts": record.working_state.conflicts,
    }


def _selected(record: PreparationRecord, fields: set[WorkingField]) -> dict[WorkingField, str]:
    return {
        entry.field: entry.value
        for entry in record.working_state.entries
        if entry.field in fields and entry.status == "user_provided"
    }


def mapping_from_working(
    record: PreparationRecord, draft: ColumnMappingDraft, fields: set[WorkingField] | None = None
) -> ColumnMappingDraft:
    values = _selected(record, _MAPPING if fields is None else fields)
    data = draft.model_dump()
    for field, value in values.items():
        if field == WorkingField.OTHER_FEATURES:
            data[field.value] = json.loads(value)
        elif field == WorkingField.TEMPERATURE_AVAILABLE:
            data[field.value] = value == "true"
            if value == "false":
                data["temperature_column"] = None
        else:
            data[field.value] = None if value == "none" else value
            if field == WorkingField.TEMPERATURE_COLUMN:
                data["temperature_available"] = value != "none"
    chosen = {
        data.get("timestamp_column"),
        data.get("target_column"),
        data.get("temperature_column"),
    }
    data["other_features"] = [name for name in data["other_features"] if name not in chosen]
    return ColumnMappingDraft.model_validate(data)


def plan_from_working(
    record: PreparationRecord, plan: PreparationPlan, fields: set[WorkingField] | None = None
) -> PreparationPlan:
    data = plan.model_dump()
    for field, value in _selected(record, _PLAN if fields is None else fields).items():
        data[field.value] = (
            value == "true"
            if field
            in {WorkingField.SORT_CHRONOLOGICALLY, WorkingField.NORMALIZE_THOUSANDS_SEPARATORS}
            else value
        )
    return PreparationPlan.model_validate(data)


def task_from_working(
    record: PreparationRecord, draft: ForecastTaskDraft, fields: set[WorkingField] | None = None
) -> ForecastTaskDraft:
    data = draft.model_dump(mode="json")
    values = _selected(record, _TASK if fields is None else fields)
    for field, value in values.items():
        if field == WorkingField.OUTPUT:
            data["output"] = {
                "representation": value,
                "quantile_levels": [0.1, 0.5, 0.9] if value == "quantile" else [],
            }
            if WorkingField.OBJECTIVE_ID not in values and value == "quantile":
                data["objective_id"] = "crps"
            elif WorkingField.OBJECTIVE_ID not in values and value == "point":
                data["objective_id"] = "mae"
        elif field in {
            WorkingField.METRIC_SPEC,
            WorkingField.TEMPERATURE_POLICY,
            WorkingField.OTHER_POLICIES,
        }:
            data[field.value] = json.loads(value)
        elif field in {WorkingField.DELTA, WorkingField.HORIZON, WorkingField.SEED}:
            data[field.value] = int(value)
        elif field in {
            WorkingField.TRAIN_FRACTION,
            WorkingField.VALIDATION_FRACTION,
            WorkingField.TEST_FRACTION,
        }:
            data[field.value] = float(value)
        else:
            data[field.value] = value
    if WorkingField.OBJECTIVE_ID in values and values[WorkingField.OBJECTIVE_ID] in {
        "mae",
        "mape",
        "crps",
    }:
        data["metric_spec"] = None
    return ForecastTaskDraft.model_validate(data)
