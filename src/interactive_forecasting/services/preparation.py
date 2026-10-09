"""Deterministic Preparation calculations; never uses agent text as data."""

from __future__ import annotations

from io import BytesIO
from typing import Literal, cast
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from interactive_forecasting.domain.forecasting import AuxiliaryPolicy
from interactive_forecasting.domain.models import ColumnMapping, DatasetSnapshot, DatasetSource
from interactive_forecasting.domain.preparation import (
    ChartSeries,
    ColumnMappingDraft,
    ColumnProfile,
    DataQualityReport,
    DatasetCapabilities,
    DatasetOverview,
    PreparationPlan,
    PreparedSnapshot,
    SchemaInspection,
)
from interactive_forecasting.services.data.core import SOURCE_CLOCK, NormalizedSeries
from interactive_forecasting.storage.artifacts import ArtifactStore

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_COLUMNS = 128
MAX_PREPARED_ROWS = 2_000_000
_CHART_POINTS = 400


def read_tabular(content: bytes, extension: str) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("upload exceeds 32 MiB limit")
    if extension == ".csv":
        frame = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=True)
    elif extension == ".parquet":
        frame = pd.read_parquet(BytesIO(content))
    else:
        raise ValueError("supported uploads are CSV and Parquet")
    if len(frame.columns) > MAX_COLUMNS:
        raise ValueError("dataset exceeds 128-column Preparation limit")
    if frame.empty or not frame.columns.is_unique or any(not str(x).strip() for x in frame.columns):
        raise ValueError("dataset must contain rows and distinct, nonempty column names")
    return frame.rename(columns={name: str(name) for name in frame.columns})


def _thousands_mask(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.str.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?").fillna(False)


def _numeric(values: pd.Series, *, strip_thousands: bool = True) -> pd.Series:
    text = values.astype("string").str.strip()
    if strip_thousands:
        mask = _thousands_mask(values)
        text = text.where(~mask, text.str.replace(",", "", regex=False))
    return pd.to_numeric(text, errors="coerce")


def _times(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=False, format="mixed")


def _time_clues(values: pd.Series) -> tuple[bool | None, bool | None, tuple[str, ...], int | None]:
    parsed = _times(values).dropna()
    if len(parsed) < 2:
        return None, None, (), None
    ordered = bool(parsed.is_monotonic_increasing)
    unique = bool(parsed.is_unique)
    deltas = parsed.sort_values().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return ordered, unique, (), None
    modes = positive.value_counts().head(3).index
    primary = pd.Timedelta(modes[0])
    ordered_unique = pd.DatetimeIndex(parsed.drop_duplicates().sort_values())
    gaps = ordered_unique.to_series().diff().dropna()
    missing = sum(max(int(delta / primary) - 1, 0) for delta in gaps)
    return ordered, unique, tuple(str(pd.Timedelta(x)) for x in modes), missing


def inspect_schema(frame: pd.DataFrame) -> SchemaInspection:
    profiles: list[ColumnProfile] = []
    time_candidates: list[str] = []
    for name in frame.columns:
        values = frame[name]
        nonmissing = values.dropna().astype("string").str.strip()
        nonmissing = nonmissing[nonmissing.ne("")]
        count = len(nonmissing)
        numeric = _numeric(nonmissing)
        # Numeric columns parsed as epoch nanoseconds are not evidence of dates.
        date_like = nonmissing.str.contains(r"[-/:T ]", regex=True).mean() if count else 0.0
        time_ratio = float(_times(nonmissing).notna().mean()) if count and date_like > 0.5 else 0.0
        numeric_ratio = float(numeric.notna().mean()) if count else 0.0
        unique = int(nonmissing.nunique())
        identifier = bool(
            count > 20
            and unique / count > 0.98
            and time_ratio < 0.2
            and (numeric_ratio < 0.2 or str(name).lower().endswith(("id", "_id")))
        )
        binary = unique <= 2 and count > 0
        constant = unique <= 1
        hints = []
        if time_ratio > 0.95:
            hints.append("datetime")
            time_candidates.append(str(name))
        if numeric_ratio > 0.95:
            hints.append("numeric")
        if binary:
            hints.append("binary")
        if identifier:
            hints.append("identifier")
        if constant:
            hints.append("constant")
        profiles.append(
            ColumnProfile(
                name=str(name),
                physical_dtype=str(values.dtype),
                missing_count=int(
                    values.isna().sum()
                    + values.astype("string").str.strip().eq("").fillna(False).sum()
                ),
                unique_count=unique,
                sample_values=tuple(str(x)[:80] for x in nonmissing.head(5)),
                datetime_parse_ratio=time_ratio,
                numeric_parse_ratio=numeric_ratio,
                binary=binary,
                identifier_like=identifier,
                constant=constant,
                semantic_hints=tuple(hints),
            )
        )
    best = max(
        (p for p in profiles if p.name in time_candidates),
        key=lambda p: p.datetime_parse_ratio,
        default=None,
    )
    ordered, timestamp_unique, intervals, missing = (
        _time_clues(frame[best.name]) if best else (None, None, (), None)
    )
    return SchemaInspection(
        row_count=len(frame),
        columns=tuple(profiles),
        plausible_time_columns=tuple(time_candidates),
        interval_candidates=intervals,
        time_ordered=ordered,
        timestamp_unique=timestamp_unique,
        missing_timestamp_count=missing,
    )


def propose_mapping(schema: SchemaInspection) -> ColumnMappingDraft:
    def select(role: str, candidates: list[ColumnProfile]) -> tuple[str | None, bool]:
        scored: list[tuple[int, str]] = []
        for column in candidates:
            name = column.name.lower().replace("_", "")
            score = 0
            if role == "time":
                score += 3 if column.datetime_parse_ratio >= 0.95 else 0
                score += (
                    3 if any(x in name for x in ("timestamp", "datetime", "date", "time")) else 0
                )
            elif role == "load":
                score += 3 if column.numeric_parse_ratio >= 0.95 else 0
                score += (
                    4 if any(x in name for x in ("load", "demand", "power", "consumption")) else 0
                )
                score -= 3 if column.binary or column.constant or column.identifier_like else 0
            else:
                score += 3 if column.numeric_parse_ratio >= 0.95 else 0
                score += (
                    4
                    if any(x in name for x in ("temperature", "temp", "drybulb")) or name == "t"
                    else 0
                )
                score -= 3 if column.binary or column.constant else 0
            if score > 3:
                scored.append((score, column.name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored or len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None, bool(scored)
        return scored[0][1], False

    columns = list(schema.columns)
    time, amb_time = select("time", columns)
    load, amb_load = select(
        "load", [p for p in columns if p.name != time and p.datetime_parse_ratio < 0.5]
    )
    temperature, amb_temp = select("temp", [p for p in columns if p.name not in {time, load}])
    # A generic numeric column alone is not enough to declare temperature.
    if temperature:
        name = temperature.lower()
        if not (any(x in name for x in ("temperature", "temp", "drybulb")) or name == "t"):
            temperature = None
    chosen = {time, load, temperature}
    other: list[str] = []
    excluded: list[str] = []
    for column in columns:
        if column.name in chosen:
            continue
        if (
            column.numeric_parse_ratio >= 0.95
            and not column.constant
            and not column.identifier_like
        ):
            other.append(column.name)
        else:
            excluded.append(column.name)
    ambiguous = tuple(
        role
        for role, flag in (
            ("timestamp", amb_time),
            ("target_load", amb_load),
            ("temperature", amb_temp),
        )
        if flag
    )
    return ColumnMappingDraft(
        timestamp_column=time,
        target_column=load,
        temperature_column=temperature,
        temperature_available=None,
        other_features=tuple(other),
        excluded_features=tuple(excluded),
        ambiguous_roles=ambiguous,
        warnings=("Confirm time, load, and temperature availability before continuing.",),
    )


def confirm_mapping(
    draft: ColumnMappingDraft, schema: SchemaInspection, dataset_id: UUID
) -> ColumnMapping:
    names = {p.name: p for p in schema.columns}
    if not draft.timestamp_column or not draft.target_column or draft.temperature_available is None:
        raise ValueError("time, load and temperature availability require explicit confirmation")
    if draft.temperature_available != (draft.temperature_column is not None):
        raise ValueError("temperature choice and availability disagree")
    chosen = [draft.timestamp_column, draft.target_column]
    if draft.temperature_column:
        chosen.append(draft.temperature_column)
    chosen.extend(draft.other_features)
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(names):
        raise ValueError("mapping contains missing or repeated columns")
    if names[draft.timestamp_column].datetime_parse_ratio < 0.8:
        raise ValueError("timestamp column is not sufficiently date-parseable")
    if names[draft.target_column].numeric_parse_ratio < 0.8:
        raise ValueError("load column is not sufficiently numeric")
    for column in chosen[2:]:
        if (
            names[column].numeric_parse_ratio < 0.8
            or names[column].constant
            or names[column].identifier_like
        ):
            raise ValueError(f"{column} is not an eligible numeric auxiliary")
    auxiliaries = {}
    if draft.temperature_column:
        auxiliaries[draft.temperature_column] = "temperature"
    auxiliaries.update({name: "other" for name in draft.other_features})
    return ColumnMapping(
        dataset_id=dataset_id,
        timestamp_column=draft.timestamp_column,
        target_column=draft.target_column,
        auxiliary_roles=auxiliaries,
        availability={
            name: AuxiliaryPolicy(kind="observed", role=cast(Literal["temperature", "other"], role))
            for name, role in auxiliaries.items()
        },
        confirmed=True,
    )


def analyze_quality(
    frame: pd.DataFrame, mapping: ColumnMapping, schema: SchemaInspection
) -> DataQualityReport:
    times = _times(frame[mapping.timestamp_column])
    parsed = times.dropna()
    ordered = bool(parsed.is_monotonic_increasing)
    duplicate = int(parsed.duplicated().sum())
    _, _, intervals, missing = _time_clues(frame[mapping.timestamp_column])
    irregular = (
        int(
            (
                parsed.drop_duplicates().sort_values().diff().dropna() != pd.Timedelta(intervals[0])
            ).sum()
        )
        if intervals
        else 0
    )
    numeric_columns = [mapping.target_column, *mapping.auxiliary_roles]
    missing_values = {name: int(frame[name].isna().sum()) for name in numeric_columns}
    invalid = {}
    thousands = {}
    outliers = {}
    constants = []
    for name in numeric_columns:
        raw = frame[name]
        numeric = _numeric(raw)
        invalid[name] = int((numeric.isna() & raw.notna()).sum())
        thousands[name] = int(_thousands_mask(raw).sum())
        nonmissing = numeric.dropna()
        if nonmissing.nunique() <= 1:
            constants.append(name)
        if len(nonmissing) >= 4:
            q1, q3 = nonmissing.quantile([0.25, 0.75])
            spread = q3 - q1
            outliers[name] = (
                int(((nonmissing < q1 - 3 * spread) | (nonmissing > q3 + 3 * spread)).sum())
                if spread
                else 0
            )
        else:
            outliers[name] = 0
    warnings = []
    if not ordered:
        warnings.append("timestamps are not sorted")
    if duplicate:
        warnings.append(f"{duplicate} duplicate timestamps")
    if missing:
        warnings.append(f"{missing} missing timestamps")
    if any(invalid.values()) or any(missing_values.values()):
        warnings.append("missing or invalid numerical values")
    if any(thousands.values()):
        warnings.append("numeric strings contain thousands separators")
    if any(outliers.values()):
        warnings.append("extreme-value diagnostics warrant review")
    return DataQualityReport(
        row_count=len(frame),
        timestamp_parse_failures=int(times.isna().sum()),
        ordered=ordered,
        duplicate_timestamps=duplicate,
        missing_timestamps=missing or 0,
        frequency=intervals[0] if intervals else None,
        irregular_intervals=irregular,
        missing_values=missing_values,
        invalid_numeric=invalid,
        thousands_separator_counts=thousands,
        constant_columns=tuple(constants),
        outlier_counts=outliers,
        usable_start=str(parsed.min()) if not parsed.empty else None,
        usable_end=str(parsed.max()) if not parsed.empty else None,
        warnings=tuple(warnings),
    )


def propose_plan(quality: DataQualityReport) -> PreparationPlan:
    rationale = ["Parse timestamps and numeric values; normalize thousands separators."]
    if not quality.ordered:
        rationale.append("Sort timestamps chronologically.")
    if quality.duplicate_timestamps:
        rationale.append("Duplicates require an explicit user policy.")
    if quality.missing_timestamps:
        rationale.append("Missing timestamp rows require an explicit user policy.")
    if any(quality.missing_values.values()) or any(quality.invalid_numeric.values()):
        rationale.append("Missing or invalid values require an explicit user policy.")
    return PreparationPlan(rationale=tuple(rationale))


def capabilities(mapping: ColumnMapping, frequency: str) -> DatasetCapabilities:
    temperature = any(role == "temperature" for role in mapping.auxiliary_roles.values())
    other = tuple(name for name, role in mapping.auxiliary_roles.items() if role == "other")
    groups = ["calendar", "historical_load"]
    if temperature:
        groups.extend(
            ("historical_temperature", "temperature_selection", "calendar_temperature_interaction")
        )
    if other:
        groups.append("other_features")
    return DatasetCapabilities(
        has_temperature=temperature,
        available_other_features=other,
        timestamp_frequency=frequency,
        supported_feature_groups=tuple(groups),
    )


def _prepare_frame(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    plan: PreparationPlan,
    *,
    minimum_rows: int = 3,
    frequency_override: str | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if plan.timezone_name != SOURCE_CLOCK:
        ZoneInfo(plan.timezone_name)
    names = [mapping.timestamp_column, mapping.target_column, *mapping.auxiliary_roles]
    availability = [
        f"{name}__available_at"
        for name in mapping.auxiliary_roles
        if f"{name}__available_at" in frame.columns
    ]
    if len(names) != len(set(names)):
        raise ValueError("mapped source columns must be distinct")
    if set(mapping.auxiliary_roles) & {"timestamp", "target", "series_id"}:
        raise ValueError(
            "auxiliary names collide with canonical columns; rename them before mapping"
        )
    if set(availability) & set(names):
        raise ValueError("auxiliary issue columns must be distinct from mapped value columns")
    numeric_names = ["target", *mapping.auxiliary_roles]
    # Read every mapped value from the untouched source, never from canonical output.
    data = pd.DataFrame(index=frame.index)
    times = _times(frame[mapping.timestamp_column])
    if plan.timezone_name == SOURCE_CLOCK:
        if times.dt.tz is not None:
            raise ValueError("source-clock timestamps must not carry timezone offsets")
    elif times.dt.tz is None:
        times = times.dt.tz_localize(plan.timezone_name, ambiguous="raise", nonexistent="raise")
    else:
        times = times.dt.tz_convert(plan.timezone_name)
    data["timestamp"] = times
    transformations = ["parse timestamps", "normalize numerical columns"]
    if plan.normalize_thousands_separators:
        transformations.append("normalize thousands separators")
    data["target"] = _numeric(
        frame[mapping.target_column], strip_thousands=plan.normalize_thousands_separators
    )
    for name in mapping.auxiliary_roles:
        data[name] = _numeric(frame[name], strip_thousands=plan.normalize_thousands_separators)
    for name in availability:
        parsed = _times(frame[name])
        if plan.timezone_name == SOURCE_CLOCK:
            if parsed.dt.tz is not None:
                raise ValueError("source-clock issue times must not carry timezone offsets")
        elif parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(
                plan.timezone_name, ambiguous="raise", nonexistent="raise"
            )
        else:
            parsed = parsed.dt.tz_convert(plan.timezone_name)
        if parsed.isna().any():
            raise ValueError(f"{name} has missing forecast issue timestamps")
        data[name] = parsed
    invalid_time = data["timestamp"].isna()
    if invalid_time.any():
        if plan.invalid_row_policy != "drop":
            raise ValueError("invalid timestamps require approved row dropping")
        data = data.loc[~invalid_time].copy()
        transformations.append(f"drop {int(invalid_time.sum())} invalid timestamp rows")
    if plan.sort_chronologically:
        data = data.sort_values("timestamp").reset_index(drop=True)
        transformations.append("sort chronologically")
    elif not data["timestamp"].is_monotonic_increasing:
        raise ValueError("unsorted data requires approved chronological sorting")
    if data["timestamp"].duplicated().any():
        if plan.duplicate_policy == "reject":
            raise ValueError("duplicate timestamps require an approved policy")
        count = int(data["timestamp"].duplicated().sum())
        if plan.duplicate_policy == "mean":
            if availability:
                raise ValueError("mean duplicate policy cannot combine issue timestamps")
            data = data.groupby("timestamp", as_index=False)[numeric_names].mean()
        else:
            data = data.drop_duplicates("timestamp", keep=plan.duplicate_policy)
        transformations.append(f"resolve {count} duplicate timestamps: {plan.duplicate_policy}")
    if len(data) < minimum_rows:
        raise ValueError(f"prepared dataset requires at least {minimum_rows} rows")
    if plan.missing_value_policy == "reject" and data[numeric_names].isna().any().any():
        raise ValueError("existing missing values require an approved value policy")
    diffs = data["timestamp"].sort_values().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty and frequency_override is None:
        raise ValueError("cannot infer a positive sampling frequency")
    frequency = (
        pd.Timedelta(frequency_override)
        if frequency_override
        else pd.Timedelta(positive.mode().iloc[0])
    )
    if frequency <= pd.Timedelta(0):
        raise ValueError("sampling frequency must be positive")
    origin = data["timestamp"].min()
    off_grid = (data["timestamp"] - origin) % frequency != pd.Timedelta(0)
    if off_grid.any():
        if plan.invalid_row_policy != "drop":
            raise ValueError("off-grid timestamps require approved invalid-row dropping")
        count = int(off_grid.sum())
        data = data.loc[~off_grid].copy()
        transformations.append(f"drop {count} off-grid timestamp rows")
    expected_count = int((data["timestamp"].max() - origin) / frequency) + 1
    if expected_count > MAX_PREPARED_ROWS:
        raise ValueError("filled dataset exceeds row limit")
    expected = pd.date_range(origin, periods=expected_count, freq=frequency)
    missing_count = len(expected.difference(pd.DatetimeIndex(data["timestamp"])))
    if missing_count:
        if plan.missing_timestamp_policy == "reject":
            raise ValueError("missing timestamps require an approved policy")
        data = data.set_index("timestamp").reindex(expected)
        data.index.name = "timestamp"
        data = data.reset_index()
        transformations.append(
            f"fill {missing_count} missing timestamps: {plan.missing_timestamp_policy}"
        )
    if data[["target", *mapping.auxiliary_roles]].isna().any().any():
        if plan.missing_value_policy == "reject" and plan.missing_timestamp_policy == "reject":
            raise ValueError("missing values require an approved policy")
        if plan.missing_value_policy == "drop":
            data = data.dropna(subset=["target", *mapping.auxiliary_roles])
            transformations.append("drop rows with missing numerical values")
        else:
            method = plan.missing_value_policy
            if method == "reject":
                method = plan.missing_timestamp_policy
            cols = ["target", *mapping.auxiliary_roles]
            if method == "interpolate":
                data[cols] = data[cols].interpolate(method="linear", limit_direction="both")
            elif method == "forward_fill":
                data[cols] = data[cols].ffill().bfill()
            transformations.append(f"impute missing numerical values: {method}")
    if data[["target", *mapping.auxiliary_roles]].isna().any().any():
        raise ValueError("approved preparation leaves missing numerical values")
    data["series_id"] = "default"
    data = data[["timestamp", "series_id", "target", *mapping.auxiliary_roles, *availability]]
    freq_text = str(frequency)
    NormalizedSeries.validate(
        data,
        freq_text,
        plan.timezone_name,
        {
            name: AuxiliaryPolicy(kind="observed", role=cast(Literal["temperature", "other"], role))
            for name, role in mapping.auxiliary_roles.items()
        },
    )
    return data, tuple(transformations)


def apply_plan(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    plan: PreparationPlan,
    store: ArtifactStore,
    task_id: UUID,
    snapshot_id: UUID,
) -> PreparedSnapshot:
    data, transformations = _prepare_frame(frame, mapping, plan)
    frequency = str(data["timestamp"].sort_values().diff().dropna().mode().iloc[0])
    serialized = data.to_csv(index=False).encode("utf-8")
    artifact = store.put_bytes(f"prepared/{task_id}/{snapshot_id}/data.csv", serialized)
    snapshot = DatasetSnapshot(
        snapshot_id=snapshot_id,
        dataset_id=mapping.dataset_id,
        mapping_version="1",
        cleaning_version=str(plan.version),
        artifact=artifact,
        frequency=frequency,
        timezone_name=plan.timezone_name,
        series_id="default",
    )
    return PreparedSnapshot(
        dataset=snapshot,
        mapping=mapping,
        plan=plan,
        row_count=len(data),
        time_start=data["timestamp"].iloc[0].isoformat(),
        time_end=data["timestamp"].iloc[-1].isoformat(),
        columns=tuple(data.columns),
        applied_transformations=transformations,
        capabilities=capabilities(mapping, frequency),
    )


def overview(
    prepared: PreparedSnapshot, store: ArtifactStore, quality: DataQualityReport
) -> DatasetOverview:
    data = pd.read_csv(BytesIO(store.read_bytes(prepared.dataset.artifact)))
    times = pd.to_datetime(data["timestamp"])
    stride = max(1, int(np.ceil(len(data) / _CHART_POINTS)))

    def chart(name: str) -> ChartSeries:
        return ChartSeries(
            name=name,
            points=tuple(
                (pd.Timestamp(t).isoformat(), float(v))
                for t, v in zip(times.iloc[::stride], data[name].iloc[::stride], strict=True)
            ),
        )

    secondary_name = next(iter(prepared.mapping.auxiliary_roles), None)
    stats = {
        name: {
            "min": float(data[name].min()),
            "max": float(data[name].max()),
            "mean": float(data[name].mean()),
        }
        for name, role in prepared.mapping.auxiliary_roles.items()
        if role == "other"
    }
    return DatasetOverview(
        row_count=prepared.row_count,
        time_start=prepared.time_start,
        time_end=prepared.time_end,
        frequency=prepared.dataset.frequency,
        mapped_columns={
            "timestamp": prepared.mapping.timestamp_column,
            "target_load": prepared.mapping.target_column,
            **prepared.mapping.auxiliary_roles,
        },
        missing_values=quality.missing_values,
        duplicate_timestamps=quality.duplicate_timestamps,
        load_min=float(data["target"].min()),
        load_max=float(data["target"].max()),
        load_mean=float(data["target"].mean()),
        other_feature_stats=stats,
        primary=chart("target"),
        secondary=chart(secondary_name) if secondary_name else None,
        missing_timestamp_count=quality.missing_timestamps,
    )


def source_for_upload(
    dataset_id: UUID, filename: str, artifact_uri: str, checksum: str
) -> DatasetSource:
    return DatasetSource(
        dataset_id=dataset_id,
        adapter_kind="uploaded_tabular",
        source_uri=artifact_uri,
        sha256=checksum,
        provenance={
            "original_filename": filename.replace("\\", "/").split("/")[-1],
            "upload_kind": "user",
        },
        series_ids=["default"],
    )
