"""Small, explicit Task Manager command grammar over persisted Preparation drafts."""

from __future__ import annotations

import re

from interactive_forecasting.domain.preparation import (
    ColumnMappingDraft,
    ForecastTaskDraft,
    PreparationRecord,
    PreparationStep,
)


def parse_mapping_command(record: PreparationRecord, text: str) -> ColumnMappingDraft | None:
    """Resolve only the supported mapping edits against the inspected schema."""
    if record.step != PreparationStep.CONFIRM_COLUMN_MAPPING or record.mapping_draft is None:
        return None
    command = text.strip()
    match = re.fullmatch(
        r"(?:use|set)\s+(.+?)\s+as\s+(temperature|time|timestamp|load|target)[.!]?",
        command,
        flags=re.IGNORECASE,
    )
    none_temp = command.lower() in {"no temperature", "temperature none", "use no temperature"}
    if not match and not none_temp:
        return None
    draft = record.mapping_draft.model_dump()
    if none_temp:
        draft.update(temperature_column=None, temperature_available=False)
    else:
        assert match is not None
        column = match.group(1).strip()
        role = match.group(2).lower()
        if record.schema_inspection is None or column not in {
            profile.name for profile in record.schema_inspection.columns
        }:
            raise ValueError("requested column is not in inspected schema")
        key = {
            "time": "timestamp_column",
            "timestamp": "timestamp_column",
            "load": "target_column",
            "target": "target_column",
            "temperature": "temperature_column",
        }[role]
        draft[key] = column
        if role == "temperature":
            draft["temperature_available"] = True
            draft["other_features"] = tuple(
                name for name in draft["other_features"] if name != column
            )
    return ColumnMappingDraft.model_validate(draft)


def has_actionable_intent(text: str) -> bool:
    """Require an explicit current-turn request before applying a suggested edit."""
    lowered = text.strip().lower()
    return lowered.startswith(
        (
            "use ",
            "set ",
            "change ",
            "don't ",
            "do not ",
            "leave ",
            "let's ",
            "please ",
            "can you ",
            "could you ",
            "i want to ",
            "i'd like to ",
            "okay, let's ",
            "ok, let's ",
        )
    )


def is_confirmation_request(text: str) -> bool:
    """Recognize explicit approval, never a question about approval."""
    return bool(
        re.fullmatch(
            r"(?:please )?(?:confirm|approve)(?: (?:this|the|current) "
            r"(?:mapping|plan|task|draft|settings))?[.!]?|"
            r"(?:looks good|that looks good|yes|okay|ok)[,.! ]+"
            r"(?:go ahead|confirm it|approve it)[.!]?",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def is_pure_hypothetical(text: str) -> bool:
    """Protect read-only questions from storing speculative preferences."""
    return bool(
        re.match(
            r"^(?:what if|what would|what happens if|why|how|should i|suppose|"
            r"hypothetically|would it|could i|can i)\b",
            text.strip(),
            flags=re.IGNORECASE,
        )
    ) and not bool(re.search(r"\b(?:i want|i choose|let's use|use it)\b", text, re.I))


def parse_task_command(record: PreparationRecord, text: str) -> ForecastTaskDraft | None:
    """Parse only supported commands; never infer an experimental weather assumption."""
    if record.task_draft is None:
        return None
    command = text.strip().rstrip(".")
    lowered = command.lower()
    draft = record.task_draft.model_dump()
    weighted = re.fullmatch(
        r"set (?:objective|metric) to weighted mae from "
        r"(\d{2}:\d{2}) to (\d{2}:\d{2}) weight (\d+(?:\.\d+)?)",
        lowered,
    )
    if weighted:
        if record.prepared is None:
            raise ValueError("prepared dataset timezone is required for weighted MAE")
        start, end, weight = weighted.groups()
        draft["objective_id"] = "weighted_mae"
        draft["metric_spec"] = {
            "kind": "weighted",
            "base_metric": "mae",
            "timezone_name": record.prepared.dataset.timezone_name,
            "time_range": {"start_local": start, "end_local": end, "weight": float(weight)},
        }
        return ForecastTaskDraft.model_validate(draft)
    if re.fullmatch(r"set (?:objective|metric) to weighted mae", lowered):
        raise ValueError("weighted MAE needs start/end local times and a positive weight")
    asymmetric = re.fullmatch(
        r"set (?:objective|metric) to asymmetric mae over weight "
        r"(\d+(?:\.\d+)?) under weight (\d+(?:\.\d+)?)",
        lowered,
    )
    if asymmetric:
        over, under = map(float, asymmetric.groups())
        draft["objective_id"] = "asymmetric_mae"
        draft["metric_spec"] = {
            "kind": "asymmetric",
            "base_metric": "mae",
            "over_weight": over,
            "under_weight": under,
        }
        return ForecastTaskDraft.model_validate(draft)
    if re.fullmatch(r"set (?:objective|metric) to asymmetric mae", lowered):
        raise ValueError("asymmetric MAE needs positive over and under weights")
    match = re.fullmatch(r"set (?:delta|lead) to (\d+)", lowered)
    if match:
        draft["delta"] = int(match.group(1))
    else:
        match = re.fullmatch(r"set horizon to (\d+)", lowered)
        if match:
            draft["horizon"] = int(match.group(1))
        else:
            match = re.fullmatch(r"set lead unit to (samples|hours)", lowered)
            if match:
                draft["time_unit"] = match.group(1)
            else:
                match = re.fullmatch(r"set (?:forecast type|output) to (point|quantile)", lowered)
                if match:
                    representation = match.group(1)
                    draft["output"] = (
                        {"representation": "quantile", "quantile_levels": (0.1, 0.5, 0.9)}
                        if representation == "quantile"
                        else {"representation": "point", "quantile_levels": ()}
                    )
                    draft["objective_id"] = "crps" if representation == "quantile" else "mae"
                    draft["metric_spec"] = None
                else:
                    match = re.fullmatch(r"set (?:objective|metric) to (mae|mape|crps)", lowered)
                    if match:
                        draft["objective_id"] = match.group(1)
                        draft["metric_spec"] = None
                    else:
                        match = re.fullmatch(r"set quantile levels to ([\d.,\s]+)", lowered)
                        if match:
                            levels = tuple(
                                float(value.strip())
                                for value in match.group(1).split(",")
                                if value.strip()
                            )
                            draft["output"] = {
                                "representation": "quantile",
                                "quantile_levels": levels,
                            }
                            draft["objective_id"] = "crps"
                            draft["metric_spec"] = None
                        else:
                            match = re.fullmatch(
                                r"set temperature (?:availability|policy) to "
                                r"(observed|known_ahead|perfect_forecast)"
                                r"(?: protocol ([\w.-]+))?",
                                lowered,
                            )
                            if match:
                                if (
                                    record.capabilities is None
                                    or not record.capabilities.has_temperature
                                ):
                                    raise ValueError("confirmed dataset has no temperature")
                                kind, protocol = match.groups()
                                draft["temperature_policy"] = {
                                    "kind": kind,
                                    "role": "temperature",
                                    "protocol_id": protocol if kind == "perfect_forecast" else None,
                                }
                            else:
                                match = re.fullmatch(
                                    r"set temperature policy to forecast source ([\w./-]+)",
                                    lowered,
                                )
                                if match:
                                    if (
                                        record.capabilities is None
                                        or not record.capabilities.has_temperature
                                    ):
                                        raise ValueError("confirmed dataset has no temperature")
                                    draft["temperature_policy"] = {
                                        "kind": "forecast",
                                        "role": "temperature",
                                        "source_ref": match.group(1),
                                    }
                                else:
                                    match = re.fullmatch(
                                        r"set ([\w.-]+) availability to "
                                        r"(observed|known_ahead|forecast)"
                                        r"(?: source ([\w./-]+))?",
                                        lowered,
                                    )
                                    if match:
                                        name, kind, source = match.groups()
                                        if (
                                            record.capabilities is None
                                            or name
                                            not in record.capabilities.available_other_features
                                        ):
                                            raise ValueError("other-feature name is not confirmed")
                                        draft["other_policies"][name] = {
                                            "kind": kind,
                                            "role": "other",
                                            "source_ref": source if kind == "forecast" else None,
                                        }
                                    else:
                                        match = re.fullmatch(
                                            r"set split to ([\d.]+),\s*([\d.]+),\s*([\d.]+)",
                                            lowered,
                                        )
                                        if match:
                                            train, validation, test = map(float, match.groups())
                                            draft.update(
                                                train_fraction=train,
                                                validation_fraction=validation,
                                                test_fraction=test,
                                            )
                                        else:
                                            return None
    return ForecastTaskDraft.model_validate(draft)


def offline_reply(record: PreparationRecord, text: str) -> tuple[str, str]:
    """Contextual read-only fallback when no agent runtime is configured."""
    lowered = text.strip().lower()
    if any(word in lowered for word in ("confirm", "approve", "looks good", "go ahead")):
        return (
            "confirmation",
            "Nothing has been confirmed yet. Review the draft and use its approval button.",
        )
    is_question = "?" in lowered or lowered.startswith(
        ("why ", "what ", "how ", "can ", "could ", "is ", "does ")
    )
    if not is_question:
        return (
            "correction",
            "I haven't changed the draft. You can edit it in the workspace "
            "or use a supported chat command.",
        )
    if (
        any(word in lowered for word in ("mapping", "column", "temperature", "time", "load"))
        and record.mapping_draft
    ):
        mapping = record.mapping_draft
        roles = (
            ("time", mapping.timestamp_column),
            ("load", mapping.target_column),
            ("temperature", mapping.temperature_column),
        )
        profiles = (
            {col.name: col for col in record.schema_inspection.columns}
            if record.schema_inspection
            else {}
        )
        details = []
        for role, name in roles:
            if name:
                profile = profiles.get(name)
                hints = (
                    f" (hints: {', '.join(profile.semantic_hints)})"
                    if profile and profile.semantic_hints
                    else ""
                )
                details.append(f"{name} for {role}{hints}")
        return (
            "question",
            f"The current mapping suggests {', '.join(details)}. "
            "It is based on the inspected column profiles; you can change it before confirming.",
        )
    if any(word in lowered for word in ("quality", "missing", "duplicate", "outlier")):
        quality = record.quality
        if quality is None:
            return ("question", "Data-quality analysis has not been run yet.")
        warnings = " ".join(quality.warnings[:2]) or "No additional warning was recorded."
        return (
            "question",
            f"The analysis found {quality.missing_timestamps} missing timestamps "
            f"and {quality.duplicate_timestamps} duplicates. {warnings}",
        )
    if any(word in lowered for word in ("clean", "preparation", "interpolat", "operation")):
        plan = record.plan_draft
        if plan is None:
            return ("question", "No preparation plan has been proposed yet.")
        reasons = " ".join(plan.rationale[:2])
        return (
            "question",
            f"The current plan uses {plan.missing_value_policy} for missing values "
            f"and {plan.duplicate_policy} for duplicate timestamps. {reasons}",
        )
    if any(
        word in lowered
        for word in ("metric", "forecast", "setting", "objective", "mae", "mape", "crps")
    ):
        task_draft = record.task_draft
        if task_draft is None:
            return ("question", "Forecast settings have not been configured yet.")
        return (
            "question",
            f"The current draft uses {task_draft.objective_id.upper()} to evaluate "
            f"a {task_draft.output.representation} forecast. Another metric would score "
            "validation results differently without itself changing the training loss.",
        )
    step = record.step.value.replace("_", " ").lower()
    return ("question", f"We are at {step}. What part of the current preparation can I explain?")
