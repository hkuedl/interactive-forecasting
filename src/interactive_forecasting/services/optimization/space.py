"""Research-space importer and complete candidate resolution above Forecasting Core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from interactive_forecasting.domain.forecasting import (
    CoreFeatureRecipe,
    OutputConfig,
    PreprocessConfig,
    ResolvedCandidate,
)
from interactive_forecasting.domain.models import EvaluationProtocol, TaskDefinition
from interactive_forecasting.domain.preparation import DatasetCapabilities
from interactive_forecasting.domain.search import (
    CandidateRequest,
    FamilySpace,
    ParameterDomain,
    Scalar,
    SearchSpace,
    TrialCandidate,
)
from interactive_forecasting.domain.types import ModelFamily
from interactive_forecasting.services.models.core import ModelRegistry

_RESEARCH_ROOT = Path(__file__).resolve().parent / "specs"


def _domain(
    name: str, source: Any, *, condition_on: str | None = None, condition_value: Scalar = None
) -> ParameterDomain:
    if isinstance(source, list):
        return ParameterDomain(
            name=name,
            kind="categorical",
            choices=tuple(source),
            condition_on=condition_on,
            condition_value=condition_value,
        )
    if "choices" in source:
        return _domain(
            name, source["choices"], condition_on=condition_on, condition_value=condition_value
        )
    low, high = source["low"], source["high"]
    kind: Literal["integer", "float"] = (
        "integer" if isinstance(low, int) and isinstance(high, int) else "float"
    )
    return ParameterDomain(
        name=name,
        kind=kind,
        low=low,
        high=high,
        step=source.get("step"),
        log=source.get("sampling") == "log",
        condition_on=condition_on,
        condition_value=condition_value,
    )


def canonical_research_space(spec_root: Path = _RESEARCH_ROOT) -> SearchSpace:
    """Import Table III/IV dimensions without inventing missing experiment settings."""
    model_bytes = (spec_root / "table_03_hyperparameter_search_space.json").read_bytes()
    feature_bytes = (spec_root / "table_04_feature_search_space.json").read_bytes()
    model_spec = json.loads(model_bytes)
    feature_spec = json.loads(feature_bytes)
    source_digest = hashlib.sha256(model_bytes + b"\0" + feature_bytes).hexdigest()
    families: list[FamilySpace] = []
    name_map = {"C": "c", "hidden_layers": "layers"}
    for family in ModelFamily:
        raw = model_spec["models"][family.value]
        params = []
        for name, source in raw.items():
            params.append(
                _domain(
                    name_map.get(name, name),
                    source,
                    condition_on="regularization" if name == "alpha" else None,
                    condition_value="ridge" if name == "alpha" else None,
                )
            )
        families.append(FamilySpace(family=family, parameters=tuple(params)))
    raw_features = feature_spec["feature_groups"]
    features = (
        _domain("calendar", raw_features["calendar_features"]),
        _domain("temperature_mode", raw_features["temperature_lag_values"]["choices"]),
        _domain(
            "temperature_ratio",
            {"low": 0.0, "high": 1.0},
            condition_on="temperature_mode",
            condition_value="correlation",
        ),
        _domain("interaction_mode", raw_features["calendar_temperature_interaction"]),
        _domain("load_mode", raw_features["load_lag_regression"]["choices"]),
        _domain(
            "load_ratio",
            {"low": 0.0, "high": 1.0},
            condition_on="load_mode",
            condition_value="correlation",
        ),
        _domain("sequence_frequency", raw_features["historical_load_sequence"]["frequency"]),
        _domain(
            "sequence_length",
            {
                "low": raw_features["historical_load_sequence"]["length"][0],
                "high": raw_features["historical_load_sequence"]["length"][1],
                "step": 1,
            },
        ),
        _domain("other_mode", raw_features["other_features"]["choices"]),
        _domain(
            "other_ratio",
            {"low": 0.0, "high": 1.0},
            condition_on="other_mode",
            condition_value="correlation",
        ),
    )
    return SearchSpace(
        version=f"table-03-04-{source_digest[:12]}",
        provenance="packaged research specs/table_03,table_04",
        families=tuple(families),
        features=features,
    )


def space_for_capabilities(space: SearchSpace, capabilities: DatasetCapabilities) -> SearchSpace:
    """Remove unavailable feature dimensions before any search trial is generated."""
    blocked: set[str] = set()
    if not capabilities.has_temperature:
        blocked.update({"temperature_mode", "temperature_ratio", "interaction_mode"})
    if not capabilities.available_other_features:
        blocked.update({"other_mode", "other_ratio"})
    return SearchSpace(
        version=f"{space.version}-cap-{int(capabilities.has_temperature)}-{int(bool(capabilities.available_other_features))}",
        provenance=f"{space.provenance}; prepared dataset capability filter",
        parent_space_id=space.space_id,
        families=space.families,
        features=tuple(item for item in space.features if item.name not in blocked),
    )


def space_for_output(space: SearchSpace, output: OutputConfig) -> SearchSpace:
    """Limit quantile tasks to adapter families with declared quantile capability."""
    if output.representation == "point":
        return space
    eligible = {ModelFamily.MLP, ModelFamily.LSTM, ModelFamily.GRU, ModelFamily.CNN}
    return SearchSpace(
        version=f"{space.version}-quantile",
        provenance=f"{space.provenance}; quantile-capable families",
        parent_space_id=space.space_id,
        families=tuple(item for item in space.families if item.family in eligible),
        features=space.features,
    )


class CandidateResolver:
    """Resolve sampled or supplied scalars into the core's complete typed candidate."""

    def __init__(
        self,
        task: TaskDefinition,
        protocol: EvaluationProtocol,
        templates: dict[ModelFamily, ResolvedCandidate],
    ):
        self.task = task
        self.protocol = protocol
        self.templates = dict(templates)

    def resolve(self, request: CandidateRequest, space: SearchSpace) -> TrialCandidate:
        space.validate_values(request.family, request.values)
        if request.family not in self.templates:
            raise ValueError("missing complete family candidate template")
        base = self.templates[request.family]
        if (
            base.family != request.family
            or base.index.delta != self.task.delta
            or base.index.horizon != self.task.horizon
            or base.index.offset_unit != self.task.time_unit
            or base.auxiliary_policies != self.task.auxiliary_policies
        ):
            raise ValueError("candidate template disagrees with task protocol")
        hp = dict(base.hyperparameters)
        training = base.training.model_dump()
        training["seed"] = request.seed
        recipe = base.features.model_dump()
        preprocessing = base.preprocessing.model_dump()
        for name, value in request.values.items():
            if name == "learning_rate" and base.family != ModelFamily.XGBOOST:
                training["learning_rate"] = value
            elif name == "calendar":
                recipe["calendar"] = value
            elif name == "sequence_frequency":
                recipe["sequence_frequency"] = value
            elif name == "sequence_length":
                recipe["sequence_length"] = value
            elif name in {
                "temperature_mode",
                "temperature_ratio",
                "interaction_mode",
                "load_mode",
                "load_ratio",
                "other_mode",
                "other_ratio",
            }:
                pass
            else:
                hp[name] = value
        if hp.get("regularization") == "none":
            hp.pop("alpha", None)
            hp.pop("ridge_tol", None)
            hp.pop("ridge_max_iter", None)
        if self.task.forecast_output.representation == "quantile":
            training["loss"] = "pinball"
        if (
            base.family in {ModelFamily.LSTM, ModelFamily.GRU, ModelFamily.CNN}
            and request.values.get("temperature_mode") == "correlation"
            and not recipe.get("temperature_column")
        ):
            temperature_columns = [
                name
                for name, policy in self.task.auxiliary_policies.items()
                if policy.role == "temperature"
            ]
            if len(temperature_columns) != 1:
                raise ValueError("temperature search needs one configured temperature variable")
            recipe["temperature_column"] = temperature_columns[0]
        self._resolve_features(request.values, recipe, preprocessing, base.index.frequency)
        candidate = ResolvedCandidate.model_validate(
            {
                **base.model_dump(),
                "hyperparameters": hp,
                "training": training,
                "features": CoreFeatureRecipe.model_validate(recipe),
                "preprocessing": PreprocessConfig.model_validate(preprocessing),
                "output": self.task.forecast_output,
                "configuration_version": f"{base.configuration_version}@{space.version}",
            }
        )
        ModelRegistry().resolve(candidate)
        return TrialCandidate(
            candidate=candidate,
            request=request,
            space_id=space.space_id,
            task_definition_id=self.task.definition_id,
            protocol_id=self.protocol.protocol_id,
        )

    @staticmethod
    def _resolve_features(
        values: dict[str, Scalar],
        recipe: dict[str, Any],
        preprocessing: dict[str, Any],
        frequency: str,
    ) -> None:
        if values.get("interaction_mode") == "none":
            recipe["interactions"] = ()
        if any(key in values for key in ("temperature_mode", "load_mode", "other_mode")):
            if preprocessing["selection"] != "none":
                raise ValueError("group selection cannot combine with global Pearson selection")
            preprocessing["selection_ratio"] = None

        temperature_mode = values.get("temperature_mode")
        if temperature_mode is not None:
            if temperature_mode == "none":
                recipe.update(temperature_lags=(), temperature_daily_days=())
            else:
                if not recipe.get("temperature_column"):
                    raise ValueError(
                        "temperature correlation needs a configured temperature column"
                    )
                step = pd.Timedelta(frequency)
                hourly_steps = pd.Timedelta(hours=1) / step
                if hourly_steps < 1 or int(hourly_steps) != hourly_steps:
                    raise ValueError(
                        "temperature hourly candidates require a frequency dividing one hour"
                    )
                recipe["temperature_lags"] = tuple(i * int(hourly_steps) for i in range(72))
                recipe["temperature_daily_days"] = (1, 2, 3)
            recipe["temperature_selection"] = {
                "mode": temperature_mode,
                "ratio": values.get("temperature_ratio"),
            }

        load_mode = values.get("load_mode")
        if load_mode is not None:
            if load_mode == "none":
                recipe["load_lags"] = ()
            else:
                step = pd.Timedelta(frequency)
                hourly_steps = pd.Timedelta(hours=1) / step
                if hourly_steps < 1 or int(hourly_steps) != hourly_steps:
                    raise ValueError("hourly load candidates require a frequency dividing one hour")
                hours = range(168) if load_mode == "correlation" else range(24, 169, 24)
                recipe["load_lags"] = tuple(i * int(hourly_steps) for i in hours)
            recipe["load_selection"] = {"mode": load_mode, "ratio": values.get("load_ratio")}

        other_mode = values.get("other_mode")
        if other_mode is not None:
            if other_mode == "none":
                recipe.update(auxiliary_lags={}, auxiliary_leads={})
            elif not recipe.get("auxiliary_lags") and not recipe.get("auxiliary_leads"):
                raise ValueError("other correlation needs explicit auxiliary candidates")
            recipe["other_selection"] = {"mode": other_mode, "ratio": values.get("other_ratio")}

        if values.get("interaction_mode") == "all":
            calendar_names: tuple[str, ...]
            calendar = recipe.get("calendar")
            if calendar == "numerical":
                calendar_names = ("calendar_hour", "calendar_weekday", "calendar_month")
            elif calendar == "categorical":
                calendar_names = (
                    *(f"hour_{i}" for i in range(24)),
                    *(f"weekday_{i}" for i in range(7)),
                    *(f"month_{i}" for i in range(1, 13)),
                )
            elif calendar == "trigonometric":
                calendar_names = (
                    "hour_sin",
                    "hour_cos",
                    "weekday_sin",
                    "weekday_cos",
                    "month_sin",
                    "month_cos",
                )
            else:
                calendar_names = ()
            temperature_names = (
                *(f"temperature_lag_{lag}" for lag in recipe.get("temperature_lags", ())),
                *(f"temperature_day_{day}_avg" for day in recipe.get("temperature_daily_days", ())),
            )
            if values.get("temperature_ratio") == 0:
                temperature_names = ()
            recipe["interactions"] = tuple(
                (calendar_name, temperature_name)
                for calendar_name in calendar_names
                for temperature_name in temperature_names
            )
