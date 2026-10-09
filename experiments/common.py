"""Shared public-benchmark orchestration; numerical work stays in production services."""

# ruff: noqa: E402 -- direct script entry points must find this checkout's src package.
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from interactive_forecasting.domain.forecasting import (
    AuxiliaryPolicy,
    CoreFeatureRecipe,
    IndexSpec,
    OriginSchedule,
    OutputConfig,
    PreprocessConfig,
    ResolvedCandidate,
    TrainingConfig,
)
from interactive_forecasting.domain.models import (
    ColumnMapping,
    DatasetSnapshot,
    EvaluationProtocol,
    ExperimentRun,
    Task,
    TaskDefinition,
    TimeWindow,
)
from interactive_forecasting.domain.preparation import (
    DatasetCapabilities,
    PreparationPlan,
    PreparationRecord,
    PreparationStep,
    PreparedSnapshot,
)
from interactive_forecasting.domain.search import (
    BackendConfig,
    CandidateRequest,
    FamilySpace,
    ParameterDomain,
    Scalar,
    SearchSpace,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import (
    ModelFamily,
    OptimizationState,
    Stage,
    WorkflowStatus,
)
from interactive_forecasting.services.data.core import SOURCE_CLOCK, NormalizedSeries
from interactive_forecasting.services.optimization.backend import OptunaBackend
from interactive_forecasting.services.optimization.engine import SearchEngine
from interactive_forecasting.services.optimization.execution import (
    FinalTestEvaluator,
    TrialExecutor,
    ValidationEvaluator,
)
from interactive_forecasting.services.optimization.space import (
    canonical_research_space,
    space_for_capabilities,
    space_for_output,
)
from interactive_forecasting.storage.artifacts import ArtifactStore
from interactive_forecasting.storage.preparation import PreparationRepository
from interactive_forecasting.storage.sql import (
    Database,
    ExperimentRepository,
    LLMCallRepository,
    MessageRepository,
    OptimizationSessionRepository,
    TaskRepository,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "examples"
MANIFESTS = Path(__file__).resolve().parent / "datasets"
FAMILIES = tuple(ModelFamily)
SEQUENTIAL = {ModelFamily.CNN, ModelFamily.LSTM, ModelFamily.GRU}
QUANTILE = {ModelFamily.MLP, *SEQUENTIAL}

# These are the stated fixed/default values, not optimizer priors.
FIXED_HP: dict[ModelFamily, dict[str, int | float | str | bool]] = {
    ModelFamily.LINEAR: {"regularization": "ridge", "alpha": 0.1, "fit_intercept": True},
    ModelFamily.SVR: {"c": 10.0, "gamma": 0.01, "kernel": "rbf", "epsilon": 0.0},
    ModelFamily.XGBOOST: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
    },
    ModelFamily.MLP: {"layers": 3, "hidden_size": 256, "dropout": 0.1},
    ModelFamily.LSTM: {"layers": 2, "hidden_size": 128, "fc_size": 128, "dropout": 0.1},
    ModelFamily.GRU: {"layers": 2, "hidden_size": 128, "fc_size": 128, "dropout": 0.1},
    ModelFamily.CNN: {
        "layers": 2,
        "kernel_size": 1,
        "filters": 64,
        "fc_size": 128,
        "dropout": 0.0,
    },
}
FIXED_LR = {
    ModelFamily.MLP: 1e-3,
    ModelFamily.LSTM: 5e-4,
    ModelFamily.GRU: 5e-4,
    ModelFamily.CNN: 5e-4,
}


def manifest(dataset: str) -> dict:
    path = MANIFESTS / f"{dataset}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown public dataset: {dataset}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("id") != dataset:
        raise ValueError(f"invalid dataset manifest: {path}")
    return value


def discover(dataset: str | None = None) -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for name in [dataset] if dataset else ("gefcom2014", "gefcom2012", "gefcom2017"):
        spec = manifest(name)
        for path in sorted(DATA.glob(spec["file_glob"])):
            series = path.name.removesuffix(spec["file_suffix"])
            if not re.fullmatch(spec["series_pattern"], series):
                continue
            found.append((name, series, path))
    return found


def _numbers(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="raise")


def normalize(path: Path, spec: dict, timezone_name: str) -> tuple[pd.DataFrame, dict]:
    """Manifest-driven CSV adaptation; no train-fitted numerical transformation."""
    raw = pd.read_csv(path)
    timestamps = pd.to_datetime(raw[spec["timestamp"]], errors="raise")
    if timezone_name == SOURCE_CLOCK:
        if timestamps.dt.tz is not None:
            raise ValueError("source-clock inputs must not contain timezone offsets")
    elif timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            timezone_name, ambiguous="raise", nonexistent="raise"
        )
    else:
        timestamps = timestamps.dt.tz_convert(timezone_name)
    columns: dict[str, object] = {
        "timestamp": timestamps,
        "series_id": path.name.removesuffix(spec["file_suffix"]),
        "target": _numbers(raw[spec["load"]]),
    }
    temperature = spec.get("temperature")
    if temperature:
        # The released 2012/2017 columns are already station averages.
        if isinstance(temperature, str):
            columns["temperature"] = _numbers(raw[temperature])
        else:
            station_columns = [col for col in raw if re.fullmatch(temperature["pattern"], col)]
            if not station_columns:
                raise ValueError("no weather-station columns matched the manifest")
            columns["temperature"] = pd.concat(
                [_numbers(raw[col]) for col in station_columns], axis=1
            ).mean(axis=1)
    for name, source in spec.get("other", {}).items():
        columns[name] = _numbers(raw[source])
    frame = pd.DataFrame(columns).sort_values("timestamp")
    if frame["timestamp"].duplicated().any():
        raise ValueError("duplicate source timestamps require a declared resolution policy")
    original_rows = len(frame)
    full = pd.date_range(
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
        freq=spec["frequency"],
        tz=None if timezone_name == SOURCE_CLOCK else timezone_name,
    )
    missing = len(full) - original_rows
    policy = spec["missing_timestamps"]
    imputed_targets = full.difference(pd.DatetimeIndex(frame["timestamp"]))
    if missing:
        if policy != "causal_forward_fill":
            raise ValueError(f"{missing} missing timestamps; manifest policy is {policy}")
        frame = frame.set_index("timestamp").reindex(full)
        frame.index.name = "timestamp"
        frame["series_id"] = path.name.removesuffix(spec["file_suffix"])
        numeric = [name for name in frame if name != "series_id"]
        frame[numeric] = frame[numeric].ffill()
        frame = frame.reset_index()
    if frame.isna().any().any() or not len(frame):
        raise ValueError("unresolved missing values in canonical dataset")
    return frame, {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": original_rows,
        "canonical_rows": len(frame),
        "filled_timestamps": missing,
        "imputed_target_timestamps": [value.isoformat() for value in imputed_targets],
        "missing_timestamp_policy": policy,
        "source_temperature": temperature,
        "timestamp_policy": (
            "source_clock_no_timezone_or_dst" if timezone_name == SOURCE_CLOCK else "zoned"
        ),
        "timezone_attached_or_converted": timezone_name != SOURCE_CLOCK,
        "dst_normalization": False if timezone_name == SOURCE_CLOCK else "zone_rules",
    }


def require_protocol(config: dict, dataset: str, series: str) -> dict:
    """Scientific choices must be supplied, not guessed from the table specs."""
    datasets = config.get("datasets", {})
    entry = datasets.get(dataset, {})
    selected = {
        **config.get("task", {}),
        **config.get("training", {}),
        **(
            {"mape_zero_policy": config["mape_zero_policy"]} if "mape_zero_policy" in config else {}
        ),
        **entry,
        **entry.get("series", {}).get(series, {}),
    }
    for key in (
        "timezone",
        "delta",
        "horizon",
        "offset_unit",
        "anchor",
        "neural_epochs",
        "neural_batch_size",
        "mape_zero_policy",
    ):
        if key not in selected:
            raise ValueError(f"missing frozen protocol setting datasets.{dataset}.{key}")
    if "split" not in selected and not all(
        key in selected for key in ("train", "validation", "test")
    ):
        raise ValueError("frozen protocol requires chronological split or explicit windows")
    if not isinstance(selected["timezone"], str) or not selected["timezone"]:
        raise ValueError(f"timestamp basis for {dataset} must be configured")
    source_clock_policy = config.get("time_handling") == "source_clock_no_timezone_or_dst"
    if (selected["timezone"] == SOURCE_CLOCK) != source_clock_policy:
        raise ValueError("source-clock timestamp basis must match the frozen time policy")
    if selected["mape_zero_policy"] not in {"error", "omit", "epsilon"}:
        raise ValueError("mape_zero_policy must be error, omit or epsilon")
    if selected["mape_zero_policy"] == "epsilon" and "mape_epsilon" not in selected:
        raise ValueError("epsilon MAPE policy requires mape_epsilon")
    if selected["horizon"] != 1:
        raise ValueError("current production adapters support H=1")
    return selected


def resolve_split(frame: pd.DataFrame, settings: dict, frequency: str) -> dict:
    """Derive disjoint target windows from the chronological usable target grid."""
    if "split" not in settings:
        return settings
    fractions = settings["split"]
    if fractions != [0.70, 0.15, 0.15]:
        raise ValueError("formal chronological split must be 70/15/15")
    # Warm-up rows supply lookback but are never scored targets.
    eligible = frame.loc[
        frame["timestamp"] >= frame["timestamp"].iloc[0] + pd.Timedelta(days=31),
        "timestamp",
    ].reset_index(drop=True)
    count = len(eligible)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if not (0 < train_end < validation_end < count):
        raise ValueError("not enough usable target timestamps for 70/15/15 split")
    end = eligible.iloc[-1] + pd.Timedelta(frequency)
    return {
        **settings,
        "train": [eligible.iloc[0].isoformat(), eligible.iloc[train_end].isoformat()],
        "validation": [
            eligible.iloc[train_end].isoformat(),
            eligible.iloc[validation_end].isoformat(),
        ],
        "test": [eligible.iloc[validation_end].isoformat(), end.isoformat()],
        "split_target_rows": {
            "train": train_end,
            "validation": validation_end - train_end,
            "test": count - validation_end,
            "warmup": len(frame) - count,
        },
    }


def backend_config(config: dict, budget: int | None = None) -> BackendConfig:
    raw = config.get("backend")
    if not isinstance(raw, dict):
        raise ValueError("missing frozen backend config")
    if budget is not None:
        raw = {**raw, "trial_budget": budget, "max_rounds": budget}
    return BackendConfig.model_validate(raw)


def fixed_values(family: ModelFamily, has_temperature: bool, has_other: bool) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {
        "calendar": "numerical",
        "interaction_mode": "none",
        "temperature_mode": "correlation" if has_temperature else "none",
        "load_mode": "none" if family in SEQUENTIAL else "correlation",
        "other_mode": "correlation" if has_other else "none",
    }
    if has_temperature:
        values["temperature_ratio"] = 0.5
    if has_other:
        values["other_ratio"] = 0.25
    if family in SEQUENTIAL:
        values.update(sequence_frequency=24, sequence_length=7)
    else:
        values["load_ratio"] = 0.2
    hp = FIXED_HP[family]
    values.update(
        {
            name: value
            for name, value in hp.items()
            if name
            in {
                "regularization",
                "alpha",
                "c",
                "gamma",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "layers",
                "hidden_size",
                "dropout",
                "fc_size",
                "kernel_size",
                "filters",
            }
        }
    )
    if family in FIXED_LR:
        values["learning_rate"] = FIXED_LR[family]
    return values


def _fixed_space(space: SearchSpace, family: ModelFamily, values: dict) -> SearchSpace:
    def fixed(domain: ParameterDomain) -> ParameterDomain:
        return ParameterDomain(
            name=domain.name,
            kind="fixed",
            value=values[domain.name],
            condition_on=domain.condition_on,
            condition_value=domain.condition_value,
        )

    relevant = space.parameters_for(family)
    # Conditional dimensions are represented only when active.
    params = tuple(fixed(item) for item in relevant if item.name in values)
    features = tuple(fixed(item) for item in space.features if item.name in values)
    return SearchSpace(
        version=f"{space.version}-fixed-{family.value}",
        provenance="manuscript fixed/default",
        families=(FamilySpace(family=family, parameters=params),),
        features=features,
    )


@dataclass
class Context:
    dataset: str
    series: str
    store: ArtifactStore
    db: Database
    runs: ExperimentRepository
    snapshot: DatasetSnapshot
    task: TaskDefinition
    protocol: EvaluationProtocol
    schedule: OriginSchedule
    space: SearchSpace
    templates: dict[ModelFamily, ResolvedCandidate]
    backend: BackendConfig
    metric: ValidationMetricConfig
    prepared: PreparedSnapshot
    source_metadata: dict
    seed: int
    mape_zero_policy: Literal["error", "omit", "epsilon"]
    mape_epsilon: float | None


def build_context(
    dataset: str,
    series: str,
    path: Path,
    config: dict,
    workspace: Path,
    *,
    quantile: bool = False,
    family: ModelFamily | None = None,
    budget: int | None = None,
) -> Context:
    spec = manifest(dataset)
    settings = require_protocol(config, dataset, series)
    frame, source_metadata = normalize(path, spec, settings["timezone"])
    settings = resolve_split(frame, settings, spec["frequency"])
    source_metadata["resolved_windows"] = {
        key: settings[key] for key in ("train", "validation", "test")
    }
    if "split_target_rows" in settings:
        source_metadata["split_target_rows"] = settings["split_target_rows"]
    policies = {}
    # Keep all permissible lookback for Table IV (24 samples x 24 h = 576 h),
    # while excluding unrelated years from the evaluator's lookup frame.
    first = pd.Timestamp(settings["train"][0]) - pd.Timedelta(days=31)
    last = pd.Timestamp(settings["test"][1])
    frame = frame.loc[(frame["timestamp"] >= first) & (frame["timestamp"] < last)].reset_index(
        drop=True
    )
    source_metadata["task_window_rows"] = len(frame)
    source_metadata["task_window_lookback_days"] = 31
    if "temperature" in frame:
        policies["temperature"] = AuxiliaryPolicy(
            kind=settings.get("temperature_policy", "observed"),
            role="temperature",
            protocol_id=settings.get("temperature_protocol_id"),
            source_ref=settings.get("temperature_source_ref"),
        )
    for name in spec.get("other", {}):
        policies[name] = AuxiliaryPolicy(kind="observed", role="other")
    data = NormalizedSeries.validate(frame, spec["frequency"], settings["timezone"], policies)
    workspace.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(workspace / "artifacts")
    ref = store.put_bytes(f"data/{uuid4()}.csv", data.frame.to_csv(index=False).encode())
    dataset_id = uuid4()
    snapshot = DatasetSnapshot(
        dataset_id=dataset_id,
        mapping_version=spec["version"],
        cleaning_version=f"{spec['missing_timestamps']}-v1",
        artifact=ref,
        frequency=spec["frequency"],
        timezone_name=settings["timezone"],
        series_id=series,
    )
    seed = int(config["seed"])
    output = (
        OutputConfig(
            representation="quantile",
            quantile_levels=tuple(config.get("quantiles", (0.1, 0.5, 0.9))),
        )
        if quantile
        else OutputConfig()
    )
    metric = ValidationMetricConfig(objective="crps" if quantile else "mae")
    task = TaskDefinition(
        task_id=uuid4(),
        dataset_id=dataset_id,
        delta=settings["delta"],
        horizon=settings["horizon"],
        time_unit=settings["offset_unit"],
        timezone_name=settings["timezone"],
        objective_id=metric.objective,
        forecast_output=output,
        auxiliary_policies=policies,
    )
    protocol = EvaluationProtocol(
        train=TimeWindow(start=settings["train"][0], end=settings["train"][1]),
        validation=TimeWindow(start=settings["validation"][0], end=settings["validation"][1]),
        test=TimeWindow(start=settings["test"][0], end=settings["test"][1]),
        seed=seed,
        metric_ids=[metric.objective],
        protocol_version=str(config["protocol_version"]),
    )
    index = IndexSpec(
        delta=settings["delta"],
        horizon=settings["horizon"],
        offset_unit=settings["offset_unit"],
        anchor=settings["anchor"],
        frequency=spec["frequency"],
    )
    from interactive_forecasting.services.data.core import ForecastIndex

    imputed = {pd.Timestamp(value) for value in source_metadata["imputed_target_timestamps"]}
    forecaster = ForecastIndex(index)
    times = tuple(
        value.to_pydatetime()
        for value in frame["timestamp"]
        if all(target not in imputed for target in forecaster.targets(pd.Timestamp(value)))
    )
    schedule = OriginSchedule(version=str(config["protocol_version"]), origins={series: times})
    caps = DatasetCapabilities(
        has_temperature="temperature" in frame,
        available_other_features=tuple(spec.get("other", {})),
        timestamp_frequency=spec["frequency"],
        supported_feature_groups=("calendar", "load", "temperature", "other"),
    )
    space = space_for_output(space_for_capabilities(canonical_research_space(), caps), output)
    if family is not None:
        space = SearchSpace(
            version=f"{space.version}-{family.value}",
            provenance=f"{space.provenance}; fixed family",
            parent_space_id=space.space_id,
            families=tuple(item for item in space.families if item.family == family),
            features=space.features,
        )
    families = (family,) if family is not None else tuple(item.family for item in space.families)
    templates = {}
    for item in families:
        hp: dict[str, int | float | str | bool | None] = dict(FIXED_HP[item])
        # Only paper-search dimensions are sampled; fixed non-search parameters remain in templates.
        features = CoreFeatureRecipe(
            version="table-iv-v1",
            calendar="numerical",
            load_lags=(0,) if item not in SEQUENTIAL else (),
            temperature_column="temperature" if "temperature" in policies else None,
            auxiliary_lags={name: (0,) for name in spec.get("other", {})},
            sequence_length=7 if item in SEQUENTIAL else None,
            sequence_frequency=24 if item in SEQUENTIAL else None,
        )
        training = (
            TrainingConfig(seed=seed)
            if item not in QUANTILE
            else TrainingConfig(
                seed=seed,
                epochs=settings["neural_epochs"],
                batch_size=settings["neural_batch_size"],
                learning_rate=FIXED_LR[item],
                optimizer=settings.get("optimizer", "adam"),
                patience=settings.get("early_stopping_patience"),
                weight_decay=settings.get("weight_decay", 0),
                gradient_clip_norm=settings.get("gradient_clip_norm"),
                validation_metric=settings.get("validation_metric", "training_loss"),
                loss="pinball" if quantile else "mae",
            )
        )
        templates[item] = ResolvedCandidate(
            family=item,
            auxiliary_policies=policies,
            hyperparameters=hp,
            features=features,
            preprocessing=PreprocessConfig(scale="standard"),
            training=training,
            output=output,
            index=index,
            configuration_version="paper-table-iii-iv-v1",
        )
    backend = backend_config(config, budget)
    db = Database(f"sqlite:///{workspace / 'runs.sqlite3'}")
    db.create_schema()
    runs = ExperimentRepository(db)
    prepared = PreparedSnapshot(
        dataset=snapshot,
        mapping=ColumnMapping(
            dataset_id=dataset_id,
            timestamp_column="timestamp",
            target_column="target",
            auxiliary_roles={name: policy.role for name, policy in policies.items()},
            availability=policies,
            confirmed=True,
        ),
        plan=PreparationPlan(timezone_name=settings["timezone"]),
        row_count=len(frame),
        time_start=frame["timestamp"].iloc[0].isoformat(),
        time_end=frame["timestamp"].iloc[-1].isoformat(),
        columns=tuple(frame.columns),
        applied_transformations=(
            f"{source_metadata['filled_timestamps']} timestamps causal-forward-filled",
        ),
        capabilities=caps,
    )
    return Context(
        dataset,
        series,
        store,
        db,
        runs,
        snapshot,
        task,
        protocol,
        schedule,
        space,
        templates,
        backend,
        metric,
        prepared,
        source_metadata,
        seed,
        settings["mape_zero_policy"],
        settings.get("mape_epsilon"),
    )


def engine(context: Context) -> SearchEngine:
    evaluator = ValidationEvaluator.from_snapshot(
        context.store,
        context.snapshot,
        context.task,
        next(iter(context.templates.values())).index,
        context.protocol,
        context.schedule,
    )
    return SearchEngine(
        context.runs, OptunaBackend(context.backend), TrialExecutor(evaluator, context.store)
    )


def selected_metrics(context: Context, run: ExperimentRun) -> dict[str, float | str]:
    """Unlock test only after the production engine has frozen its selection."""
    if run.selected_trial_id is None:
        return {}
    evaluated = FinalTestEvaluator(context.runs, context.store).evaluate(run.run_id)
    scores: dict[str, float | str] = dict(evaluated.final_test_metrics or {})
    if context.metric.objective == "mae":
        from interactive_forecasting.domain.forecasting import Partition, PointForecast
        from interactive_forecasting.services.data.core import (
            ForecastIndex,
            final_test_labels,
            split_examples,
        )
        from interactive_forecasting.services.metrics.core import mape
        from interactive_forecasting.services.models.core import FittedCoreModel

        selected = next(
            item for item in evaluated.trials if item.trial_id == evaluated.selected_trial_id
        )
        ref = next(item for item in evaluated.artifact_refs if item.uri == selected.artifact_uri)
        fitted = FittedCoreModel.load(context.store, ref)
        data = NormalizedSeries.from_csv_artifact(
            context.store,
            context.snapshot.artifact,
            frequency=context.snapshot.frequency,
            timezone_name=context.snapshot.timezone_name,
            auxiliary_availability=context.task.auxiliary_policies,
        )
        test = split_examples(
            data, ForecastIndex(fitted.candidate.index), context.protocol, context.schedule
        )[Partition.TEST]
        forecast = fitted.predict(data, test)
        if not isinstance(forecast, PointForecast):
            raise ValueError("selected point task returned non-point forecast")
        scores["mape"] = mape(
            final_test_labels(data, test),
            forecast,
            zero_policy=context.mape_zero_policy,
            epsilon=context.mape_epsilon,
        )
        scores["mape_zero_policy"] = context.mape_zero_policy
    return scores


def summarize(context: Context, run, experiment: str, *, extra: dict | None = None) -> dict:
    selected = next(
        (trial for trial in run.trials if trial.trial_id == run.selected_trial_id), None
    )
    result = {
        "experiment": experiment,
        "dataset": context.dataset,
        "series": context.series,
        "status": run.status.value,
        "run_id": str(run.run_id),
        "seed": context.seed,
        "budget": context.backend.trial_budget,
        "protocol_version": context.protocol.protocol_version,
        "source": context.source_metadata,
        "snapshot_sha256": context.snapshot.artifact.sha256,
        "space_version": context.space.version,
        "backend": context.backend.model_dump(mode="json"),
        "trials": [trial.model_dump(mode="json") for trial in run.trials],
        "selected_trial_id": str(run.selected_trial_id) if run.selected_trial_id else None,
        "selected_trial_number": selected.number if selected else None,
        "selected_candidate": selected.candidate.candidate.model_dump(mode="json")
        if selected and selected.candidate
        else None,
        "final_metrics": selected_metrics(context, run) if selected else {},
        "audit": extra or {},
    }
    return result


def run_search(context: Context, experiment: str, *, fixed: ModelFamily | None = None) -> dict:
    search = engine(context)
    space = context.space
    if experiment == "fixed_default":
        if fixed is None:
            raise ValueError("fixed_default requires a model family")
        values = fixed_values(
            fixed,
            context.prepared.capabilities.has_temperature,
            bool(context.prepared.capabilities.available_other_features),
        )
        space = _fixed_space(space, fixed, values)
        values = {
            key: val
            for key, val in values.items()
            if key in {item.name for item in space.domains_for(fixed)}
        }
    run = search.create_run(
        spec_id=experiment,
        task=context.task,
        snapshot=context.snapshot,
        protocol=context.protocol,
        schedule=context.schedule,
        space=space,
        templates=context.templates,
        metric=context.metric,
        experiment_seed=context.seed,
    )
    if experiment == "fixed_default":
        assert fixed is not None
        request = CandidateRequest(
            family=fixed, values=values, seed=context.seed, source="fixed_reference"
        )
        run = search.run_fixed(run.run_id, request)
    else:
        while run.status.value in {"queued", "running"}:
            run = search.run_round(run.run_id)
    return summarize(context, run, experiment)


def make_workflow(context: Context, runtime):
    from interactive_forecasting.domain.types import OptimizationMode
    from interactive_forecasting.orchestration.optimization import OptimizationWorkflow, RunSetup
    from interactive_forecasting.orchestration.state_machine import WorkflowStateMachine

    tasks = TaskRepository(context.db)
    tasks.create(
        Task(
            task_id=context.task.task_id,
            stage=Stage.OPTIMIZATION,
            substate=OptimizationState.INITIALIZE_SEARCH.value,
            workflow_status=WorkflowStatus.ACTIVE,
            definition_id=context.task.definition_id,
        )
    )
    PreparationRepository(context.db).create(
        PreparationRecord(
            task_id=context.task.task_id,
            step=PreparationStep.PREPARATION_READY,
            prepared=context.prepared,
            definition=context.task,
            protocol=context.protocol,
            effective_search_space=context.space,
            frozen=True,
        )
    )
    workflow = OptimizationWorkflow(
        tasks,
        PreparationRepository(context.db),
        context.runs,
        OptimizationSessionRepository(context.db),
        MessageRepository(context.db),
        LLMCallRepository(context.db),
        context.store,
        WorkflowStateMachine(),
        runtime,
    )
    setup = RunSetup(
        mode=OptimizationMode.LLM_GUIDED,
        spec_id="llm_guided",
        backend=context.backend,
        schedule=context.schedule,
        templates=context.templates,
        metric=context.metric,
        experiment_seed=context.seed,
    )
    return workflow, setup


def fingerprint(
    experiment: str,
    dataset: str,
    series: str,
    path: Path,
    config: dict,
    family: ModelFamily | None,
    budget: int | None,
) -> str:
    implementation = hashlib.sha256()
    for relative in (
        "pyproject.toml",
        "experiments/common.py",
        "experiments/run_all.py",
        "experiments/fixed_default.py",
        "experiments/vanilla_bo.py",
        "experiments/llm_guided.py",
        "experiments/probabilistic.py",
        f"experiments/datasets/{dataset}.yaml",
        "experiments/architecture_specific.py",
        "src/interactive_forecasting/services/optimization/space.py",
        "src/interactive_forecasting/services/optimization/engine.py",
        "src/interactive_forecasting/services/optimization/backend.py",
        "src/interactive_forecasting/services/features/core.py",
        "src/interactive_forecasting/services/models/core.py",
        "src/interactive_forecasting/services/models/neural.py",
        "src/interactive_forecasting/services/metrics/core.py",
        "src/interactive_forecasting/orchestration/optimization.py",
        "src/interactive_forecasting/agents/runtime.py",
        "src/interactive_forecasting/agents/transport.py",
        "src/interactive_forecasting/agents/contracts.py",
        "src/interactive_forecasting/agents/model_manager_policy.py",
        "src/interactive_forecasting/agents/skills/model_manager.md",
    ):
        implementation.update(relative.encode())
        implementation.update((ROOT / relative).read_bytes())
    payload = {
        "experiment": experiment,
        "dataset": dataset,
        "series": series,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": config,
        "family": family.value if family else None,
        "budget_override": budget,
        "canonical_space_version": canonical_research_space().version,
        "dependency_versions": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "torch",
                "optuna",
                "openai-agents",
                "PyYAML",
            )
        },
        "implementation_sha256": implementation.hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
