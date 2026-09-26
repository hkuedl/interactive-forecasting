"""Build the deterministic, read-only Deployment paper-demo fixture from GEF14.csv."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from interactive_forecasting.domain.forecasting import (
    AuxiliaryPolicy,
    CoreFeatureRecipe,
    ForecastKey,
    IndexSpec,
    LabelVector,
    OutputConfig,
    Partition,
    PreprocessConfig,
    ResolvedCandidate,
    TrainingConfig,
)
from interactive_forecasting.domain.models import (
    Adjustment,
    DeploymentSession,
    Forecast,
    ForecastOrigin,
    ForecastVersion,
    FutureAuxiliaryValue,
)
from interactive_forecasting.domain.types import ModelFamily
from interactive_forecasting.services.data.core import ForecastExample, NormalizedSeries
from interactive_forecasting.services.deployment.core import DeploymentSeries, forecast_examples
from interactive_forecasting.services.deployment.postprocessing import apply_adjustment
from interactive_forecasting.services.deployment.references import analyze_references
from interactive_forecasting.services.models.core import FittedCoreModel, fit_candidate
from interactive_forecasting.storage.artifacts import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/examples/GEF14.csv"
OUTPUT = ROOT / "data/examples/deployment_paper_demo.json"
ORIGIN = pd.Timestamp("2014-08-21T20:00:00Z")
TARGET = ORIGIN + pd.Timedelta(hours=1)
CREATED = datetime(2014, 8, 21, 20, 5, tzinfo=timezone.utc)
PROTOCOL = "paper-demo-perfect-temperature-v1"


def stable(label: str):
    return uuid5(NAMESPACE_URL, "interactive-forecasting/paper-demo/gef14/" + label)


def main() -> None:
    source = pd.read_csv(SOURCE)
    source["load"] = source["load"] / 1000.0
    # GEF14 has no timezone marker; interpret its recorded hourly clock as UTC here.
    source["timestamp"] = pd.to_datetime(source.pop("DateTime"), utc=True)
    source["temperature"] = source.pop("T")
    source = source[["timestamp", "load", "temperature"]].sort_values("timestamp")
    if source["timestamp"].duplicated().any():
        raise ValueError("GEF14 source contains duplicate timestamps")
    if not (source["timestamp"].min() <= TARGET - pd.Timedelta(days=365)):
        raise ValueError("GEF14 source lacks D-365")
    historical_source = source.loc[source.timestamp <= ORIGIN].copy()
    canonical = historical_source.rename(columns={"load": "target"}).copy()
    canonical.insert(1, "series_id", "default")
    canonical = canonical[["timestamp", "series_id", "target", "temperature"]]
    policy = AuxiliaryPolicy(kind="perfect_forecast", role="temperature", protocol_id=PROTOCOL)
    policies = {"temperature": policy}
    data = NormalizedSeries.validate(canonical, "1h", "UTC", policies)
    recipe = CoreFeatureRecipe(
        version="paper-demo-v1",
        sequence_length=8,
        sequence_frequency=1,
        temperature_column="temperature",
        temperature_leads=(1,),
    )
    candidate = ResolvedCandidate(
        family=ModelFamily.CNN,
        auxiliary_policies=policies,
        hyperparameters={"layers": 2, "kernel_size": 3, "filters": 8, "fc_size": 8, "dropout": 0.0},
        features=recipe,
        preprocessing=PreprocessConfig(scale="standard"),
        training=TrainingConfig(seed=17, epochs=100, batch_size=32, learning_rate=0.01, loss="mse"),
        output=OutputConfig(),
        index=IndexSpec(
            delta=1,
            horizon=1,
            frequency="1h",
            offset_unit="samples",
            anchor="start_at_delta",
        ),
        configuration_version="paper-demo-v1",
    )
    times = tuple(canonical.timestamp)

    def examples(start: int, end: int, part: Partition):
        return tuple(
            ForecastExample(
                ForecastKey(
                    series_id="default",
                    origin=times[i - 1].to_pydatetime(),
                    target=times[i].to_pydatetime(),
                ),
                None,
                part,
            )
            for i in range(start, end)
        )

    first_validation = len(times) - 7 * 24
    train = examples(first_validation - 28 * 24, first_validation, Partition.TRAIN)
    valid = examples(first_validation, len(times), Partition.VALIDATION)

    def labels(batch):
        return LabelVector(
            keys=tuple(item.key for item in batch),
            values=tuple(
                float(canonical.target.iloc[times.index(pd.Timestamp(item.key.target))])
                for item in batch
            ),
        )

    fitted = fit_candidate(candidate, data, train, labels(train), valid, labels(valid))
    task_id, run_id, trial_id = (stable(name) for name in ("task", "run", "trial"))
    session_id, forecast_id = (stable(name) for name in ("session", "forecast"))
    v0_id, v1_id, adjustment_id = (stable(name) for name in ("v0", "v1", "adjustment"))
    future_source = source.loc[
        (source.timestamp > ORIGIN) & (source.timestamp < ORIGIN.normalize() + pd.Timedelta(days=1))
    ]
    future = tuple(
        FutureAuxiliaryValue(
            column="temperature",
            valid_at=row.timestamp.to_pydatetime(),
            value=float(row.temperature),
            protocol_id=PROTOCOL,
        )
        for row in future_source.itertuples()
    )
    snapshot = canonical.loc[canonical.timestamp < ORIGIN.normalize()].copy()
    context = canonical.loc[canonical.timestamp >= ORIGIN.normalize()].copy()
    with TemporaryDirectory(prefix="iforecast-paper-demo-") as temporary:
        store = ArtifactStore(Path(temporary))
        model_ref = fitted.save(store, "models/paper_demo_cnn.zip")
        # The model bundle is real, but zipfile stamps entries with wall-clock time.
        # Normalize only ZIP headers so the read-only fixture is byte-reproducible.
        stable_bundle = BytesIO()
        with zipfile.ZipFile(BytesIO(store.read_bytes(model_ref))) as source_bundle:
            with zipfile.ZipFile(
                stable_bundle, "w", compression=zipfile.ZIP_DEFLATED
            ) as target_bundle:
                for name in source_bundle.namelist():
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    target_bundle.writestr(
                        info, source_bundle.read(name), compress_type=zipfile.ZIP_DEFLATED
                    )
        model_ref = store.put_bytes("models/paper_demo_cnn_stable.zip", stable_bundle.getvalue())
        raw_ref = store.put_bytes(
            "input/new_observations.csv",
            historical_source.loc[historical_source.timestamp >= ORIGIN.normalize()]
            .to_csv(index=False)
            .encode(),
        )
        snapshot_ref = store.put_bytes(
            "input/prepared_snapshot.csv", snapshot.to_csv(index=False).encode()
        )
        context_ref = store.put_bytes(
            "input/validated_context.csv", context.to_csv(index=False).encode()
        )
        future_ref = store.put_bytes(
            "input/future_temperature.json",
            json.dumps([item.model_dump(mode="json") for item in future], sort_keys=True).encode(),
        )
        origin = ForecastOrigin(
            task_id=task_id,
            selected_trial_id=trial_id,
            model_artifact=model_ref,
            latest_observed_at=ORIGIN.to_pydatetime(),
            targets=(TARGET.to_pydatetime(),),
            delta=1,
            horizon=1,
            offset_unit="samples",
            anchor="start_at_delta",
            timezone_name="UTC",
        )
        deployment_data = DeploymentSeries(
            frame=data.frame,
            frequency=data.frequency,
            timezone_name=data.timezone_name,
            auxiliary_availability=data.auxiliary_availability,
            future={(item.column, pd.Timestamp(item.valid_at)): item for item in future},
            origin=ORIGIN,
        )
        prediction = FittedCoreModel.load(store, model_ref).predict(
            deployment_data, forecast_examples(origin)
        )
        forecast = Forecast(
            forecast_id=forecast_id,
            task_id=task_id,
            optimization_run_id=run_id,
            selected_trial_id=trial_id,
            deployment_session_id=session_id,
            model_artifact=model_ref,
            origin=origin,
            target_timestamps=origin.targets,
            prediction_representation="point",
            prediction=prediction,
            raw_upload=raw_ref,
            context_artifact=context_ref,
            future_auxiliary_artifact=future_ref,
            prepared_snapshot=snapshot_ref,
            created_at=CREATED,
        )
        reference = analyze_references(
            forecast,
            store,
            frequency="1h",
            temperature_column="temperature",
            auxiliary_policies=policies,
        )
        reference = reference.model_copy(
            update={"analysis_id": stable("reference"), "created_at": CREATED}
        )
        if not all(
            (
                reference.d_minus_1.available,
                reference.d_minus_7.available,
                reference.d_minus_365.available,
                reference.weather_available,
                len(reference.weather_analogs) == 3,
            )
        ):
            raise ValueError("required paper-demo references unavailable")
        latest_load = float(canonical.target.iloc[-1])
        suggested_adjustment = 0.02  # Conservative user choice, not inferred from analogs.
        origin_gap = latest_load / float(prediction.values[0]) - 1
        if not 0 < suggested_adjustment < origin_gap <= 0.05:
            raise ValueError("demo adjustment must be smaller than the at-origin gap")
        hour_label = TARGET.strftime("%H:%M")
        percent_label = f"{suggested_adjustment * 100:.0f}%"
        original = ForecastVersion(
            version_id=v0_id,
            forecast_id=forecast_id,
            version_number=0,
            prediction_representation="point",
            prediction=prediction,
            provenance={"source": "saved_model", "protocol_id": PROTOCOL},
            created_at=CREATED,
        )
        draft = Adjustment(
            adjustment_id=adjustment_id,
            task_id=task_id,
            forecast_id=forecast_id,
            parent_version_id=v0_id,
            adjustment_type="time_scaling",
            start_at=TARGET.to_pydatetime(),
            lambda_value=suggested_adjustment,
            source="user",
            user_request_text=(
                f"Please apply the confirmed {percent_label} increase at {hour_label} UTC."
            ),
            created_at=CREATED,
        )
        adjusted = apply_adjustment(forecast, original, draft)
        adjusted = adjusted.model_copy(
            update={
                "version_id": v1_id,
                "created_at": CREATED,
            }
        )
        applied = draft.model_copy(
            update={
                "status": "applied",
                "confirmed_at": CREATED,
                "confirmation_id": stable("confirmation"),
                "applied_version_id": v1_id,
            }
        )
        session = DeploymentSession(
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
            selected_trial_id=trial_id,
            model_artifact=model_ref,
            state="FORECAST_VERSION_UPDATED",
            version=6,
            raw_upload=raw_ref,
            upload_extension=".csv",
            future_auxiliaries=future,
            context_artifact=context_ref,
            forecast_id=forecast_id,
            reference_analysis_id=reference.analysis_id,
            current_version_id=v1_id,
            created_at=CREATED,
        )
        original_value = float(prediction.values[0])
        adjusted_value = float(adjusted.prediction.values[0])
        # Presentation-only: D-1 shape is scaled to the single saved-model target hour.
        # This does not create model predictions for the other 23 hours.
        day_times = pd.date_range(ORIGIN.normalize(), periods=24, freq="h", tz="UTC")
        previous_load = [point.value for point in reference.d_minus_1.load_profile]
        target_index = int((TARGET - TARGET.normalize()) / pd.Timedelta(hours=1))
        if len(previous_load) != len(day_times) or previous_load[target_index] == 0:
            raise ValueError("cannot construct an honest illustrative daily profile")
        factor = original_value / previous_load[target_index]
        original_profile = [float(value * factor) for value in previous_load]
        original_profile[target_index] = original_value
        adjusted_profile = original_profile.copy()
        adjusted_profile[target_index] = adjusted_value
        presentation = {
            "dataset_label": "GEF14.csv · load in thousands",
            "model_label": "CNN",
            "day_profile": {
                "timestamps": [stamp.isoformat() for stamp in day_times],
                "original_values": original_profile,
                "adjusted_values": adjusted_profile,
                "target_index": target_index,
                "disclosure": (
                    "Illustrative full-day shape: the D-1 load profile is scaled to the saved "
                    f"model's {hour_label} forecast. Only {hour_label} is a "
                    "model-predicted target. "
                    f"The confirmed +{percent_label} adjustment changes that target hour only."
                ),
            },
        }
        messages = [
            {
                "role": "user",
                "text": "The evening forecast looks a little low. Can you show me "
                "some similar days?",
            },
            {
                "role": "task_manager",
                "text": "Sure. I've displayed calendar and weather-based reference days "
                "for comparison.",
            },
            {
                "role": "user",
                "text": "I've looked at those. I still think the evening forecast is a "
                f"little low. Can we preview a {percent_label} increase at {hour_label}?",
            },
            {
                "role": "task_manager",
                "text": "Here's the adjusted preview. The original forecast is still shown "
                "for comparison.",
            },
            {"role": "user", "text": "Looks good. Apply it."},
            {
                "role": "task_manager",
                "text": "Done. The updated forecast has been saved.",
            },
        ]
        fixture = {
            "metadata": {
                "task_id": str(task_id),
                "dataset": "GEF14.csv",
                "target_date": str(TARGET.date()),
                "target_timestamp": TARGET.isoformat(),
                "model": "CNN, H=1, Δ=1 h",
                "temperature_policy": policy.model_dump(mode="json"),
                "source": "data/examples/GEF14.csv",
                "note": (
                    "Paper preview · GEF14 load shown in thousands. The CNN and weather analogs "
                    "use the declared perfect-temperature protocol. The CNN forecasts "
                    f"{hour_label} "
                    "only; the timezone-free source clock is treated as UTC."
                ),
            },
            "session": session.model_dump(mode="json"),
            "forecast": forecast.model_dump(mode="json"),
            "original": original.model_dump(mode="json"),
            "versions": [original.model_dump(mode="json"), adjusted.model_dump(mode="json")],
            "adjustments": [applied.model_dump(mode="json")],
            "reference": reference.model_dump(mode="json"),
            "variables": ["temperature"],
            "messages": messages,
            "presentation": presentation,
        }
        OUTPUT.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"Fixture: {OUTPUT}")
    print(f"Target {TARGET.isoformat()}, v0 {original_value:.3f}, v1 {adjusted_value:.3f}")
    print(
        "Analogs:",
        [(str(item.date), round(item.distance, 3)) for item in reference.weather_analogs],
    )


if __name__ == "__main__":
    main()
