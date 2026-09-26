"""Deterministic calendar and weather analogs from frozen historical deployment inputs."""

from __future__ import annotations

import json
from datetime import date, timedelta
from io import BytesIO
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from interactive_forecasting.domain.forecasting import AuxiliaryPolicy
from interactive_forecasting.domain.models import (
    Forecast,
    FutureAuxiliaryValue,
    ProfilePoint,
    ReferenceAnalysis,
    ReferenceDay,
    WeatherAnalog,
)
from interactive_forecasting.services.deployment.postprocessing import _external_at
from interactive_forecasting.storage.artifacts import ArtifactStore


def _day_times(day: date, timezone_name: str, frequency: str) -> tuple[pd.Timestamp, ...]:
    zone = ZoneInfo(timezone_name)
    start = pd.Timestamp(day).tz_localize(zone)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(zone)
    return tuple(pd.date_range(start, end, freq=frequency, inclusive="left").tz_convert("UTC"))


def analyze_references(
    forecast: Forecast,
    store: ArtifactStore,
    *,
    frequency: str,
    temperature_column: str | None,
    auxiliary_policies: dict[str, AuxiliaryPolicy],
    top_k: int = 3,
) -> ReferenceAnalysis:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    snapshot = pd.read_csv(BytesIO(store.read_bytes(forecast.prepared_snapshot)))
    context = pd.read_csv(BytesIO(store.read_bytes(forecast.context_artifact)))
    if tuple(snapshot.columns) != tuple(context.columns):
        raise ValueError("historical snapshot and deployment context schema differ")
    history = pd.concat((snapshot, context), ignore_index=True)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history = history.loc[
        history["timestamp"] <= pd.Timestamp(forecast.origin.latest_observed_at)
    ].drop_duplicates(["series_id", "timestamp"], keep="last")
    if set(history["series_id"]) != {"default"}:
        raise ValueError("reference analysis supports the frozen single series only")
    history = history.set_index("timestamp").sort_index()
    future = tuple(
        FutureAuxiliaryValue.model_validate(item)
        for item in json.loads(store.read_bytes(forecast.future_auxiliary_artifact))
    )
    target_date = (
        pd.Timestamp(forecast.target_timestamps[0]).tz_convert(forecast.origin.timezone_name).date()
    )
    if any(
        pd.Timestamp(target).tz_convert(forecast.origin.timezone_name).date() != target_date
        for target in forecast.target_timestamps
    ):
        raise ValueError("reference analysis currently requires targets on one local date")

    def profile(day: date, column: str) -> tuple[ProfilePoint, ...] | None:
        expected = _day_times(day, forecast.origin.timezone_name, frequency)
        if not expected or any(time not in history.index for time in expected):
            return None
        return tuple(
            ProfilePoint(timestamp=time.to_pydatetime(), value=float(history.at[time, column]))
            for time in expected
        )

    def calendar_reference(label: Literal["D-1", "D-7", "D-365"], offset: int) -> ReferenceDay:
        day = target_date - timedelta(days=offset)
        load = profile(day, "target")
        weather = (
            profile(day, temperature_column)
            if temperature_column and temperature_column in history
            else None
        )
        return ReferenceDay(
            label=label,
            date=day,
            available=load is not None,
            reason=None if load is not None else "historical load profile is incomplete or absent",
            load_profile=load or (),
            weather_profile=weather or (),
        )

    references = (
        calendar_reference("D-1", 1),
        calendar_reference("D-7", 7),
        calendar_reference("D-365", 365),
    )
    target_weather: tuple[ProfilePoint, ...] = ()
    analogs: tuple[WeatherAnalog, ...] = ()
    reason: str | None = None
    policy = auxiliary_policies.get(temperature_column) if temperature_column else None
    if temperature_column is None or policy is None or policy.role != "temperature":
        reason = "no frozen temperature variable"
    elif temperature_column not in history:
        reason = "temperature is absent from prepared history"
    else:
        expected = _day_times(target_date, forecast.origin.timezone_name, frequency)
        readings: list[ProfilePoint] = []
        for time in expected:
            if time <= pd.Timestamp(forecast.origin.latest_observed_at) and time in history.index:
                value = float(history.at[time, temperature_column])
            else:
                try:
                    value = _external_at(
                        forecast, temperature_column, time, future, auxiliary_policies
                    )
                except ValueError:
                    break
            readings.append(ProfilePoint(timestamp=time.to_pydatetime(), value=value))
        if len(readings) != len(expected) or not expected:
            reason = "target-day weather profile is incomplete or unavailable at origin"
        else:
            target_weather = tuple(readings)
            available_days = sorted(
                {
                    stamp.tz_convert(forecast.origin.timezone_name).date()
                    for stamp in history.index
                    if stamp.tz_convert(forecast.origin.timezone_name).date() < target_date
                }
            )
            candidates = []
            for day in available_days:
                weather = profile(day, temperature_column)
                load = profile(day, "target")
                if weather is not None and load is not None and len(weather) == len(target_weather):
                    candidates.append((day, load, weather))
            if not candidates:
                reason = "no complete historical weather and load profiles"
            else:
                scale = float(np.std([point.value for _, _, row in candidates for point in row]))
                if scale == 0:
                    scale = 1.0
                target_values = np.asarray([point.value for point in target_weather])
                scored = []
                for day, load, weather in candidates:
                    comparison = np.asarray([point.value for point in weather])
                    distance = float(np.sqrt(np.mean(((comparison - target_values) / scale) ** 2)))
                    scored.append(
                        WeatherAnalog(
                            date=day,
                            distance=distance,
                            load_profile=load,
                            weather_profile=weather,
                        )
                    )
                analogs = tuple(sorted(scored, key=lambda row: (row.distance, row.date))[:top_k])
    return ReferenceAnalysis(
        forecast_id=forecast.forecast_id,
        target_date=target_date,
        target_timestamps=forecast.target_timestamps,
        d_minus_1=references[0],
        d_minus_7=references[1],
        d_minus_365=references[2],
        target_weather_profile=target_weather,
        weather_analogs=analogs,
        weather_available=reason is None,
        weather_unavailable_reason=reason,
        top_k=top_k,
        provenance={
            "prepared_snapshot_sha256": forecast.prepared_snapshot.sha256,
            "context_sha256": forecast.context_artifact.sha256,
            "future_auxiliary_sha256": forecast.future_auxiliary_artifact.sha256,
            "distance": "normalized_rmse_v1; historical_std_zero_to_one; ties_by_date",
            "frequency": frequency,
            "timezone_name": forecast.origin.timezone_name,
        },
    )
