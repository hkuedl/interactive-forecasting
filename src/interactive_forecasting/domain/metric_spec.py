"""Frozen, serializable evaluation-metric choices; never executable user code."""

from __future__ import annotations

from datetime import time
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TimeRangeWeight(MetricRecord):
    """Half-open target-local clock interval; one positive override weight."""

    start_local: time
    end_local: time
    weight: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self) -> TimeRangeWeight:
        if self.start_local.tzinfo is not None or self.end_local.tzinfo is not None:
            raise ValueError("time-range endpoints must be local clock times without offsets")
        if self.start_local >= self.end_local:
            raise ValueError("time range must be increasing and cannot cross midnight")
        return self


class MetricSpec(MetricRecord):
    kind: Literal["standard", "weighted", "asymmetric"]
    base_metric: Literal["mae", "mape", "crps"]
    time_range: TimeRangeWeight | None = None
    timezone_name: str | None = None
    over_weight: float | None = Field(default=None, gt=0)
    under_weight: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def supported_choice(self) -> MetricSpec:
        if self.kind == "standard":
            if any(
                value is not None
                for value in (
                    self.time_range,
                    self.timezone_name,
                    self.over_weight,
                    self.under_weight,
                )
            ):
                raise ValueError("standard metric cannot have weighting settings")
        elif self.kind == "weighted":
            if self.base_metric != "mae":
                raise ValueError("only weighted MAE is supported")
            if self.time_range is None or not self.timezone_name:
                raise ValueError("weighted MAE needs a time range and task timezone")
            if self.over_weight is not None or self.under_weight is not None:
                raise ValueError("weighted MAE cannot have asymmetric penalties")
            try:
                ZoneInfo(self.timezone_name)
            except ZoneInfoNotFoundError as exc:
                raise ValueError("weighted metric timezone is not recognized") from exc
        else:
            if self.base_metric != "mae":
                raise ValueError("only asymmetric MAE is supported")
            if self.over_weight is None or self.under_weight is None:
                raise ValueError("asymmetric MAE needs positive over and under weights")
            if self.time_range is not None or self.timezone_name is not None:
                raise ValueError("asymmetric MAE cannot have time-range weighting")
        return self

    @property
    def objective_id(self) -> str:
        if self.kind == "standard":
            return self.base_metric
        return "weighted_mae" if self.kind == "weighted" else "asymmetric_mae"
