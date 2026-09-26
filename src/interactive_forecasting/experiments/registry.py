"""Typed experiment definitions; execution requires declared scientific inputs."""

from __future__ import annotations

from pydantic import Field, model_validator

from interactive_forecasting.domain.forecasting import OriginSchedule, ResolvedCandidate
from interactive_forecasting.domain.models import (
    DatasetSnapshot,
    EvaluationProtocol,
    TaskDefinition,
)
from interactive_forecasting.domain.search import (
    BackendConfig,
    SearchRecord,
    SearchSpace,
    ValidationMetricConfig,
)
from interactive_forecasting.domain.types import ModelFamily


class ExperimentDefinition(SearchRecord):
    spec_id: str = Field(min_length=1)
    entrypoint: str = Field(min_length=1)
    task: TaskDefinition
    snapshot: DatasetSnapshot
    protocol: EvaluationProtocol
    schedule: OriginSchedule
    search_space: SearchSpace
    backend: BackendConfig
    metric: ValidationMetricConfig
    templates: dict[ModelFamily, ResolvedCandidate]
    experiment_seed: int = Field(ge=0, le=2**32 - 1)
    artifact_requirements: tuple[str, ...] = ()

    @model_validator(mode="after")
    def mandatory_settings(self) -> ExperimentDefinition:
        if self.task.dataset_id != self.snapshot.dataset_id:
            raise ValueError("experiment dataset snapshot does not match task")
        if self.task.objective_id != self.metric.objective:
            raise ValueError("experiment validation objective does not match task")
        if self.metric.objective not in self.protocol.metric_ids:
            raise ValueError("experiment protocol does not authorize the objective")
        if not self.templates:
            raise ValueError("experiment requires complete family templates")
        return self


class ExperimentRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ExperimentDefinition] = {}

    def register(self, definition: ExperimentDefinition) -> None:
        if definition.spec_id in self._items:
            raise ValueError("experiment spec id is already registered")
        self._items[definition.spec_id] = definition

    def get(self, spec_id: str) -> ExperimentDefinition:
        if spec_id not in self._items:
            raise KeyError(f"experiment settings are missing for {spec_id}")
        return self._items[spec_id]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)
