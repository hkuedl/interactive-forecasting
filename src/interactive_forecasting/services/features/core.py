"""One origin-safe feature builder and train-fitted transforms for all model families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from interactive_forecasting.domain.forecasting import (
    CoreFeatureRecipe,
    FeatureProvenance,
    ForecastKey,
    LabelVector,
    PreprocessConfig,
)
from interactive_forecasting.services.data.core import (
    SOURCE_CLOCK,
    ForecastExample,
    ForecastIndex,
    NormalizedSeries,
)


@dataclass(frozen=True)
class FeatureBatch:
    keys: tuple[ForecastKey, ...]
    names: tuple[str, ...]
    values: np.ndarray
    representation: str
    structured: FeatureBatch | None = None

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values) or not self.keys:
            raise ValueError("feature rows must align with keys")
        dimensions = {"tabular": 2, "sequence": 3}
        if (
            self.representation not in dimensions
            or self.values.ndim != dimensions[self.representation]
            or self.values.shape[-1] != len(self.names)
            or len(set(self.names)) != len(self.names)
            or len(set(self.keys)) != len(self.keys)
        ):
            raise ValueError("invalid feature shape, schema or duplicate keys")
        if not np.isfinite(self.values).all():
            raise ValueError("non-finite feature values")
        if self.structured is not None and (
            self.representation != "sequence"
            or self.structured.representation != "tabular"
            or self.structured.keys != self.keys
        ):
            raise ValueError("structured features must be aligned tabular rows")


class FeatureBuilder:
    def __init__(self, recipe: CoreFeatureRecipe, index: ForecastIndex):
        self.recipe = recipe
        self.index = index

    def build(
        self,
        data: NormalizedSeries,
        examples: tuple[ForecastExample, ...],
        *,
        representation: str,
    ) -> FeatureBatch:
        if data.frequency != self.index.spec.frequency:
            raise ValueError("feature frequency differs from forecast index")
        keys = tuple(example.key for example in examples)
        for key in keys:
            if pd.Timestamp(key.target) not in self.index.targets(pd.Timestamp(key.origin)):
                raise ValueError("forecast key does not match configured index")
        if representation == "sequence":
            if self.recipe.sequence_length is None or self.recipe.sequence_frequency is None:
                raise ValueError("sequence representation requires length and frequency")
            sequence_rows = [
                [
                    [
                        self._value(
                            data,
                            key,
                            "target",
                            self.index.lag(
                                pd.Timestamp(key.origin), step * self.recipe.sequence_frequency
                            ),
                        )
                    ]
                    for step in reversed(range(self.recipe.sequence_length))
                ]
                for key in keys
            ]
            values = np.asarray(sequence_rows, dtype=float)
            names: tuple[str, ...] = ("load",)
            structured_names = self._tabular_names()
            structured = (
                FeatureBatch(
                    keys,
                    structured_names,
                    np.asarray([self._tabular_row(data, key) for key in keys], dtype=float),
                    "tabular",
                )
                if structured_names
                else None
            )
        elif representation == "tabular":
            names = self._tabular_names()
            tabular_rows = [self._tabular_row(data, key) for key in keys]
            values = np.asarray(tabular_rows, dtype=float)
            if values.ndim != 2 or values.shape[1] == 0:
                raise ValueError("tabular recipe produced no features")
        else:
            raise ValueError("unknown feature representation")
        return FeatureBatch(
            keys,
            names,
            values,
            representation,
            structured if representation == "sequence" else None,
        )

    def _value(
        self, data: NormalizedSeries, key: ForecastKey, column: str, timestamp: pd.Timestamp
    ) -> float:
        origin = pd.Timestamp(key.origin)
        if column == "target":
            if timestamp > origin:
                raise ValueError("target feature would read after forecast origin")
        else:
            policy = data.auxiliary_availability.get(column)
            if policy is None:
                raise ValueError(f"unmapped auxiliary variable: {column}")
            if policy.kind == "observed":
                if timestamp > origin:
                    raise ValueError("observed auxiliary is unavailable after origin")
                if (
                    data.has_availability_time(column)
                    and data.available_at(key.series_id, timestamp, column) > origin
                ):
                    raise ValueError("observed auxiliary was unavailable at origin")
            elif policy.kind == "known_ahead":
                if (
                    data.has_availability_time(column)
                    and data.available_at(key.series_id, timestamp, column) > origin
                ):
                    raise ValueError("known-ahead auxiliary was unavailable at origin")
            elif policy.kind == "forecast":
                if data.available_at(key.series_id, timestamp, column) > origin:
                    raise ValueError("forecast product was not issued by origin")
            elif policy.kind == "perfect_forecast" and column != self.recipe.temperature_column:
                raise ValueError("perfect_forecast is restricted to configured temperature")
        return data.at(key.series_id, timestamp, column)

    def _tabular_names(self) -> tuple[str, ...]:
        names: list[str] = []
        mode = self.recipe.calendar
        if mode == "numerical":
            names += ["calendar_hour", "calendar_weekday", "calendar_month"]
        elif mode == "categorical":
            names += [f"hour_{i}" for i in range(24)]
            names += [f"weekday_{i}" for i in range(7)]
            names += [f"month_{i}" for i in range(1, 13)]
        elif mode == "trigonometric":
            names += [
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
                "month_sin",
                "month_cos",
            ]
        names += [f"load_lag_{lag}" for lag in self.recipe.load_lags]
        names += [f"temperature_lag_{lag}" for lag in self.recipe.temperature_lags]
        names += [f"temperature_lead_{lead}" for lead in self.recipe.temperature_leads]
        names += [f"temperature_day_{day}_avg" for day in self.recipe.temperature_daily_days]
        for column, lags in sorted(self.recipe.auxiliary_lags.items()):
            names += [f"{column}_lag_{lag}" for lag in lags]
        for column, leads in sorted(self.recipe.auxiliary_leads.items()):
            names += [f"{column}_lead_{lead}" for lead in leads]
        names += [f"{left}_x_{right}" for left, right in self.recipe.interactions]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature names")
        return tuple(names)

    def _tabular_row(self, data: NormalizedSeries, key: ForecastKey) -> list[float]:
        origin = pd.Timestamp(key.origin)
        target = pd.Timestamp(key.target)
        if data.timezone_name != SOURCE_CLOCK:
            target = target.tz_convert(data.timezone_name)
        mode = self.recipe.calendar
        values: dict[str, float] = {}
        if mode == "numerical":
            values.update(
                calendar_hour=float(target.hour),
                calendar_weekday=float(target.dayofweek),
                calendar_month=float(target.month),
            )
        elif mode == "categorical":
            values.update({f"hour_{i}": float(target.hour == i) for i in range(24)})
            values.update({f"weekday_{i}": float(target.dayofweek == i) for i in range(7)})
            values.update({f"month_{i}": float(target.month == i) for i in range(1, 13)})
        elif mode == "trigonometric":
            for name, number, period in (
                ("hour", target.hour, 24),
                ("weekday", target.dayofweek, 7),
                ("month", target.month - 1, 12),
            ):
                values[f"{name}_sin"] = float(np.sin(2 * np.pi * number / period))
                values[f"{name}_cos"] = float(np.cos(2 * np.pi * number / period))
        for lag in self.recipe.load_lags:
            values[f"load_lag_{lag}"] = self._value(
                data, key, "target", self.index.lag(origin, lag)
            )
        temp = self.recipe.temperature_column
        if temp is not None:
            for lag in self.recipe.temperature_lags:
                values[f"temperature_lag_{lag}"] = self._value(
                    data, key, temp, self.index.lag(origin, lag)
                )
            for lead in self.recipe.temperature_leads:
                values[f"temperature_lead_{lead}"] = self._value(
                    data, key, temp, origin + self.index.step * lead
                )
            steps_per_day = pd.Timedelta(days=1) / self.index.step
            if self.recipe.temperature_daily_days and int(steps_per_day) != steps_per_day:
                raise ValueError("daily averages require frequency dividing one day")
            day_steps = int(steps_per_day)
            for day in self.recipe.temperature_daily_days:
                samples = [
                    self._value(data, key, temp, self.index.lag(origin, (day - 1) * day_steps + i))
                    for i in range(day_steps)
                ]
                values[f"temperature_day_{day}_avg"] = float(np.mean(samples))
        for column, lags in sorted(self.recipe.auxiliary_lags.items()):
            for lag in lags:
                values[f"{column}_lag_{lag}"] = self._value(
                    data, key, column, self.index.lag(origin, lag)
                )
        for column, leads in sorted(self.recipe.auxiliary_leads.items()):
            for lead in leads:
                values[f"{column}_lead_{lead}"] = self._value(
                    data, key, column, origin + self.index.step * lead
                )
        for left, right in self.recipe.interactions:
            if left not in values or right not in values:
                raise ValueError("interaction references unknown base feature")
            values[f"{left}_x_{right}"] = values[left] * values[right]
        return [values[name] for name in self._tabular_names()]


@dataclass(frozen=True)
class FittedFeatureTransform:
    config: PreprocessConfig
    provenance: FeatureProvenance

    @staticmethod
    def group_columns(recipe: CoreFeatureRecipe) -> dict[str, tuple[str, ...]]:
        """Historical temperature and load candidates; other auxiliaries are separate."""
        return {
            "temperature": (
                tuple(f"temperature_lag_{lag}" for lag in recipe.temperature_lags)
                + tuple(f"temperature_day_{day}_avg" for day in recipe.temperature_daily_days)
            ),
            "load": tuple(f"load_lag_{lag}" for lag in recipe.load_lags),
            "other": (
                tuple(
                    f"{column}_lag_{lag}"
                    for column, lags in sorted(recipe.auxiliary_lags.items())
                    for lag in lags
                )
                + tuple(
                    f"{column}_lead_{lead}"
                    for column, leads in sorted(recipe.auxiliary_leads.items())
                    for lead in leads
                )
            ),
        }

    @staticmethod
    def _pearson_indices(values: np.ndarray, target: np.ndarray, count: int) -> list[int]:
        if count < 0:
            raise ValueError("selection count cannot be negative")
        if count == 0:
            return []
        scores = []
        for column in values.T:
            if np.std(column) == 0 or np.std(target) == 0:
                scores.append(0.0)
            else:
                score = abs(float(np.corrcoef(column, target)[0, 1]))
                scores.append(score if np.isfinite(score) else 0.0)
        return sorted(sorted(range(values.shape[1]), key=lambda i: (-scores[i], i))[:count])

    @classmethod
    def fit(
        cls,
        train: FeatureBatch,
        labels: LabelVector,
        config: PreprocessConfig,
        recipe: CoreFeatureRecipe | None = None,
    ) -> FittedFeatureTransform:
        if train.keys != labels.keys:
            raise ValueError("training feature/label keys must align")
        names = train.names
        values = train.values
        settings = (
            {
                name: setting
                for name, setting in (
                    ("temperature", recipe.temperature_selection),
                    ("load", recipe.load_selection),
                    ("other", recipe.other_selection),
                )
                if setting is not None
            }
            if recipe is not None and train.representation == "tabular"
            else {}
        )
        group_candidates: dict[str, tuple[str, ...]] = {}
        group_selected: dict[str, tuple[str, ...]] = {}
        if settings:
            if config.selection != "none" or train.representation != "tabular":
                raise ValueError("group selection requires tabular features and no global selector")
            assert recipe is not None
            group_candidates = cls.group_columns(recipe)
            positions = {name: i for i, name in enumerate(names)}
            if any(
                column not in positions for group in group_candidates.values() for column in group
            ):
                raise ValueError("group candidates differ from feature schema")
            selected_set = set(names)
            target = np.asarray(labels.values, dtype=float)
            for group, setting in settings.items():
                candidates = group_candidates[group]
                if setting.mode == "correlation":
                    assert setting.ratio is not None
                    indices = [positions[name] for name in candidates]
                    chosen = cls._pearson_indices(
                        values[:, indices], target, int(np.ceil(len(indices) * setting.ratio))
                    )
                    kept = tuple(candidates[i] for i in chosen)
                elif setting.mode == "fixed":
                    kept = candidates
                else:
                    kept = ()
                group_selected[group] = kept
                selected_set.difference_update(set(candidates) - set(kept))
            selected = tuple(name for name in names if name in selected_set)
        elif config.selection == "pearson":
            if train.representation != "tabular":
                raise ValueError("Pearson selection requires tabular features")
            ratio = config.selection_ratio
            assert ratio is not None
            indices = cls._pearson_indices(
                values, np.asarray(labels.values, dtype=float), int(np.ceil(len(names) * ratio))
            )
            selected = tuple(names[i] for i in indices)
        else:
            selected = names
        if not selected:
            raise ValueError("feature selection produced no features")
        selected_values = cls._select(values, names, selected, train.representation)
        if config.scale == "standard":
            if train.representation == "sequence":
                flattened = selected_values.reshape(-1, selected_values.shape[-1])
            else:
                flattened = selected_values
            means = tuple(float(v) for v in flattened.mean(axis=0))
            scales = tuple(float(v) if v > 0 else 1.0 for v in flattened.std(axis=0))
        else:
            means, scales = (), ()
        structured_provenance = (
            cls.fit(train.structured, labels, config, recipe).provenance
            if train.structured is not None
            else None
        )
        return cls(
            config,
            FeatureProvenance(
                train_keys=train.keys,
                input_names=names,
                selected_names=selected,
                group_candidates=group_candidates,
                group_selected=group_selected,
                group_settings=settings,
                means=means,
                scales=scales,
                structured=structured_provenance,
            ),
        )

    @classmethod
    def validate_provenance(cls, recipe: CoreFeatureRecipe, state: FeatureProvenance) -> None:
        if state.structured is not None:
            cls.validate_provenance(recipe, state.structured)
            return
        if state.input_names == ("load",):
            return
        settings = {
            name: setting
            for name, setting in (
                ("temperature", recipe.temperature_selection),
                ("load", recipe.load_selection),
                ("other", recipe.other_selection),
            )
            if setting is not None
        }
        if not settings:
            if state.group_candidates or state.group_selected or state.group_settings:
                raise ValueError("artifact group selector state disagrees with candidate")
            return
        candidates = cls.group_columns(recipe)
        if state.group_settings != settings or state.group_candidates != candidates:
            raise ValueError("artifact group selector settings or candidates mismatch")
        if set(state.group_selected) != set(settings):
            raise ValueError("artifact selected feature groups mismatch")
        retained = set(state.input_names)
        for group, selection in settings.items():
            pool = set(candidates[group])
            chosen = set(state.group_selected[group])
            if not chosen.issubset(pool):
                raise ValueError("artifact selected columns are not group candidates")
            if selection.mode == "none" and chosen:
                raise ValueError("artifact none group contains selected columns")
            if selection.mode == "fixed" and chosen != pool:
                raise ValueError("artifact fixed group does not retain all columns")
            if selection.mode == "correlation":
                assert selection.ratio is not None
                if len(chosen) != int(np.ceil(len(pool) * selection.ratio)):
                    raise ValueError("artifact group selected count mismatch")
            retained.difference_update(pool - chosen)
        if state.selected_names != tuple(name for name in state.input_names if name in retained):
            raise ValueError("artifact final feature order mismatch")

    @staticmethod
    def _select(
        values: np.ndarray,
        names: tuple[str, ...],
        selected: tuple[str, ...],
        representation: str,
    ) -> np.ndarray:
        if representation == "sequence":
            if names != selected:
                raise ValueError("sequence feature selection is unsupported")
            return values
        positions = [names.index(name) for name in selected]
        return values[:, positions]

    def transform(self, batch: FeatureBatch) -> FeatureBatch:
        if batch.names != self.provenance.input_names:
            raise ValueError("feature schema differs from fitted transform")
        values = self._select(
            batch.values, batch.names, self.provenance.selected_names, batch.representation
        )
        if self.config.scale == "standard":
            values = (values - np.asarray(self.provenance.means)) / np.asarray(
                self.provenance.scales
            )
        if (batch.structured is None) != (self.provenance.structured is None):
            raise ValueError("structured feature schema differs from fitted transform")
        structured = (
            FittedFeatureTransform(self.config, self.provenance.structured).transform(
                batch.structured
            )
            if batch.structured is not None and self.provenance.structured is not None
            else None
        )
        return FeatureBatch(
            batch.keys, self.provenance.selected_names, values, batch.representation, structured
        )
