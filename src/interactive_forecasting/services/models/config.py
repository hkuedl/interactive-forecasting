"""Family-specific resolved hyperparameters and executable capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from interactive_forecasting.domain.forecasting import ResolvedCandidate, TrainingConfig
from interactive_forecasting.domain.types import ModelFamily


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class LinearHP(Hyperparameters):
    regularization: Literal["none", "ridge"]
    alpha: float | None = Field(default=None, gt=0)
    fit_intercept: bool
    ridge_tol: float | None = Field(default=None, gt=0)
    ridge_max_iter: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def conditional_alpha(self) -> LinearHP:
        if (self.regularization == "ridge") != (self.alpha is not None):
            raise ValueError("alpha is required only for ridge regularization")
        if self.regularization != "ridge" and (
            self.ridge_tol is not None or self.ridge_max_iter is not None
        ):
            raise ValueError("ridge solver settings require ridge regularization")
        return self


class SVRHP(Hyperparameters):
    c: float = Field(gt=0)
    gamma: float = Field(gt=0)
    epsilon: float = Field(ge=0)
    kernel: Literal["rbf"]
    tol: float = Field(default=1e-3, gt=0)
    shrinking: bool = True
    max_iter: int = Field(default=-1, ge=-1)

    @model_validator(mode="after")
    def iteration_limit(self) -> SVRHP:
        if self.max_iter == 0:
            raise ValueError("SVR max_iter must be -1 or positive")
        return self


class XGBoostHP(Hyperparameters):
    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_lambda: float = Field(ge=0)
    tree_method: Literal["hist"]
    min_child_weight: float = Field(default=1.0, ge=0)
    gamma: float = Field(default=0.0, ge=0)
    reg_alpha: float = Field(default=0.0, ge=0)
    max_bin: int = Field(default=256, ge=2)


class MLPHP(Hyperparameters):
    layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class RecurrentHP(Hyperparameters):
    layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    fc_size: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def conditional_dropout(self) -> RecurrentHP:
        if self.layers == 1 and self.dropout != 0:
            raise ValueError("recurrent dropout requires at least two layers")
        return self


class CNNHP(Hyperparameters):
    layers: int = Field(gt=0)
    kernel_size: int = Field(gt=0)
    filters: int = Field(gt=0)
    fc_size: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


HYPERPARAMETER_SCHEMAS: dict[ModelFamily, type[Hyperparameters]] = {
    ModelFamily.LINEAR: LinearHP,
    ModelFamily.SVR: SVRHP,
    ModelFamily.XGBOOST: XGBoostHP,
    ModelFamily.MLP: MLPHP,
    ModelFamily.LSTM: RecurrentHP,
    ModelFamily.GRU: RecurrentHP,
    ModelFamily.CNN: CNNHP,
}


@dataclass(frozen=True)
class ModelCapabilities:
    representation: Literal["tabular", "sequence"]
    point: bool
    quantile: bool
    custom_differentiable_loss: bool
    sample_weighting: bool
    gpu: bool
    multi_output: bool


CAPABILITIES: dict[ModelFamily, ModelCapabilities] = {
    ModelFamily.LINEAR: ModelCapabilities("tabular", True, False, False, True, False, False),
    ModelFamily.SVR: ModelCapabilities("tabular", True, False, False, True, False, False),
    ModelFamily.XGBOOST: ModelCapabilities("tabular", True, False, False, True, False, False),
    ModelFamily.MLP: ModelCapabilities("tabular", True, True, False, False, True, False),
    ModelFamily.LSTM: ModelCapabilities("sequence", True, True, False, False, True, False),
    ModelFamily.GRU: ModelCapabilities("sequence", True, True, False, False, True, False),
    ModelFamily.CNN: ModelCapabilities("sequence", True, True, False, False, True, False),
}


def validate_candidate(candidate: ResolvedCandidate) -> Hyperparameters:
    schema = HYPERPARAMETER_SCHEMAS[candidate.family]
    hp = schema.model_validate(candidate.hyperparameters)
    from interactive_forecasting.services.data.core import ForecastIndex

    ForecastIndex(candidate.index)
    recipe = candidate.features
    for name in recipe.auxiliary_lags.keys() | recipe.auxiliary_leads.keys():
        if name not in candidate.auxiliary_policies:
            raise ValueError("feature references an unmapped auxiliary")
    leads = set(recipe.auxiliary_leads)
    if recipe.temperature_leads and recipe.temperature_column:
        leads.add(recipe.temperature_column)
    if not leads.issubset(candidate.auxiliary_policies):
        raise ValueError("feature references an unmapped auxiliary")
    if any(candidate.auxiliary_policies[name].kind == "observed" for name in leads):
        raise ValueError("observed auxiliary cannot supply future leads")
    group_settings = (
        recipe.temperature_selection,
        recipe.load_selection,
        recipe.other_selection,
    )
    if any(setting is not None for setting in group_settings):
        if candidate.preprocessing.selection != "none":
            raise ValueError("global and group Pearson selection cannot be combined")
    if recipe.load_selection is not None and recipe.load_selection.mode == "fixed":
        daily_steps = pd.Timedelta(days=1) / pd.Timedelta(candidate.index.frequency)
        if int(daily_steps) != daily_steps or daily_steps < 1:
            raise ValueError("fixed same-hour load lags require a frequency dividing one day")
        expected = tuple(int(daily_steps) * day for day in range(1, 8))
        if recipe.load_lags != expected:
            raise ValueError("fixed load lags must be the previous seven same-hour days")
    if candidate.preprocessing.selection_ratio == 0:
        raise ValueError("selection ratio yields zero features")
    caps = CAPABILITIES[candidate.family]
    temperature = candidate.features.temperature_column
    if temperature is not None:
        policy = candidate.auxiliary_policies.get(temperature)
        if policy is None or policy.role != "temperature":
            raise ValueError("configured temperature needs a temperature availability policy")
    if candidate.output.representation == "quantile" and not caps.quantile:
        raise ValueError(f"{candidate.family.value} does not support quantile output")
    if candidate.index.horizon != 1:
        raise ValueError("current adapters require exactly one target per origin")
    if caps.representation == "sequence":
        length = candidate.features.sequence_length
        if length is None or candidate.features.sequence_frequency is None:
            raise ValueError("sequential model requires a sequence feature recipe")
        if candidate.features.load_lags or candidate.features.load_selection is not None:
            raise ValueError("sequential model uses sequence history, not regression load lags")
        if candidate.family == ModelFamily.CNN:
            assert isinstance(hp, CNNHP)
            if length - hp.layers * (hp.kernel_size - 1) < 1:
                raise ValueError("CNN kernel/layers exceed sequence length")
    elif candidate.features.sequence_length is not None:
        raise ValueError("tabular model cannot ignore sequence feature settings")
    if caps.representation == "sequence" and candidate.preprocessing.selection != "none":
        raise ValueError("Pearson selection is only defined for tabular features")
    if candidate.family in {ModelFamily.LINEAR, ModelFamily.SVR, ModelFamily.XGBOOST}:
        defaults = TrainingConfig(seed=candidate.training.seed)
        for name, value in candidate.training.model_dump().items():
            if name not in {"seed", "loss", "device"} and value != getattr(defaults, name):
                raise ValueError(f"classical model cannot use neural training setting: {name}")
        if candidate.training.loss != "native":
            raise ValueError("classical models use their native training objective")
        if candidate.training.device != "cpu":
            raise ValueError("classical adapters are CPU-only")
    elif candidate.output.representation == "quantile":
        if candidate.training.loss != "pinball":
            raise ValueError("quantile neural models require pinball training loss")
    elif candidate.training.loss not in {"mae", "mse"}:
        raise ValueError("point neural models require mae or mse training loss")
    if candidate.training.device == "cuda" and not caps.gpu:
        raise ValueError("family does not support GPU")
    return hp
