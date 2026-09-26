"""Single registry and adapter execution path for fully resolved forecasting candidates."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

import joblib
import numpy as np
import torch
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from interactive_forecasting.domain.forecasting import (
    FeatureProvenance,
    LabelVector,
    Partition,
    PointForecast,
    QuantileForecast,
    ResolvedCandidate,
)
from interactive_forecasting.domain.models import ArtifactRef
from interactive_forecasting.domain.types import ModelFamily
from interactive_forecasting.services.data.core import (
    ForecastExample,
    ForecastIndex,
    NormalizedSeries,
)
from interactive_forecasting.services.features.core import (
    FeatureBatch,
    FeatureBuilder,
    FittedFeatureTransform,
)
from interactive_forecasting.services.models.config import (
    CAPABILITIES,
    CNNHP,
    HYPERPARAMETER_SCHEMAS,
    MLPHP,
    SVRHP,
    LinearHP,
    ModelCapabilities,
    RecurrentHP,
    XGBoostHP,
    validate_candidate,
)
from interactive_forecasting.services.models.neural import (
    NeuralModelInput,
    build_network,
    fit_neural,
    predict_neural,
    seed_libraries,
)
from interactive_forecasting.storage.artifacts import ArtifactStore


class ModelAdapter(Protocol):
    family: ModelFamily
    capabilities: ModelCapabilities

    def validate(self, candidate: ResolvedCandidate) -> BaseModel: ...
    def build(self, candidate: ResolvedCandidate, input_width: int) -> Any: ...
    def fit(
        self,
        model: Any,
        train: FeatureBatch,
        labels: LabelVector,
        validation: FeatureBatch,
        validation_labels: LabelVector,
        candidate: ResolvedCandidate,
        weights: np.ndarray | None,
    ) -> tuple[Any, dict[str, Any]]: ...
    def predict(self, model: Any, batch: FeatureBatch) -> np.ndarray: ...
    def save_state(self, model: Any) -> bytes: ...
    def load_state(self, candidate: ResolvedCandidate, input_width: int, payload: bytes) -> Any: ...


class ClassicalAdapter:
    def __init__(self, family: ModelFamily):
        self.family = family
        self.capabilities = CAPABILITIES[family]

    def validate(self, candidate: ResolvedCandidate) -> BaseModel:
        return validate_candidate(candidate)

    def build(self, candidate: ResolvedCandidate, input_width: int) -> Any:
        hp = self.validate(candidate)
        if self.family == ModelFamily.LINEAR:
            assert isinstance(hp, LinearHP)
            if hp.regularization == "ridge":
                assert hp.alpha is not None
                return Ridge(
                    alpha=hp.alpha,
                    fit_intercept=hp.fit_intercept,
                    tol=hp.ridge_tol if hp.ridge_tol is not None else 1e-4,
                    max_iter=hp.ridge_max_iter,
                )
            return LinearRegression(fit_intercept=hp.fit_intercept)
        if self.family == ModelFamily.SVR:
            assert isinstance(hp, SVRHP)
            return SVR(
                C=hp.c,
                gamma=hp.gamma,
                epsilon=hp.epsilon,
                kernel=hp.kernel,
                tol=hp.tol,
                shrinking=hp.shrinking,
                max_iter=hp.max_iter,
            )
        if self.family == ModelFamily.XGBOOST:
            assert isinstance(hp, XGBoostHP)
            return XGBRegressor(
                n_estimators=hp.n_estimators,
                max_depth=hp.max_depth,
                learning_rate=hp.learning_rate,
                subsample=hp.subsample,
                colsample_bytree=hp.colsample_bytree,
                reg_lambda=hp.reg_lambda,
                tree_method=hp.tree_method,
                min_child_weight=hp.min_child_weight,
                gamma=hp.gamma,
                reg_alpha=hp.reg_alpha,
                max_bin=hp.max_bin,
                objective="reg:squarederror",
                random_state=candidate.training.seed,
                n_jobs=1,
            )
        raise ValueError("not a classical family")

    def fit(
        self,
        model: Any,
        train: FeatureBatch,
        labels: LabelVector,
        validation: FeatureBatch,
        validation_labels: LabelVector,
        candidate: ResolvedCandidate,
        weights: np.ndarray | None,
    ) -> tuple[Any, dict[str, Any]]:
        if train.keys != labels.keys or validation.keys != validation_labels.keys:
            raise ValueError("features and labels must align")
        model.fit(train.values, np.asarray(labels.values), sample_weight=weights)
        return model, {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost") if self.family == ModelFamily.XGBOOST else None,
            "estimator_parameters": {
                key: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
                for key, value in model.get_params().items()
            },
            "seed": candidate.training.seed,
            "device": "cpu",
        }

    def predict(self, model: Any, batch: FeatureBatch) -> np.ndarray:
        return np.asarray(model.predict(batch.values), dtype=float).reshape(-1)

    def save_state(self, model: Any) -> bytes:
        stream = io.BytesIO()
        joblib.dump(model, stream)
        return stream.getvalue()

    def load_state(self, candidate: ResolvedCandidate, input_width: int, payload: bytes) -> Any:
        # Joblib uses pickle. Only load trusted, checksum-verified local artifacts.
        model = joblib.load(io.BytesIO(payload))
        if not hasattr(model, "predict"):
            raise ValueError("invalid model artifact")
        return model


class NeuralAdapter:
    def __init__(self, family: ModelFamily):
        self.family = family
        self.capabilities = CAPABILITIES[family]

    def validate(self, candidate: ResolvedCandidate) -> BaseModel:
        return validate_candidate(candidate)

    def build(self, candidate: ResolvedCandidate, input_width: int) -> Any:
        hp = self.validate(candidate)
        assert isinstance(hp, (MLPHP, RecurrentHP, CNNHP))
        seed_libraries(candidate.training.seed, deterministic=candidate.training.deterministic)
        return build_network(
            self.family,
            hp,
            input_width,
            len(candidate.output.quantile_levels)
            if candidate.output.representation == "quantile"
            else 1,
        )

    def fit(
        self,
        model: Any,
        train: FeatureBatch,
        labels: LabelVector,
        validation: FeatureBatch,
        validation_labels: LabelVector,
        candidate: ResolvedCandidate,
        weights: np.ndarray | None,
    ) -> tuple[Any, dict[str, Any]]:
        if weights is not None:
            raise ValueError("neural sample weighting is not implemented")
        if train.keys != labels.keys or validation.keys != validation_labels.keys:
            raise ValueError("features and labels must align")
        fit = fit_neural(
            model,
            self._input(train),
            np.asarray(labels.values, dtype=float),
            self._input(validation),
            np.asarray(validation_labels.values, dtype=float),
            candidate.training,
            candidate.output,
        )
        return fit.model, {
            **fit.environment,
            "python": platform.python_version(),
            "device": candidate.training.device,
            "epochs_completed": fit.epochs_completed,
            "best_validation_loss": fit.best_validation_loss,
            "training_loss_curve": list(fit.training_loss_curve),
            "validation_loss_curve": list(fit.validation_loss_curve),
        }

    @staticmethod
    def _input(batch: FeatureBatch) -> np.ndarray | NeuralModelInput:
        if batch.representation == "tabular":
            return batch.values
        structured = (
            batch.structured.values
            if batch.structured is not None
            else np.empty((len(batch.keys), 0), dtype=float)
        )
        return NeuralModelInput(batch.values, structured)

    def predict(self, model: Any, batch: FeatureBatch) -> np.ndarray:
        return predict_neural(model, self._input(batch))

    def save_state(self, model: Any) -> bytes:
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        return stream.getvalue()

    def load_state(self, candidate: ResolvedCandidate, input_width: int, payload: bytes) -> Any:
        model = self.build(candidate, input_width)
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[ModelFamily, ModelAdapter] = {
            family: (
                ClassicalAdapter(family)
                if family in {ModelFamily.LINEAR, ModelFamily.SVR, ModelFamily.XGBOOST}
                else NeuralAdapter(family)
            )
            for family in ModelFamily
        }

    def resolve(self, candidate: ResolvedCandidate) -> ModelAdapter:
        adapter = self._adapters[candidate.family]
        adapter.validate(candidate)
        return adapter

    def capabilities(self, family: ModelFamily) -> ModelCapabilities:
        return self._adapters[family].capabilities

    @property
    def families(self) -> tuple[ModelFamily, ...]:
        return tuple(self._adapters)


@dataclass(frozen=True)
class FittedCoreModel:
    candidate: ResolvedCandidate
    transform: FittedFeatureTransform
    model: Any
    metadata: dict[str, Any]

    def predict(
        self, data: NormalizedSeries, examples: tuple[ForecastExample, ...]
    ) -> PointForecast | QuantileForecast:
        registry = ModelRegistry()
        adapter = registry.resolve(self.candidate)
        if data.auxiliary_availability != self.candidate.auxiliary_policies:
            raise ValueError("inference auxiliary policies differ from saved candidate")
        if data.timezone_name != self.metadata.get("timezone_name"):
            raise ValueError("inference timezone differs from training timezone")
        index = ForecastIndex(self.candidate.index)
        raw = FeatureBuilder(self.candidate.features, index).build(
            data, examples, representation=adapter.capabilities.representation
        )
        transformed = self.transform.transform(raw)
        values = adapter.predict(self.model, transformed)
        if not np.isfinite(values).all():
            raise ValueError("model output shape or values invalid")
        if self.candidate.output.representation == "quantile":
            expected = (len(examples), len(self.candidate.output.quantile_levels))
            if values.shape != expected:
                raise ValueError("model quantile output shape invalid")
            # Rearrange only at inference; training heads keep their fixed level association.
            ordered = np.sort(values, axis=1)
            return QuantileForecast(
                keys=transformed.keys,
                levels=self.candidate.output.quantile_levels,
                values=tuple(tuple(float(v) for v in row) for row in ordered),
            )
        if values.shape != (len(examples),):
            raise ValueError("model point output shape invalid")
        return PointForecast(keys=transformed.keys, values=tuple(float(v) for v in values))

    def save(self, store: ArtifactStore, uri: str) -> ArtifactRef:
        adapter = ModelRegistry().resolve(self.candidate)
        metadata = {
            "candidate": self.candidate.model_dump(mode="json"),
            "auxiliary_policies": {
                name: policy.model_dump(mode="json")
                for name, policy in self.candidate.auxiliary_policies.items()
            },
            "transform_config": self.transform.config.model_dump(mode="json"),
            "provenance": self.transform.provenance.model_dump(mode="json"),
            "environment": self.metadata,
            "representation": adapter.capabilities.representation,
            "input_width": len(self.transform.provenance.selected_names),
            "structured_width": (
                len(self.transform.provenance.structured.selected_names)
                if self.transform.provenance.structured is not None
                else 0
            ),
            "format_version": 3,
            "quantile_crossing_policy": (
                "sort_ascending" if self.candidate.output.representation == "quantile" else None
            ),
        }
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
            bundle.writestr("model.bin", adapter.save_state(self.model))
        return store.put_bytes(uri, stream.getvalue())

    @classmethod
    def load(cls, store: ArtifactStore, reference: ArtifactRef) -> FittedCoreModel:
        content = store.read_bytes(reference)
        with zipfile.ZipFile(io.BytesIO(content)) as bundle:
            metadata = json.loads(bundle.read("metadata.json"))
            payload = bundle.read("model.bin")
        if metadata.get("format_version") != 3:
            raise ValueError(
                "model artifact lacks audited timezone/configuration provenance (requires format 3)"
            )
        candidate = ResolvedCandidate.model_validate(metadata["candidate"])
        if metadata["auxiliary_policies"] != {
            name: policy.model_dump(mode="json")
            for name, policy in candidate.auxiliary_policies.items()
        }:
            raise ValueError("artifact auxiliary policy metadata mismatch")
        adapter = ModelRegistry().resolve(candidate)
        expected_crossing = (
            "sort_ascending" if candidate.output.representation == "quantile" else None
        )
        if metadata.get("quantile_crossing_policy") != expected_crossing:
            raise ValueError("artifact quantile crossing policy mismatch")
        if metadata["transform_config"] != candidate.preprocessing.model_dump(mode="json"):
            raise ValueError("artifact preprocessing configuration mismatch")
        if metadata["representation"] != adapter.capabilities.representation:
            raise ValueError("artifact input representation mismatch")
        provenance = FeatureProvenance.model_validate(metadata["provenance"])
        if metadata["input_width"] != len(provenance.selected_names):
            raise ValueError("artifact input width mismatch")
        expected_structured = (
            FeatureBuilder(candidate.features, ForecastIndex(candidate.index))._tabular_names()
            if adapter.capabilities.representation == "sequence"
            else ()
        )
        actual_structured = (
            provenance.structured.input_names if provenance.structured is not None else ()
        )
        if expected_structured != actual_structured:
            raise ValueError("artifact structured feature schema mismatch")
        structured_width = (
            len(provenance.structured.selected_names) if provenance.structured is not None else 0
        )
        if metadata.get("structured_width", 0) != structured_width:
            raise ValueError("artifact structured input width mismatch")
        FittedFeatureTransform.validate_provenance(candidate.features, provenance)
        transform = FittedFeatureTransform(
            candidate.preprocessing,
            provenance,
        )
        network_width = (
            structured_width
            if adapter.capabilities.representation == "sequence"
            else int(metadata["input_width"])
        )
        model = adapter.load_state(candidate, network_width, payload)
        return cls(candidate, transform, model, metadata["environment"])


def fit_candidate(
    candidate: ResolvedCandidate,
    data: NormalizedSeries,
    train: tuple[ForecastExample, ...],
    train_labels: LabelVector,
    validation: tuple[ForecastExample, ...],
    validation_labels: LabelVector,
    *,
    train_weights: np.ndarray | None = None,
) -> FittedCoreModel:
    """Fit only train observations; validation guides early stopping, never feature fitting."""
    registry = ModelRegistry()
    adapter = registry.resolve(candidate)
    candidate = candidate.model_copy(
        update={"hyperparameters": adapter.validate(candidate).model_dump(exclude_none=True)}
    )
    if data.frequency != candidate.index.frequency:
        raise ValueError("candidate frequency differs from dataset")
    if data.auxiliary_availability != candidate.auxiliary_policies:
        raise ValueError("candidate auxiliary policies differ from dataset")
    if not train or not validation:
        raise ValueError("training and validation examples must be nonempty")
    if any(example.partition != Partition.TRAIN for example in train) or any(
        example.partition != Partition.VALIDATION for example in validation
    ):
        raise ValueError("fit requires training and validation partition provenance")
    if max(example.key.target for example in train) > min(
        example.key.origin for example in validation
    ):
        raise ValueError("training labels extend beyond the first validation origin; add a gap")
    if train_weights is not None:
        if not adapter.capabilities.sample_weighting:
            raise ValueError("family does not support sample weighting")
        train_weights = np.asarray(train_weights, dtype=float)
        if (
            train_weights.shape != (len(train),)
            or not np.isfinite(train_weights).all()
            or np.any(train_weights < 0)
            or train_weights.sum() <= 0
        ):
            raise ValueError("invalid training sample weights")
    builder = FeatureBuilder(candidate.features, ForecastIndex(candidate.index))
    train_raw = builder.build(data, train, representation=adapter.capabilities.representation)
    valid_raw = builder.build(data, validation, representation=adapter.capabilities.representation)
    transform = FittedFeatureTransform.fit(
        train_raw, train_labels, candidate.preprocessing, candidate.features
    )
    train_x = transform.transform(train_raw)
    valid_x = transform.transform(valid_raw)
    input_width = (
        len(train_x.structured.names)
        if train_x.structured is not None
        else train_x.values.shape[-1]
        if train_x.representation == "tabular"
        else 0
    )
    model = adapter.build(candidate, input_width)
    fitted, metadata = adapter.fit(
        model, train_x, train_labels, valid_x, validation_labels, candidate, train_weights
    )
    metadata["fit_input_sha256"] = hashlib.sha256(
        train_x.values.tobytes()
        + valid_x.values.tobytes()
        + (train_x.structured.values.tobytes() if train_x.structured is not None else b"")
        + (valid_x.structured.values.tobytes() if valid_x.structured is not None else b"")
        + np.asarray(train_labels.values, dtype=float).tobytes()
        + np.asarray(validation_labels.values, dtype=float).tobytes()
    ).hexdigest()
    metadata["training_labels"] = list(train_labels.values)
    metadata["validation_labels"] = list(validation_labels.values)
    metadata["timezone_name"] = data.timezone_name
    metadata["train_weights"] = train_weights.tolist() if train_weights is not None else None
    metadata["validation_keys"] = [e.key.model_dump(mode="json") for e in validation]
    metadata["pandas"] = version("pandas")
    metadata["platform"] = platform.platform()
    metadata["machine"] = platform.machine()
    metadata["auxiliary_policies"] = {
        name: policy.model_dump(mode="json")
        for name, policy in candidate.auxiliary_policies.items()
    }
    metadata["hyperparameters"] = (
        HYPERPARAMETER_SCHEMAS[candidate.family]
        .model_validate(candidate.hyperparameters)
        .model_dump(mode="json")
    )
    return FittedCoreModel(candidate, transform, fitted, metadata)
