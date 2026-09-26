"""Shared deterministic-as-practical PyTorch trainer and family-specific architectures."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import torch
from torch import nn

from interactive_forecasting.domain.forecasting import OutputConfig, TrainingConfig
from interactive_forecasting.domain.types import ModelFamily
from interactive_forecasting.services.models.config import CNNHP, MLPHP, RecurrentHP


def seed_libraries(seed: int, *, deterministic: bool) -> dict[str, str | int | bool]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return {
        "seed": seed,
        "deterministic_requested": deterministic,
        "torch": version("torch"),
        "numpy": version("numpy"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda or "none",
        "cpu_threads": torch.get_num_threads(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "cudnn_version": torch.backends.cudnn.version() or 0,
    }


class MLP(nn.Module):
    def __init__(self, input_width: int, hp: MLPHP, output_width: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_width
        for _ in range(hp.layers):
            layers.extend([nn.Linear(width, hp.hidden_size), nn.ReLU(), nn.Dropout(hp.dropout)])
            width = hp.hidden_size
        layers.append(nn.Linear(width, output_width))
        self.network = nn.Sequential(*layers)
        self.output_width = output_width

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.network(features)
        return output.squeeze(-1) if self.output_width == 1 else output


@dataclass(frozen=True)
class NeuralModelInput:
    sequence: np.ndarray | torch.Tensor
    structured: np.ndarray | torch.Tensor


class Recurrent(nn.Module):
    def __init__(
        self, family: ModelFamily, hp: RecurrentHP, output_width: int = 1, structured_width: int = 0
    ):
        super().__init__()
        cell = nn.LSTM if family == ModelFamily.LSTM else nn.GRU
        self.recurrent = cell(
            input_size=1,
            hidden_size=hp.hidden_size,
            num_layers=hp.layers,
            dropout=hp.dropout if hp.layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hp.hidden_size + structured_width, hp.fc_size),
            nn.ReLU(),
            nn.Linear(hp.fc_size, output_width),
        )
        self.output_width = output_width

    def forward(self, features: NeuralModelInput | torch.Tensor) -> torch.Tensor:
        sequence: np.ndarray | torch.Tensor
        structured: np.ndarray | torch.Tensor
        if isinstance(features, torch.Tensor):
            sequence = features
            structured = features.new_empty((len(features), 0))
        else:
            sequence = features.sequence
            structured = features.structured
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(structured, torch.Tensor)
        output, _ = self.recurrent(sequence)
        encoded = torch.cat((output[:, -1, :], structured), dim=1)
        prediction = self.head(encoded)
        return prediction.squeeze(-1) if self.output_width == 1 else prediction


class CNN(nn.Module):
    def __init__(self, hp: CNNHP, output_width: int = 1, structured_width: int = 0):
        super().__init__()
        layers: list[nn.Module] = []
        width = 1
        for _ in range(hp.layers):
            layers.extend(
                [
                    nn.Conv1d(width, hp.filters, hp.kernel_size),
                    nn.ReLU(),
                    nn.Dropout(hp.dropout),
                ]
            )
            width = hp.filters
        self.convolutions = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hp.filters + structured_width, hp.fc_size),
            nn.ReLU(),
            nn.Linear(hp.fc_size, output_width),
        )
        self.output_width = output_width

    def forward(self, features: NeuralModelInput | torch.Tensor) -> torch.Tensor:
        sequence: np.ndarray | torch.Tensor
        structured: np.ndarray | torch.Tensor
        if isinstance(features, torch.Tensor):
            sequence = features
            structured = features.new_empty((len(features), 0))
        else:
            sequence = features.sequence
            structured = features.structured
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(structured, torch.Tensor)
        output = self.convolutions(sequence.transpose(1, 2))
        encoded = torch.cat((output.mean(dim=2), structured), dim=1)
        prediction = self.head(encoded)
        return prediction.squeeze(-1) if self.output_width == 1 else prediction


def build_network(
    family: ModelFamily, hp: MLPHP | RecurrentHP | CNNHP, input_width: int, output_width: int = 1
) -> nn.Module:
    if family == ModelFamily.MLP:
        assert isinstance(hp, MLPHP)
        return MLP(input_width, hp, output_width)
    if family in {ModelFamily.LSTM, ModelFamily.GRU}:
        assert isinstance(hp, RecurrentHP)
        return Recurrent(family, hp, output_width, input_width)
    if family == ModelFamily.CNN:
        assert isinstance(hp, CNNHP)
        return CNN(hp, output_width, input_width)
    raise ValueError("not a neural family")


@dataclass(frozen=True)
class NeuralFitResult:
    model: nn.Module
    epochs_completed: int
    best_validation_loss: float | None
    environment: dict[str, str | int | bool]
    training_loss_curve: tuple[float, ...] = ()
    validation_loss_curve: tuple[float, ...] = ()


def fit_neural(
    model: nn.Module,
    train_x: np.ndarray | NeuralModelInput,
    train_y: np.ndarray,
    valid_x: np.ndarray | NeuralModelInput,
    valid_y: np.ndarray,
    config: TrainingConfig,
    output: OutputConfig | None = None,
) -> NeuralFitResult:
    if isinstance(train_x, NeuralModelInput) != isinstance(valid_x, NeuralModelInput):
        raise ValueError("neural train and validation input types differ")
    train_rows = len(train_x.sequence) if isinstance(train_x, NeuralModelInput) else len(train_x)
    valid_rows = len(valid_x.sequence) if isinstance(valid_x, NeuralModelInput) else len(valid_x)
    if (
        train_y.shape != (train_rows,)
        or valid_y.shape != (valid_rows,)
        or (
            isinstance(train_x, NeuralModelInput)
            and isinstance(valid_x, NeuralModelInput)
            and (
                train_x.sequence.ndim != 3
                or train_x.structured.ndim != 2
                or train_x.structured.shape[0] != train_rows
                or valid_x.structured.shape[0] != valid_rows
                or train_x.sequence.shape[1:] != valid_x.sequence.shape[1:]
                or train_x.structured.shape[1:] != valid_x.structured.shape[1:]
            )
        )
        or (
            isinstance(train_x, np.ndarray)
            and isinstance(valid_x, np.ndarray)
            and (train_x.ndim not in {2, 3} or train_x.shape[1:] != valid_x.shape[1:])
        )
    ):
        raise ValueError("neural features and targets have incompatible dimensions")
    arrays: list[np.ndarray | torch.Tensor] = [train_y, valid_y]
    if isinstance(train_x, NeuralModelInput) and isinstance(valid_x, NeuralModelInput):
        arrays.extend((train_x.sequence, train_x.structured, valid_x.sequence, valid_x.structured))
    elif isinstance(train_x, np.ndarray) and isinstance(valid_x, np.ndarray):
        arrays.extend((train_x, valid_x))
    if not all(np.isfinite(np.asarray(values)).all() for values in arrays):
        raise ValueError("neural inputs must be finite")
    output = output or OutputConfig()
    if (output.representation == "quantile") != (config.loss == "pinball"):
        raise ValueError(
            "quantile output requires pinball loss and point output requires mae or mse"
        )
    environment = seed_libraries(config.seed, deterministic=config.deterministic)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(config.device)
    model.to(device)
    if config.optimizer == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
    levels = (
        torch.as_tensor(output.quantile_levels, dtype=torch.float32, device=device)
        if output.representation == "quantile"
        else None
    )
    features = _tensor_input(train_x, device)
    targets = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    validation_features = _tensor_input(valid_x, device)
    validation_targets = torch.as_tensor(valid_y, dtype=torch.float32, device=device)
    if train_rows == 0 or valid_rows == 0:
        raise ValueError("training and validation data must be nonempty")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    completed = 0
    training_curve: list[float] = []
    validation_curve: list[float] = []
    for epoch in range(config.epochs):
        model.train()
        order = (
            torch.randperm(train_rows, generator=generator)
            if config.shuffle
            else torch.arange(train_rows)
        )
        for batch in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_slice_input(features, batch))
            loss = _training_loss(prediction, targets[batch], config.loss, levels)
            if not torch.isfinite(loss):
                raise ValueError("non-finite neural training loss")
            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = float(
                _training_loss(model(features), targets, config.loss, levels).item()
            )
            validation_prediction = model(validation_features)
            if config.validation_metric == "mae":
                if levels is not None:
                    if 0.5 not in output.quantile_levels:
                        raise ValueError("validation MAE requires a median quantile")
                    median = output.quantile_levels.index(0.5)
                    validation_prediction = torch.sort(validation_prediction, dim=1).values[
                        :, median
                    ]
                validation_loss = float(
                    nn.functional.l1_loss(validation_prediction, validation_targets).item()
                )
            else:
                validation_loss = float(
                    _training_loss(
                        validation_prediction, validation_targets, config.loss, levels
                    ).item()
                )
        if not np.isfinite(training_loss) or not np.isfinite(validation_loss):
            raise ValueError("non-finite neural epoch loss")
        training_curve.append(training_loss)
        validation_curve.append(validation_loss)
        completed = epoch + 1
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if config.patience is not None and stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    return NeuralFitResult(
        model, completed, best_loss, environment, tuple(training_curve), tuple(validation_curve)
    )


def _tensor_input(
    values: np.ndarray | NeuralModelInput, device: torch.device
) -> torch.Tensor | NeuralModelInput:
    if isinstance(values, NeuralModelInput):
        return NeuralModelInput(
            torch.as_tensor(values.sequence, dtype=torch.float32, device=device),
            torch.as_tensor(values.structured, dtype=torch.float32, device=device),
        )
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _slice_input(
    values: torch.Tensor | NeuralModelInput, rows: torch.Tensor
) -> torch.Tensor | NeuralModelInput:
    if isinstance(values, NeuralModelInput):
        return NeuralModelInput(values.sequence[rows], values.structured[rows])
    return values[rows]


def _training_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    loss_kind: str,
    levels: torch.Tensor | None,
) -> torch.Tensor:
    expected = (targets.shape[0], len(levels)) if levels is not None else targets.shape
    if tuple(prediction.shape) != tuple(expected):
        raise ValueError("neural prediction and target dimensions are incompatible")
    if levels is not None:
        errors = targets[:, None] - prediction
        return torch.maximum(levels * errors, (levels - 1) * errors).mean()
    if loss_kind == "mae":
        return nn.functional.l1_loss(prediction, targets)
    if loss_kind == "mse":
        return nn.functional.mse_loss(prediction, targets)
    raise ValueError("neural point training loss must be mae or mse")


def predict_neural(model: nn.Module, values: np.ndarray | NeuralModelInput) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = _tensor_input(values, torch.device("cpu"))
        output = model(tensor).cpu().numpy()
    return np.asarray(output, dtype=float)
