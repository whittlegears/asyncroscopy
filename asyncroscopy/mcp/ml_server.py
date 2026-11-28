"""Machine-learning suggestions as an MCP server, kept apart so torch lives only here.

Every tool *suggests*; nothing here touches an instrument. Data come from the same
sources the pyTEMlib analysis server reads (a file path, or a data key on the Tiled
server), and settings come back to the caller, typically the calibration
procedures server, which owns the hardware limits.

- ``bo_suggest_next_setting``: Bayesian optimisation of one score with a Gaussian
  process (BoTorch SingleTaskGP, log expected improvement or upper confidence bound).
- ``bo_suggest_pareto_setting``: two or more scores at once (expected hypervolume
  improvement), as in Utkarsh Pratiush's aberration-tuning notebooks
  (github.com/utkarshp1161/Active-learning-in-microscopy).
- ``dkl_suggest_locations``: deep-kernel active learning. A small CNN maps image
  patches around candidate probe positions to features, an exact GP maps features to a
  measured value (for example an EELS-derived quantity), and the next positions are
  the ones with the best acquisition value. The notebook version paired a variational
  GP with the exact-GP likelihood; this one is an exact GP trained on its own
  marginal likelihood, the standard deep-kernel setup for tens to hundreds of points.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import torch
import yaml
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from fastmcp import FastMCP
from gpytorch.mlls import ExactMarginalLogLikelihood
from pyTEMlib.mcp_analysis_tools import load_array
from torch import nn

dtype = torch.float64


class SuggestionError(ValueError):
    pass


def _bounds(bounds, dimensions: int) -> torch.Tensor:
    array = np.asarray(bounds, dtype=float).reshape(dimensions, 2)
    if np.any(array[:, 0] >= array[:, 1]):
        raise SuggestionError("every bound must be [low, high] with low < high")
    return torch.tensor(array.T, dtype=dtype)


def _points(points, minimum: int) -> torch.Tensor:
    x = torch.tensor(np.asarray(points, dtype=float), dtype=dtype)
    x = x[:, None] if x.ndim == 1 else x
    if x.shape[0] < minimum:
        raise SuggestionError(f"at least {minimum} measured points are needed, got {x.shape[0]}")
    return x


def _fit(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(x, y, input_transform=Normalize(d=x.shape[-1], bounds=bounds),
                         outcome_transform=Standardize(m=y.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def suggest_setting(points, scores, bounds, maximize=True, acquisition="log_ei", beta=2.0, seed=0,
                    num_restarts=10, raw_samples=256) -> dict:
    """Next setting for one score from a Gaussian-process model of all measurements."""
    x = _points(points, 2)
    sign = 1.0 if maximize else -1.0
    y = sign * torch.tensor(np.asarray(scores, dtype=float), dtype=dtype).reshape(-1, 1)
    if y.shape[0] != x.shape[0]:
        raise SuggestionError("points and scores differ in length")
    box = _bounds(bounds, x.shape[1])
    torch.manual_seed(seed)
    model = _fit(x, y, box)
    if acquisition == "log_ei":
        acquisition_function = LogExpectedImprovement(model, best_f=y.max())
    elif acquisition == "ucb":
        acquisition_function = UpperConfidenceBound(model, beta=beta)
    else:
        raise SuggestionError("acquisition must be 'log_ei' or 'ucb'")
    candidate, _ = optimize_acqf(acquisition_function, bounds=box, q=1, num_restarts=num_restarts,
                                 raw_samples=raw_samples)
    with torch.no_grad():
        posterior = model.posterior(candidate)
    best = int(torch.argmax(y))
    return {"next_point": candidate[0].tolist(),
            "predicted_score": float(sign * posterior.mean.item()),
            "predicted_std": float(posterior.variance.sqrt().item()),
            "best_measured_point": x[best].tolist(),
            "best_measured_score": float(sign * y[best].item()),
            "measurements": int(x.shape[0])}


def suggest_pareto_setting(points, objectives, bounds, maximize=None, ref_point=None, seed=0,
                           num_restarts=10, raw_samples=256) -> dict:
    """Next setting that most increases the Pareto hypervolume of several scores."""
    x = _points(points, 2)
    y = torch.tensor(np.asarray(objectives, dtype=float), dtype=dtype)
    if y.ndim != 2 or y.shape[1] < 2 or y.shape[0] != x.shape[0]:
        raise SuggestionError("objectives must be one row per point with at least two scores")
    signs = torch.tensor([1.0 if flag else -1.0 for flag in (maximize or [True] * y.shape[1])], dtype=dtype)
    y = y * signs
    box = _bounds(bounds, x.shape[1])
    reference = (y.min(dim=0).values - 0.1 * y.std(dim=0)) if ref_point is None \
        else torch.tensor(ref_point, dtype=dtype) * signs
    torch.manual_seed(seed)
    model = _fit(x, y, box)
    pareto = is_non_dominated(y)
    acquisition_function = qLogExpectedHypervolumeImprovement(
        model=model, ref_point=reference.tolist(),
        partitioning=NondominatedPartitioning(ref_point=reference, Y=y[pareto]))
    candidate, _ = optimize_acqf(acquisition_function, bounds=box, q=1, num_restarts=num_restarts,
                                 raw_samples=raw_samples)
    with torch.no_grad():
        predicted = model.posterior(candidate).mean[0] * signs
    return {"next_point": candidate[0].tolist(), "predicted_objectives": predicted.tolist(),
            "pareto_indices": torch.where(pareto)[0].tolist(), "measurements": int(x.shape[0])}


class PatchFeatures(nn.Module):
    """Two conv / pool stages and a linear layer: square patch -> feature vector."""

    def __init__(self, window_size: int, features: int = 8):
        super().__init__()
        if window_size % 4:
            raise SuggestionError("window_size must be a multiple of 4")
        self.window_size = window_size
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * (window_size // 4) ** 2, features))

    def forward(self, flattened: torch.Tensor) -> torch.Tensor:
        return self.layers(flattened.reshape(-1, 1, self.window_size, self.window_size))


class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, patches, values, likelihood, window_size: int):
        super().__init__(patches, values, likelihood)
        self.features = PatchFeatures(window_size)
        self.scale = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, patches):
        projected = self.scale(self.features(patches))
        return gpytorch.distributions.MultivariateNormal(self.mean_module(projected),
                                                         self.covar_module(projected))


def extract_patches(image: np.ndarray, positions: np.ndarray, window_size: int) -> torch.Tensor:
    """Flattened square patches centred on (row, column) positions, image normalised to [0, 1].

    The exact GP takes 2D inputs (points x features), so each patch is one flattened row;
    PatchFeatures reshapes it back to an image.
    """
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
    half = window_size // 2
    padded = np.pad(image, half, mode="reflect")
    stack = np.stack([padded[row:row + window_size, column:column + window_size] for row, column in positions])
    return torch.tensor(stack.reshape(len(stack), -1), dtype=torch.float32)


def suggest_locations(image, measurements, window_size=16, step=4, count=1, acquisition="ei", beta=4.0,
                      epochs=150, learning_rate=0.01, min_separation=None, seed=0) -> dict:
    """Train a deep-kernel GP on measured positions and pick where to measure next.

    Returns positions as (row, column) image pixels, predicted value and standard deviation
    there, and the predicted mean and standard deviation on the whole candidate grid.
    Suggestions are at least ``min_separation`` pixels from each other and from every
    measurement; by default one patch width, so a batch does not pile up on overlapping patches.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise SuggestionError(f"dkl_suggest_locations needs a 2D structure image, got {image.shape}")
    measured = np.asarray([[int(row), int(column)] for row, column, _ in measurements])
    values = np.asarray([float(value) for _, _, value in measurements])
    if len(measured) < 3:
        raise SuggestionError("at least 3 measurements are needed")
    if np.any(measured < 0) or np.any(measured[:, 0] >= image.shape[0]) or np.any(measured[:, 1] >= image.shape[1]):
        raise SuggestionError("a measurement lies outside the image")
    if acquisition not in ("ei", "ucb", "max_variance"):
        raise SuggestionError("acquisition must be 'ei', 'ucb' or 'max_variance'")
    torch.manual_seed(seed)
    rows, columns = np.meshgrid(np.arange(0, image.shape[0], step), np.arange(0, image.shape[1], step), indexing="ij")
    grid = np.column_stack([rows.ravel(), columns.ravel()])

    centre, spread = values.mean(), values.std() if values.std() > 0 else 1.0
    train_x = extract_patches(image, measured, window_size)
    train_y = torch.tensor((values - centre) / spread, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DeepKernelGP(train_x, train_y, likelihood, window_size)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    marginal = ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -marginal(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    candidates = extract_patches(image, grid, window_size)
    means, variances = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(candidates), 1024):
            prediction = likelihood(model(candidates[start:start + 1024]))
            means.append(prediction.mean)
            variances.append(prediction.variance)
    mean = torch.cat(means).numpy().astype(float)
    std = np.sqrt(torch.cat(variances).numpy().astype(float))
    if acquisition == "ei":
        best = train_y.max().item()
        improvement = mean - best
        z = improvement / np.maximum(std, 1e-12)
        normal = torch.distributions.Normal(0.0, 1.0)
        cdf = normal.cdf(torch.tensor(z)).numpy()
        pdf = np.exp(normal.log_prob(torch.tensor(z)).numpy())
        score = improvement * cdf + std * pdf
    elif acquisition == "ucb":
        score = mean + np.sqrt(beta) * std
    else:
        score = std.copy()

    separation = float(window_size if min_separation is None else min_separation)
    distance_to_measured = np.min(np.hypot(*(grid[:, None, :] - measured[None, :, :]).transpose(2, 0, 1)), axis=1)
    score[distance_to_measured < separation] = -np.inf
    chosen = []
    for _ in range(int(count)):
        index = int(np.argmax(score))
        if not np.isfinite(score[index]):
            break
        chosen.append(index)
        score[np.hypot(*(grid - grid[index]).T) < separation] = -np.inf
    shape = rows.shape
    return {"next_positions": [grid[index].tolist() for index in chosen],
            "predicted_values": [float(centre + spread * mean[index]) for index in chosen],
            "predicted_std": [float(spread * std[index]) for index in chosen],
            "mean_map": (centre + spread * mean).reshape(shape),
            "std_map": (spread * std).reshape(shape),
            "grid_step": int(step), "measurements": int(len(measured)), "final_loss": float(loss.item())}


@dataclass
class MLConfig:
    name: str = "ML_Suggestions"
    http_host: str = "127.0.0.1"
    http_port: int = 8011
    tiled_uri: str = ""
    output_dir: str = "outputs/ml"

    @classmethod
    def from_yaml(cls, path: Path) -> "MLConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        section = raw.get("ml", raw)
        unknown = sorted(set(section) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown ml config keys: {unknown}")
        return cls(**section)


class MLServer:
    def __init__(self, config: MLConfig):
        self.config = config
        self.mcp = FastMCP(config.name)
        self._register()

    def _register(self) -> None:
        config = self.config

        @self.mcp.tool()
        def bo_suggest_next_setting(points: list, scores: list, bounds: list, maximize: bool = True,
                                    acquisition: str = "log_ei", beta: float = 2.0, seed: int = 0) -> str:
            """Bayesian optimisation: the next setting to try for one score.

            points: settings measured so far, one list per measurement (any number of parameters, at
            least two measurements). scores: the measured score of each. bounds: [[low, high], ...] per
            parameter; the suggestion stays inside them. acquisition: log_ei (expected improvement) or
            ucb (upper confidence bound, beta sets exploration). Returns the suggestion with the model's
            predicted score and standard deviation there. It only suggests; nothing is moved.
            """
            return json.dumps(suggest_setting(points, scores, bounds, maximize, acquisition, beta, seed))

        @self.mcp.tool()
        def bo_suggest_pareto_setting(points: list, objectives: list, bounds: list, maximize: list = None,
                                      ref_point: list = None, seed: int = 0) -> str:
            """Multi-objective Bayesian optimisation (expected hypervolume improvement).

            objectives: one row per measured point with two or more scores (for example image contrast
            and FFT signal-to-noise). maximize: one flag per score, all true by default. Returns the next
            setting, its predicted scores and which measured points are currently Pareto-optimal.
            """
            return json.dumps(suggest_pareto_setting(points, objectives, bounds, maximize, ref_point, seed))

        @self.mcp.tool()
        def dkl_suggest_locations(source: str, measurements: list, dataset_key: str = None, window_size: int = 16,
                                  step: int = 4, count: int = 1, acquisition: str = "ei", beta: float = 4.0,
                                  epochs: int = 150, seed: int = 0) -> str:
            """Deep-kernel active learning: where to measure next, judged from the structure image.

            source: the structure image (HAADF etc.), a file path or data key. measurements: list of
            [row, column, value] in that image's pixels, e.g. a spectroscopic quantity measured there
            (at least 3). window_size: patch side in pixels (multiple of 4). step: candidate grid
            spacing. acquisition: ei (find high values), ucb, or max_variance (learn the whole map).
            Returns the next positions, the predicted value and uncertainty there, and .npy files of
            the predicted mean and standard deviation on the candidate grid.
            """
            image = load_array(source, dataset_key, None, config.tiled_uri or None)[0]
            result = suggest_locations(image, measurements, window_size, step, count, acquisition, beta, epochs,
                                       seed=seed)
            directory = Path(config.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            stem = "".join(character if character.isalnum() or character in "-_." else "_" for character in source)
            for name in ("mean_map", "std_map"):
                path = directory / f"{Path(stem).stem}_dkl_{name}.npy"
                np.save(path, result.pop(name))
                result[f"{name}_file"] = str(path.resolve())
            return json.dumps(result)

        @self.mcp.tool()
        def ml_status() -> str:
            """Library versions and the compute device this server uses."""
            import botorch
            return json.dumps({"torch": torch.__version__, "gpytorch": gpytorch.__version__,
                               "botorch": botorch.__version__,
                               "device": "cuda" if torch.cuda.is_available() else "cpu"})

    def run(self) -> None:
        self.mcp.run(transport="streamable-http", host=self.config.http_host, port=self.config.http_port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yaml", type=Path, required=True, help="ML server config (see configs/ml.yaml)")
    args = parser.parse_args(argv)
    config = MLConfig.from_yaml(args.yaml)
    if config.tiled_uri:
        os.environ.setdefault("PYTEMLIB_TILED_URI", config.tiled_uri)
    MLServer(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
