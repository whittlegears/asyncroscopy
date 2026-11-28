"""Machine-learning suggestions as an MCP server, kept apart so torch lives only here.

Every tool *suggests* or *analyses*; nothing here touches an instrument. Data come from
the same sources the pyTEMlib analysis server reads (a file path, or a data key on the
Tiled server), and settings come back to the caller, typically the calibration
procedures server, which owns the hardware limits.

Only methods that already exist in asyncroscopy belong here:

- ``bo_suggest_next_setting``: Bayesian optimisation of one score with a Gaussian
  process (BoTorch SingleTaskGP, log expected improvement or upper confidence bound),
  from notebooks/AberrationsBO.ipynb.
- ``bo_suggest_pareto_setting``: several scores at once with expected hypervolume
  improvement, also from notebooks/AberrationsBO.ipynb.
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
