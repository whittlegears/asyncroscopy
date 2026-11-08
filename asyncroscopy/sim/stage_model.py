"""Stage with separate commanded, true and reported positions.

A real stage never lands exactly where it was told and never reports exactly
where it is. The image is rendered from the *true* position; ``get_stage``
returns the *reported* one. With every parameter at zero the three agree,
which is how the original twin behaved.

``StageModel`` is what ``DigitalTwinSTAGE`` runs. It does not keep time: each
move is told the simulated time it starts at and says how long it takes, so
the twin microscope's clock stays the single time source.
``StageTrajectory`` is a small copy of the model's state that a client can
evaluate at any time without asking the stage again (one per frame, not one
per pixel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rng import EventRng

WALK_STEP_S = 60.0  # the drift's random walk takes one step per simulated minute


@dataclass
class StageParams:
    bias_m: tuple[float, float] = (0.0, 0.0)
    noise_std_m: float = 0.0
    reported_noise_std_m: float = 0.0
    backlash_m: tuple[float, float] = (0.0, 0.0)
    speed_m_s: float = 5e-6
    move_overhead_s: float = 0.0
    settle_drift_m_s: float = 0.0
    settle_tau_s: float = 30.0
    drift_m_per_min: float = 0.0
    drift_random_walk_m: float = 0.0


@dataclass
class MoveRecord:
    index: int
    time_s: float
    commanded: list[float]
    true: list[float]
    reported: list[float]


@dataclass(frozen=True)
class StageTrajectory:
    """The stage's positions and everything needed to work out its drift at a time in the window it was made for."""

    commanded: np.ndarray = field(default_factory=lambda: np.zeros(5))
    true: np.ndarray = field(default_factory=lambda: np.zeros(5))
    reported: np.ndarray = field(default_factory=lambda: np.zeros(5))
    drift_velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(2))
    settle_start_s: float | None = None
    settle_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    settle_drift_m_s: float = 0.0
    settle_tau_s: float = 30.0
    walk_first_minute: int = 0
    walk_m: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))  # the walk's position from each minute of the window on

    def drift_offset_at(self, t: float) -> np.ndarray:
        """Specimen drift (thermal plus settling) relative to the stage, at simulated time t."""
        steady = self.drift_velocity_m_s * t
        last = len(self.walk_m) - 1
        walk = self.walk_m[max(0, min(last, int(t // WALK_STEP_S) - self.walk_first_minute))]
        settle = np.zeros(2)
        if self.settle_drift_m_s > 0 and self.settle_start_s is not None and t >= self.settle_start_s:
            elapsed = t - self.settle_start_s
            settle = self.settle_direction * self.settle_drift_m_s * self.settle_tau_s * (1.0 - np.exp(-elapsed / self.settle_tau_s))
        return steady + walk + settle

    def true_xy_at(self, t: float) -> np.ndarray:
        return self.true[:2] + self.drift_offset_at(t)

    def to_dict(self) -> dict:
        return {
            "commanded": self.commanded.tolist(),
            "true": self.true.tolist(),
            "reported": self.reported.tolist(),
            "drift_velocity_m_s": self.drift_velocity_m_s.tolist(),
            "settle_start_s": self.settle_start_s,
            "settle_direction": self.settle_direction.tolist(),
            "settle_drift_m_s": self.settle_drift_m_s,
            "settle_tau_s": self.settle_tau_s,
            "walk_first_minute": self.walk_first_minute,
            "walk_m": self.walk_m.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageTrajectory":
        return cls(
            commanded=np.asarray(data["commanded"], dtype=np.float64),
            true=np.asarray(data["true"], dtype=np.float64),
            reported=np.asarray(data["reported"], dtype=np.float64),
            drift_velocity_m_s=np.asarray(data["drift_velocity_m_s"], dtype=np.float64),
            settle_start_s=None if data["settle_start_s"] is None else float(data["settle_start_s"]),
            settle_direction=np.asarray(data["settle_direction"], dtype=np.float64),
            settle_drift_m_s=float(data["settle_drift_m_s"]),
            settle_tau_s=float(data["settle_tau_s"]),
            walk_first_minute=int(data["walk_first_minute"]),
            walk_m=np.asarray(data["walk_m"], dtype=np.float64).reshape(-1, 2),
        )


@dataclass
class StageModel:
    params: StageParams
    rng: EventRng
    commanded: np.ndarray = field(default_factory=lambda: np.zeros(5))
    true: np.ndarray = field(default_factory=lambda: np.zeros(5))
    reported: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self._move_index = 0
        self._last_direction = np.zeros(2)
        self._settle_start: float | None = None
        self._settle_dir = np.zeros(2)
        drift_rng = self.rng.for_event("drift_direction")
        angle = drift_rng.uniform(0.0, 2.0 * np.pi)
        self._drift_dir = np.array([np.cos(angle), np.sin(angle)])
        self._walk = [np.zeros(2)]  # the walk's position from each minute on
        self.history: list[MoveRecord] = []

    def move_seconds(self, target: list[float]) -> float:
        """Simulated time a move to ``target`` takes."""
        target = np.asarray(target, dtype=np.float64)
        distance = float(np.hypot(*(target[:2] - self.commanded[:2])))
        p = self.params
        return p.move_overhead_s + (distance / p.speed_m_s if p.speed_m_s > 0 else 0.0)

    def move(self, target: list[float], at_s: float = 0.0) -> float:
        """Move to ``target``, starting at simulated time ``at_s``; returns the simulated seconds the move takes."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (5,):
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        p = self.params
        self._move_index += 1
        rng = self.rng.for_event("stage", self._move_index)

        distance = float(np.hypot(*(target[:2] - self.commanded[:2])))
        delta = target[:2] - self.commanded[:2]
        direction = np.sign(delta)
        backlash = np.zeros(2)
        for axis in range(2):
            if direction[axis] != 0 and self._last_direction[axis] != 0 and direction[axis] != self._last_direction[axis]:
                # A reversal undershoots: the stage stops short by the backlash.
                backlash[axis] = -direction[axis] * float(p.backlash_m[axis])
            if direction[axis] != 0:
                self._last_direction[axis] = direction[axis]

        true = target.copy()
        true[:2] += np.asarray(p.bias_m, dtype=np.float64) + backlash
        if p.noise_std_m > 0:
            true[:3] += rng.normal(0.0, p.noise_std_m, size=3)
        reported = target.copy()
        if p.reported_noise_std_m > 0:
            reported[:3] += rng.normal(0.0, p.reported_noise_std_m, size=3)

        duration = p.move_overhead_s + (distance / p.speed_m_s if p.speed_m_s > 0 else 0.0)
        finished = float(at_s) + duration
        if distance > 0:
            self._settle_start = finished
            angle = rng.uniform(0.0, 2.0 * np.pi)
            self._settle_dir = np.array([np.cos(angle), np.sin(angle)])

        self.commanded, self.true, self.reported = target, true, reported
        self.history.append(MoveRecord(self._move_index, finished, target.tolist(), true.tolist(), reported.tolist()))
        return duration

    def _walk_through(self, minute: int) -> None:
        """Take the walk's steps up to ``minute``; each comes from its own stream, so it does not depend on how often it is sampled."""
        while len(self._walk) <= minute:
            step = self.rng.for_event("drift_walk", len(self._walk) - 1).normal(0.0, self.params.drift_random_walk_m, size=2)
            self._walk.append(self._walk[-1] + step)

    def trajectory(self, start_s: float, end_s: float | None = None) -> StageTrajectory:
        """The stage as it is now, with what a client needs to work out the drift from ``start_s`` to ``end_s``."""
        p = self.params
        end_s = start_s if end_s is None else max(start_s, end_s)
        if p.drift_random_walk_m > 0:
            first, last = int(start_s // WALK_STEP_S), int(end_s // WALK_STEP_S)
            self._walk_through(last)
            walk = np.array(self._walk[first : last + 1], dtype=np.float64)
        else:
            first, walk = 0, np.zeros((1, 2))
        return StageTrajectory(
            commanded=self.commanded.copy(),
            true=self.true.copy(),
            reported=self.reported.copy(),
            drift_velocity_m_s=self._drift_dir * (p.drift_m_per_min / 60.0),
            settle_start_s=self._settle_start,
            settle_direction=self._settle_dir.copy(),
            settle_drift_m_s=p.settle_drift_m_s,
            settle_tau_s=p.settle_tau_s,
            walk_first_minute=first,
            walk_m=walk,
        )

    def drift_offset_at(self, t: float) -> np.ndarray:
        """Specimen drift (thermal plus settling) relative to the stage, at simulated time t."""
        return self.trajectory(t).drift_offset_at(t)

    def true_xy_at(self, t: float) -> np.ndarray:
        return self.true[:2] + self.drift_offset_at(t)
