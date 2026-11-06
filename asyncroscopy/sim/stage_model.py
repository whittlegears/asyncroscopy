"""Stage with separate commanded, true and reported positions.

A real stage never lands exactly where it was told and never reports exactly
where it is. The image is rendered from the *true* position; ``get_stage``
returns the *reported* one. With every parameter at zero the three agree,
which is how the original twin behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clock import SimClock
from .rng import EventRng


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


@dataclass
class StageModel:
    params: StageParams
    rng: EventRng
    clock: SimClock
    commanded: np.ndarray = field(default_factory=lambda: np.zeros(5))
    true: np.ndarray = field(default_factory=lambda: np.zeros(5))
    reported: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self._move_index = 0
        self._last_direction = np.zeros(2)
        self._settle_start = -np.inf
        self._settle_dir = np.zeros(2)
        drift_rng = self.rng.for_event("drift_direction")
        angle = drift_rng.uniform(0.0, 2.0 * np.pi)
        self._drift_dir = np.array([np.cos(angle), np.sin(angle)])
        self._walk = np.zeros(2)
        self._walk_time = 0.0
        self.history: list[MoveRecord] = []

    def move_seconds(self, target: list[float]) -> float:
        """Simulated time a move to ``target`` takes."""
        target = np.asarray(target, dtype=np.float64)
        distance = float(np.hypot(*(target[:2] - self.commanded[:2])))
        p = self.params
        return p.move_overhead_s + (distance / p.speed_m_s if p.speed_m_s > 0 else 0.0)

    def move(self, target: list[float], wait: bool = True) -> None:
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

        self.clock.advance(p.move_overhead_s + (distance / p.speed_m_s if p.speed_m_s > 0 else 0.0), wait=wait)
        if distance > 0:
            self._settle_start = self.clock.now()
            angle = rng.uniform(0.0, 2.0 * np.pi)
            self._settle_dir = np.array([np.cos(angle), np.sin(angle)])

        self.commanded, self.true, self.reported = target, true, reported
        self.history.append(
            MoveRecord(self._move_index, self.clock.now(), target.tolist(), true.tolist(), reported.tolist())
        )

    def drift_offset_at(self, t: float) -> np.ndarray:
        """Specimen drift (thermal plus settling) relative to the stage, at simulated time t."""
        p = self.params
        steady = self._drift_dir * (p.drift_m_per_min / 60.0) * t
        if p.drift_random_walk_m > 0:
            # One step per simulated minute, each from its own stream, so the walk
            # does not depend on how often it is sampled.
            while self._walk_time + 60.0 <= t:
                minute = int(self._walk_time // 60.0)
                self._walk += self.rng.for_event("drift_walk", minute).normal(0.0, p.drift_random_walk_m, size=2)
                self._walk_time += 60.0
        settle = np.zeros(2)
        if p.settle_drift_m_s > 0 and np.isfinite(self._settle_start) and t >= self._settle_start:
            elapsed = t - self._settle_start
            settle = self._settle_dir * p.settle_drift_m_s * p.settle_tau_s * (1.0 - np.exp(-elapsed / p.settle_tau_s))
        return steady + self._walk + settle

    def true_xy_at(self, t: float) -> np.ndarray:
        return self.true[:2] + self.drift_offset_at(t)
