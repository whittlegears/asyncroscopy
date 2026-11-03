"""Seeded random streams, one per kind of event.

Every random draw in the twin comes from here so that the same seed and the
same sequence of commands always produce the same run, while repeating an
acquisition still gets fresh noise (the event index advances).
"""

from __future__ import annotations

import numpy as np


class EventRng:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def for_event(self, event: str, index: int = 0) -> np.random.Generator:
        event_code = sum((position + 1) * ord(character) for position, character in enumerate(event))
        return np.random.default_rng([self.seed % (2**32), event_code % (2**32), int(index) % (2**32)])
