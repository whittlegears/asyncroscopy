"""Where asyncroscopy's servers put what they write, resolved once from the environment when asked.

``Settings.from_env()`` reads the variables at call time (never at import), so a test or a launcher
can set them first. Library code takes explicit paths; entry points (command lines, launchers) build
their defaults from these settings.

ASYNCROSCOPY_OUTPUTS   the folder every server writes under (traces, notes, placements). Default: ``outputs``
                       in the working directory the entry point was started from.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

OUTPUTS_ENV = "ASYNCROSCOPY_OUTPUTS"


@dataclass(frozen=True)
class Settings:
    outputs_root: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        return cls(outputs_root=Path(env.get(OUTPUTS_ENV) or "outputs").resolve())

    def output(self, *parts: str) -> Path:
        """A path under ``outputs_root``, e.g. ``output("trace", "analysis.jsonl")``."""
        return self.outputs_root.joinpath(*parts)
