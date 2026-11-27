"""
Concrete STAGE implementation for tests and local simulated configs.
"""

from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE


class TestStage(STAGE):
    """Cached stage implementation used when no hardware-backed stage exists."""

    def _read_position(self) -> list[float]:
        return list(self._position)

    def _write_position(self, value) -> None:
        position = [float(component) for component in value]
        if len(position) != 5:
            raise ValueError("Stage position must be [x, y, z, alpha, beta]")
        self._position = position

    def _read_beta_tilt_enabled(self) -> bool:
        return True

    def _is_moving(self) -> bool:
        """Its moves set the position at once, so it is never between two."""
        return False


if __name__ == "__main__":
    TestStage.run_server()
