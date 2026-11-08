"""SAMPLE Tango device: which sample is in the instrument, and the dose it has had.

The sample is not part of the stage: ``load_sample`` records what was loaded and gives it a new ``sample_load_id``,
``get_sample`` says what is loaded, and ``get_dose`` answers from the dose ledger for that load. Every dataset
records the identity (asyncroscopy.data.record), and job devices and microscopes hold a proxy to this device. It works
the same for any instrument, so it has no vendor backends.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from tango import DevState
from tango.server import Device, device_property

from asyncroscopy.data.dose_ledger import dose_at
from asyncroscopy.utils.commands import action_hints_attribute, command


class SAMPLE(Device):
    """The sample loaded in an instrument: its identity, its load id and its dose ledger."""

    # What each command is (risk, role), as its @command declares it, for the MCP server.
    action_hints = action_hints_attribute()

    sample_file = device_property(dtype=str, default_value="", doc="Where load_sample records the loaded sample, so a restart keeps it; blank: outputs/sample/<device name>_sample.json.")

    def init_device(self) -> None:
        Device.init_device(self)
        self._sample_path = Path(self.sample_file or f"outputs/sample/{self.get_name().replace('/', '_')}_sample.json")
        self.set_state(DevState.ON)

    @command(dtype_out=str, risk="low", role="setting.read")
    def get_sample(self) -> str:
        """The sample in the holder: JSON {"sample_id", "sample_load_id", "loaded_at", "who"}; nulls when none was recorded.
        Every dataset records these, so a position from an earlier load is not mistaken for one on this sample."""
        empty = {"sample_id": None, "sample_load_id": None, "loaded_at": None, "who": None}
        if not self._sample_path.exists():
            return json.dumps(empty)
        try:
            saved = json.loads(self._sample_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return json.dumps({**empty, "error": f"the sample record in {self._sample_path} could not be read ({exc}); load the sample again"})
        return json.dumps({key: saved.get(key) for key in empty})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"sample_id": "...", "who": "..."}: the sample just put in the holder.', risk="high", operator_only=True, role="setting.set")
    def load_sample(self, request: str) -> str:
        """Record that a sample was loaded (operator only): it gets a new sample_load_id, so stage positions
        recorded before this load are known to belong to another load, even of the same sample."""
        body = json.loads(request) if request.strip().startswith("{") else {"sample_id": request.strip()}
        sample_id = str(body.get("sample_id") or "").strip()
        if not sample_id or len(sample_id) > 200:
            raise ValueError("load_sample needs a sample_id of 1 to 200 characters")
        record = {"sample_id": sample_id, "sample_load_id": uuid.uuid4().hex, "loaded_at": time.time(), "who": str(body.get("who") or "")[:80] or None}
        self._sample_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._sample_path.with_name(self._sample_path.name + ".partial")
        partial.write_text(json.dumps(record), encoding="utf-8")
        partial.replace(self._sample_path)
        return json.dumps(record)

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"x_m": ..., "y_m": ...} for the dose at one stage point, or {} for the exposure count only.', risk="low", role="setting.read")
    def get_dose(self, request: str) -> str:
        """Electron dose already put on the loaded sample, from the dose ledger: exposures counted (scanned frames and
        parked-beam spectra), and at a stage point the exposures that covered it and their summed dose in e/A^2;
        "not_counted" says what is not counted."""
        body = json.loads(request) if request.strip() else {}
        point = [body.get("x_m"), body.get("y_m")]
        if any(v is not None for v in point) and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point):
            raise ValueError("give both x_m and y_m in metres, or neither")
        sample = json.loads(self.get_sample())
        if sample.get("sample_load_id") is None:
            why = sample.get("error") or "no sample load is recorded: an operator calls the SAMPLE device's load_sample when a sample goes in"
            return json.dumps({"sample_load_id": None, "sample_id": sample.get("sample_id"), "exposures": None, "error": why})
        x_m, y_m = (float(point[0]), float(point[1])) if point[0] is not None else (None, None)
        return json.dumps({**dose_at(sample["sample_load_id"], x_m, y_m), "sample_id": sample.get("sample_id")})


if __name__ == "__main__":
    SAMPLE.run_server()
