"""The instrument clock a trace writer stamps its events with (the instrument's ``clock_s`` attribute)."""

from __future__ import annotations

import threading
import time


class InstrumentClock:
    """Reads ``clock_s`` from one Tango device: simulated seconds on a twin, Unix time on hardware.

    A TraceWriter calls it for each event. A read that fails returns None (the event then
    carries this computer's time and ``"clock": "wall"``), and the device is not asked again
    for ``retry_after_s``, so a stopped instrument never slows the trace down.
    """

    def __init__(self, address: str, timeout_ms: int = 1000, retry_after_s: float = 10.0):
        self.address = address
        self.timeout_ms = timeout_ms
        self.retry_after_s = retry_after_s
        self._proxy = None
        self._failed_at: float | None = None
        self._lock = threading.Lock()

    def __call__(self) -> float | None:
        if not self.address:
            return None
        with self._lock:
            if self._failed_at is not None and time.monotonic() - self._failed_at < self.retry_after_s:
                return None
            try:
                if self._proxy is None:
                    import tango

                    self._proxy = tango.DeviceProxy(self.address)
                    self._proxy.set_timeout_millis(self.timeout_ms)
                value = float(self._proxy.read_attribute("clock_s").value)
            except Exception:
                self._failed_at = time.monotonic()
                self._proxy = None
                return None
            self._failed_at = None
            return value
