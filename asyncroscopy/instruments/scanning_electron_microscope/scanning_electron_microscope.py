"""Scanning electron microscope Tango device: the contract the ``sem`` map profile drives.

Everything an SEM has in common with a STEM is ``ScanningBeamInstrument`` (asyncroscopy/instruments/scanning_beam.py):
the beam, blanker, field of view, image shift, stage, beam current, limits and command lock. This base names the
modality and its default detector (secondary electrons); a backend implements the ``_x`` methods. The simulated
backend is ``DigitalTwinSEM`` in digital_twin.py. No vendor backend exists yet.
"""

from tango.server import device_property

from asyncroscopy.instruments.scanning_beam import ScanningBeamInstrument


class ScanningElectronMicroscope(ScanningBeamInstrument):
    """Top-level scanning electron microscope device."""

    default_scan_detector = device_property(
        dtype=str,
        default_value="se",
        doc="Scanning detector used when a caller names none: acquire_scanned_image with an empty list, a job plan without detectors.",
    )

    def read_instrument_type(self) -> str:
        return "SEM"
