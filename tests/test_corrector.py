import json
from datetime import datetime, timedelta, timezone

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR
from asyncroscopy.instruments.electron_microscope.hardware.corrector_ceos import CEOSCorrector


def test_digital_twin_corrector_reports_ceos_compatible_info(corrector_proxy):
    payload = json.loads(corrector_proxy.get_info())

    assert payload["jsonrpc"] == "2.0"
    assert payload["result"]["manufacturer"] == "CEOS"
    assert payload["result"]["simulation"] is True
    assert corrector_proxy.state() == tango.DevState.ON


def test_digital_twin_tableau_and_correction_round_trip(corrector_proxy):
    initial = json.loads(corrector_proxy.acquire_tableau("Fast 1"))
    initial_c1 = initial["result"]["aberrations"]["C1"][0]

    response = json.loads(corrector_proxy.correct_aberration("C1 8e-9"))
    measured = json.loads(corrector_proxy.acquire_tableau("Fast 1"))

    assert response["result"]["corrected"] is True
    assert measured["result"]["aberrations"]["C1"][0] == pytest.approx(
        initial_c1 - 8e-9
    )


def test_simulation_coefficients_are_absolute_and_validate_shape(corrector_proxy):
    coefficients = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    coefficients["C1"] = [12e-9]
    corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))

    updated = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    assert updated["C1"] == pytest.approx([12e-9])

    coefficients["C1"] = [1.0, 2.0]
    with pytest.raises(tango.DevFailed):
        corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))


def test_last_aberrations_is_empty_until_a_tableau_is_measured(corrector_proxy):
    corrector_proxy.Init()
    assert corrector_proxy.last_aberrations == ""

    corrector_proxy.measure_c1a1()
    assert corrector_proxy.last_aberrations == ""


def test_tableau_records_last_aberrations_as_reported_with_time_and_command(corrector_proxy):
    before = datetime.now(timezone.utc)
    response = json.loads(corrector_proxy.acquire_tableau("Fast 1"))

    record = json.loads(corrector_proxy.last_aberrations)

    assert record["aberrations"] == response["result"]["aberrations"]
    assert record["command"] == "acquire_tableau"
    measured_at = datetime.fromisoformat(record["measured_at"])
    assert before - timedelta(seconds=1) <= measured_at <= datetime.now(timezone.utc)


class Recorder:
    _last_aberrations = None


def test_ceos_tableau_response_is_recorded_verbatim():
    reported = {"C1": [1.5e-9], "A1": [2.0e-9, -3.0e-9], "C3": [1.2e-6]}
    wire = CEOSCorrector._encode_netstring({"jsonrpc": "2.0", "id": 4, "result": {"aberrations": reported}})
    device = Recorder()

    CORRECTOR._record_aberrations(device, CEOSCorrector._decode_netstring(wire), "acquire_tableau")

    assert device._last_aberrations["aberrations"] == reported
    assert device._last_aberrations["command"] == "acquire_tableau"


@pytest.mark.parametrize(
    "response",
    [
        json.dumps({"jsonrpc": "2.0", "id": 5, "error": {"code": -1, "message": "no beam"}}),
        json.dumps({"jsonrpc": "2.0", "id": 6, "result": {"aberrations": {}}}),
        "not json",
    ],
)
def test_responses_without_aberrations_are_not_recorded(response):
    device = Recorder()

    CORRECTOR._record_aberrations(device, response, "acquire_tableau")

    assert device._last_aberrations is None


def test_ceos_netstring_codec_round_trip():
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "acquireTableau",
        "params": {"tabType": "Fast", "angle": 1.0},
    }

    encoded = CEOSCorrector._encode_netstring(payload)
    decoded = json.loads(CEOSCorrector._decode_netstring(encoded))

    assert decoded == payload


class FragmentedSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def recv(self, _bufsize):
        return self.chunks.pop(0) if self.chunks else b""


def test_ceos_netstring_receiver_handles_fragmented_payload():
    raw = CEOSCorrector._encode_netstring({"result": {"ok": True}})
    socket = FragmentedSocket([raw[:3], raw[3:9], raw[9:]])

    assert CEOSCorrector._recv_netstring(socket) == raw


def test_ceos_without_an_address_faults_and_says_where_to_set_it():
    from tango import DevState

    class Unconfigured:
        ceos_host = ""
        ceos_port = 9092
        socket_timeout = 1.0
        _last_status = ""
        state = None
        errors: list = []

        def set_state(self, state):
            self.state = state

        def error_stream(self, message):
            self.errors.append(message)

    device = Unconfigured()
    CEOSCorrector._connect_backend(device)
    assert device.state == DevState.FAULT
    assert "ceos_host" in device._last_status and "YAML" in device._last_status


def test_ceos_has_no_built_in_server_address():
    assert CEOSCorrector.TangoClassClass.device_property_list["ceos_host"][2] in ([], [""], "")
