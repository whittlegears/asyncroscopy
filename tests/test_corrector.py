import json

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.hardware.corrector_ceos import CEOSCorrector


def test_digital_twin_corrector_reports_ceos_compatible_info(corrector_proxy):
    payload = json.loads(corrector_proxy.get_info())

    assert payload["jsonrpc"] == "2.0"
    assert payload["result"]["manufacturer"] == "CEOS"
    assert payload["result"]["simulation"] is True
    assert corrector_proxy.state() == tango.DevState.ON


def test_digital_twin_tableau_and_correction_round_trip(corrector_proxy):
    initial = json.loads(corrector_proxy.acquire_tableau("Fast 9"))
    initial_c1 = initial["result"]["aberrations"]["C1"][0]

    response = json.loads(corrector_proxy.correct_aberration("C1 8e-9"))
    measured = json.loads(corrector_proxy.acquire_tableau("Fast 9"))

    # Like the CEOS server, a successful correction returns a null result.
    assert response["result"] is None
    assert measured["result"]["aberrations"]["C1"][0] == pytest.approx(
        initial_c1 - 8e-9
    )


@pytest.mark.parametrize(
    ("tab_type", "expected"),
    [
        ("Fast", {"WD", "C1", "A1", "A2", "B2"}),
        ("Standard", {"WD", "C1", "A1", "A2", "B2", "C3", "A3", "S3", "A4"}),
        (
            "Enhanced",
            {"WD", "C1", "A1", "A2", "B2", "C3", "A3", "S3", "A4", "D4", "B4"},
        ),
    ],
)
def test_digital_twin_tableau_reports_only_its_fit_order(corrector_proxy, tab_type, expected):
    payload = json.loads(corrector_proxy.acquire_tableau(f"{tab_type} 18"))

    assert set(payload["result"]["aberrations"]) == expected


def test_digital_twin_sends_every_aberration_as_an_xy_pair(corrector_proxy):
    tableau = json.loads(corrector_proxy.acquire_tableau("Enhanced 34"))
    c1a1 = json.loads(corrector_proxy.measure_c1a1())

    for aberrations in (tableau["result"]["aberrations"], c1a1["result"]["aberrations"]):
        assert all(len(value) == 2 for value in aberrations.values())
        assert aberrations["C1"][1] == 0.0
    assert tableau["result"]["aberrations"]["C3"][1] == 0.0


def test_digital_twin_accepts_xy_pair_for_scalar_correction(corrector_proxy):
    before = json.loads(corrector_proxy.measure_c1a1())["result"]["aberrations"]["C1"][0]

    corrector_proxy.correct_aberration("C1 3e-9 0")
    after = json.loads(corrector_proxy.measure_c1a1())["result"]["aberrations"]["C1"][0]

    assert after == pytest.approx(before - 3e-9)


@pytest.mark.parametrize("args", ["fast 9", "Quick 9", "Fast 0", "Standard -5"])
def test_digital_twin_rejects_invalid_tableau_requests(corrector_proxy, args):
    with pytest.raises(tango.DevFailed):
        corrector_proxy.acquire_tableau(args)


def test_ceos_driver_sends_scalar_corrections_as_xy_pairs():
    class Recorder:
        def _call(self, method, params):
            self.sent = (method, params)
            return "{}"

    recorder = Recorder()
    CEOSCorrector._correct_aberration(recorder, "C1", [4e-9])

    assert recorder.sent == ("correctAberration", {"name": "C1", "value": [4e-9, 0.0]})


def test_simulation_coefficients_are_absolute_and_validate_shape(corrector_proxy):
    coefficients = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    coefficients["C1"] = [12e-9]
    corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))

    updated = json.loads(corrector_proxy.get_aberrations_coeff_sim())
    assert updated["C1"] == pytest.approx([12e-9])

    coefficients["C1"] = [1.0, 2.0]
    with pytest.raises(tango.DevFailed):
        corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))


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


def test_correction_larger_than_step_limit_is_refused(limited_corrector_proxy):
    before = json.loads(limited_corrector_proxy.measure_c1a1())["result"]["aberrations"]["C1"][0]

    with pytest.raises(tango.DevFailed, match="max_step"):
        limited_corrector_proxy.correct_aberration("C1 20e-9")

    after = json.loads(limited_corrector_proxy.measure_c1a1())["result"]["aberrations"]["C1"][0]
    assert after == pytest.approx(before)  # nothing reached the corrector


def test_step_limit_uses_the_vector_length(limited_corrector_proxy):
    # 4 nm and 4 nm are each under the 5 nm A1 limit, but together 5.7 nm.
    with pytest.raises(tango.DevFailed, match="max_step"):
        limited_corrector_proxy.correct_aberration("A1 4e-9 4e-9")
    limited_corrector_proxy.correct_aberration("A1 3e-9 3e-9")


def test_small_steps_cannot_add_up_past_the_total_until_a_new_tableau(limited_corrector_proxy):
    limited_corrector_proxy.acquire_tableau("Fast 9")
    limited_corrector_proxy.correct_aberration("C1 10e-9")
    limited_corrector_proxy.correct_aberration("C1 10e-9")

    with pytest.raises(tango.DevFailed, match="max_total_since_tableau"):
        limited_corrector_proxy.correct_aberration("C1 10e-9")

    limited_corrector_proxy.acquire_tableau("Fast 9")
    limited_corrector_proxy.correct_aberration("C1 10e-9")


def test_limits_are_readable(limited_corrector_proxy):
    limited_corrector_proxy.acquire_tableau("Fast 9")
    limited_corrector_proxy.correct_aberration("C1 4e-9")

    limits = json.loads(limited_corrector_proxy.get_correction_limits())

    assert limits["max_step"] == {"C1": 10e-9, "A1": 5e-9}
    assert limits["max_total_since_tableau"] == {"C1": 25e-9}
    assert limits["applied_since_tableau"]["C1"] == pytest.approx(4e-9)


def test_unlimited_corrector_accepts_large_corrections(corrector_proxy):
    corrector_proxy.correct_aberration("C1 1e-6")
    corrector_proxy.correct_aberration("C1 -1e-6")


def test_a_nan_correction_is_refused(limited_corrector_proxy):
    with pytest.raises(tango.DevFailed, match="finite"):
        limited_corrector_proxy.correct_aberration("C1 nan")


def test_a_round_aberration_with_a_direction_is_refused(corrector_proxy):
    with pytest.raises(tango.DevFailed, match="no direction"):
        corrector_proxy.correct_aberration("C1 3e-9 5e-9")


def test_limits_must_name_real_aberrations():
    from asyncroscopy.instruments.electron_microscope.hardware.corrector import _parse_limits

    with pytest.raises(ValueError, match="unknown aberrations"):
        _parse_limits('{"C 1": 1e-9}', "max_step")
    with pytest.raises(ValueError, match="> 0"):
        _parse_limits('{"C1": 0}', "max_step")


def test_a_tableau_error_does_not_reset_the_running_total(limited_corrector_proxy, monkeypatch):
    from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DigitalTwinCorrector

    limited_corrector_proxy.acquire_tableau("Fast 9")
    limited_corrector_proxy.correct_aberration("C1 10e-9")
    monkeypatch.setattr(
        DigitalTwinCorrector, "_acquire_tableau",
        lambda self, tab_type, angle: '{"jsonrpc": "2.0", "id": 1, "error": {"message": "Server error"}}',
    )

    limited_corrector_proxy.acquire_tableau("Fast 9")

    used = json.loads(limited_corrector_proxy.get_correction_limits())["applied_since_tableau"]
    assert used["C1"] == pytest.approx(10e-9)


def test_ceos_driver_connects_fast_and_waits_long_for_replies(monkeypatch):
    from types import SimpleNamespace

    from asyncroscopy.instruments.electron_microscope.hardware import corrector_ceos

    reply = CEOSCorrector._encode_netstring({"jsonrpc": "2.0", "id": 1, "result": None})
    seen = {}

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def settimeout(self, seconds):
            seen["reply_timeout"] = seconds

        def sendall(self, data):
            pass

        def recv(self, size):
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            return chunk

    def fake_connect(address, timeout):
        seen["connect_timeout"] = timeout
        sock = FakeSocket()
        sock.remaining = reply
        return sock

    monkeypatch.setattr(corrector_ceos.socket, "create_connection", fake_connect)
    device = SimpleNamespace(
        ceos_host="ceos.example", ceos_port=9092, connect_timeout_s=5.0, command_timeout_s=300.0,
        _message_id=1, _encode_netstring=CEOSCorrector._encode_netstring,
        _recv_netstring=CEOSCorrector._recv_netstring, _decode_netstring=CEOSCorrector._decode_netstring,
        set_state=lambda state: None,
    )

    CEOSCorrector._call(device, "getInfo")

    assert seen == {"connect_timeout": 5.0, "reply_timeout": 300.0}
