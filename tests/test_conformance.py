"""Backend conformance: the twin passes; a backend reporting nanometres fails."""

import json

from asyncroscopy.conformance import FAIL, run


def test_twin_is_conformant_with_writes_and_acquire(twin_proxy, scan_proxy):
    scan_proxy.imsize = 32
    rep = run(twin_proxy, "twin", writes=True, acquire=True)
    assert rep.ok, [r for r in rep.results if r.level == FAIL]
    assert rep.features["return-to-site (register_images skill)"]["status"] == "yes"
    assert rep.features["beam tilt / image shift"]["status"] == "no"     # twin leaves these out, cleanly


class _Cmd:
    def __init__(self, name):
        self.cmd_name = name


class NanometreBackend:
    """A vendor backend that forgot to convert: stage in nm, and a crash on get_fov."""

    def info(self):
        return type("I", (), {"dev_class": "VendorScope"})()

    def state(self):
        return "ON"

    def command_list_query(self):
        return [_Cmd(n) for n in ("get_stage", "get_fov", "get_defocus", "get_parameters")]

    def command_inout(self, name, *args):
        if name == "get_stage":
            return [1500.0, -200.0, 0.0, 0.0, 0.0]
        if name == "get_fov":
            raise KeyError("fov")
        if name == "get_defocus":
            return 0.0
        if name == "get_parameters":
            return json.dumps({"vendor": "x"})
        raise AssertionError(name)


def test_unit_mistake_and_crash_are_failures():
    rep = run(NanometreBackend(), "vendor/scope/1")
    levels = {r.name: (r.level, r.detail) for r in rep.results}
    assert levels["get_stage"][0] == FAIL and "metres" in levels["get_stage"][1]
    assert levels["get_fov"][0] == FAIL and "crashed" in levels["get_fov"][1]
    assert not rep.ok
    assert rep.features["imaging"]["status"] == "no"
