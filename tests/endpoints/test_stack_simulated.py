"""A stack counts as simulated only when every device that can act says it is simulated."""

from asyncroscopy.mcp.descriptors import ToolDescriptor
from asyncroscopy.mcp.mcp_server import MCPServer


class _Exported:
    value_string: list[str] = []

    def get_device_exported(self, pattern):
        return self


class _Proxy:
    def __init__(self, simulated):
        self.simulated = simulated

    def get_property(self, names):
        return {"simulated": [] if self.simulated is None else [str(self.simulated).lower()]}


def _server(devices: dict) -> MCPServer:
    server = MCPServer("t", "localhost", 0, needs_approval={}, blocked_classes=[], data_device_address="", database=_Exported(), verbose=False)
    for prefix, (simulated, acts) in devices.items():
        server.add_device(prefix, _Proxy(simulated))
        server.descriptors[f"{prefix}_go"] = ToolDescriptor(name=f"{prefix}_go", device=prefix, command="go", native=False,
                                                            action_role="other", risk="high", acts=acts, operator_only=False)
    return server


def test_one_real_actuating_device_makes_the_stack_real():
    assert _server({"Twin": (True, True), "Stage": (True, True)}).stack_simulated()
    assert not _server({"Twin": (True, True), "Stage": (False, True)}).stack_simulated()
    assert not _server({"Twin": (True, True), "Stage": (None, True)}).stack_simulated()
    assert _server({"Twin": (True, True), "Readout": (False, False)}).stack_simulated()  # a read-only device does not decide
    assert not _server({}).stack_simulated()
