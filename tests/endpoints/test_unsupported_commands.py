"""A command a device always refuses is marked unsupported in its action_hints and not offered to agents."""

import json

import tango

from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.digital_twin_particles import DigitalTwinParticles
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt
from asyncroscopy.instruments.scanning_beam import unsupported_commands
from asyncroscopy.mcp.mcp_server import MCPServer

PARTICLES = "asyncroscopy/digitaltwinparticles/default"


class _Exported:
    def __init__(self, names):
        self.value_string = names

    def get_device_exported(self, pattern):
        return self


def test_the_refusing_defaults_name_their_commands():
    assert "acquire_camera_image" in unsupported_commands(DigitalTwinParticles)
    assert "acquire_camera_image" in unsupported_commands(DigitalTwin)
    assert "acquire_camera_image" not in unsupported_commands(DigitalTwinTilt)
    assert "blank_beam" not in unsupported_commands(DigitalTwinParticles)


def test_an_unsupported_command_is_not_offered_to_agents(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access(PARTICLES))
    assert json.loads(proxy.action_hints)["acquire_camera_image"]["unsupported"] is True
    offered = {}
    for role in ("agent", "operator"):
        server = MCPServer("t", "localhost", 0, blocked_functions={}, blocked_classes=[], data_device_address="", endpoint_role=role,
                           database=_Exported([tango_ctx.get_device_access(PARTICLES)]), verbose=False)
        server.setup(print_summary=False)
        offered[role] = set(server.descriptors)
    assert "DigitalTwinParticles_acquire_camera_image" not in offered["agent"]
    assert "DigitalTwinParticles_acquire_camera_image" in offered["operator"]
    assert "DigitalTwinParticles_blank_beam" in offered["agent"] and "DigitalTwinParticles_acquire_scanned_image" in offered["agent"]
