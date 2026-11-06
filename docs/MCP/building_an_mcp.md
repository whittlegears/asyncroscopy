# Adding MCP Capabilities

The instrument MCP server is `asyncroscopy.mcp.mcp_server.MCPServer`. It is configured by [configs/mcp.yaml](../../configs/mcp.yaml) and started with `startup_scripts/run_mcp.py`.

The LLM agent swarm is served by a second server; see [agent_mcp_server.md](agent_mcp_server.md).

The server exposes two kinds of tools:

- Tango commands discovered from the live Tango database.
- Native MCP helpers defined by the server's components (`asyncroscopy/mcp/components/`).

## Discovery Model

MCP does not scan Python packages or require a custom server subclass. At
startup it:

1. Connects to the Tango database from `tango.host` and `tango.port`.
2. Calls `get_device_exported("*")`.
3. Opens each exported device with `DeviceProxy`.
4. Reads each device's `command_list_query()`.
5. Decides once what each command is (a `ToolDescriptor`: action role, risk, whether it acts,
   operator-only, routine in auto mode, a stop) from what the device declares and the MCP config.
6. Registers every non-blocked command as a FastMCP tool.

That means the Tango database is the source of truth. If a device is registered,
exported, and not blocked, MCP can expose its commands.

## Add A Device Command

Add a Tango command to the relevant device class:

```python
from tango.server import Device, command


class CAMERA(Device):
    @command(dtype_in=float, dtype_out=str)
    def set_exposure(self, exposure_ms: float) -> str:
        self.exposure_ms = exposure_ms
        return f'exposure set to {exposure_ms} ms'
```

Then start the Tango/device stack and MCP:

```bash
uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

If the device is live, MCP discovers `CAMERA.set_exposure` automatically.

## Declare What A Command Is

Declare risk, operator-only, stop and role with `asyncroscopy.utils.commands.command`, and give
the device an `action_hints` attribute that serves the declarations:

```python
from tango.server import Device

from asyncroscopy.utils.commands import action_hints_attribute, command


class CAMERA(Device):
    action_hints = action_hints_attribute()

    @command(dtype_in=float, dtype_out=str, risk="low")
    def set_exposure(self, exposure_ms: float) -> str:
        """Set the exposure time in milliseconds."""

    @command(stop=True, role="beam.blank", risk="low")
    def close_shutter(self) -> None:
        """Close the camera shutter."""
```

Declare on the base class: every vendor backend and twin that subclasses it inherits the
declarations and redeclares nothing (`declared_hints` merges them along the class hierarchy). A
`role` from the read roles (`setting.read`, `stage.read`, `data.read`, `job.status`,
`job.results`) also tells the server the command only observes. `routine=True` lets auto mode run a
high-risk command without a person; `operator_only=True` keeps it off agent endpoints. A device
that works out more about its commands (units and ranges at its own limits, as the
`ElectronMicroscope` base does with `min_fov`, `max_fov` and `max_defocus`) passes
`action_hints_attribute(extra=lambda device: device.argument_hints())`.

A command with no declaration falls back to the name rules in `asyncroscopy/safety/policy.py`
(`get_*` low, unknown high), and the server lists those commands at startup. `mcp.safety.risk`
in the MCP config overrides both. `stop_all` calls the no-argument commands whose role is
`beam.blank` or `probe.retract`, or that declare `stop=True`.

Use `asyncroscopy.utils.commands.command` rather than `tango.server.command`: it also carries the
docstring's first paragraph to the tool description and names the tool's argument after the
Python parameter.

## Add A Native MCP Helper

Native helpers live in components (`asyncroscopy/mcp/components/`): `data_tools` (`list_data`,
`view_data`), `jobs` (`run_job`, `wait_for_job`), `gate` (`approval_status`), `discovery`
(`find_tools`, `call_tool`, `endpoint_info`, `action_space`), `skills_tools`, `trace_tools`
(`trace_since`) and `estop` (`stop_all`). Each has `register(server, cfg)` and a `natives()`
method returning the tools it offers on this endpoint:

```python
from fastmcp.tools import tool


class DataTools:
    def natives(self) -> list:
        return [self.list_data, self.view_data]

    @tool()
    def list_data(self, kind: str | None = None, search: str | None = None, limit: int = 20, offset: int = 0) -> dict:
        ...
```

A component that changes how device calls run adds an interceptor
(`asyncroscopy/mcp/interceptors.py`: `before_call`, `after_call`, `on_error`) with
`server.add_interceptor(...)`.

Use this path for cross-device helpers, data lookups, or MCP-specific
convenience commands. Do not create a separate subclass for normal asyncroscopy
behavior.

## Block Commands

Use [configs/mcp.yaml](../../configs/mcp.yaml):

```yaml
mcp:
  blocked_classes:
    - DataBase
    - DServer
  blocked_functions:
    "*":
      - Init
      - Kill
      - RestartServer
    DATA:
      - stop_tiled_server
```

`blocked_classes` hides whole Tango classes. `blocked_functions` can hide global
command names, class-specific command names, or fully qualified
`Class.command` entries.

## Test Changes

Use the focused MCP tests:

```bash
uv run pytest tests/test_mcp_server.py tests/test_run_servers.py
```

Add or adjust tests when you change tool discovery, command filtering, argument
mapping, or native MCP helpers.
