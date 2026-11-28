# MCP server

The MCP server (`asyncroscopy/mcp/mcp_server.py`) is a FastMCP HTTP server over a live Tango database. It turns Tango commands into MCP tools, so a model client can call the devices. Start it after the stack is up.

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

The default endpoint is `http://127.0.0.1:8000/mcp`. To run an LLM agent on these tools, see the [agent MCP server](agent_mcp_server.md).

## Config

```yaml
tango:
  host: localhost              # the computer that runs the Tango database
  port: 9094

mcp:
  name: Spectra300_MCP
  transport: streamable-http
  http_host: 127.0.0.1         # 0.0.0.0 when other computers must connect
  http_port: 8000
  data_device_address: asyncroscopy/data/default
  quiet: true
  tool_log: ""                 # JSONL file of tool calls; blank turns it off
  include_only_functions: []   # blank: every command that is not blocked
  blocked_classes:
    - DataBase
    - DServer
  blocked_functions:
    "*":
      - Init
      - Kill
      - RestartServer
```

## How it finds tools

At startup the server:

1. Connects to the Tango database at `tango.host` and `tango.port`.
2. Calls `get_device_exported("*")` and opens each device with `DeviceProxy`.
3. Skips devices whose class is in `blocked_classes`.
4. Reads each device's `command_list_query()` and skips blocked commands.
5. Registers each remaining command as a tool named `<TangoClass>_<command>`, for example `AutoScriptMicroscope_acquire_scanned_image`.
6. Registers the native tools `list_devices` and `get_data_from_key`.

The Tango database is the source of truth. If a device is registered, exported and not blocked, its commands become tools. The tool set changes only when the server restarts.

Tool signatures come from the Tango command types. The server converts NumPy values and `DevEncoded` results to JSON-safe values.

## Adding tools

For device behavior, add a Tango `@command` to the device class:

```python
from tango.server import Device, command


class CAMERA(Device):
    @command(dtype_in=float, dtype_out=str)
    def set_exposure(self, exposure_ms: float) -> str:
        self.exposure_ms = exposure_ms
        return f"exposure set to {exposure_ms} ms"
```

Restart the stack and the MCP server. The tool `CAMERA_set_exposure` appears.

For behavior that is not one device command, such as a lookup across devices, add a method to `MCPServer` and add it to the `native_tools` list in `MCPServer.setup()`. Do not make a subclass for this.

## Reading data

`get_data_from_key` reads a key from the Tiled server that the DATA device reports. It returns JSON-safe metadata and a small preview. The MCP computer does not need access to the instrument's file system.

## Blocking commands

The MCP server exposes hardware control. The block lists are part of the safety boundary, so set them before you connect an autonomous agent.

- `blocked_classes` hides whole Tango classes.
- `blocked_functions` hides commands. Under `"*"`, use a command name or `Class.command`. Under a class name, list that class's commands.
- `include_only_functions`, if not empty, keeps only the listed commands.

```yaml
blocked_functions:
  "*":
    - Init
    - DATA.stop_tiled_server
  AutoScriptMicroscope:
    - Disconnect
```

## Commands that wait for an operator

Blocking hides a command. To keep a command available but put a person in the
loop, list it under `needs_approval` instead:

```yaml
mcp:
  needs_approval:
    "*": [move_stage, acquire_map_grid, correct_aberration, set_column_valves, unblank_beam]
  approvals_db: outputs/approvals.sqlite
```

Patterns match by Tango class or `"*"`, and `Class.command` also works.

- **Agent endpoint** (the default `role: agent`). A listed command does not
  run. The call is stored in `approvals_db`, and the tool answers with an error
  that starts `HELD <id>`. The agent can check it with `call_status`.
- **Operator endpoint.** A second bridge with `role: operator` and the same
  `approvals_db` runs every command directly. It also offers `list_pending`,
  `approve_call` and `deny_call`. Approving runs the held command there and
  stores the result for `call_status`. Give this endpoint only to a person or
  their GUI, never to an agent.

For the digital twin, `configs/mcp_dt.yaml` has this on, and
`configs/mcp_dt_operator.yaml` is its operator endpoint, on port 8010. On the
microscope config it is commented out until it has been tried there.
Workflows (`asyncroscopy/agent_workflows`) wait on held calls by themselves.

## Running without the startup script

`run_mcp.py` passes the config to the module as arguments. To run the module directly against a stack that is up:

```bash
uv run python -m asyncroscopy.mcp.mcp_server \
  --name Spectra300_MCP \
  --tango-host localhost \
  --tango-port 9094 \
  --transport streamable-http \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --data-device-address asyncroscopy/data/default \
  --blocked-classes-json '["DataBase", "DServer"]' \
  --blocked-functions-json '{"*": ["Init", "Kill", "RestartServer"]}'
```

## Tests

```bash
uv run pytest tests/test_mcp_server.py tests/test_run_servers.py
```

Add or change tests when you change tool discovery, command filtering, argument mapping or the native tools.
