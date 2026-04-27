# Building Custom MCP Servers

Build Model Context Protocol servers to expose hardware or services to LLM agents.

## Quick Start

Create an MCP server that discovers and wraps Tango device commands:

```python
from asyncroscopy.mcp.mcp_server import MCPServer

server = MCPServer(
    name="MyServer",
    tango_host="localhost",
    tango_port=9094
)
server.start()
```

The server automatically:
- Connects to a Tango database
- Discovers all exported devices
- Extracts command signatures and types
- Generates MCP tools from Tango commands
- Starts an MCP server for LLM agents

## Architecture

### Discovery Pipeline

```
Tango Database
    ↓
MCPServer.__init__()  → Connect to DB
    ↓
MCPServer.setup()     → Query devices and commands
    ↓
_find_tools()         → Extract device classes and command info
    ↓
_create_wrapper()     → Convert Tango types to Python types
    ↓
MCP tool registration → Expose to LLM agents
```

### Type Mapping

Tango command types are automatically mapped to Python types for MCP:

| Tango Type | Python Type |
|-----------|------------|
| `DevVoid` | `None` |
| `DevBoolean` | `bool` |
| `DevFloat64` | `float` |
| `DevInt32` | `int` |
| `DevString` | `str` |
| `DevEncoded` | `dict` (base64) |
| Arrays | `list[type]` |

DevEncoded binary data is base64-encoded:

```json
{
  "encoding": "base64",
  "metadata": "header_string",
  "payload": "base64_encoded_data"
}
```

## Configuration

### Block Lists

Exclude specific commands or device classes from MCP exposure:

```python
server = MCPServer(
    name="MyServer",
    tango_host="localhost",
    tango_port=9094,
    blocked_classes=["DataBase", "DServer", "MyUnwantedClass"],
    blocked_functions={
        "*": ["Init", "Status"],  # Global blocks
        "Microscope": ["Connect", "Disconnect"],  # Per-class blocks
    },
    search_packages=["mymodule", "asyncroscopy"]
)
```

### Parameters

- **`name`** (str): Display name for the server
- **`tango_host`** (str): Tango database hostname
- **`tango_port`** (int): Tango database port
- **`blocked_classes`** (list[str]): Tango classes to skip (default: `["DataBase", "DServer"]`)
- **`blocked_functions`** (dict | list): Commands to exclude
  - List: Applied globally to all classes
  - Dict: Map class names to command lists; `"*"` for global blocks
- **`search_packages`** (list[str]): Python packages to search for Tango Device source code (default: `["asyncroscopy"]`)
- **`verbose`** (bool): Print discovery and registration progress (default: `True`)

## Adding Custom Tools

Extend the MCPServer class to add custom tools, resources, and prompts:

### Custom Tool

```python
from fastmcp.tools import tool

class MyMCPServer(MCPServer):
    @tool()
    def calculate_exposure(self, gain: int) -> float:
        """Calculate optimal exposure based on gain."""
        return gain * 2.5
```

### Custom Resource

```python
from fastmcp.resources import resource

class MyMCPServer(MCPServer):
    @resource("config://system")
    def get_system_config(self) -> str:
        """Return system configuration."""
        return "TIMEOUT=30\nRETRIES=3"
```

### Custom Prompt

```python
from fastmcp.prompts import prompt

class MyMCPServer(MCPServer):
    @prompt()
    def focus_procedure(self, voltage: float) -> str:
        """Prompt template for focusing procedure."""
        return f"Please focus the beam at {voltage}kV and report any drift."
```

Custom tools, resources, and prompts are automatically registered during `setup()`.

## Implementation Details

### Source-Level Introspection

The server introspects Tango Device source code to improve tool descriptions:

1. Search for the Device subclass in `search_packages`
2. Extract the actual parameter names (not generic `arg`)
3. Pull docstrings from the command method
4. Build rich descriptions for LLM agents

```python
class Microscope(Device):
    @command(dtype_in=int, dtype_out=float)
    def acquire_image(self, exposure_ms: int) -> float:
        """Acquire a STEM image with specified exposure."""
        # implementation
```

The MCP tool parameter is named `exposure_ms` (from source), not `arg`.

### Wrapper Generation

Commands are wrapped with proper Python signatures using `exec()`:

```python
def _create_wrapper(self, func, cmd_info, command_name, dev_class):
    # Resolve parameter name from source
    param_name = self._get_param_name(dev_class, command_name)
    
    # Map Tango type to Python type
    py_type = self._tango_type_to_python(cmd_info.in_type)
    
    # Generate function with proper signature
    exec(f"def wrapper({param_name}: py_type): ...")
    
    # Normalize DevEncoded output to JSON
    return self._normalize_command_result(...)
```

### Tool Registration

Tools are registered via FastMCP:

```python
tool_obj = Tool.from_function(wrapped_func)
self.mcp.add_tool(tool_obj)
```

Each tool has:
- Parameter names from source code
- Type hints for validation
- Full docstrings with Tango metadata
- Proper return type annotations

## Transport Options

### Stdio (Default)

For local connections to agents:

```python
server.start()
```

Uses JSON-RPC over stdin/stdout. Connect agents directly to the process.

### HTTP

For remote access:

```python
server.start_http(host="0.0.0.0", port=8000)
```

Exposes MCP tools via HTTP. Agents connect via HTTP client.

## Usage Example

### Standalone Server

```python
from asyncroscopy.mcp.mcp_server import MCPServer

# Create server
server = MCPServer(
    name="Microscope",
    tango_host="microscope.lab.local",
    tango_port=9094,
    blocked_functions={"*": ["Init"]},
    verbose=True
)

# Add custom tools
from fastmcp.tools import tool

class CustomServer(MCPServer):
    @tool()
    def suggest_parameters(self, voltage: int) -> str:
        """Suggest imaging parameters for given voltage."""
        return f"For {voltage}kV: gain=50, exposure=10ms"

# Create instance and start
custom = CustomServer(
    name="Microscope",
    tango_host="localhost",
    tango_port=9094
)
custom.start()
```

### With Custom Device Classes

```python
class MyServer(MCPServer):
    @tool()
    def list_available_modes(self) -> list[str]:
        """List available imaging modes."""
        return ["STEM", "BF", "DF", "HAADF"]

# Ensure your Device subclasses are importable
import mymodule  # Contains MyDevice(Device)

server = MyServer(
    name="MyServer",
    tango_host="localhost",
    tango_port=9094,
    search_packages=["mymodule"]
)
server.start()
```

## Testing

### Unit Tests

Test custom tools in isolation:

```python
def test_custom_tool():
    server = MyServer(name="Test", tango_host="localhost", tango_port=9094)
    result = server.suggest_parameters(voltage=200)
    assert "gain" in result
```

### Integration Tests

Test with a real Tango database:

```python
import tango

def test_mcp_with_tango():
    # Start Tango services (database, device server)
    # Create MCPServer
    server = MCPServer(
        name="Test",
        tango_host="localhost",
        tango_port=9094
    )
    server.setup()
    
    # Verify tools are registered
    assert len(server.tools) > 0
```

See `tests/test_mcp_server.py` for full test examples.

## Advanced Patterns

### Conditional Tool Registration

```python
class ConditionalServer(MCPServer):
    def setup(self):
        super().setup()
        
        # Add tools based on discovered devices
        available_devices = self.list_devices()
        if any("EDS" in d for d in available_devices):
            self.mcp.add_tool(self.analyze_eds_spectrum)
```

### Dynamic Blocking

```python
class FilterServer(MCPServer):
    def _is_blocked_function(self, dev_class, command_name):
        # Custom logic: block based on runtime state
        if command_name.startswith("_"):
            return True
        return super()._is_blocked_function(dev_class, command_name)
```

### Multi-Device Coordination

```python
class CoordinatedServer(MCPServer):
    @tool()
    def acquire_multimodal(self, exposure_ms: int) -> dict:
        """Acquire STEM + EDS simultaneously."""
        stem_dev = tango.DeviceProxy("test/microscope/stem")
        eds_dev = tango.DeviceProxy("test/detector/eds")
        
        stem_data = stem_dev.command_inout("AcquireImage", exposure_ms)
        eds_data = eds_dev.command_inout("Acquire", exposure_ms)
        
        return {"stem": stem_data, "eds": eds_data}
```

## Troubleshooting

### No Devices Discovered

Check:
1. Tango database is running: `tango_host` and `tango_port` are correct
2. Devices are exported: `server.list_devices()` returns non-empty list
3. Devices are not blocked: Check `blocked_classes` and `blocked_functions`

### Tools Not Appearing in Agent

Check:
1. `setup()` is called before agent connects
2. Tool registration succeeded (check verbose output)
3. Tool wrapper function has valid signature
4. Parameter types are JSON-serializable

### Source Introspection Not Working

Verify:
1. Device subclass is in a module under `search_packages`
2. Module is importable: `import mymodule` works
3. Class name matches Tango class name exactly
4. Source code has proper type hints

## References

- [Tango Python Documentation](http://www.tango-controls.org/developers/python-api/)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://modelcontextprotocol.io/)
