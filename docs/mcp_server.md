# MCP Server Documentation

The [`MCPServer`](../asyncroscopy/mcp/mcp_server.py#L43) is a bridge between a Tango control system and the Model Context Protocol (MCP). It allows LLM agents to interact directly with hardware by exposing Tango device commands as MCP tools.

---

## Core Functionality

### 1. Dynamic Device Discovery
On startup, the server queries the Tango Database to find all exported devices via [`_list_all_devices()`](../asyncroscopy/mcp/mcp_server.py#L103). It then:
- Filters out infrastructure classes (e.g., `DataBase`, `DServer`) using [`_is_blocked_class()`](../asyncroscopy/mcp/mcp_server.py#L99).
- Excludes devices or classes specified in the block lists — see [`_is_blocked_function()`](../asyncroscopy/mcp/mcp_server.py#L136).
- Dynamically queries each device for its available commands in [`_find_tools()`](../asyncroscopy/mcp/mcp_server.py#L469).

### 2. Automatic Tool Generation
Each discovered Tango command is wrapped into an MCP tool via [`_create_wrapper()`](../asyncroscopy/mcp/mcp_server.py#L393). The server:
- Maps Tango types to Python types for parameter validation — see [`_tango_type_to_python()`](../asyncroscopy/mcp/mcp_server.py#L247).
- **Source-Level Introspection**: Uses [`_get_tango_device_class()`](../asyncroscopy/mcp/mcp_server.py#L294) to search specified Python packages (default: `["asyncroscopy"]`) and `inspect` to retrieve real parameter names via [`_get_param_name()`](../asyncroscopy/mcp/mcp_server.py#L372) and docstrings via [`_get_docstring()`](../asyncroscopy/mcp/mcp_server.py#L330) from the source implementation.
- Handles `DevEncoded` data by base64-encoding the payload for [JSON-safe transport](#data-transport--encoding) — see [`_normalize_command_result()`](../asyncroscopy/mcp/mcp_server.py#L264).

---

## Configuration & Customization

### Block Lists
You can restrict which commands or classes are exposed through the following [`__init__()`](../asyncroscopy/mcp/mcp_server.py#L47) arguments:

- **`blocked_classes`**: List of Tango class names to skip entirely (defaults to `["DataBase", "DServer"]`).
- **`blocked_functions`**: 
  - A simple list (e.g., `["Init", "Status"]`) applied globally.
  - Or a dictionary mapping class names to command lists (e.g. `{"Microscope": ["Connect"]}`).
  - Use `"*"` as a dictionary key to apply global overrides (e.g. `{"*": ["Init"]}`).
- **`search_packages`**: List of Python package names to search for Tango Device subclasses when resolving docstrings and parameter names (defaults to `["asyncroscopy"]`).

### Adding Native MCP Tools, Resources, and Prompts
Beyond dynamic Tango commands, you can add native Python methods directly to the `MCPServer` instance using decorators. These methods are automatically registered during [`setup()`](../asyncroscopy/mcp/mcp_server.py#L531) via [`_register_instance_methods()`](../asyncroscopy/mcp/mcp_server.py#L150).

#### Native Tools
Use `@tool()` to define custom logic that requires arbitrary Python code.

```python
from fastmcp.tools import tool
from asyncroscopy.mcp.mcp_server import MCPServer

class MyCustomMCPServer(MCPServer):
    @tool()
    def custom_helper_tool(self, data: str) -> str:
        """This tool will be automatically registered alongside Tango commands."""
        return f"Processed: {data}"
```

#### Resources
Use `@resource()` to expose static or dynamic content (like configuration files or documentation) as data sources for LLMs.

```python
from fastmcp.resources import resource
from asyncroscopy.mcp.mcp_server import MCPServer

class MyCustomMCPServer(MCPServer):
    @resource("config://network")
    def get_network_config(self) -> str:
        """Expose current network configuration."""
        return "TANGO_HOST=localhost:9094"
```

#### Prompts
Use `@prompt()` to provide pre-defined templates that help LLMs structure their interactions with the hardware.

```python
from fastmcp.prompts import prompt
from asyncroscopy.mcp.mcp_server import MCPServer

class MyCustomMCPServer(MCPServer):
    @prompt()
    def optimize_beam_setup(self, voltage: float) -> str:
        """A prompt template for optimizing beam alignment."""
        return f"Please check the alignment for {voltage}kV setup and report any deviation."
```

---

## Data Transport & Encoding

Tango `DevEncoded` commands often return binary data (like images). The [`_normalize_command_result()`](../asyncroscopy/mcp/mcp_server.py#L264) method normalizes these into a standard JSON structure:

```json
{
  "encoding": "base64",
  "metadata": "header_string",
  "payload": "base64_encoded_binary_data"
}
```

---

## Running the Server

The server can be started as a standalone process. It requires a connection to a running Tango Database.

```python
from asyncroscopy.mcp.mcp_server import MCPServer

# Initialize and start the server
server = MCPServer(
    name="AsyncroscopyServer",
    tango_host="localhost",
    tango_port=9094
)

# Use server.start() for stdio (default) or server.start_http() for HTTP
server.start()
```

By default, [`start()`](../asyncroscopy/mcp/mcp_server.py#L603) uses `stdio` transport for piping to agents. To expose the server over HTTP, use [`start_http()`](../asyncroscopy/mcp/mcp_server.py#L599) (wraps `host="0.0.0.0"`, `port=8000`).
