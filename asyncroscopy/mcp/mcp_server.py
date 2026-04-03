"""
Bridge between a Tango control system and an MCP (Model Context Protocol) server.

This module queries a Tango database for exported devices, introspects their
commands, and dynamically registers each command as an MCP tool to make physical
hardware controllable via LLM agents.

Usage:
    server = MCPServer("MyServer", tango_host="localhost", tango_port=9094)
    server.start()   # discovers devices, registers tools, starts HTTP server
"""
import os
import inspect
import importlib
import pkgutil
import base64
from inspect import signature, getdoc
from typing import Any, Dict, Callable, Annotated
from pydantic import Field

from tango import Database, DeviceProxy, CommandInfo, CmdArgType
from tango.utils import (
    TO_TANGO_TYPE,
    is_array_type,
    is_scalar_type,
    is_bool_type,
    is_float_type,
    is_int_type,
    is_str_type,
)
from tango.server import Device

from fastmcp import FastMCP
from fastmcp.tools import tool, Tool
from fastmcp.resources import resource
from fastmcp.prompts import prompt
from fastmcp.tools.function_tool import ToolMeta
from fastmcp.resources.function_resource import ResourceMeta
from fastmcp.prompts.function_prompt import PromptMeta
from fastmcp.server.server import Transport


class MCPServer:
    DEFAULT_BLOCKED_CLASSES = ["DataBase", "DServer"]
    DEFAULT_BLOCKED_FUNCTIONS = ["Init"]

    def __init__(
        self,
        name: str,
        tango_host: str,
        tango_port: int,
        blocked_functions: list[str] | dict[str, list[str]] | None = None,
        blocked_classes: list[str] | None = None,
        search_packages: list[str] | None = None,
        verbose: bool = True,
    ):
        """
        Args:
            name (str): Display name for the MCP server instance.
            tango_host (str): Hostname of the Tango database server (e.g. "localhost").
            tango_port (int): Port of the Tango database server (e.g. 9094).
            blocked_functions (list[str] | dict[str, list[str]] | None, optional): 
                Command names to exclude. Can be a simple list for global blocks, 
                or a dictionary mapping Tango class names to command lists. 
                Use "*" as a dictionary key for global blocks. Defaults to None, 
                which applies the built-in block list: ["Init"].
            blocked_classes (list[str] | None, optional): Tango device class names to
                skip entirely. Defaults to None, which applies the built-in block list
                ["DataBase", "DServer"] (Tango infrastructure classes not useful as tools).
            search_packages (list[str] | None, optional): Python package names to search
                for Tango Device subclasses when resolving richer docstrings and parameter
                names. Defaults to None, which searches ["asyncroscopy"].
            verbose (bool, optional): If True, print device discovery and tool registration
                progress to stdout. Defaults to True.
        """
        self.database = Database(tango_host, tango_port)
        self.mcp = FastMCP(name)
        
        # Normalize to storage format: dict[class_name, list[command_name]]
        if blocked_functions is None:
            self.blocked_functions = {"*": self.DEFAULT_BLOCKED_FUNCTIONS.copy()}
        elif isinstance(blocked_functions, list):
            self.blocked_functions = {"*": blocked_functions}
        else:
            self.blocked_functions = blocked_functions
            
        self.blocked_classes = blocked_classes or self.DEFAULT_BLOCKED_CLASSES.copy()
        self._blocked_classes_normalized = {
            cls_name.lower() for cls_name in self.blocked_classes
        }

        self.search_packages = search_packages if search_packages is not None else ["asyncroscopy"]
        
        self.verbose = verbose

        # Tools are keyed by Tango class, then command name, with the value being the wrapped function
        self.tools: Dict[str, Dict[str, Callable]] = {}

    def _is_blocked_class(self, class_name: str) -> bool:
        """Return True when a Tango class should be filtered out."""
        return class_name.lower() in self._blocked_classes_normalized
        
    def _list_all_devices(self) -> list[str]:
        """List all devices exported in the Tango DB."""
        devices = self.database.get_device_exported("*")
        return list(devices.value_string)

    @staticmethod
    def _is_admin_device(device_name: str) -> bool:
        """Return True for Tango admin (dserver) devices."""
        return device_name.lower().startswith("dserver/")

    # @tool cannot register instance methods, but it still adds metadata
    @tool()
    def list_devices(self) -> list[str]:
        """List available devices filtered by blocked classes."""
        all_devices = self._list_all_devices()
        available = []
        for device_name in all_devices:
            if self._is_admin_device(device_name):
                continue
            try:
                # Create a DeviceProxy from the found name
                dev = DeviceProxy(device_name)
                dev_class = dev.info().dev_class
                if not self._is_blocked_class(dev_class):
                    available.append(device_name)
            except Exception:
                pass
        return available
    
    def get_blocked_functions(self) -> dict[str, list[str]]:
        """Get the list of blocked functions."""
        return self.blocked_functions

    def _is_blocked_function(self, dev_class: str, command_name: str) -> bool:
        """Check if a command is blocked."""
        # Global blocks apply to every class
        if command_name in self.blocked_functions.get("*", []):
            return True
        # Class-specific overrides
        if command_name in self.blocked_functions.get(dev_class, []):
            return True
        return False

    def get_blocked_classes(self) -> list[str]:
        """Get the list of blocked Tango classes."""
        return self.blocked_classes

    def _register_instance_methods(self) -> int:
        """Discover and register all methods decorated with @tool, @resource, or @prompt.
        
        Returns:
            Number of methods successfully registered.
        """
        registered_count = 0
        
        # Get all members from this instance's class (including inherited)
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        
        for name, method in methods:
            # Skip private/internal methods
            if name.startswith('_'):
                continue
            
            func = method.__func__
            
            # decorators attach a __fastmcp__ metadata object
            try:
                if hasattr(func, '__fastmcp__'):
                    meta = func.__fastmcp__
                    
                    if isinstance(meta, ToolMeta):
                        self.mcp.add_tool(method)
                        mcp_type = "tool"
                    elif isinstance(meta, ResourceMeta):
                        self.mcp.add_resource(method)
                        mcp_type = "resource"
                    elif isinstance(meta, PromptMeta):
                        self.mcp.add_prompt(method)
                        mcp_type = "prompt"
                    else:
                        if self.verbose:
                            print(f"Unknown MCP type for {name}")
                        continue
                        
                    registered_count += 1
                    if self.verbose:
                        print(f"Auto-registered {mcp_type}: {name}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to auto-register {name}: {e}")
        
        return registered_count

    @staticmethod
    def _is_dev_encoded_type(cmd_type: CmdArgType) -> bool:
        """Check if the command type is DevEncoded."""
        return cmd_type == CmdArgType.DevEncoded

    @staticmethod
    def _tango_scalar_to_python_type(cmd_type: CmdArgType) -> Any:
        """Map a Tango scalar CmdArgType to a Python type."""
        if not is_scalar_type(cmd_type):
            return None

        if is_bool_type(cmd_type):
            return bool
        if is_float_type(cmd_type):
            return float
        if is_int_type(cmd_type):
            return int
        if is_str_type(cmd_type):
            return str

        # Keep compatibility with less common or non-builtin mapped scalar types.
        candidates = [
            py_type
            for py_type, tango_type in TO_TANGO_TYPE.items()
            if tango_type == cmd_type and isinstance(py_type, type)
        ]
        if not candidates:
            return Any

        for py_type in candidates:
            if py_type.__module__ == "builtins":
                return py_type
        return candidates[0]

    @staticmethod
    def _tango_array_to_python_list(cmd_type: CmdArgType) -> Any:
        """Map a Tango array type to a typed Python list when possible."""
        if not is_array_type(cmd_type):
            return None

        if is_bool_type(cmd_type, inc_array=True):
            return list[bool]
        if is_float_type(cmd_type, inc_array=True):
            return list[float]
        if is_int_type(cmd_type, inc_array=True):
            return list[int]
        if is_str_type(cmd_type, inc_array=True):
            return list[str]
        return list
        
    @staticmethod
    def _tango_type_to_python(cmd_type: CmdArgType) -> Any:
        if cmd_type == CmdArgType.DevVoid:
            return type(None)
        if MCPServer._is_dev_encoded_type(cmd_type):
            return dict

        scalar_type = MCPServer._tango_scalar_to_python_type(cmd_type)
        if scalar_type is not None:
            return scalar_type

        typed_list = MCPServer._tango_array_to_python_list(cmd_type)
        if typed_list is not None:
            return typed_list

        return Any

    @staticmethod
    def _normalize_command_result(out_type: CmdArgType, result: Any) -> Any:
        """Convert Tango command output into JSON-safe data for MCP transport."""
        if not MCPServer._is_dev_encoded_type(out_type):
            return result

        if not isinstance(result, tuple) or len(result) != 2:
            return result

        metadata_raw, payload_raw = result
        if isinstance(metadata_raw, bytes):
            metadata = metadata_raw.decode("utf-8", errors="replace")
        else:
            metadata = str(metadata_raw)

        if isinstance(payload_raw, memoryview):
            payload_bytes = payload_raw.tobytes()
        elif isinstance(payload_raw, bytearray):
            payload_bytes = bytes(payload_raw)
        elif isinstance(payload_raw, bytes):
            payload_bytes = payload_raw
        else:
            payload_bytes = str(payload_raw).encode("utf-8")

        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        return {
            "encoding": "base64",
            "metadata": metadata,
            "payload": payload_b64,
        }

    def _get_tango_device_class(self, dev_class: str) -> type[Device] | None:
        """Find or import the Tango Device class for a given class name."""
        # Existing loaded subclasses
        for cls in Device.__subclasses__():
            if cls.__name__ == dev_class:
                return cls

        # Try direct import
        try:
            mod = importlib.import_module(dev_class)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, Device) and cls.__name__ == dev_class:
                    return cls
        except Exception:
            pass

        # Search packages
        for pkg_name in self.search_packages or []:
            try:
                pkg = importlib.import_module(pkg_name)
                if not hasattr(pkg, "__path__"):
                    continue

                for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                    try:
                        mod = importlib.import_module(modname)
                        for _, cls in inspect.getmembers(mod, inspect.isclass):
                            if issubclass(cls, Device) and cls.__name__ == dev_class:
                                return cls
                    except Exception:
                        continue
            except ImportError:
                continue

        return None

    def _get_docstring(self, dev_class: str, command_name: str) -> str | None:
        cls = self._get_tango_device_class(dev_class)
        if not cls:
            return None
        func = getattr(cls, command_name, None)
        return inspect.getdoc(func) if func else None

    def _build_command_docstring(
        self,
        func: Callable,
        cmd_info: CommandInfo,
        command_name: str,
        dev_class: str,
    ) -> str:
        """Build a tool description combining source docstrings with Tango metadata."""
        # Preference: Actual source docstring, then proxy docstring, then command name
        header_doc = self._get_docstring(dev_class, command_name) or getdoc(func)

        lines = []
        if header_doc:
            lines.append(header_doc)
            lines.append("")
        
        lines.append(f"Tango Device Class: {dev_class}")
        lines.append(f"Tango Command: {command_name}")

        if not header_doc:
            in_type = cmd_info.in_type
            out_type = cmd_info.out_type
            in_desc = cmd_info.in_type_desc
            out_desc = cmd_info.out_type_desc

            lines.append(f"Input Type: {self.in_type.name}")
            if in_desc:
                lines.append(f"Input Description: {in_desc}")

            lines.append(f"Output Type: {self.out_type.name}")
            if out_desc:
                lines.append(f"Output Description: {out_desc}")

        return "\n".join(lines).strip()
    
    def _get_param_name(self, dev_class: str, command_name: str) -> str:
        """Pull the first non-self parameter name from the source method signature."""
        cls = self._get_tango_device_class(dev_class)
        if not cls:
            return "arg"

        method = getattr(cls, command_name, None)
        if method is None:
            return "arg"

        try:
            params = list(inspect.signature(method).parameters.values())
            # Skip 'self' if present
            for p in params:
                if p.name != "self":
                    return p.name
        except (ValueError, TypeError):
            pass

        return "arg"

    def _create_wrapper(
        self,
        func: Callable,
        cmd_info: CommandInfo,
        command_name: str,
        dev_class: str,
    ) -> Callable:
        """Create a wrapper function with a proper signature for a Tango command.
        
        Args:
            func: The raw Tango device command method
            cmd_info: The CommandInfo object from Tango
            command_name: The name of the command
            dev_class: The Tango device class name
            
        Returns:
            A wrapper function with a proper signature
        """
        doc = self._build_command_docstring(
            func=func,
            cmd_info=cmd_info,
            command_name=command_name,
            dev_class=dev_class,
        )

        in_type = cmd_info.in_type
        py_type = self._tango_type_to_python(in_type)
        in_desc = cmd_info.in_type_desc
        
        out_type = cmd_info.out_type
        py_return_type = self._tango_type_to_python(out_type)

        if in_desc and in_desc.lower() not in ("uninitialised", "none", "", "uninitialized"):
            arg_type = Annotated[py_type, Field(description=in_desc)]
        else:
            arg_type = py_type

        if in_type == CmdArgType.DevVoid:
            def wrapper():
                result = func()
                return self._normalize_command_result(out_type, result)
            wrapper.__annotations__ = {"return": py_return_type}
        else:
            param_name = self._get_param_name(dev_class, command_name)

            # Build the wrapper with the real param name so pydantic/FastMCP
            # advertises the correct keyword in the tool schema.
            ns: dict = {
                "func": func,
                "arg_type": arg_type,
                "py_return_type": py_return_type,
                "self": self,
                "out_type": out_type,
            }
            # FastMCP inspects the actual parameter *name* in the function signature
            # to build its JSON schema (e.g. "exposure_time" not generic "arg").
            # Python's exec() is the only way to set a runtime-determined param name
            # on a function — functools.wraps and __wrapped__ don't affect introspection.
            exec(
                f"def wrapper({param_name}: arg_type) -> py_return_type:\n"
                f"    result = func({param_name})\n"
                f"    return self._normalize_command_result(out_type, result)\n",
                ns,
            )
            wrapper = ns["wrapper"]
            wrapper.__annotations__ = {param_name: arg_type, "return": py_return_type}

        wrapper.__doc__ = doc
        
        # Set unique function name for FastMCP tool registration
        unique_name = f"{dev_class}_{command_name}".replace("/", "_").replace("-", "_")
        wrapper.__name__ = unique_name
        wrapper.__qualname__ = unique_name
        
        return wrapper
    
    def _find_tools(self) -> Dict[str, Dict[str, tuple[Callable, CommandInfo]]]:
        """Discover tools by querying Tango DB for devices and their commands.
        
        Returns a dict mapping dev_class -> command_name -> (func, cmd_info)
        """
        devices = self._list_all_devices()
        tools: Dict[str, Dict[str, tuple[Callable, CommandInfo]]] = {}
        for device_name in devices:
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                info = dev.info()
                dev_class = info.dev_class
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to open proxy ({exc})")
                continue

            if self._is_blocked_class(dev_class):
                continue

            try:
                commands = dev.command_list_query()
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to query commands ({exc})")
                continue

            for cmd in commands:
                command_name = cmd.cmd_name if hasattr(cmd, "cmd_name") else str(cmd)
                if self._is_blocked_function(dev_class, command_name):
                    continue
                try:
                    func = getattr(dev, command_name)
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"Skipping {device_name}.{command_name}: "
                            f"failed to resolve command ({exc})"
                        )
                    continue
                if dev_class not in tools:
                    tools[dev_class] = {}
                tools[dev_class][command_name] = (func, cmd)
        return tools

    def _print_discovered_tools(self, tools: Dict[str, Dict[str, tuple[Callable, CommandInfo]]]) -> None:
        """Print discovered tools before registration.
        
        Args:
            tools: Dictionary of discovered tools by class and command name.
        """
        if not self.verbose:
            return
        print("Discovered tools by Tango class:")
        for dev_class in sorted(tools):
            command_names = sorted(tools[dev_class].keys())
            print(f"- {dev_class}: {len(command_names)}")
            for command_name in command_names:
                print(f"    • {command_name}")

    def setup(self, print_summary: bool = True):
        """Configure tools and add them to the MCP instance.
        
        Args:
            print_summary: If True, print tool discovery and registration summary.
        """
        raw_tools = self._find_tools()
        
        # Convert to final wrapped tools
        wrapped_tools: Dict[str, Dict[str, Callable]] = {}
        for dev_class in raw_tools:
            wrapped_tools[dev_class] = {}
            for command_name, (func, cmd_info) in raw_tools[dev_class].items():
                wrapped = self._create_wrapper(func, cmd_info, command_name, dev_class)
                wrapped_tools[dev_class][command_name] = wrapped
        
        self.tools = wrapped_tools
        if print_summary:
            self._print_discovered_tools(raw_tools)

        # Auto-register all @tool, @resource, and @prompt decorated instance methods
        num_instance_tools = self._register_instance_methods()
        
        # Register wrapped tools with MCP
        num_device_tools = 0
        for dev_class in wrapped_tools:
            for command_name, wrapped_func in wrapped_tools[dev_class].items():
                try:
                    tool_obj = Tool.from_function(wrapped_func)
                    self.mcp.add_tool(tool_obj)
                    num_device_tools += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {dev_class}.{command_name}: {e}")
        
        # Print all registered MCP tools
        if print_summary:
            self._print_registration_summary(num_device_tools, num_instance_tools)
    
    def _print_registration_summary(self, num_device_tools: int, num_instance_tools: int) -> None:
        """Print all registered MCP tools.
        
        Args:
            num_device_tools: Number of Tango device command tools registered
            num_instance_tools: Number of instance method tools registered
        """
        if not self.verbose:
            return
        
        print(f"\nRegistered {num_instance_tools} instance method tool(s)")
        print(f"Registered {num_device_tools} Tango device command tool(s)")
        print(f"Total: {num_instance_tools + num_device_tools} tools")
        print("\nAll MCP tools available:")
        
        for dev_class in sorted(self.tools.keys()):
            command_names = sorted(self.tools[dev_class].keys())
            for command_name in command_names:
                wrapped_func = self.tools[dev_class][command_name]
                sig = signature(wrapped_func)
                print(f"  • {dev_class}.{command_name}{sig}")
                if wrapped_func.__doc__:
                    for line in wrapped_func.__doc__.split('\n'):
                        stripped = line.strip()
                        if stripped:
                            print(f"{stripped}")
                print("")
            print("")

    def start_http(self, host: str = "127.0.0.1", port: int = 8000):
        """Exposes MCP tools via HTTP for cross-process or remote agent access."""
        self.start(transport="streamable-http", host=host, port=port)

    def start(self, transport: Transport | None = None, **kwargs):
        """
        Synchronizes with Tango DB and begins serving the MCP protocol.
        
        Args:
            transport: Transport protocol to use ("stdio", "http", "sse", or "streamable-http").
                       Defaults to None, which uses stdio for local piping to agents.
            **kwargs: Additional keyword arguments to pass to the MCP server
        """
        self.setup()
        self.mcp.run(transport=transport, **kwargs)

if __name__ == "__main__":
    tango_host = os.environ.get("TANGO_HOST", "localhost:9094")
    if ":" not in tango_host:
        raise SystemExit(f"Invalid TANGO_HOST value: {tango_host}. Expected host:port")
    host, port_str = tango_host.rsplit(":", maxsplit=1)
    port = int(port_str)

    server = MCPServer(name="MCPServer", tango_host=host, tango_port=port)
    print(f"Connected to Tango DB at {host}:{port}")
    print("Exported devices:", server.list_devices())
    server.start()