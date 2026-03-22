import os
from inspect import signature, getdoc
from typing import Any, Dict, Callable

from fastmcp import FastMCP
from tango import Database, DeviceProxy, CommandInfo, CmdArgType
from fastmcp.tools import tool, Tool

DEFAULT_BLOCKED_CLASSES = ["DataBase", "DServer"]

class MCPServer:
    def __init__(
        self,
        name: str,
        tango_host: str,
        tango_port: int,
        blocked_functions: list[str] | None = None,
        blocked_classes: list[str] | None = None,
        verbose: bool = True,
    ):
        self.database = Database(tango_host, tango_port)
        self.mcp = FastMCP(name)
        self.blocked_functions = blocked_functions or []
        self.blocked_classes = blocked_classes or DEFAULT_BLOCKED_CLASSES.copy()
        self.verbose = verbose
        self._blocked_classes_normalized = {
            cls_name.lower() for cls_name in self.blocked_classes
        }

        # Tools are keyed by Tango class, then command name, with the value being the wrapped function
        self.tools: Dict[str, Dict[str, Callable]] = {}

    def _is_blocked_class(self, class_name: str) -> bool:
        """Return True when a Tango class should be filtered out."""
        return class_name.lower() in self._blocked_classes_normalized
        
    def _list_all_devices(self) -> list[str]:
        """List all devices exported in the Tango DB."""
        devices = self.database.get_device_exported("*")
        return list(devices.value_string)
    
    @tool()
    def list_devices(self) -> list[str]:
        """List available devices filtered by blocked classes."""
        all_devices = self._list_all_devices()
        available = []
        for device_name in all_devices:
            try:
                # Create a DeviceProxy from the found name
                dev = DeviceProxy(device_name)
                dev_class = dev.info().dev_class
                if not self._is_blocked_class(dev_class):
                    available.append(device_name)
            except Exception:
                pass
        return available
    
    def get_blocked_functions(self) -> list[str]:
        """Get the list of blocked functions."""
        return self.blocked_functions

    def get_blocked_classes(self) -> list[str]:
        """Get the list of blocked Tango classes."""
        return self.blocked_classes

    @staticmethod
    def _cmd_type_name(cmd_type: CmdArgType) -> str:
        """Return a readable Tango command type name."""
        return cmd_type.name

    def _build_command_docstring(
        self,
        func: Callable,
        cmd_info: CommandInfo,
        command_name: str,
        dev_class: str,
    ) -> str:
        """Build a tool description from function doc or Tango metadata."""
        func_doc = getdoc(func)
        if func_doc:
            return func_doc

        in_type = cmd_info.in_type
        out_type = cmd_info.out_type
        in_desc = cmd_info.in_type_desc
        out_desc = cmd_info.out_type_desc

        lines = [
            f"{dev_class}.{command_name}",
            "",
            f"Input: {self._cmd_type_name(in_type)}",
        ]
        if in_desc:
            lines.append(f"Input description: {in_desc}")

        lines.append(f"Output: {self._cmd_type_name(out_type)}")
        if out_desc:
            lines.append(f"Output description: {out_desc}")

        return "\n".join(lines)
    
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

        if in_type == CmdArgType.DevVoid:
            def wrapper() -> Any:
                return func()
        else:
            def wrapper(arg: Any) -> Any:
                return func(arg)

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
                if command_name in self.blocked_functions:
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

    def _print_tools(self, tools: Dict[str, Dict[str, tuple[Callable, CommandInfo]]]) -> None:
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
            self._print_tools(raw_tools)

        # Register wrapped tools with MCP
        num_tools = 0
        for dev_class in wrapped_tools:
            for command_name, wrapped_func in wrapped_tools[dev_class].items():
                try:
                    tool_obj = Tool.from_function(wrapped_func)
                    self.mcp.add_tool(tool_obj)
                    num_tools += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {dev_class}.{command_name}: {e}")
        
        # Print all registered MCP tools
        if print_summary:
            self._print_mcp_tools(num_tools)
    
    def _print_mcp_tools(self, num_registered: int) -> None:
        """Print all registered MCP tools."""
        if not self.verbose:
            return
        print(f"\nRegistered {num_registered} Tango device tools")
        print("All MCP tools available:")
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

    def start(self):
        """Setup and start the MCP server."""
        self.setup()
        self.mcp.run()

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