import os

from fastmcp import FastMCP
from tango import Database
from fastmcp.tools import tool

class MCPServer:
    def __init__(self, name: str, tango_host: str, tango_port: int, blocked_functions: list[str] = []):
        self.database = Database(tango_host, tango_port)
        self.mcp = FastMCP(name)
        self.blocked_functions = blocked_functions

    @tool
    def list_devices(self) -> list[str]:
        devices = self.database.get_device_exported("*")
        return list(devices.value_string)
    
    @tool
    def get_blocked_functions(self) -> list[str]:
        return self.blocked_functions
    
    def find_tools(self) -> list:
        pass

    def start(self):
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