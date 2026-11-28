"""Run by tests/endpoints/test_stack_fingerprint.py in a fresh interpreter: an MCP server discovers a real Tango database and prints its stack_fingerprint."""

import json
import sys

from asyncroscopy.mcp.mcp_server import MCPServer

host, port, permissions = sys.argv[1], int(sys.argv[2]), sys.argv[3]
server = MCPServer("fingerprint", host, port, needs_approval={}, blocked_classes=["DataBase", "DServer"], data_device_address="",
                   verbose=False, permissions_file=permissions)
server.setup(print_summary=False)
print(json.dumps(server.discovery.endpoint_info()["stack_fingerprint"]))
