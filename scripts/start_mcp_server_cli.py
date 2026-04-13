#!/usr/bin/env python
"""Interactive CLI to start MCP server connected to a chosen Tango DB."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.mcp.mcp_server import MCPServer


def prompt_host(default: str = "127.0.0.1") -> str:
    print(f"Enter Tango DB host [{default}]: ", end="", flush=True)
    value = input().strip()
    return value or default

def prompt_port(default: int = 8000) -> int:
    while True:
        print(f"Enter Tango DB port [{default}]: ", end="", flush=True)
        raw = input().strip()
        if raw == "":
            return default

        try:
            port = int(raw)
        except ValueError:
            print("Invalid port: must be an integer.")
            continue

        if 1 <= port <= 65535:
            return port

        print("Invalid port: must be between 1 and 65535.")


def main() -> None:
    tango_db_host = prompt_host(default="127.0.0.1")
    tango_db_port = prompt_port(default=9094)
    os.environ["TANGO_HOST"] = f"{tango_db_host}:{tango_db_port}"

    server = MCPServer(name="MCPServer", tango_host=tango_db_host, tango_port=tango_db_port)
    print(f"Connected to Tango DB at {tango_db_host}:{tango_db_port}")
    print("Starting MCP server on 127.0.0.1:8000")
    print("Exported devices:", server.list_devices())
    server.start_http(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
