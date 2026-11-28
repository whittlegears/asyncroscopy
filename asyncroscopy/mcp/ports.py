"""Port check shared by the MCP servers; imports nothing from Tango."""

import socket


def check_port_free(host: str, port: int) -> str | None:
    """Return an error string if (host, port) cannot be bound, else None."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        return str(exc)
    finally:
        probe.close()
    return None


def report_busy_port(host: str, port: int, server_kind: str) -> bool:
    """Print the "MCP ERROR:" line the startup GUIs parse and return True if the port is taken."""
    bind_error = check_port_free(host, port)
    if bind_error:
        print(
            f"MCP ERROR: http://{host}:{port} is already in use - "
            f"another {server_kind} is likely still running with a different config. "
            f"Stop it or choose a different port. ({bind_error})",
            flush=True,
        )
    return bool(bind_error)
