__all__ = ["MCPServer", "AgentMCPServer"]

def __getattr__(name):
    if name == "MCPServer":
        from .mcp_server import MCPServer
        return MCPServer
    if name == "AgentMCPServer":
        from .agent_mcp_server import AgentMCPServer
        return AgentMCPServer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")