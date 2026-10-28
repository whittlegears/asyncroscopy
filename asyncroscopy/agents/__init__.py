"""Code-defined LangGraph agents for asyncroscopy.

``graphs.mapping`` is a fixed workflow (plan -> reserve -> approval -> run ->
stitch -> score -> report) where the language model only writes the plan
and the report; ``graphs.general`` is a tool-using agent that loads skills on
demand. Both are listed in langgraph.json so LangGraph Studio can run them.

Configuration comes from environment variables so Studio, scripts and the
agent MCP server share it:

=========================== ==================================== ==========================
Variable                    Meaning                              Default
=========================== ==================================== ==========================
ASYNCROSCOPY_MODEL          Ollama tag or provider model name    qwen3:8b
ASYNCROSCOPY_MODEL_PROVIDER ollama, or any init_chat_model one   ollama
ASYNCROSCOPY_NUM_CTX        Context window for Ollama            8192
ASYNCROSCOPY_MCP_URL        asyncroscopy agent MCP endpoint      http://127.0.0.1:8000/mcp
ASYNCROSCOPY_ANALYSIS_URL   pyTEMlib MCP server ("" to skip)     http://127.0.0.1:8003/mcp
ASYNCROSCOPY_CALLER         name recorded in the trace           agent
=========================== ==================================== ==========================
"""

from .config import AgentConfig

__all__ = ["AgentConfig"]
