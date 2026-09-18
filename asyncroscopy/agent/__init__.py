"""LangGraph-based agent layer for asyncroscopy.

The package is split so that the LLM Tango device (``asyncroscopy.mcp.llm``),
the example notebook, and LangGraph Studio (``langgraph.json``) all share the
same graphs:

- ``config``      settings dataclasses and ``.env`` loading (stdlib only)
- ``models``      chat-model factory: Ollama, API-key providers, or the LLM Tango device
- ``tools``       MCP tool loading and glob filtering
- ``graphs``      ReAct agent, supervisor swarm, and deterministic workflows
- ``skills``      Hermes-style ``SKILL.md`` registry with full-text search
- ``studio``      graph factories referenced by ``langgraph.json``

Heavy dependencies (``langgraph``, ``langchain``) are imported lazily inside the
modules that need them so the stdlib parts stay importable without the
``agent`` extra installed.
"""
