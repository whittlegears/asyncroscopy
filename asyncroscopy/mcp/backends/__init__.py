"""Pluggable agent backends for the LLM Tango device."""

from .base import Agent, AgentBackend, BackendConfig, BackendUnsupported

__all__ = ["Agent", "AgentBackend", "BackendConfig", "BackendUnsupported", "create_backend"]

VALID_BACKENDS = ("langgraph", "hermes")


def create_backend(name: str, config: BackendConfig, agents: list[Agent]) -> AgentBackend:
    """Instantiate the named backend, importing its module (and heavy deps) lazily."""
    normalized = (name or "langgraph").strip().lower()
    if normalized == "langgraph":
        try:
            from .langgraph_backend import LangGraphBackend
        except ImportError as exc:
            raise RuntimeError(
                "The langgraph backend needs the agent dependencies. Run: uv sync --extra agent"
            ) from exc
        return LangGraphBackend(config, agents)
    if normalized == "hermes":
        from .hermes_backend import HermesBackend

        return HermesBackend(config, agents)
    raise RuntimeError(
        f"Unknown agent_backend '{name}'. Valid backends: {', '.join(VALID_BACKENDS)}"
    )
