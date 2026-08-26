"""Abstract agent backend contract shared by all backend implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Agent:
    """Represents a single AI agent in the swarm."""
    name: str
    system_prompt: str
    tools: list[str]
    model: str | None = None
    description: str = ""


@dataclass
class BackendConfig:
    """Snapshot of the LLM device's properties handed to a backend at creation."""
    model_name: str = ""
    model_provider: str = "ollama"
    ollama_model: str = ""
    ollama_host: str = "http://localhost:11434"
    local_model_path: str = ""
    api_key: str = ""
    use_init_chat_model: bool = False
    hermes_url: str = ""
    hermes_model: str = "hermes-agent"
    hermes_api_key: str = ""
    reflection_min_tool_steps: int = 4


class BackendUnsupported(RuntimeError):
    """Raised when a backend is asked for an operation it does not provide."""


class AgentBackend(ABC):
    """One agent framework behind the LLM Tango device.

    The device owns Tango plumbing (properties, state, commands); a backend owns
    the model and the agent loop. The ``agents`` list is the device's own mutable
    list, so SpawnAgent on the device is visible to the backend without any
    extra API surface.
    """

    name: str = ""
    supports_complete: bool = False
    supports_connect_mcp: bool = False

    def __init__(self, config: BackendConfig, agents: list[Agent]):
        self.config = config
        self.agents = agents
        self._skills_service = None

    def set_skills_service(self, service) -> None:
        """Receive the device's skill store handle; backends override to grow tools from it."""
        self._skills_service = service

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the backend (model init, reachability checks). Raise on failure."""

    @abstractmethod
    async def query(self, prompt: str, max_steps: int) -> str:
        """Run a full agent turn on the prompt and return the final answer."""

    async def complete(self, request: dict) -> dict:
        """Run one raw model step on an OpenAI-style request, returning the assistant message."""
        raise BackendUnsupported(
            f"The {self.name} backend does not support single-step completion."
        )

    async def connect_mcp(self, url: str, transport: str) -> int:
        """Connect to an MCP server and inherit its tools, returning the tool count."""
        raise BackendUnsupported(
            f"The {self.name} backend does not connect to MCP servers from the device."
        )

    def tool_names(self) -> list[str]:
        return []

    def capabilities(self) -> dict:
        return {
            "complete": self.supports_complete,
            "connect_mcp": self.supports_connect_mcp,
        }
