"""Settings for the agent layer.

Everything here is stdlib (plus PyYAML, a core dependency) so it can be imported
by launchers and tests without the ``agent`` extra installed.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_DIR = PROJECT_DIR / "skills"
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
ENV_PREFIX = "ASYNCROSCOPY_AGENT_"

# Conventional API-key environment variables per init_chat_model provider.
DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Providers that need no API key.
KEYLESS_PROVIDERS = frozenset({"ollama", "tango"})


@dataclass
class Agent:
    """A single worker agent in the supervisor swarm.

    ``tools`` is a list of glob patterns matched against MCP tool names
    (e.g. ``["*_acquire_scanned_image", "get_data_from_key"]`` or ``["*"]``).
    """

    name: str
    system_prompt: str
    tools: list[str]
    model: str | None = None
    description: str = ""


@dataclass
class AgentSettings:
    """How to build a chat model and where to find tools and skills.

    ``provider`` is one of:

    - ``"ollama"``: local model through ``langchain_ollama.ChatOllama``.
    - ``"tango"``: the asyncroscopy LLM Tango device; ``model`` is then the
      device name (e.g. ``asyncroscopy/llm/default``). Inference runs on the
      LLM computer while the graph runs locally.
    - any provider accepted by ``langchain.chat_models.init_chat_model``
      (``"openai"``, ``"anthropic"``, ``"google_genai"``, ...). The API key is
      read from the environment variable named by ``api_key_env`` (or the
      provider's conventional variable); it is never stored in settings.
    """

    provider: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    api_key_env: str | None = None
    base_url: str | None = None
    mcp_url: str | None = DEFAULT_MCP_URL
    mcp_transport: str = "streamable_http"
    skills_dirs: list[str] = field(default_factory=lambda: [str(DEFAULT_SKILLS_DIR)])
    temperature: float = 0.0
    max_steps: int = 10
    use_init_chat_model: bool = False

    def __post_init__(self) -> None:
        self.provider = (self.provider or "ollama").strip().lower()
        if isinstance(self.skills_dirs, str):
            self.skills_dirs = _split_paths(self.skills_dirs)
        self.skills_dirs = [str(p) for p in self.skills_dirs]
        self.max_steps = int(self.max_steps)
        self.temperature = float(self.temperature)

    @property
    def resolved_skills_dirs(self) -> list[Path]:
        """Skills directories as absolute paths (relative ones resolve against the repo)."""
        resolved = []
        for entry in self.skills_dirs:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = PROJECT_DIR / path
            resolved.append(path)
        return resolved

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AgentSettings":
        """Build settings from a dict (e.g. YAML), ignoring unknown keys.

        Accepts the legacy LLM-device spellings ``model_provider``,
        ``model_name`` and ``ollama_model``. A literal ``api_key`` is refused:
        keys belong in the environment or ``.env``.
        """
        if "api_key" in raw and raw["api_key"]:
            raise ValueError(
                "'api_key' must not be stored in configuration files. "
                "Put it in the environment or a .env file and set 'api_key_env' to its name."
            )
        data: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key in known:
                data[key] = value
        if "provider" not in data and raw.get("model_provider"):
            data["provider"] = raw["model_provider"]
        if "model" not in data:
            if raw.get("model_name"):
                data["model"] = raw["model_name"]
            elif raw.get("ollama_model"):
                data["model"] = raw["ollama_model"]
        unknown = set(raw) - known - {"model_provider", "model_name", "ollama_model", "api_key", "tango", "startup_agents"}
        if unknown:
            warnings.warn(f"Ignoring unknown agent settings: {sorted(unknown)}", stacklevel=2)
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = ENV_PREFIX, environ: Mapping[str, str] | None = None) -> "AgentSettings":
        """Build settings from ``ASYNCROSCOPY_AGENT_*`` environment variables.

        Recognised suffixes: PROVIDER, MODEL, API_KEY_ENV, BASE_URL, MCP_URL,
        MCP_TRANSPORT, SKILLS_DIRS (os.pathsep or comma separated), TEMPERATURE,
        MAX_STEPS, USE_INIT_CHAT_MODEL.
        """
        env = os.environ if environ is None else environ
        data: dict[str, Any] = {}
        for f in fields(cls):
            value = env.get(prefix + f.name.upper())
            if value is None or value == "":
                continue
            if f.name == "skills_dirs":
                data[f.name] = _split_paths(value)
            elif f.name == "use_init_chat_model":
                data[f.name] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                data[f.name] = value
        return cls(**data)

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "AgentSettings":
        """Return a copy with non-empty values from ``overrides`` applied."""
        if not overrides:
            return self
        known = {f.name for f in fields(self)}
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in overrides.items():
            if key in known and value not in (None, ""):
                data[key] = value
        return AgentSettings(**data)


def _split_paths(value: str) -> list[str]:
    separators = [os.pathsep, ","]
    parts = [value]
    for sep in separators:
        parts = [piece for part in parts for piece in part.split(sep)]
    return [p.strip() for p in parts if p.strip()]


def load_dotenv_if_present(start: Path | None = None) -> bool:
    """Load a ``.env`` file into ``os.environ`` if python-dotenv is installed.

    Searches ``start`` (default: the repository root) and its parents. Existing
    environment variables win over the file. Returns True when a file was loaded.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return False

    candidates: Iterable[Path] = [start] if start is not None else [Path.cwd(), PROJECT_DIR]
    for base in candidates:
        base = Path(base)
        if base.is_file():
            return bool(load_dotenv(base, override=False))
        for parent in [base, *base.parents]:
            candidate = parent / ".env"
            if candidate.is_file():
                return bool(load_dotenv(candidate, override=False))
    found = find_dotenv(usecwd=True)
    return bool(found and load_dotenv(found, override=False))


def default_api_key_env(provider: str) -> str | None:
    """Conventional API-key variable for a provider, or None if it needs none."""
    provider = (provider or "").lower()
    if provider in KEYLESS_PROVIDERS:
        return None
    return DEFAULT_API_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")


def resolve_api_key(settings: AgentSettings, environ: Mapping[str, str] | None = None) -> str | None:
    """Read the API key named by ``settings.api_key_env`` (or the provider default)."""
    env = os.environ if environ is None else environ
    name = settings.api_key_env or default_api_key_env(settings.provider)
    if not name:
        return None
    value = env.get(name)
    return value or None
