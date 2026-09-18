"""Register and launch the LLM Tango device (asyncroscopy/llm/default).

Usage:
    uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml [--interactive]

API keys are never read from YAML: set ``api_key_env`` to the name of an
environment variable (for example ``OPENAI_API_KEY``) and put the key in your
shell environment or in a ``.env`` file at the repository root.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tango
import yaml
from tango import DeviceProxy

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.agent.config import Agent, load_dotenv_if_present  # noqa: E402  (stdlib-only module)
from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402

DEVICE_NAME = "asyncroscopy/llm/default"
INSTANCE_NAME = "llm_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'

# Device properties from older configs that must not linger in the Tango database.
STALE_PROPERTIES = ["api_key", "local_model_path"]


@dataclass
class TangoConfig:
    host: str
    port: int


@dataclass
class LLMConfig:
    tango: TangoConfig
    mcp_url: str = "http://127.0.0.1:8000/mcp"
    model_provider: str = "ollama"
    model_name: str = "gemma4:31b"
    api_key_env: str | None = None
    base_url: str | None = None
    skills_dirs: list[str] = field(default_factory=lambda: ["skills"])
    max_steps: int = 10
    temperature: float = 0.0
    use_init_chat_model: bool = False
    startup_agents: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.tango, dict):
            self.tango = TangoConfig(**self.tango)
        if isinstance(self.skills_dirs, str):
            self.skills_dirs = [self.skills_dirs]
        self.model_provider = (self.model_provider or "ollama").strip().lower()
        if self.model_provider == "tango":
            raise ValueError("model_provider 'tango' is only valid for clients, not for the LLM device itself.")

    def device_properties(self) -> dict:
        """Tango device properties to register (explicit allow-list, no secrets)."""
        properties = {
            "mcp_url": self.mcp_url,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "ollama_model": self.model_name,  # legacy property still read by older device builds
            "api_key_env": self.api_key_env or "",
            "base_url": self.base_url or "",
            "skills_dirs": [str(PROJECT_DIR / d) if not Path(d).is_absolute() else d for d in self.skills_dirs],
            "temperature": self.temperature,
            "use_init_chat_model": self.use_init_chat_model,
            "startup_agents": [json.dumps(agent) for agent in self.startup_agents],
        }
        return properties


def _require(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Config section '{where}' is missing required key '{key}'")
    return mapping[key]


def load_config(path: Path) -> LLMConfig:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if raw.get("api_key"):
        raise ValueError(
            "'api_key' must not be stored in the YAML config. Put the key in the environment or in .env "
            "and set 'api_key_env' to the variable name (e.g. OPENAI_API_KEY)."
        )
    raw.pop("api_key", None)
    raw.pop("local_model_path", None)  # legacy key from the transformers-based device
    _require(raw, "tango", "root")
    for agent in raw.get("startup_agents") or []:
        Agent(**agent)  # validates the shape early
    known = set(LLMConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        print(f"[WARNING]: Ignoring unknown config keys: {sorted(unknown)}")
    return LLMConfig(**{k: v for k, v in raw.items() if k in known})


def ensure_database_running(config: LLMConfig) -> tango.Database:
    host = config.tango.host
    port = config.tango.port
    tango_host = f"{host}:{port}"

    try:
        database = tango.Database(host, port)
        database.get_class_list("*")
        return database
    except tango.DevFailed:
        pass

    print(f"[SYSTEM]: Tango database not responding at {tango_host}. Launching database server...")
    env = {**os.environ, 'TANGO_HOST': tango_host}
    subprocess.Popen(
        ["uv", "run", "python", "-m", "tango.databaseds.database", "2"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            database = tango.Database(host, port)
            database.get_class_list("*")
            print(f"[SYSTEM]: Tango database at {tango_host} is now ready!")
            return database
        except tango.DevFailed:
            time.sleep(1)

    raise RuntimeError(f"Could not connect to Tango database at {tango_host}.")


def register_device(config: LLMConfig):
    database = ensure_database_running(config)
    try:
        device_info = tango.DbDevInfo()
        device_info.server = f"LLM/{INSTANCE_NAME}"
        device_info._class = "LLM"
        device_info.name = DEVICE_NAME
        database.add_device(device_info)
        print(f"Registered device: {DEVICE_NAME}")
    except tango.DevFailed as e:
        print(f"Device already registered or error: {e}")

    try:
        database.delete_device_property(DEVICE_NAME, STALE_PROPERTIES)
    except tango.DevFailed:
        pass

    properties = config.device_properties()
    database.put_device_property(DEVICE_NAME, properties)
    print(f"Set device properties: {properties}")


def build_command(config: LLMConfig) -> list[str]:
    command = ["uv", "run", "--extra", "agent"]
    if config.model_provider == "ollama":
        command += ["--extra", "ollama"]
    command += ["python", "-m", "asyncroscopy.mcp.llm", INSTANCE_NAME]
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='LLM YAML config to start from.')
    parser.add_argument('--interactive', action='store_true', default=False, help='Run in interactive mode, allowing user to send prompts to the LLM device.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1

    # Load .env so the child process inherits API keys named by api_key_env.
    load_dotenv_if_present(PROJECT_DIR)
    if config.api_key_env and not os.environ.get(config.api_key_env):
        print(f"[WARNING]: {config.api_key_env} is not set; provider '{config.model_provider}' will fail to initialise.")

    tango_host = f'{config.tango.host}:{config.tango.port}'
    os.environ['TANGO_HOST'] = tango_host

    register_device(config)

    command = build_command(config)
    env = {**os.environ, 'TANGO_HOST': tango_host, 'PYTHONUNBUFFERED': '1'}

    try:
        with ProcessManager() as manager:
            managed: ManagedProcess = manager.start_process(
                key="llm",
                label="LLM Server",
                command=command,
                env=env,
                stdout=None,
                stderr=None,
            )

            print("Waiting for LLM device to start and initialize...")

            proxy = None
            max_wait_seconds = 120

            for _ in range(max_wait_seconds):
                try:
                    if proxy is None:
                        proxy = DeviceProxy(DEVICE_NAME)
                        proxy.ping()

                    state = proxy.state()
                    if state == tango.DevState.ON:
                        print("Device initialized and ready.")
                        break
                    elif state == tango.DevState.FAULT:
                        print(f"Device initialization failed. Status: {proxy.status()}")
                        return 1
                except Exception:
                    proxy = None

                time.sleep(1)
            else:
                print("Timeout waiting for device to initialize.")
                return 1

            if args.interactive:
                print("Entering interactive mode. Type 'exit' to quit.")
                while True:
                    prompt = input("LLM Prompt (or 'exit'): ")
                    if prompt.lower() == 'exit':
                        break

                    try:
                        response = proxy.Query(prompt)
                        print(f"Response: {response}")
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                print("Press Ctrl+C to terminate.")
                # Loop with timeout so Windows handles SIGINT / Ctrl+C cleanly
                while managed.running:
                    try:
                        managed.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass

    except KeyboardInterrupt:
        print("\nShutting down server...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
