"""Register and start the asyncroscopy LLM Tango device from a YAML config."""

import json
import sys
import argparse
import os
import subprocess
import yaml
from pathlib import Path
from dataclasses import dataclass
import time
import tango
from tango import DeviceProxy


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager
from startup_scripts.config_schema import ConfigError, Setting, read_settings
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers

DEVICE_NAME = "asyncroscopy/llm/default"
INSTANCE_NAME = "llm_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'

TITLE = 'Asyncroscopy LLM Startup'
LAUNCHER = 'startup_scripts/run_llm.py'
SETTINGS = [
    Setting('tango.host', required=True, section='Database', label='Tango host'),
    Setting('tango.port', 'int', required=True, section='Database', label='Tango port'),
    Setting('mcp_url', required=True, section='Model', label='MCP URL'),
    Setting('model_provider', section='Model', choices=('ollama', 'anthropic', 'openai'), help='Blank: the device default.'),
    Setting('model_name', section='Model'),
    Setting('local_model_path', 'path', section='Model', label='Local model path'),
    Setting('api_key', section='Model', label='API key'),
    Setting('startup_agents', 'yaml', [], section='Agents', label='Startup agents'),
]


def is_config(raw) -> bool:
    return isinstance(raw, dict) and 'mcp_url' in raw

@dataclass 
class TangoConfig:
    host: str
    port: int

@dataclass
class LLMConfig:
    tango: TangoConfig
    mcp_url: str
    local_model_path: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    startup_agents: list[dict] | None = None  # Agent configs, as asyncroscopy.mcp.llm.Agent reads them

    def __post_init__(self):
        # Convert tango dict to TangoConfig
        if isinstance(self.tango, dict):
            self.tango = TangoConfig(**self.tango)

def load_config(path: Path) -> LLMConfig:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f'{path.name} is not valid YAML: {exc}']) from exc
    values = read_settings(raw, SETTINGS)
    known = {setting.path.split('.')[0] for setting in SETTINGS}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError([f"unknown key(s) {', '.join(unknown)}; expected {', '.join(sorted(known))}"])
    agents = values['startup_agents'] or []
    if not all(isinstance(agent, dict) and agent.get('name') for agent in agents):
        raise ConfigError(['every startup agent needs at least a name'])
    return LLMConfig(
        tango=TangoConfig(host=values['tango.host'], port=values['tango.port']),
        mcp_url=values['mcp_url'],
        local_model_path=values['local_model_path'] or None,
        model_provider=values['model_provider'] or None,
        model_name=values['model_name'] or None,
        api_key=values['api_key'] or None,
        startup_agents=agents,
    )

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

    if config:
        properties = {}
        for key, value in config.__dict__.items():
            if key != "tango":
                properties[key] = value
                if key == "startup_agents":
                    properties[key] = [json.dumps(agent) for agent in value]
        
        database.put_device_property(DEVICE_NAME, properties)
        print(f"Set device properties: {properties}")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='LLM YAML config to start from.')
    parser.add_argument('--interactive', action='store_true', default=False, help='Run in interactive mode, allowing user to send prompts to the LLM device.')
    parser.add_argument('--name', default='default', help='Run as asyncroscopy/llm/<name>, beside the default LLM device.')
    add_lifecycle_args(parser)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    global DEVICE_NAME, INSTANCE_NAME
    args = parse_args(argv)
    if args.name != 'default':
        DEVICE_NAME = f"asyncroscopy/llm/{args.name}"
        INSTANCE_NAME = f"llm_{args.name}_instance"
    install_shutdown_handlers(args.parent_pid)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1
    # One state file per device so a second LLM never reaps the first.
    state_name = "run_llm" if args.name == "default" else f"run_llm_{INSTANCE_NAME}"
    if args.cleanup:
        with ProcessManager(name=state_name):
            pass
        print('Stopped processes left by an earlier run.')
        return 0

    tango_host = f'{config.tango.host}:{config.tango.port}'
    os.environ['TANGO_HOST'] = tango_host

    register_device(config)
    
    command = ["uv", "run", "--extra", "agent", "--extra", "ollama", "python", "-m", "asyncroscopy.mcp.llm", INSTANCE_NAME]
    env = {**os.environ, 'TANGO_HOST': tango_host, 'PYTHONUNBUFFERED': '1'}

    try:
        with ProcessManager(name=state_name) as manager:
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
                        return
                except Exception:
                    proxy = None
                
                time.sleep(1)
            else:
                print("Timeout waiting for device to initialize.")
                return

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
    raise SystemExit(main())
