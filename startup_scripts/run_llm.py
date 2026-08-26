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

DEVICE_NAME = "asyncroscopy/llm/default"
INSTANCE_NAME = "llm_instance"
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'

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
    ollama_model: str | None = None
    ollama_host: str | None = None
    api_key: str | None = None
    use_init_chat_model: bool | None = None
    agent_backend: str | None = None
    hermes_url: str | None = None
    hermes_model: str | None = None
    hermes_api_key: str | None = None
    skills_dir: str | None = None
    embedding_model: str | None = None
    startup_agents: list[dict] | None = None

    def __post_init__(self):
        # Convert tango dict to TangoConfig
        if isinstance(self.tango, dict):
            self.tango = TangoConfig(**self.tango)

def load_config(path: Path) -> LLMConfig:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return LLMConfig(**raw)

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

    server_name = f"LLM/{INSTANCE_NAME}"
    try:
        exported = database.get_device_exported(DEVICE_NAME)
        if exported and len(exported.value_string) > 0:
            try:
                proxy = DeviceProxy(DEVICE_NAME)
                proxy.set_timeout_millis(1000)
                proxy.ping()
                print(f"[SYSTEM]: Tango device {DEVICE_NAME} is actively running.")
            except Exception:
                print(f"[SYSTEM]: Device {DEVICE_NAME} is not responsive. Cleaning up stale registration for {server_name}...")
                database.unexport_server(server_name)
        else:
            database.unexport_server(server_name)
    except Exception as e:
        print(f"[SYSTEM]: Warning checking device export status: {e}")

    try:
        device_info = tango.DbDevInfo()
        device_info.server = server_name
        device_info._class = "LLM"
        device_info.name = DEVICE_NAME
        database.add_device(device_info)
        print(f"Registered device: {DEVICE_NAME}")
    except tango.DevFailed as e:
        print(f"Device already registered or error: {e}")

    if config:
        properties = {}
        unset_keys = []
        for key, value in config.__dict__.items():
            if key == "tango":
                continue
            if value is None:
                unset_keys.append(key)
                continue
            properties[key] = value
            if key == "startup_agents":
                properties[key] = [json.dumps(agent) for agent in value]

        if unset_keys:
            database.delete_device_property(DEVICE_NAME, unset_keys)
        database.put_device_property(DEVICE_NAME, properties)
        printable = {
            key: ("****" if key in ("api_key", "hermes_api_key") and value else value)
            for key, value in properties.items()
        }
        print(f"Set device properties: {printable}")

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

    tango_host = f'{config.tango.host}:{config.tango.port}'
    os.environ['TANGO_HOST'] = tango_host

    register_device(config)
    
    command = ["uv", "run", "--extra", "agent", "--extra", "ollama", "python", "-m", "asyncroscopy.mcp.llm", INSTANCE_NAME]
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
    main()
