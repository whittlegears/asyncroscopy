#!/usr/bin/env python
"""Start the Tango database and asyncroscopy device servers."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from io import BufferedReader
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import tango
import yaml


DATABASE_TIMEOUT_SECONDS = 120
TILED_COMMAND_TIMEOUT_MILLIS = 120_000
TILED_STARTUP_REGISTRATION_TIMEOUT_MILLIS = 3_600_000

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402
from startup_scripts.config_schema import ConfigError, DeviceTable, Setting, coerce, read_settings  # noqa: E402
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers  # noqa: E402

# Default config used in interactive mode (no --yaml). Passing --yaml <file>
# selects a different config AND runs headlessly (no prompts).
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'Spectra300.yaml'
# The GUI opens on the digital twin so a fresh window never targets the real microscope.
GUI_DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'DigitalTwin.yaml'


class Style:
    enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    reset = "\033[0m" if enabled else ""
    bold = "\033[1m" if enabled else ""
    dim = "\033[2m" if enabled else ""
    green = "\033[32m" if enabled else ""
    yellow = "\033[33m" if enabled else ""
    red = "\033[31m" if enabled else ""
    cyan = "\033[36m" if enabled else ""


@dataclass(frozen=True)
class DeviceConfig:
    key: str
    class_name: str
    module_name: str
    properties: dict[str, list[str]] = field(default_factory=dict)
    start_after_dependencies: bool = False

    @property
    def server_name(self) -> str:
        return f"{self.class_name}/{self.instance_name}"

    @property
    def device_name(self) -> str:
        return f"asyncroscopy/{self.key}/default"

    @property
    def command(self) -> list[str]:
        return ["uv", "run", "python", "-u", "-m", self.module_name, self.instance_name]

    @property
    def instance_name(self) -> str:
        return f"{self.key}_instance"


@dataclass(frozen=True)
class InstrumentConfig:
    class_name: str
    file: Path
    description: str = ""
    hardware_host: str | None = None
    hardware_port: int | None = None
    timeout_seconds: int | None = None
    properties: dict[str, list[str]] = field(default_factory=dict)

    @property
    def module_name(self) -> str:
        return python_file_to_module(self.file)


@dataclass(frozen=True)
class TiledConfig:
    host: str
    port: int
    acquisition_dir: str
    autostart: bool = True
    register_on_startup: bool = False


@dataclass(frozen=True)
class Config:
    path: Path
    instrument: InstrumentConfig
    support_devices: list[DeviceConfig]
    tango_host: str
    tango_port: int
    reset_database_file: bool
    tiled: TiledConfig
    device_timeout_seconds: int
    disabled_devices: tuple[str, ...] = ()


def _instrument_choices() -> tuple[str, ...]:
    """Instrument files the GUI suggests: modules under asyncroscopy/instruments/<kind>/ that run a server."""
    files = sorted((PROJECT_DIR / "asyncroscopy" / "instruments").glob("*/*.py"))
    return tuple(
        path.relative_to(PROJECT_DIR).as_posix()
        for path in files
        if path.name != "__init__.py" and "run_server(" in path.read_text(encoding="utf-8", errors="ignore")
    )


TITLE = "Asyncroscopy Server Startup"
LAUNCHER = "startup_scripts/run_servers.py"
SETTINGS = [
    Setting("instrument.file", "path", required=True, section="Instrument", label="Instrument file", choices=_instrument_choices()),
    Setting("instrument.class_name", section="Instrument", label="Instrument class", help="Blank: the class the file serves."),
    Setting("instrument.description", section="Instrument"),
    Setting("instrument.hardware_host", section="Instrument", label="Hardware host", help="Blank for digital twins."),
    Setting("instrument.hardware_port", "int", section="Instrument", label="Hardware port"),
    Setting("instrument.timeout_seconds", "int", section="Instrument", label="Hardware timeout (s)"),
    Setting("tango.host", required=True, section="Database", label="Tango host"),
    Setting("tango.port", "int", required=True, section="Database", label="Tango port"),
    Setting("tango.reset_database_file", "bool", False, section="Database", label="Delete tango_database.db before start"),
    Setting("tiled.host", required=True, section="Data server", label="Tiled host"),
    Setting("tiled.port", "int", required=True, section="Data server", label="Tiled port"),
    Setting("tiled.acquisition_dir", "dir", required=True, section="Data server", label="Acquisition dir"),
    Setting("tiled.autostart", "bool", True, section="Data server", label="Start Tiled HTTP server"),
    Setting("tiled.register_on_startup", "bool", False, section="Data server", label="Register acquisition dir on startup (slow for large folders)"),
    Setting("device_timeout_seconds", "int", 120, section="Startup", label="Device startup timeout (s)"),
]
DEVICE_SPEC_KEYS = ("module_name", "class_name", "properties", "enabled", "description")
DEVICE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
DEVICES = DeviceTable(
    summary=lambda key, spec: spec.get("class_name") or str(spec.get("module_name", "?")).rsplit(".", 1)[-1],
)


def is_config(raw) -> bool:
    return isinstance(raw, dict) and isinstance(raw.get("instrument"), dict)


def python_file_to_module(path: Path) -> str:
    file_path = path.expanduser()
    if not file_path.is_absolute():
        file_path = PROJECT_DIR / file_path
    relative_path = file_path.resolve().relative_to(PROJECT_DIR)
    return ".".join(relative_path.with_suffix("").parts)


def _instrument_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def module_source(module_name: str) -> Path:
    """The .py file a module name resolves to, without importing it."""
    if not MODULE_NAME_PATTERN.match(module_name):
        raise ValueError(f"'{module_name}' is not a valid Python module name")
    relative = Path(*module_name.split("."))
    for candidate in (PROJECT_DIR / relative.with_suffix(".py"), PROJECT_DIR / relative / "__main__.py"):
        if candidate.is_file():
            return candidate
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin and spec.origin.endswith(".py"):
        return Path(spec.origin)
    raise ValueError(f"module '{module_name}' not found")


def find_device_class(path: Path, requested: str | None = None, fallback: str | None = None) -> str:
    """The Tango device class a server file runs: the requested one, else the one passed to run_server()."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    imported = [alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names]
    if requested:
        if requested in classes or requested in imported:
            return requested
        raise ValueError(f"class '{requested}' is not defined in {path.name} (it defines: {', '.join(classes) or 'none'})")
    served = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_server" and isinstance(node.func.value, ast.Name)
    }
    if len(served) == 1:
        return served.pop()
    if fallback in classes:
        return fallback
    if len(classes) == 1:
        return classes[0]
    raise ValueError(f"cannot tell which class in {path.name} is the device; set class_name (classes: {', '.join(classes) or 'none'})")


def _property_values(raw, where: str) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.properties must be a mapping")
    return {
        name: [str(item) for item in value] if isinstance(value, list) else [str(value)]
        for name, value in raw.items()
    }


def _instrument_config(raw: dict, values: dict) -> InstrumentConfig:
    instrument_file = _instrument_file(values["instrument.file"])
    if not instrument_file.is_file():
        raise ValueError(f"instrument.file: {instrument_file} not found")
    return InstrumentConfig(
        class_name=find_device_class(instrument_file, values["instrument.class_name"]),
        file=instrument_file,
        description=values["instrument.description"] or "",
        hardware_host=values["instrument.hardware_host"] or None,
        hardware_port=values["instrument.hardware_port"] or None,
        timeout_seconds=values["instrument.timeout_seconds"] or None,
        properties=_property_values(raw.get("properties"), "instrument"),
    )


def device_config(key, spec) -> tuple[DeviceConfig | None, bool]:
    """(DeviceConfig, enabled) for one devices entry; a disabled entry is only shape-checked."""
    where = f"devices.{key}"
    if not isinstance(key, str) or not DEVICE_NAME_PATTERN.match(key):
        raise ValueError(f"device name {key!r} must start with a letter and use only letters, digits and _")
    if key == "instrument":
        raise ValueError("'instrument' is reserved for the instrument device; rename this device")
    if not isinstance(spec, dict):
        raise ValueError(f"{where} needs at least a module_name")
    unknown = sorted(set(spec) - set(DEVICE_SPEC_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}; expected {', '.join(DEVICE_SPEC_KEYS)}")
    enabled = coerce(Setting(f"{where}.enabled", "bool"), spec.get("enabled", True))
    if enabled is False:
        return None, False
    module_name = spec.get("module_name")
    if not module_name:
        raise ValueError(f"{where} is missing module_name")
    try:
        class_name = find_device_class(module_source(str(module_name)), spec.get("class_name"), fallback=key.upper())
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    return DeviceConfig(
        key=key,
        class_name=class_name,
        module_name=str(module_name),
        properties=_property_values(spec.get("properties"), where),
    ), True


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f"{path.name} is not valid YAML: {exc}"]) from exc
    values = read_settings(raw, SETTINGS)
    problems: list[str] = []
    instrument = None
    try:
        instrument = _instrument_config(raw["instrument"], values)
    except ValueError as exc:
        problems.append(str(exc))

    devices = raw.get("devices") or {}
    if not isinstance(devices, dict):
        problems.append("devices must be a mapping of device name to its settings")
        devices = {}
    support_devices, disabled = [], []
    for key, spec in devices.items():
        try:
            device, enabled = device_config(key, spec)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if enabled:
            support_devices.append(device)
        else:
            disabled.append(key)
    if problems:
        raise ConfigError(problems)

    return Config(
        path=path,
        instrument=instrument,
        support_devices=support_devices,
        tango_host=values["tango.host"],
        tango_port=values["tango.port"],
        reset_database_file=values["tango.reset_database_file"],
        tiled=TiledConfig(
            host=values["tiled.host"],
            port=values["tiled.port"],
            acquisition_dir=values["tiled.acquisition_dir"],
            autostart=values["tiled.autostart"],
            register_on_startup=values["tiled.register_on_startup"],
        ),
        device_timeout_seconds=values["device_timeout_seconds"],
        disabled_devices=tuple(disabled),
    )


def build_devices(config: Config) -> list[DeviceConfig]:
    return [
        *config.support_devices,
        DeviceConfig(
            key="instrument",
            class_name=config.instrument.class_name,
            module_name=config.instrument.module_name,
            start_after_dependencies=True,
        ),
    ]


def device_address_properties(config: Config) -> dict[str, list[str]]:
    return {
        f"{device.key}_device_address": [device.device_name]
        for device in config.support_devices
    }


def instrument_properties(config: Config) -> dict[str, list[str]]:
    properties = {**config.instrument.properties, **device_address_properties(config)}
    if config.instrument.hardware_host is not None:
        properties['hardware_host'] = [str(config.instrument.hardware_host)]
    if config.instrument.hardware_port is not None:
        properties['hardware_port'] = [str(config.instrument.hardware_port)]
    if config.instrument.timeout_seconds is not None:
        properties['hardware_timeout_seconds'] = [str(config.instrument.timeout_seconds)]
    return properties


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML config to start from. When given, runs headlessly (no prompts). "
        "When omitted, uses the bundled default config and prompts interactively.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stream each server's output (stdout+stderr) live to a per-device log "
        "file under output_tango_devices_logs/<timestamp>/ for troubleshooting.",
    )
    parser.add_argument(
        "--isolated",
        type=Path,
        default=None,
        metavar="DIR",
        help="Run a separate stack with its database and process state in DIR, "
        "leaving other stacks alone. Use distinct tango/tiled ports in the YAML.",
    )
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def instrument_cleanup_patterns(config: Config) -> set[str]:
    return {f"{config.instrument.class_name} instrument_instance"}


def selected_instrument(devices: list[DeviceConfig]) -> DeviceConfig:
    return next(device for device in devices if device.key == "instrument")


def color(text: str, code: str) -> str:
    return f"{code}{text}{Style.reset}" if Style.enabled else text


def prompt_str(label: str, default: str) -> str:
    try:
        answer = input(f"{color(label, Style.bold)} [{default}]: ").strip()
    except EOFError:
        print(default)
        return default
    return answer or default


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = prompt_str(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  {color('Invalid number.', Style.red)} Please enter an integer or press Enter.")


def prompt_bool(label: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            answer = input(f"{color(label, Style.bold)} [{suffix}]: ").strip().lower()
        except EOFError:
            print("yes" if default else "no")
            return default
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print(f"  {color('Invalid choice.', Style.red)} Enter y, n, or press Enter.")


def print_banner(title: str) -> None:
    width = 78
    print()
    print(color("=" * width, Style.cyan))
    print(color(title.center(width), Style.bold + Style.cyan))
    print(color("=" * width, Style.cyan))


def print_section(step: int, total: int, title: str) -> None:
    print()
    print(color(f"[{step}/{total}] {title}", Style.bold))
    print(color("-" * 78, Style.dim))


def status_line(status: str, message: str, detail: str = "") -> None:
    colors = {
        "OK": Style.green,
        "RUN": Style.cyan,
        "WAIT": Style.yellow,
        "FAIL": Style.red,
        "SKIP": Style.dim,
    }
    tag = color(f"{status:>4}", colors.get(status, ""))
    if detail:
        print(f"  {tag}  {message:<32} {color(detail, Style.dim)}")
    else:
        print(f"  {tag}  {message}")


def make_environment(
    host: str, port: int, tiled_host: str, tiled_port: int, acquisition_dir: str
) -> dict[str, str]:
    tango_host = f"{host}:{port}"
    os.environ["TANGO_HOST"] = tango_host
    return {
        **os.environ,
        "TANGO_HOST": tango_host,
        "ASYNCROSCOPY_TILED_URI": f"http://{tiled_host}:{tiled_port}",
        "ASYNCROSCOPY_ACQUISITION_DIR": acquisition_dir,
        "PYTHONUNBUFFERED": "1",
    }


def drain_process_output(
    stream: BufferedReader | None,
    output: deque[str],
    log_path: Path | None = None,
    stream_name: str = '',
) -> None:
    if stream is None:
        return

    def drain() -> None:
        log_file = log_path.open('a', encoding='utf-8') if log_path is not None else None
        for line in iter(stream.readline, b""):
            text = line.decode(errors='replace').rstrip()
            output.append(text)
            if log_file is not None:
                prefix = f'[{stream_name}] ' if stream_name else ''
                log_file.write(f'{prefix}{text}\n')
                log_file.flush()
        if log_file is not None:
            log_file.close()
        stream.close()

    threading.Thread(target=drain, daemon=True).start()


def buffered_output(lines: deque[str]) -> str:
    return "\n".join(line for line in lines if line)
    

def wait_for_database(host: str, port: int, timeout: int) -> float:
    start = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - start < timeout:
        try:
            db = tango.Database(host, port)
            db.get_db_host()
            return time.monotonic() - start
        except Exception as exc:
            last_error = exc
            print(color(".", Style.dim), end="", flush=True)
            time.sleep(1)
    raise TimeoutError(f"Tango database did not become ready after {timeout}s. Last error: {last_error}")


def wait_for_device(device_name: str, timeout: int) -> float:
    start = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - start < timeout:
        try:
            proxy = tango.DeviceProxy(device_name)
            proxy.ping()
            return time.monotonic() - start
        except Exception as exc:
            last_error = exc
            print(color(".", Style.dim), end="", flush=True)
            time.sleep(1)
    raise TimeoutError(f"{device_name} did not become ready after {timeout}s. Last error: {last_error}")


def register_devices(
    devices: list[DeviceConfig], instrument_properties: dict[str, list[str]], disabled: Iterable[str] = ()
) -> None:
    database = tango.Database()
    status_line("OK", "database", f"{database.get_db_host()}:{database.get_db_port()}")
    instrument_name = selected_instrument(devices).device_name

    # A device switched off must not linger from an earlier run in a kept database.
    for key in disabled:
        try:
            database.delete_device(f"asyncroscopy/{key}/default")
        except tango.DevFailed:
            pass
        database.delete_device_property(instrument_name, [f"{key}_device_address"])
        status_line("SKIP", key, "disabled")

    for device in devices:
        device_info = tango.DbDevInfo()
        device_info.server = device.server_name
        device_info._class = device.class_name
        device_info.name = device.device_name
        database.add_device(device_info)
        try:
            database.unexport_server(device.server_name)
        except Exception:
            pass
        status_line("OK", device.device_name)
        for property_name, property_value in device.properties.items():
            database.put_device_property(device.device_name, {property_name: property_value})
            status_line("OK", f"property: {device.device_name} {property_name} = {property_value[0]}")

    for property_name, property_value in instrument_properties.items():
        database.put_device_property(instrument_name, {property_name: property_value})
        status_line("OK", f"property: {property_name} = {property_value[0]}")


def get_data_proxy() -> tango.DeviceProxy:
    data = tango.DeviceProxy("asyncroscopy/data/default")
    data.set_timeout_millis(TILED_COMMAND_TIMEOUT_MILLIS)
    return data


def register_tiled_save_path() -> dict:
    data = get_data_proxy()
    data.set_timeout_millis(TILED_STARTUP_REGISTRATION_TIMEOUT_MILLIS)
    return json.loads(data.register_save_path())


def stop_tiled_server() -> None:
    try:
        get_data_proxy().stop_tiled_server()
    except Exception:
        pass


def stop_tiled_server_quietly() -> None:
    try:
        stop_tiled_server()
    except Exception:
        pass


def print_debug_output(processes: Iterable[ManagedProcess]) -> None:
    print()
    print(color("Debug output", Style.bold + Style.yellow))
    print(color("-" * 78, Style.dim))
    for process in processes:
        stdout = buffered_output(process.stdout_lines)
        stderr = buffered_output(process.stderr_lines)
        print(
            f"{color(process.label, Style.bold)}  pid={process.pid}  running={process.running}  returncode={process.process.poll()}"
        )
        print(f"  command: {' '.join(process.command)}")
        print(f"  stdout: {stdout or '(empty)'}")
        print(f"  stderr: {stderr or '(empty)'}")
        print()


def print_inventory(devices: list[DeviceConfig]) -> None:
    status_line("OK", "device inventory", f"{len(devices)} declaration(s) built into this script")
    key_width = max(len(device.key) for device in devices)
    class_width = max(len(device.class_name) for device in devices)
    for device in devices:
        status_line("RUN", device.key.ljust(key_width), f"{device.class_name.ljust(class_width)}  {device.device_name}")


def print_summary(
    host: str,
    port: int,
    processes: list[ManagedProcess],
    ready_times: dict[str, float],
    tiled_config: dict | None = None,
    step: int = 5,
    total: int = 5,
) -> None:
    print_section(step, total, "Startup summary")
    print(f"  {color('TANGO_HOST', Style.bold):<18} {host}:{port}")
    print(f"  {color('PROJECT', Style.bold):<18} {PROJECT_DIR}")
    print()
    print(f"  {'SERVER':<14} {'PID':>8} {'READY':>10}  COMMAND")
    print(color("  " + "-" * 74, Style.dim))
    for process in processes:
        ready = ready_times.get(process.key)
        ready_text = f"{ready:.1f}s" if ready is not None else "-"
        print(
            f"  {process.key:<14} {process.pid:>8} {ready_text:>10}  {' '.join(process.command)}"
        )
    if tiled_config is not None:
        print()
        print(f"  {color('TILED_URI', Style.bold):<18} {tiled_config['uri']}")
        print(f"  {color('TILED_SERVING', Style.bold):<18} {tiled_config['tiled_server_serving']}")
    print()
    print(color("All asyncroscopy servers are ready.", Style.bold + Style.green))


def manager_options(args: argparse.Namespace, port: int) -> dict:
    """ProcessManager options for this run; --isolated keeps its state apart per port."""
    if args.isolated is None:
        return {}
    isolated_dir = args.isolated.resolve()
    return {"name": f"run_servers_{port}", "state_dir": isolated_dir / ".processes"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    config_path = args.yaml or DEFAULT_CONFIG_PATH
    try:
        config = load_config(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(color(f"Config error: {exc}", Style.bold + Style.red))
        return 1

    if args.cleanup:
        with ProcessManager(**manager_options(args, config.tango_port)):
            pass
        status_line("OK", "cleanup", "stopped processes left by an earlier run")
        return 0

    headless = args.yaml is not None
    devices = build_devices(config)
    instrument = selected_instrument(devices)
    instrument_config = config.instrument
    regular_devices = [device for device in devices if not device.start_after_dependencies]
    dependency_devices = [device for device in devices if device.start_after_dependencies]

    print_banner("ASYNCROSCOPY SERVER STARTUP")

    registered_instrument_properties = instrument_properties(config)
    if headless:
        print(f"Headless start from {config_path}")
        print()
        host, port = config.tango_host, config.tango_port
        tiled_host, tiled_port = config.tiled.host, config.tiled.port
        acquisition_dir = config.tiled.acquisition_dir
        should_start_tiled = config.tiled.autostart
        should_register_tiled = config.tiled.register_on_startup
        clear_first = start_database = should_register_devices = True
        reset_database_file = config.reset_database_file
        device_timeout = config.device_timeout_seconds
    else:
        print("Press Enter at any prompt to use the value shown in brackets.")
        print()
        host = prompt_str("Tango database host", config.tango_host)
        port = prompt_int("Tango database port", config.tango_port)
        default_tiled = urlsplit(os.environ.get("ASYNCROSCOPY_TILED_URI", f"http://{config.tiled.host}:{config.tiled.port}"))
        tiled_host = prompt_str("Tiled HTTP host", default_tiled.hostname or config.tiled.host)
        tiled_port = prompt_int("Tiled HTTP port", default_tiled.port or config.tiled.port)
        acquisition_dir = prompt_str(
            "Acquisition save path",
            os.environ.get("ASYNCROSCOPY_ACQUISITION_DIR", config.tiled.acquisition_dir),
        )
        should_start_tiled = prompt_bool("Start Tiled HTTP server", config.tiled.autostart)
        should_register_tiled = prompt_bool(
            "Register acquisition directory with Tiled on startup (can be slow)",
            config.tiled.register_on_startup,
        )
        clear_first = prompt_bool("Clear old processes first", True)
        reset_database_file = prompt_bool("Delete Tango database file before start", config.reset_database_file)
        start_database = prompt_bool("Start Tango database", True)
        should_register_devices = prompt_bool("Register devices", True)
        device_timeout = prompt_int("Device startup timeout seconds", config.device_timeout_seconds)
        
        if instrument_config.hardware_host is not None:
            registered_instrument_properties['hardware_host'] = [
                prompt_str("Hardware host", str(instrument_config.hardware_host))
            ]
        if instrument_config.hardware_port is not None:
            registered_instrument_properties['hardware_port'] = [
                str(prompt_int("Hardware port", int(instrument_config.hardware_port)))
            ]
        if instrument_config.timeout_seconds is not None:
            registered_instrument_properties['hardware_timeout_seconds'] = [
                str(prompt_int("Hardware timeout seconds", int(instrument_config.timeout_seconds)))
            ]

    total_steps = 5
    environment = make_environment(host, port, tiled_host, tiled_port, acquisition_dir)
    ready_times: dict[str, float] = {}
    tiled_config = None

    log_dir: Path | None = None
    if args.debug:
        log_dir = PROJECT_DIR / "output_tango_devices_logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  {color('TANGO_HOST', Style.bold):<18} {host}:{port}")
    print(f"  {color('PROJECT', Style.bold):<18} {PROJECT_DIR}")
    print(f"  {color('CONFIG', Style.bold):<18} {config_path}")
    print(f"  {color('INSTRUMENT', Style.bold):<18} {instrument.class_name}")
    if log_dir is not None:
        print(f"  {color('DEBUG LOGS', Style.bold):<18} {log_dir}")
    print_inventory(devices)

    isolated_dir = args.isolated.resolve() if args.isolated else None
    manager_kwargs = manager_options(args, port)
    if isolated_dir is not None:
        isolated_dir.mkdir(parents=True, exist_ok=True)
        clear_first = False
        environment["PYTANGO_DATABASE_NAME"] = str(isolated_dir / "tango_database.db")

    try:
        with ProcessManager(**manager_kwargs) as manager:
            print_section(1, total_steps, "Clearing old processes")
            if clear_first:
                ports_to_clear = [port]
                if should_start_tiled:
                    ports_to_clear.append(tiled_port)
                manager.scour_ports(ports_to_clear)
            else:
                status_line("SKIP", "old process cleanup")
                
            if reset_database_file and start_database and isolated_dir is not None:
                (isolated_dir / "tango_database.db").unlink(missing_ok=True)
            elif reset_database_file and start_database:
                manager.wipe_databases()
            elif reset_database_file:
                status_line("SKIP", "Tango database file", "database startup is disabled")

            print_section(2, total_steps, "Starting Tango database")
            if start_database:
                database = manager.start_process(
                    key="database",
                    label="Tango database",
                    command=["uv", "run", "python", "-m", "tango.databaseds.database", "2"],
                    env=environment,
                )
                print("  WAIT  database readiness", end="", flush=True)
                elapsed = wait_for_database(host, port, DATABASE_TIMEOUT_SECONDS)
                ready_times["database"] = elapsed
                print(f" {color('OK', Style.green)} pid={database.pid} ready in {elapsed:.1f}s")
            else:
                print("  WAIT  existing database readiness", end="", flush=True)
                elapsed = wait_for_database(host, port, DATABASE_TIMEOUT_SECONDS)
                ready_times["database"] = elapsed
                print(f" {color('OK', Style.green)} ready in {elapsed:.1f}s")

            print_section(3, total_steps, "Registering devices")
            if should_register_devices:
                register_devices(devices, registered_instrument_properties, config.disabled_devices)
            else:
                status_line("SKIP", "device registration")

            print_section(4, total_steps, "Starting device servers")
            for device in regular_devices:
                process = manager.start_process(
                    key=device.key, 
                    label=device.class_name, 
                    command=device.command, 
                    env=environment
                )
                status_line("RUN", device.key, f"{device.module_name}  pid={process.pid}")

            for device in regular_devices:
                print(f"  WAIT  {device.device_name:<34}", end="", flush=True)
                elapsed = wait_for_device(device.device_name, device_timeout)
                ready_times[device.key] = elapsed
                print(f" {color('OK', Style.green)} ready in {elapsed:.1f}s")

            if should_start_tiled:
                tiled_config = json.loads(get_data_proxy().start_tiled_server())
                if tiled_config["tiled_server"] != "yes":
                    raise RuntimeError(f"Tiled HTTP server failed to start: {tiled_config['tiled_server_status']}")
                status_line("OK", "Tiled HTTP server", f"{tiled_config['uri']} serving {tiled_config['tiled_server_serving']}")
                
                if should_register_tiled:
                    status_line("WAIT", "Tiled startup registration", "registering acquisition directory; this can take a while")
                    tiled_registration = register_tiled_save_path()
                    tiled_config.update(tiled_registration)
                    status_line("OK", "Tiled startup registration", tiled_registration["registered_path"])
                else:
                    status_line("SKIP", "Tiled startup registration", "register_on_startup=false; files register manually")
            else:
                status_line("SKIP", "Tiled HTTP server")
                if should_register_tiled:
                    status_line("SKIP", "Tiled startup registration", "Tiled HTTP server autostart is disabled")

            for device in dependency_devices:
                print()
                status_line("RUN", device.key, f"{device.module_name}  starting after dependencies")
                process = manager.start_process(
                    key=device.key, 
                    label=device.class_name, 
                    command=device.command, 
                    env=environment
                )
                print(f"  WAIT  {device.device_name:<34}", end="", flush=True)
                elapsed = wait_for_device(device.device_name, device_timeout)
                ready_times[device.key] = elapsed
                print(f" {color('OK', Style.green)} ready in {elapsed:.1f}s")

            print_summary(
                host,
                port,
                manager.active_processes,
                ready_times,
                tiled_config,
                step=total_steps,
                total=total_steps,
            )
            print()
            print(color("Leave this terminal open while you use the servers. Press Ctrl+C to stop them.", Style.dim))
            
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                # Catch the Ctrl-C INSIDE the with block so processes are still alive!
                print()
                print(color("Shutdown requested. Stopping Tiled server...", Style.yellow))
                stop_tiled_server() 
                print(color("Stopping managed processes...", Style.yellow))
                
        status_line("OK", "shutdown complete")
        return 0

    except KeyboardInterrupt:
        # Stopped during startup; the ProcessManager context has already stopped what was running.
        stop_tiled_server_quietly()
        status_line("OK", "stopped during startup")
        return 130

    except Exception as exc:   
        # Catches actual startup crashes
        print()
        print(color(f"Startup failed: {exc}", Style.bold + Style.red))
        if log_dir is not None:
            print(color(f"Per-server logs in {log_dir}:", Style.bold + Style.yellow))

        if 'manager' in locals() and manager.history:
            print_debug_output(manager.history)
            
            if log_dir is not None:
                for p in manager.history:
                    stdout = buffered_output(p.stdout_lines)
                    stderr = buffered_output(p.stderr_lines)
                    log_file = log_dir / f"{p.key}.log"
                    log_file.write_text(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}", encoding="utf-8")
                print(color(f"Saved logs to {log_dir}", Style.bold + Style.yellow))

        # Stop tiled on a crash
        try:
            stop_tiled_server() 
        except Exception:
            pass
            
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
