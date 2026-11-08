#!/usr/bin/env python
"""Start the Tango database and asyncroscopy device servers."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import signal
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

from asyncroscopy import settings
from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager, listening_pids, python_command  # noqa: E402

# Default config used in interactive mode (no --yaml). Passing --yaml <file>
# selects a different config AND runs headlessly (no prompts).
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'Spectra300.yaml'


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
        return python_command("-m", self.module_name, self.instance_name)

    @property
    def instance_name(self) -> str:
        return f"{self.key}_instance"


@dataclass(frozen=True)
class InstrumentConfig:
    class_name: str
    file: Path | None
    description: str = ""
    hardware_host: str | None = None
    hardware_port: int | None = None
    timeout_seconds: int | None = None
    properties: dict[str, list[str]] = field(default_factory=dict)
    module: str | None = None

    @property
    def module_name(self) -> str:
        return self.module or python_file_to_module(self.file)


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
    # Top-level `simulated:`. Every device gets it as its `simulated` property; False (real hardware) when absent.
    simulated: bool = False
    # tango.bind_host: the interface the Tango database listens on, when it differs from tango.host (the
    # name clients use). In a container or pod, tango.host is a service name and bind_host is 0.0.0.0.
    tango_bind_host: str = ""
    # Top-level `plugins:`: modules or .py files every device imports (asyncroscopy/plugins.py), so a
    # team's own @register_profile / @register_tracker / @register_strategy classes exist in each process.
    plugins: tuple[str, ...] = ()


YAML_INSTRUMENT_MODULE = "asyncroscopy.instruments.yaml_sim.device"
YAML_JOB_MODULE = "asyncroscopy.instruments.yaml_sim.job"


def _plugins(raw: dict) -> tuple[str, ...]:
    """The stack YAML's `plugins:` list: files resolved against the project folder and checked to exist,
    module names checked to import, so a typo fails at load rather than as "unknown instrument profile"."""
    from asyncroscopy.plugins import is_file_spec

    entries = raw.get("plugins") or []
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list):
        raise ValueError("Config 'plugins' must be a list of module names or .py files")
    resolved = []
    for entry in (str(e).strip() for e in entries):
        if not entry:
            continue
        if is_file_spec(entry):
            path = _instrument_file(entry)
            if not path.is_file():
                raise ValueError(f"plugins entry {entry!r}: no such file {path}")
            resolved.append(str(path.resolve()))
        else:
            _check_importable(entry, "plugins entry")
            resolved.append(entry)
    return tuple(resolved)


def plugin_environment(plugins: Iterable[str], current: str = "") -> str:
    """ASYNCROSCOPY_PLUGINS for the device processes: what the shell already set, then the YAML's `plugins:`."""
    from asyncroscopy.plugins import plugin_specs

    entries = plugin_specs(current)
    entries += [p for p in plugins if p not in entries]
    return os.pathsep.join(entries)


def _simulated(raw: dict, path: Path) -> bool:
    """The stack's top-level ``simulated: true|false``; a config without it is treated as real hardware."""
    if "simulated" not in raw:
        print(f"Note: {path.name} has no top-level 'simulated:'; treating the stack as real hardware (simulated: false).", file=sys.stderr)
        return False
    value = raw["simulated"]
    if not isinstance(value, bool):
        raise ValueError(f"{path}: simulated must be true or false, got {value!r}")
    return value


def _check_device_simulated(value, stack: bool, where: str, path: Path) -> None:
    """Refuse a device-level ``simulated`` property that is not the stack's own flag (or not a boolean at all)."""
    if value is None:
        return
    text = (value[0] if isinstance(value, list) and len(value) == 1 else value)
    normalised = str(text).strip().lower()
    if normalised not in ("true", "false"):
        raise ValueError(f"{path}: {where}.properties.simulated must be true or false, got {value!r}")
    if (normalised == "true") != stack:
        raise ValueError(
            f"{path}: {where}.properties.simulated is {normalised}, but the stack's top-level simulated is {str(stack).lower()}. "
            "A stack is simulated or not as a whole: remove the device's own simulated property (it gets the stack's)"
        )


def _require(mapping, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Config section '{where}' is missing required key '{key}'")
    return mapping[key]


def python_file_to_module(path: Path) -> str:
    file_path = path.expanduser()
    if not file_path.is_absolute():
        file_path = PROJECT_DIR / file_path
    try:
        relative_path = file_path.resolve().relative_to(PROJECT_DIR)
    except ValueError:
        raise ValueError(
            f"instrument.file {file_path} is outside {PROJECT_DIR}; for code kept elsewhere set instrument.module_name "
            "(and instrument.class_name) instead, and put the package's folder on PYTHONPATH (or install it with uv add --editable)"
        ) from None
    return ".".join(relative_path.with_suffix("").parts)


def _instrument_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _class_name_from_file(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    raise ValueError(f"Instrument file {path} does not define a class")


def _check_importable(module: str, where: str) -> None:
    """Fail at load, not after the device timeout, when a server's module cannot be imported."""
    try:
        found, reason = importlib.util.find_spec(module) is not None, "not found"
    except Exception as exc:  # running the package's __init__ can fail for its own reasons
        found, reason = False, f"{type(exc).__name__}: {exc}"
    if not found:
        raise ValueError(
            f"{where} {module!r} cannot be imported ({reason}); check the name, and put its package's "
            "folder on PYTHONPATH (or install it with uv add --editable)"
        )


def _instrument_config(raw: dict) -> InstrumentConfig:
    module = raw.get("module_name") if isinstance(raw, dict) else None
    if module:
        # Code outside this repository: a module the servers' Python can import.
        instrument_file = None
        if not raw.get("class_name"):
            raise KeyError("Config section 'instrument' gives module_name, so it also needs class_name")
        _check_importable(module, "instrument.module_name")
    else:
        instrument_file = _instrument_file(_require(raw, "file", "instrument"))
    properties = {
        name: [str(item) for item in value] if isinstance(value, list) else [str(value)]
        for name, value in (raw.get("properties") or {}).items()
    }
    return InstrumentConfig(
        class_name=raw.get("class_name") or _class_name_from_file(instrument_file),
        file=instrument_file,
        description=raw.get("description", ""),
        hardware_host=raw.get("hardware_host"),
        hardware_port=int(raw["hardware_port"]) if raw.get("hardware_port") else None,
        timeout_seconds=int(raw["timeout_seconds"]) if raw.get("timeout_seconds") else None,
        properties=properties,
        module=module,
    )


def _merge(base: dict, override: dict, replace_values: bool = False) -> dict:
    """``override`` laid over ``base``: mappings merge key by key, anything else replaces, and null removes.
    Under a ``properties:`` mapping each property is replaced whole (a specimen mapping is one value)."""
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif not replace_values and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value, replace_values=key == "properties")
        else:
            merged[key] = value
    return merged


def read_config(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """A stack config as a mapping, with its ``extends:`` chain merged in.

    ``extends: other.yaml`` (relative to this file) starts from that config; this file's values win.
    Sections and devices merge key by key, each property is replaced whole, and a value of null removes
    what the base set (a device, a property). Lists are replaced, not appended to.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"{path}: extends goes round in a circle ({' -> '.join(p.name for p in (*_seen, resolved))})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"{path}: extends must name one config file, got {parent!r}")
    return _merge(read_config(path.parent / parent, (*_seen, resolved)), raw)


def load_config(path: Path) -> Config:
    raw = read_config(path)

    support_devices = [
        DeviceConfig(
            key=key,
            class_name=(spec or {}).get("class_name", key.upper()),
            module_name=_require(spec or {}, "module_name", f"devices.{key}"),
            properties={
                property_name: [str(property_value)]
                for property_name, property_value in ((spec or {}).get("properties") or {}).items()
            },
        )
        for key, spec in _require(raw, "devices", "(top level)").items()
    ]
    for device in support_devices:
        _check_importable(device.module_name, f"devices.{device.key}.module_name")
    tango_section = raw.get("tango", {})
    tiled = _require(raw, "tiled", "(top level)")
    simulated = _simulated(raw, path)
    instrument = _instrument_config(_require(raw, "instrument", "(top level)"))
    # The stack flag reaches every device as its `simulated` property. A device may repeat it, but never
    # contradict it: one device saying simulated on a hardware stack would lift that device's hardware rules.
    for where, properties in [("instrument", instrument.properties), *((f"devices.{d.key}", d.properties) for d in support_devices)]:
        _check_device_simulated(properties.get("simulated"), simulated, where, path)
        properties["simulated"] = [str(simulated).lower()]

    return Config(
        path=path,
        simulated=simulated,
        instrument=instrument,
        support_devices=support_devices,
        tango_host=_require(tango_section, "host", "tango"),
        tango_port=int(_require(tango_section, "port", "tango")),
        reset_database_file=bool(tango_section.get("reset_database_file", False)),
        tango_bind_host=str(tango_section.get("bind_host") or ""),
        tiled=TiledConfig(
            host=_require(tiled, "host", "tiled"),
            port=int(_require(tiled, "port", "tiled")),
            acquisition_dir=_require(tiled, "acquisition_dir", "tiled"),
            autostart=bool(tiled.get("autostart", True)),
            register_on_startup=bool(tiled.get("register_on_startup", False)),
        ),
        device_timeout_seconds=int(raw.get("device_timeout_seconds", 120)),
        plugins=_plugins(raw),
    )


def yaml_job_device(config: Config) -> DeviceConfig | None:
    """The job device (``<class_name>Job``) of a YAML instrument whose file has a ``jobs:`` section, started from the same file."""
    if config.instrument.module_name != YAML_INSTRUMENT_MODULE:
        return None
    from asyncroscopy.instruments.yaml_sim.spec import SpecError, load_spec, resolve_path

    try:
        if not load_spec(resolve_path(config.instrument.properties["yaml_path"][0])).jobs:
            return None
    except (SpecError, KeyError):
        return None  # the instrument device starts, finds the same problem and reports it as FAULT
    return DeviceConfig(
        key="job",
        class_name=f"{config.instrument.class_name}Job",
        module_name=YAML_JOB_MODULE,
        properties={**config.instrument.properties, "instrument_device_address": ["asyncroscopy/instrument/default"]},
        start_after_dependencies=True,
    )


def build_devices(config: Config) -> list[DeviceConfig]:
    job_device = yaml_job_device(config)
    return [
        *config.support_devices,
        DeviceConfig(
            key="instrument",
            class_name=config.instrument.class_name,
            module_name=config.instrument.module_name,
            start_after_dependencies=True,
        ),
        *([job_device] if job_device else []),
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
        help="Run a separate stack with its database and process state in DIR "
        "(e.g. outputs/stacks/raman), leaving other stacks alone. Use distinct tango/tiled ports in the YAML.",
    )
    parser.add_argument(
        "--log-to-stdout",
        action="store_true",
        help="Copy every server's output to this launcher's stdout, each line prefixed with [device]. "
        "For containers and supervisors that collect stdout; without it the output is kept for error reports only.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Stop another stack (or program) that holds this config's ports or shares its "
        "Tango database file. Without it, the launcher refuses and says how to run beside it.",
    )
    return parser.parse_args(argv)


# --- Other stacks ---------------------------------------------------------
# A launcher started without --isolated scours its Tango and Tiled ports and
# deletes the project-root Tango database file. That is right for this config's
# own leftovers (a crashed earlier run) and wrong for anyone else's stack.

LAUNCHER_NAME = "run_servers.py"
ORPHAN_MARKERS = ("tango.databaseds", "tiled", "asyncroscopy.")  # processes an earlier stack leaves behind


@dataclass(frozen=True)
class ProcessInfo:
    ppid: int
    tokens: tuple[str, ...]
    cwd: str | None = None


@dataclass(frozen=True)
class OtherStack:
    identity: tuple[str, str]  # (config path, isolated directory or "")
    pid: int
    config: str  # for messages, relative to the repository where possible


def _normalized(path: Path) -> str:
    text = str(path.resolve())
    return text.lower() if os.name == "nt" else text


def stack_identity(config_path: Path, isolated_dir: Path | None) -> tuple[str, str]:
    return _normalized(config_path), _normalized(isolated_dir) if isolated_dir else ""


def _flag(tokens: list[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def launcher_identity(info: ProcessInfo) -> tuple[str, str] | None:
    """The (config, isolated) of a run_servers.py launcher's command line (or its ``uv run`` wrapper), else None."""
    tokens = list(info.tokens)
    start = next((i for i, token in enumerate(tokens) if Path(token).name == LAUNCHER_NAME), None)
    if start is None:
        return None
    args = tokens[start + 1:]
    base = Path(info.cwd) if info.cwd else PROJECT_DIR
    config, isolated = _flag(args, "--yaml"), _flag(args, "--isolated")
    config_path = DEFAULT_CONFIG_PATH if config is None else Path(config) if Path(config).is_absolute() else base / config
    isolated_dir = None if isolated is None else Path(isolated) if Path(isolated).is_absolute() else base / isolated
    return stack_identity(config_path, isolated_dir)


def _shown(identity: tuple[str, str]) -> str:
    config = Path(identity[0])
    try:
        text = config.relative_to(_normalized(PROJECT_DIR)).as_posix()
    except ValueError:
        text = str(config)
    return text + (f" (isolated in {identity[1]})" if identity[1] else "")


def process_snapshot() -> dict[int, ProcessInfo]:
    """Every visible process: parent, command line and working directory."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return {}
    snapshot = {}
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            cwd = None
        snapshot[proc.info["pid"]] = ProcessInfo(proc.info.get("ppid") or 0, tuple(proc.info.get("cmdline") or ()), cwd)
    return snapshot


def ancestors(pid: int, snapshot: dict[int, ProcessInfo]) -> list[int]:
    chain, seen = [], {pid}
    while pid in snapshot and len(chain) < 64:
        pid = snapshot[pid].ppid
        if pid in seen or pid <= 0:
            break
        seen.add(pid)
        chain.append(pid)
    return chain


def running_stacks(snapshot: dict[int, ProcessInfo], mine: tuple[str, str], my_pids: set[int]) -> dict[tuple[str, str], OtherStack]:
    """Live launchers of other stacks, one per (config, isolated), each by its outermost process (so stopping it stops the tree)."""
    found: dict[tuple[str, str], OtherStack] = {}
    for pid, info in snapshot.items():
        identity = launcher_identity(info)
        if identity is None or identity == mine or pid in my_pids:
            continue
        parent = snapshot.get(info.ppid)
        if parent is not None and launcher_identity(parent) == identity:
            continue  # the wrapper above it is the one to stop
        found[identity] = OtherStack(identity, pid, _shown(identity))
    return found


def stack_conflicts(
    snapshot: dict[int, ProcessInfo],
    mine: tuple[str, str],
    my_pids: set[int],
    listeners: dict[int, set[int]],
    share_database_file: bool,
) -> list[tuple[str, int]]:
    """(reason, pid to stop with --replace) for everything this start would wrongly stop or clobber.

    ``listeners`` maps each port this start clears to the PIDs listening on it.
    A listener belonging to this config's own stack (another or a crashed run of
    it) and an orphan an earlier stack left behind are not conflicts: clearing
    them is the launcher's job. ``share_database_file``: this start uses the
    project-root Tango database file, which no other non-isolated stack may use.
    """
    stacks = running_stacks(snapshot, mine, my_pids)
    conflicts: list[tuple[str, int]] = []
    if share_database_file:
        for stack in stacks.values():
            if not stack.identity[1]:
                conflicts.append((f"{stack.config} (launcher pid {stack.pid}) uses the same Tango database file (neither stack is --isolated)", stack.pid))
    held: dict[tuple[str, str], tuple[int, list[int]]] = {}  # another stack -> (pid to stop, its ports)
    for port, pids in sorted(listeners.items()):
        for pid in sorted(pids):
            if pid in my_pids or pid not in snapshot:
                continue
            owner = next((launcher_identity(snapshot[a]) for a in [pid, *ancestors(pid, snapshot)] if a in snapshot and launcher_identity(snapshot[a])), None)
            if owner == mine:
                continue
            if owner is not None:
                stack = stacks.get(owner)
                ports = held.setdefault(owner, (stack.pid if stack else pid, []))[1]
                if port not in ports:
                    ports.append(port)
                continue
            tokens = snapshot[pid].tokens
            if any(marker in token for token in tokens for marker in ORPHAN_MARKERS):
                continue  # left behind by an earlier stack whose launcher is gone
            name = Path(tokens[0]).name if tokens else "a program"
            conflicts.append((f"port {port} is used by {name} (pid {pid}), which is not an asyncroscopy stack", pid))
    for owner, (pid, ports) in held.items():
        listed = ", ".join(str(p) for p in ports)
        conflicts.append((f"{_shown(owner)} (launcher pid {pid}) is listening on port{'s' if len(ports) > 1 else ''} {listed}", pid))
    return conflicts


def stop_tree(pid: int) -> None:
    """Stop a process and everything it started (--replace)."""
    import psutil

    try:
        root = psutil.Process(pid)
        family = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for proc in family:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(family, timeout=10)


def explain_conflicts(conflicts: list[tuple[str, int]], config_path: Path) -> str:
    lines = ["This start would stop or clobber something that is not this config's own stack:"]
    lines += [f"  - {reason}" for reason, _ in conflicts]
    lines += [
        f"To run beside it, give {config_path.name} its own tango.port and tiled.port and start it with",
        "  --isolated outputs/stacks/<name>   (docs/Adding_New_Hardware/second_stack.md).",
        "To stop it and start this config instead, run again with --replace.",
    ]
    return "\n".join(lines)


def instrument_cleanup_patterns(config: Config) -> set[str]:
    job_device = yaml_job_device(config)
    return {f"{config.instrument.class_name} instrument_instance", *([f"{job_device.class_name} {job_device.instance_name}"] if job_device else [])}


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
        "WARN": Style.yellow,
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


def wait_for_device(device_name: str, timeout: int, process: ManagedProcess | None = None) -> float:
    start = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - start < timeout:
        if process is not None and process.process.poll() is not None:
            raise RuntimeError(
                f"{device_name}'s server exited (code {process.process.poll()}) before it was ready; "
                "check its module_name/class_name in the YAML and the server's output below"
            )
        try:
            proxy = tango.DeviceProxy(device_name)
            proxy.ping()
            return time.monotonic() - start
        except Exception as exc:
            last_error = exc
            print(color(".", Style.dim), end="", flush=True)
            time.sleep(1)
    raise TimeoutError(f"{device_name} did not become ready after {timeout}s. Last error: {last_error}")


def device_health(device_name: str) -> tuple[str, str]:
    """A device that answers its ping: its State and Status, as text. A device that started but could not
    reach its hardware (no AutoScript, a refused connection) answers too, in FAULT; reading it says so."""
    try:
        proxy = tango.DeviceProxy(device_name)
        return str(proxy.state()), str(proxy.status()).strip()
    except Exception as exc:  # it answered the ping a moment ago: say why it cannot be read, as a fault
        return "FAULT", f"State/Status could not be read: {exc}"


def report_health(device_name: str, elapsed: float, faults: dict[str, str]) -> None:
    """Print a device's State after it answered: FAULT is recorded in ``faults`` and shown as FAIL, any other
    state but ON as WARN, both with the device's Status."""
    state, status = device_health(device_name)
    if state == "FAULT":
        faults[device_name] = status
        print(f" {color('FAULT', Style.bold + Style.red)} answered in {elapsed:.1f}s")
        status_line("FAIL", device_name, status)
    elif state != "ON":
        print(f" {color(state, Style.yellow)} answered in {elapsed:.1f}s")
        status_line("WARN", device_name, status)
    else:
        print(f" {color('OK', Style.green)} ready in {elapsed:.1f}s")


def isolate_approvals(devices: list[DeviceConfig], isolated_dir: Path | None) -> None:
    """Keep an isolated stack's approval mode in its own folder.

    The Approvals device saves its mode (auto or approve) to mode_file and restores it on restart. Two
    stacks in one checkout would otherwise share outputs/approvals/mode.json, and one stack switched to
    approve would come back in the other after a restart. A mode_file the config sets is kept.
    """
    if isolated_dir is None:
        return
    for device in devices:
        if device.module_name == "asyncroscopy.safety.approvals" and "mode_file" not in device.properties:
            device.properties["mode_file"] = [str(isolated_dir / "approvals" / "mode.json")]


def register_devices(devices: list[DeviceConfig], instrument_properties: dict[str, list[str]]) -> None:
    database = tango.Database()
    status_line("OK", "database", f"{database.get_db_host()}:{database.get_db_port()}")

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

    instrument = selected_instrument(devices)
    for property_name, property_value in instrument_properties.items():
        database.put_device_property(instrument.device_name, {property_name: property_value})
        status_line("OK", f"property: {property_name} = {property_value[0]}")


def get_data_proxy() -> tango.DeviceProxy:
    data = tango.DeviceProxy("asyncroscopy/data/default")
    data.set_timeout_millis(TILED_COMMAND_TIMEOUT_MILLIS)
    return data


def reconcile_tiled() -> dict:
    """Register the acquisition files Tiled does not have yet (one by one; never the directory)."""
    data = get_data_proxy()
    data.set_timeout_millis(TILED_STARTUP_REGISTRATION_TIMEOUT_MILLIS)
    return json.loads(data.reconcile())


def tiled_key_warning(tiled_host: str, api_key: str) -> str | None:
    """A warning when Tiled is reachable from other machines with the well-known default API key."""
    if tiled_host.strip().lower() in {"127.0.0.1", "localhost", "::1"} or api_key != "secret":
        return None
    return (
        f"Tiled listens on {tiled_host} with the default API key 'secret': anyone on the network can read and "
        "write data. Set ASYNCROSCOPY_TILED_API_KEY (and the same key in SciAgentGUI's Tiled settings)."
    )


def stop_tiled_server() -> None:
    try:
        get_data_proxy().stop_tiled_server()
    except Exception as exc:
        status_line("WARN", "could not stop the Tiled server", str(exc).splitlines()[0] if str(exc) else type(exc).__name__)


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
    faults: dict[str, str] | None = None,
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
        if tiled_config.get("tiled_log"):
            print(f"  {color('TILED_LOG', Style.bold):<18} {tiled_config['tiled_log']}")
    print()
    if faults:
        # The servers run, but these devices did not reach their hardware: the stack is not live.
        print(color(f"NOT READY: {len(faults)} device(s) are in FAULT; the servers run, but the stack is not live:", Style.bold + Style.red))
        for name, status in faults.items():
            print(color(f"  {name}: {status}", Style.red))
        print(color("Fix the cause, then call Connect on the device or restart the stack.", Style.bold + Style.red))
        return
    print(color("All asyncroscopy servers are ready.", Style.bold + Style.green))


def main(argv: list[str] | None = None) -> int:
    # Line-buffer even when stdout is a pipe or file (a GUI, a log, CI), so "ready"
    # and the summary appear when they happen rather than when the launcher exits.
    sys.stdout.reconfigure(line_buffering=True)
    shutdown_requested = False

    def request_shutdown(_signum, _frame) -> None:
        # A GUI's Stop button (or a wrapper like `uv run` forwarding its own
        # copy of the same signal) can deliver SIGTERM more than once for a
        # single stop request. Only the first should raise: re-raising while
        # ProcessManager.shutdown_all() is already unwinding interrupts that
        # cleanup mid-flight and can leave child device servers orphaned.
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_shutdown)

    args = parse_args(argv)
    config_path = args.yaml or DEFAULT_CONFIG_PATH
    try:
        config = load_config(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(color(f"Config error: {exc}", Style.bold + Style.red))
        return 1

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
        default_tiled = urlsplit(settings.get().tiled_uri or f"http://{config.tiled.host}:{config.tiled.port}")
        tiled_host = prompt_str("Tiled HTTP host", default_tiled.hostname or config.tiled.host)
        tiled_port = prompt_int("Tiled HTTP port", default_tiled.port or config.tiled.port)
        acquisition_dir = prompt_str(
            "Acquisition save path",
            settings.get().acquisition_dir or config.tiled.acquisition_dir,
        )
        should_start_tiled = prompt_bool("Start Tiled HTTP server", config.tiled.autostart)
        should_register_tiled = prompt_bool(
            "Register acquisition files Tiled is missing on startup (can be slow)",
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
    if config.plugins:
        environment["ASYNCROSCOPY_PLUGINS"] = plugin_environment(config.plugins, environment.get("ASYNCROSCOPY_PLUGINS", ""))
    ready_times: dict[str, float] = {}
    faults: dict[str, str] = {}  # device name -> Status, for devices that answered in FAULT
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
    print(f"  {color('SIMULATED', Style.bold):<18} {str(config.simulated).lower()}" + ("" if config.simulated else "  (real hardware: approval auto mode refused)"))
    if log_dir is not None:
        print(f"  {color('DEBUG LOGS', Style.bold):<18} {log_dir}")
    print_inventory(devices)

    isolated_dir = args.isolated.resolve() if args.isolated else None
    manager_kwargs = {"echo_output": args.log_to_stdout}
    if isolated_dir is not None:
        isolated_dir.mkdir(parents=True, exist_ok=True)
        clear_first = False
        environment["PYTANGO_DATABASE_NAME"] = str(isolated_dir / "tango_database.db")
        manager_kwargs.update(name=f"run_servers_{port}", state_dir=isolated_dir / ".processes")

    snapshot = process_snapshot()
    my_pids = {os.getpid(), *ancestors(os.getpid(), snapshot)}
    ports = ([port] if start_database else []) + ([tiled_port] if should_start_tiled else []) if clear_first else []
    conflicts = stack_conflicts(
        snapshot, stack_identity(config_path, isolated_dir), my_pids,
        {p: listening_pids(p) or set() for p in ports},
        share_database_file=isolated_dir is None and start_database,
    )
    if conflicts and not args.replace:
        print(color(explain_conflicts(conflicts, config_path), Style.bold + Style.red))
        return 1
    if conflicts:
        for reason, pid in conflicts:
            print(f"  STOP  {reason} (--replace)")
            stop_tree(pid)

    try:
        with ProcessManager(**manager_kwargs) as manager:
            print_section(1, total_steps, "Clearing old processes")
            if clear_first:
                ports_to_clear = [port]
                if should_start_tiled:
                    ports_to_clear.append(tiled_port)
                manager.scour_ports(
                    ports_to_clear,
                    device_servers=[(device.module_name, device.instance_name) for device in devices],
                    tango_host=f"{host}:{port}",
                )
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
                    command=python_command("-m", "tango.databaseds.database", *(["--host", config.tango_bind_host] if config.tango_bind_host else []), "2"),
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
                isolate_approvals(devices, isolated_dir)
                register_devices(devices, registered_instrument_properties)
            else:
                status_line("SKIP", "device registration")

            print_section(4, total_steps, "Starting device servers")
            started = {}
            for device in regular_devices:
                process = manager.start_process(
                    key=device.key, 
                    label=device.class_name, 
                    command=device.command, 
                    env=environment
                )
                started[device.key] = process
                status_line("RUN", device.key, f"{device.module_name}  pid={process.pid}")

            for device in regular_devices:
                print(f"  WAIT  {device.device_name:<34}", end="", flush=True)
                elapsed = wait_for_device(device.device_name, device_timeout, started[device.key])
                ready_times[device.key] = elapsed
                report_health(device.device_name, elapsed, faults)

            if should_start_tiled:
                configured_key = settings.get().tiled_api_key
                warning = tiled_key_warning(tiled_host, "secret" if configured_key is None else configured_key)
                if warning:
                    status_line("WARN", "Tiled API key", warning)
                tiled_config = json.loads(get_data_proxy().start_tiled_server())
                if tiled_config["tiled_server"] != "yes":
                    raise RuntimeError(f"Tiled HTTP server failed to start: {tiled_config['tiled_server_status']}")
                status_line("OK", "Tiled HTTP server", f"{tiled_config['uri']} serving {tiled_config['tiled_server_serving']}")
                
                if should_register_tiled:
                    status_line("WAIT", "Tiled startup registration", "registering acquisition files Tiled is missing; this can take a while")
                    reconciled = reconcile_tiled()
                    detail = f"{len(reconciled['registered'])} registered, {reconciled['already']} already there, {len(reconciled.get('pending', []))} still being written"
                    if reconciled["failed"]:
                        detail += f", {len(reconciled['failed'])} failed (first: {reconciled['failed'][0]['error']})"
                    status_line("WARN" if reconciled["failed"] else "OK", "Tiled startup registration", detail)
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
                elapsed = wait_for_device(device.device_name, device_timeout, process)
                ready_times[device.key] = elapsed
                report_health(device.device_name, elapsed, faults)

            print_summary(
                host,
                port,
                manager.active_processes,
                ready_times,
                tiled_config,
                step=total_steps,
                total=total_steps,
                faults=faults,
            )
            print()
            print(color("Leave this terminal open while you use the servers. Press Ctrl+C to stop them.", Style.dim))
            
            try:
                while True:
                    time.sleep(0.1)
                    # A server that dies after startup stops the stack with an error, so a supervisor
                    # (systemd, Kubernetes) sees the failure instead of a launcher that looks healthy.
                    stopped = next((p for p in manager.active_processes if not p.running), None)
                    if stopped is not None:
                        raise RuntimeError(f"{stopped.label} ({stopped.key}, pid {stopped.pid}) exited with code {stopped.process.returncode} after startup")
            except KeyboardInterrupt:
                # Catch the Ctrl-C INSIDE the with block so processes are still alive!
                print()
                print(color("Shutdown requested. Stopping Tiled server...", Style.yellow))
                stop_tiled_server() 
                print(color("Stopping managed processes...", Style.yellow))
                
        status_line("OK", "shutdown complete")
        return 0

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

        # No stop_tiled_server() here: the process manager's tree kill has already ended Tiled with the data server.
            
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
