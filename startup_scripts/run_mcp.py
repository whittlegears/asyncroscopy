#!/usr/bin/env python
"""Start the asyncroscopy MCP server from an explicit YAML config."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.mcp.endpoint_auth import AuthSettings, is_loopback, remote_without_auth  # noqa: E402
from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager, python_command, stop_on_termination  # noqa: E402
from startup_scripts.config_schema import ConfigError, Setting, read_settings  # noqa: E402
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers  # noqa: E402

DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'mcp.yaml'

# Tango hosts this project points at day to day; the GUI opens on the first (the digital twin)
# so a fresh launch never talks to the real instrument.
HOST_PRESETS = {'Digital twin': 'localhost', 'Spectra300': '10.46.217.241'}

TITLE = 'Asyncroscopy MCP Startup'
LAUNCHER = 'startup_scripts/run_mcp.py'
SETTINGS = [
    Setting('tango.host', required=True, section='Database', label='Tango host', presets=HOST_PRESETS),
    Setting('tango.port', 'int', required=True, section='Database', label='Tango port'),
    Setting('mcp.name', required=True, section='MCP server', label='Name'),
    Setting('mcp.transport', required=True, section='MCP server', choices=('streamable-http',)),
    Setting('mcp.http_host', required=True, section='MCP server', label='HTTP host'),
    Setting('mcp.http_port', 'int', required=True, section='MCP server', label='HTTP port'),
    Setting('mcp.quiet', 'bool', True, section='MCP server', label='Quiet mode'),
    Setting('mcp.tool_log', section='MCP server', label='Tool call log (JSONL)', help='Blank disables it.'),
    Setting('mcp.data_device_address', required=True, section='Data access', label='DATA device'),
    Setting('mcp.blocked_classes', 'list', [], section='Access control'),
    Setting('mcp.include_only_functions', 'list', [], section='Access control', help='Blank: every non-blocked command.'),
    Setting('mcp.blocked_functions', 'yaml', {}, section='Access control', label='Blocked functions'),
]
# Lines mcp_server.py prints once tools are registered, or when it cannot start.
STATUS_PATTERNS = [
    (re.compile(r'MCP ready: (\d+) tool'), '{0} tools', False),
    (re.compile(r'MCP ERROR: .*in use'), 'port in use', True),
    (re.compile(r'MCP ERROR: '), 'server error', True),
]


def is_config(raw) -> bool:
    # A server config may carry an mcp: stub; it is listed with the server configs.
    return isinstance(raw, dict) and isinstance(raw.get('mcp'), dict) and 'instrument' not in raw


@dataclass(frozen=True)
class MCPConfig:
    name: str
    transport: str
    http_host: str
    http_port: int
    data_device_address: str
    quiet: bool
    blocked_classes: list[str]
    blocked_functions: dict[str, list[str]]
    include_only_functions: list[str] = field(default_factory=list)
    tool_log: str = ''
    endpoint_role: str = 'agent'
    approvals_device_address: str = ''
    risk: dict[str, str] = field(default_factory=dict)
    # mcp.safety.auto_routine: what auto mode runs without a person; None: what the devices declare.
    auto_routine: dict[str, list[str]] | None = None
    agent_profile: str = ''  # agent profile YAML whose tool selection endpoint_info reports; empty: the default
    trace: str = ''
    clock_device: str = ''  # instrument whose clock_s stamps trace events; empty: this computer's time
    trace_follow: list[str] = field(default_factory=list)  # what the operator endpoint's trace_since reads; empty: trace's directory
    data_views: list[str] = field(default_factory=list)
    analysis_url: str = ''
    job_limits: dict[str, float] = field(default_factory=dict)
    # http_host 0.0.0.0 on a stack that is not simulated (its Approvals device says) binds 127.0.0.1 unless this is true.
    allow_remote_bind: bool = False
    # Bearer-token verification (WG1's token service); None: no token check (docs/Operation/remote_access.md).
    auth: AuthSettings | None = None


@dataclass(frozen=True)
class Config:
    path: Path
    tango_host: str
    tango_port: int
    mcp: MCPConfig


def _approvals_address(mcp: dict, path) -> str:
    """The agent endpoint's approvals device. Without one every high-risk call runs unasked, so an agent
    endpoint must name one (mcp.safety.approvals_device_address) or say allow_ungated: true on purpose."""
    safety = mcp.get('safety') or {}
    address = str(safety.get('approvals_device_address') or '')
    if not address and str(mcp.get('endpoint_role', 'agent')) == 'agent' and safety.get('allow_ungated') is not True:
        raise ValueError(
            f"{path}: this agent endpoint names no approvals device, so agents could move hardware with no way for a person to hold a call. "
            "Add the Approvals device to the run_servers config and set mcp.safety.approvals_device_address: asyncroscopy/approvals/default, "
            "or, for a simulator only, mcp.safety.allow_ungated: true."
        )
    return address


def _auto_routine(mcp: dict, path) -> dict[str, list[str]] | None:
    """mcp.safety.auto_routine, checked; None when the config leaves it out (what the devices declare applies)."""
    raw = (mcp.get('safety') or {}).get('auto_routine')
    if raw is None:
        return None
    from asyncroscopy.safety.policy import check_auto_routine

    return check_auto_routine(raw, f"{path}: mcp.safety")


def _module_list(value, where: str) -> list[str]:
    """One module name or a list of them."""
    if value is None or value == '':
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"{where} must be a module name or a list of module names, not {value!r}")


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f'{path.name} is not valid YAML: {exc}']) from exc
    values = read_settings(raw, SETTINGS)
    blocked_functions = values['mcp.blocked_functions'] or {}
    if not all(isinstance(names, list) for names in blocked_functions.values()):
        raise ConfigError(['mcp.blocked_functions must map a class (or "*") to a list of command names'])
    # Settings the GUI has no widget for (safety, trace, auth, ...) are read straight from the YAML.
    mcp = raw['mcp']
    return Config(
        path=path,
        tango_host=values['tango.host'],
        tango_port=values['tango.port'],
        mcp=MCPConfig(
            name=values['mcp.name'],
            transport=values['mcp.transport'],
            http_host=values['mcp.http_host'],
            http_port=values['mcp.http_port'],
            data_device_address=values['mcp.data_device_address'],
            quiet=values['mcp.quiet'],
            blocked_classes=values['mcp.blocked_classes'],
            blocked_functions={key: [str(name) for name in names] for key, names in blocked_functions.items()},
            include_only_functions=values['mcp.include_only_functions'],
            tool_log=values['mcp.tool_log'] or '',
            endpoint_role=str(mcp.get('endpoint_role', 'agent')),
            approvals_device_address=_approvals_address(mcp, path),
            risk={str(k): str(v) for k, v in ((mcp.get('safety') or {}).get('risk') or {}).items()},
            auto_routine=_auto_routine(mcp, path),
            agent_profile=str(mcp.get('agent_profile') or ''),
            trace=str(mcp.get('trace') or ''),
            clock_device=str(mcp.get('clock_device') or ''),
            trace_follow=_module_list(mcp.get('trace_follow'), 'mcp.trace_follow'),
            data_views=_module_list(mcp.get('data_views'), 'mcp.data_views'),
            analysis_url=str(mcp.get('analysis_url') or ''),
            job_limits={str(k): float(v) for k, v in (mcp.get('job_limits') or {}).items() if v not in (None, '')},
            allow_remote_bind=mcp.get('allow_remote_bind') is True,
            # A relative public_key_file is read from the project directory (outputs/dev_tokens/... for dev tokens).
            auth=AuthSettings.from_config(mcp.get('auth'), base_dir=PROJECT_DIR),
        ),
    )


def remote_auth_problem(config: Config, simulated: Callable[[str, int, str | None], bool] | None = None) -> str | None:
    """Why this endpoint must not start: a stack that is not simulated, reachable from other machines,
    with no mcp.auth. ``simulated``: a callable (tango_host, tango_port, approvals address) -> bool; by
    default the stack's Approvals device is asked, as the server itself does."""
    mcp = config.mcp
    if mcp.auth is not None or mcp.transport != 'streamable-http' or is_loopback(mcp.http_host):
        return None
    if simulated is None:
        from asyncroscopy.mcp.mcp_server import stack_simulated as simulated
    is_simulated = bool(simulated(config.tango_host, config.tango_port, mcp.approvals_device_address or None))
    from asyncroscopy.core.net import bind_host

    bound, _note = bind_host(mcp.http_host, mcp.allow_remote_bind, is_simulated)
    return remote_without_auth(bound, is_simulated, mcp.auth)


def build_command(config: Config) -> list[str]:
    command = python_command(
        '-m',
        'asyncroscopy.mcp.mcp_server',
        '--name',
        config.mcp.name,
        '--tango-host',
        config.tango_host,
        '--tango-port',
        str(config.tango_port),
        '--transport',
        config.mcp.transport,
        '--http-host',
        config.mcp.http_host,
        '--http-port',
        str(config.mcp.http_port),
        '--data-device-address',
        config.mcp.data_device_address,
        '--blocked-classes-json',
        json.dumps(config.mcp.blocked_classes),
        '--blocked-functions-json',
        json.dumps(config.mcp.blocked_functions),
        '--include-only-functions-json',
        json.dumps(config.mcp.include_only_functions),
    )
    if config.mcp.quiet:
        command.append('--quiet')
    if config.mcp.tool_log:
        command.extend(['--tool-log', config.mcp.tool_log])
    command.extend(['--endpoint-role', config.mcp.endpoint_role])
    if config.mcp.approvals_device_address:
        command.extend(['--approvals-device-address', config.mcp.approvals_device_address])
    if config.mcp.risk:
        command.extend(['--risk-json', json.dumps(config.mcp.risk)])
    if config.mcp.auto_routine is not None:
        command.extend(['--auto-routine-json', json.dumps(config.mcp.auto_routine)])
    if config.mcp.agent_profile:
        command.extend(['--agent-profile', config.mcp.agent_profile])
    if config.mcp.trace:
        command.extend(['--trace', config.mcp.trace])
    if config.mcp.clock_device:
        command.extend(['--clock-device', config.mcp.clock_device])
    if config.mcp.trace_follow:
        command.extend(['--trace-follow-json', json.dumps(config.mcp.trace_follow)])
    if config.mcp.data_views:
        command.extend(['--data-views-json', json.dumps(config.mcp.data_views)])
    if config.mcp.analysis_url:
        command.extend(['--analysis-url', config.mcp.analysis_url])
    if config.mcp.job_limits:
        command.extend(['--job-limits-json', json.dumps(config.mcp.job_limits)])
    if config.mcp.allow_remote_bind:
        command.append('--allow-remote-bind')
    if config.mcp.auth is not None:
        command.extend(['--auth-json', config.mcp.auth.to_json()])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='MCP YAML config to start from.')
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    stop_on_termination()  # `kill <launcher>` must also stop the server it started
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1
    state_name = f"run_mcp_{config.mcp.name}_{config.mcp.http_port}"
    if args.cleanup:
        with ProcessManager(name=state_name):
            pass
        print('Stopped processes left by an earlier run.')
        return 0
    refused = remote_auth_problem(config)
    if refused:
        print(f'Config error: {config.path}: {refused}', file=sys.stderr)
        return 1

    env = {**os.environ, 'TANGO_HOST': f'{config.tango_host}:{config.tango_port}', 'PYTHONUNBUFFERED': '1'}
    command = build_command(config)
    
    print(f'Starting MCP server {config.mcp.name}')
    print(f'  config: {config.path}')
    print(f'  tango:  {env["TANGO_HOST"]}')
    print(f'  http:   http://{config.mcp.http_host}:{config.mcp.http_port}/mcp')
    print(f'  command: {" ".join(command)}')

    try:
        # One state file per name and port so bridges on other ports survive.
        with ProcessManager(name=state_name) as manager:
            managed: ManagedProcess = manager.start_process(
                key=config.mcp.name,
                label=f"MCP Server ({config.mcp.name})",
                command=command,
                env=env,
                stdout=None,
                stderr=None,
            )
            print("MCP server started. Press Ctrl+C to terminate.")
            
            # Loop with a timeout so Python catches SIGINT (Ctrl+C) on Windows
            while managed.running:
                try:
                    managed.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            # The server stopped on its own (port taken, Tango unreachable, a crash): fail, so a
            # supervisor (systemd, Kubernetes) sees it. Its own output went straight to this stdout/stderr.
            code = managed.process.returncode
            print(f'MCP server {config.mcp.name} exited with code {code}', file=sys.stderr)
            return code or 1

    except KeyboardInterrupt:
        print("\nShutting down MCP server...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())