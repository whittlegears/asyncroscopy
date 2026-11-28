#!/usr/bin/env python
"""Start the asyncroscopy MCP server from an explicit YAML config."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.core.names import endpoint_urls  # noqa: E402
from asyncroscopy.mcp.endpoint_auth import AuthSettings, is_loopback, remote_without_auth  # noqa: E402
from asyncroscopy.safety.permissions import default_approval_ttl_s  # noqa: E402
from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager, listening_pids, python_command, stop_on_termination  # noqa: E402
from startup_scripts.config_schema import ConfigError, Setting, fill_placeholders, read_settings  # noqa: E402
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers  # noqa: E402

# The config the GUI opens first. The launcher itself needs --yaml: it never picks an instrument for you.
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'mcp.yaml'
YAML_REQUIRED = ('--yaml is required: name the MCP config of the stack to serve, e.g. configs/mcp_dt.yaml for the '
                 'digital twin or configs/mcp.yaml for the Spectra 300.')


def host_presets() -> dict[str, str]:
    """Tango hosts this project points at day to day. The GUI opens on the first, the digital twin, so a fresh launch
    never talks to the real instrument. The Spectra 300's address is read from its stack YAML, the one place it is
    written; without that file there is no such preset."""
    presets = {'Digital twin': 'localhost'}
    try:
        raw = yaml.safe_load((PROJECT_DIR / 'configs' / 'Spectra300.yaml').read_text(encoding='utf-8')) or {}
        host = fill_placeholders(raw)['tango']['host']
    except (OSError, yaml.YAMLError, KeyError, TypeError, ConfigError):  # ConfigError: no site value for the address
        return presets
    if host:
        presets['Spectra300'] = str(host)
    return presets


HOST_PRESETS = host_presets()

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
    Setting('mcp.needs_approval', 'yaml', {}, section='Access control', label='Needs approval'),
]
# Lines mcp_server.py prints once tools are registered, or when it cannot start.
OPERATOR_HOST = '127.0.0.1'
STATUS_PATTERNS = [
    (re.compile(r'MCP ready: (\d+) tool'), '{0} tools', False),
    (re.compile(r'MCP ERROR: .*in use'), 'port in use', True),
    (re.compile(r'MCP ERROR: '), 'server error', True),
]


def is_config(raw) -> bool:
    # A server config may carry an mcp stub. It is listed with the server configs.
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
    needs_approval: dict[str, list[str]]
    include_only_functions: list[str] = field(default_factory=list)
    tool_log: str = ''
    endpoint_role: str = 'agent'
    risk: dict[str, str] = field(default_factory=dict)
    # mcp.safety.auto_routine is what auto mode runs without a person. None means what the devices declare.
    auto_routine: dict[str, list[str]] | None = None
    agent_profile: str = ''  # agent profile YAML whose tool selection endpoint_info reports. Empty means the default.
    trace: str = ''
    clock_device: str = ''  # instrument whose clock_s stamps trace events. Empty means this computer's time.
    trace_follow: list[str] = field(default_factory=list)  # what the operator endpoint's trace_since reads. Empty means trace's directory.
    data_views: list[str] = field(default_factory=list)
    urls: dict[str, str | None] = field(default_factory=endpoint_urls)
    permissions_file: str = ''
    approval_ttl_s: float = default_approval_ttl_s
    job_limits: dict[str, float] = field(default_factory=dict)
    # http_host 0.0.0.0 on a stack that is not simulated binds 127.0.0.1 unless this is true. The devices' simulated property says which it is.
    allow_remote_bind: bool = False
    # Bearer-token verification by the facility's token service. None means no token check. See docs/Operation/remote_access.md.
    auth: AuthSettings | None = None
    # Port of the operator endpoint started beside this agent endpoint. None starts no operator endpoint.
    operator_port: int | None = None
    # Names (besides the bind host, localhost and, on 0.0.0.0, this computer) that a request's Host and Origin may give,
    # such as the name a proxy or another machine reaches the endpoint by. Anything else is refused (asyncroscopy.mcp.host_guard).
    allowed_hosts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    path: Path
    tango_host: str
    tango_port: int
    mcp: MCPConfig


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


def _host_list(value, path) -> list[str]:
    if value in (None, ''):
        return []
    if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    raise ValueError(f"{path}: mcp.allowed_hosts must be a list of host names or addresses, not {value!r}")


def _operator_port(mcp: dict, path) -> int | None:
    port = mcp.get('operator_port')
    if port in (None, ''):
        return None
    if str(mcp.get('endpoint_role', 'agent')) != 'agent':
        raise ValueError(f"{path}: mcp.operator_port belongs on an agent endpoint, not on a {mcp.get('endpoint_role')} endpoint")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 or port == mcp.get('http_port'):
        raise ValueError(f"{path}: mcp.operator_port must be a port from 1 to 65535 other than mcp.http_port, not {port!r}")
    return port


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f'{path.name} is not valid YAML: {exc}']) from exc
    raw = fill_placeholders(raw)
    values = read_settings(raw, SETTINGS)
    if 'blocked_functions' in raw['mcp']:
        raise ConfigError([f'{path.name}: mcp.blocked_functions is now mcp.needs_approval: what it lists is held for a person, not hidden'])
    if 'analysis_url' in raw['mcp']:
        raise ConfigError([f'{path.name}: mcp.analysis_url is now mcp.urls.analysis'])
    needs_approval = values['mcp.needs_approval'] or {}
    if not all(isinstance(names, list) for names in needs_approval.values()):
        raise ConfigError(['mcp.needs_approval must map a class (or "*") to a list of command names'])
    # Settings the GUI has no widget for, such as safety, trace and auth, are read straight from the YAML.
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
            needs_approval={key: [str(name) for name in names] for key, names in needs_approval.items()},
            include_only_functions=values['mcp.include_only_functions'],
            tool_log=values['mcp.tool_log'] or '',
            endpoint_role=str(mcp.get('endpoint_role', 'agent')),
            risk={str(k): str(v) for k, v in ((mcp.get('safety') or {}).get('risk') or {}).items()},
            auto_routine=_auto_routine(mcp, path),
            agent_profile=str(mcp.get('agent_profile') or ''),
            trace=str(mcp.get('trace') or ''),
            clock_device=str(mcp.get('clock_device') or ''),
            trace_follow=_module_list(mcp.get('trace_follow'), 'mcp.trace_follow'),
            data_views=_module_list(mcp.get('data_views'), 'mcp.data_views'),
            urls=endpoint_urls(mcp.get('urls'), f'{path}: mcp.urls'),
            permissions_file=str(mcp.get('permissions_file') or ''),
            approval_ttl_s=float(mcp.get('approval_ttl_s') or default_approval_ttl_s),
            job_limits={str(k): float(v) for k, v in (mcp.get('job_limits') or {}).items() if v not in (None, '')},
            allow_remote_bind=mcp.get('allow_remote_bind') is True,
            # A relative public_key_file is read from the project directory. Dev tokens live under outputs/dev_tokens.
            auth=AuthSettings.from_config(mcp.get('auth'), base_dir=PROJECT_DIR),
            operator_port=_operator_port(mcp, path),
            allowed_hosts=_host_list(mcp.get('allowed_hosts'), path),
        ),
    )


def remote_auth_problem(config: Config, simulated: Callable[[str, int, str | None], bool] | None = None) -> str | None:
    """Why this endpoint must not start: a stack that is not simulated, reachable from other machines,
    with no mcp.auth. ``simulated``: a callable (tango_host, tango_port, unused) -> bool; by
    default the stack's devices are asked, as the server itself does."""
    mcp = config.mcp
    if mcp.auth is not None or mcp.transport != 'streamable-http' or is_loopback(mcp.http_host):
        return None
    if simulated is None:
        from asyncroscopy.mcp.mcp_server import stack_simulated as simulated
    is_simulated = bool(simulated(config.tango_host, config.tango_port, None))
    from asyncroscopy.core.net import bind_host

    bound, _note = bind_host(mcp.http_host, mcp.allow_remote_bind, is_simulated)
    return remote_without_auth(bound, is_simulated, mcp.auth)


def operator_config(config: Config) -> Config | None:
    """The operator endpoint that starts beside an agent endpoint, or None when mcp.operator_port is not set.

    It serves the same stack on 127.0.0.1 only and has no login. It offers what a person needs and an agent must
    not have, such as ground-truth exports.
    """
    mcp = config.mcp
    if mcp.operator_port is None:
        return None
    trace_dir = Path(mcp.trace).parent if mcp.trace else None
    return replace(config, mcp=replace(
        mcp,
        name=f'{mcp.name}_Operator',
        http_host=OPERATOR_HOST,
        http_port=mcp.operator_port,
        endpoint_role='operator',
        needs_approval={},
        include_only_functions=[],
        tool_log='',
        agent_profile='',
        trace=(trace_dir / 'mcp_operator.jsonl').as_posix() if trace_dir else '',
        trace_follow=[trace_dir.as_posix()] if trace_dir else [],
        allow_remote_bind=False,
        auth=None,
        operator_port=None,
        allowed_hosts=[],
    ))


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
        '--needs-approval-json',
        json.dumps(config.mcp.needs_approval),
        '--include-only-functions-json',
        json.dumps(config.mcp.include_only_functions),
    )
    if config.mcp.quiet:
        command.append('--quiet')
    if config.mcp.tool_log:
        command.extend(['--tool-log', config.mcp.tool_log])
    command.extend(['--endpoint-role', config.mcp.endpoint_role])
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
    if any(config.mcp.urls.values()):
        command.extend(['--urls-json', json.dumps(config.mcp.urls)])
    if config.mcp.permissions_file:
        command.extend(['--permissions-file', config.mcp.permissions_file])
    if config.mcp.approval_ttl_s != default_approval_ttl_s:
        command.extend(['--approval-ttl-s', str(config.mcp.approval_ttl_s)])
    if config.mcp.job_limits:
        command.extend(['--job-limits-json', json.dumps(config.mcp.job_limits)])
    if config.mcp.allow_remote_bind:
        command.append('--allow-remote-bind')
    if config.mcp.auth is not None:
        command.extend(['--auth-json', config.mcp.auth.to_json()])
    if config.mcp.allowed_hosts:
        command.extend(['--allowed-hosts-json', json.dumps(config.mcp.allowed_hosts)])
    return command


def start_operator(manager: ProcessManager, operator: Config | None, env: dict[str, str]) -> None:
    if operator is None:
        return
    port = operator.mcp.http_port
    if listening_pids(port):
        print(f'  operator endpoint: port {port} is already in use, so none is started here')
        return
    manager.start_process(key=operator.mcp.name, label=f'MCP Server ({operator.mcp.name})', command=build_command(operator),
                          env=env, stdout=None, stderr=None)
    print(f'  operator endpoint for the GUI: http://{operator.mcp.http_host}:{port}/mcp')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=None, metavar='PATH', help='MCP YAML config to start from (required).')
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    stop_on_termination()  # `kill <launcher>` must also stop the server it started
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    if args.yaml is None:
        print(f'Config error: {YAML_REQUIRED}', file=sys.stderr)
        return 1
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
            start_operator(manager, operator_config(config), env)
            print("MCP server started. Press Ctrl+C to terminate.")
            
            # Loop with a timeout so Python catches SIGINT (Ctrl+C) on Windows
            while managed.running:
                try:
                    managed.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            # The server stopped on its own, from a taken port, unreachable Tango or a crash. Fail so a
            # supervisor such as systemd or Kubernetes sees it. Its own output went straight to this stdout/stderr.
            code = managed.process.returncode
            print(f'MCP server {config.mcp.name} exited with code {code}', file=sys.stderr)
            return code or 1

    except KeyboardInterrupt:
        print("\nShutting down MCP server...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())