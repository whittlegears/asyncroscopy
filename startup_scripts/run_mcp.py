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
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager  # noqa: E402
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


@dataclass(frozen=True)
class Config:
    path: Path
    tango_host: str
    tango_port: int
    mcp: MCPConfig


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
        ),
    )


def build_command(config: Config) -> list[str]:
    command = [
        'uv',
        'run',
        'python',
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
    ]
    if config.mcp.quiet:
        command.append('--quiet')
    if config.mcp.tool_log:
        command.extend(['--tool-log', config.mcp.tool_log])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='MCP YAML config to start from.')
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        
    return 0


if __name__ == '__main__':
    raise SystemExit(main())