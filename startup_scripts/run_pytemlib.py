#!/usr/bin/env python
"""Start the pyTEMlib analysis MCP server (pytemlib_mcp.analysis_server: stitching, scoring, placements, notes) from an explicit YAML config."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.core.names import DEFAULT_ANALYSIS_URL
from asyncroscopy.core.net import is_loopback_host
from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager, python_command, stop_on_termination
from startup_scripts.config_schema import ConfigError, Setting, fill_placeholders, read_settings
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers, wait_for_exit

DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'pytemlib.yaml'

# stdio is left out because it only works when the client starts the server itself.
TRANSPORTS = ('streamable-http', 'sse')
SERVER_MODULE = 'pytemlib_mcp.analysis_server'
DEFAULT_HOST = urlsplit(DEFAULT_ANALYSIS_URL).hostname
DEFAULT_PORT = urlsplit(DEFAULT_ANALYSIS_URL).port
API_KEY_VARIABLE = 'ASYNCROSCOPY_TILED_API_KEY'

SERVER_DEFAULT = 'Blank: the server default.'
LOG_HELP = 'Blank: the default under outputs/. Write "" in the YAML to switch it off.'

TITLE = 'Asyncroscopy pyTEMlib MCP Startup'
LAUNCHER = 'startup_scripts/run_pytemlib.py'
# A setting left blank is not passed so the server uses its own default.
SETTINGS = [
    Setting('pytemlib.transport', default=TRANSPORTS[0], section='pyTEMlib MCP server', choices=TRANSPORTS),
    Setting('pytemlib.host', default=DEFAULT_HOST, section='pyTEMlib MCP server', label='HTTP host'),
    Setting('pytemlib.port', 'int', DEFAULT_PORT, section='pyTEMlib MCP server', label='HTTP port'),
    Setting('pytemlib.tiled', section='Data', label='Tiled URI', help='Blank: tools take local file paths.'),
    Setting('pytemlib.api_key', section='Data', label='Tiled API key', help=f'Blank: ${API_KEY_VARIABLE} or $TILED_API_KEY.'),
    Setting('pytemlib.local_roots', 'list', [], section='Data', label='Local roots', help='Folders whose HDF5 files tools may use. Blank: the project folder.'),
    Setting('pytemlib.local_runs', section='Data', label='Save maps in folder',
            help='Runs gathered from single images, and their maps, are saved here as files, not in Tiled. Blank: in Tiled.'),
    Setting('pytemlib.max_canvas_megapixels', section='Limits', label='Max canvas (megapixels)', help=SERVER_DEFAULT),
    Setting('pytemlib.max_jobs', 'int', section='Limits', label='Jobs at once', help=SERVER_DEFAULT),
    Setting('pytemlib.trace', section='Logs', label='Tool call trace (JSONL)', help=LOG_HELP),
    Setting('pytemlib.annotations', section='Logs', label='Notes (JSONL)', help=LOG_HELP),
    Setting('pytemlib.placements', section='Logs', label='Placements (JSONL)', help=LOG_HELP),
]
STATUS_PATTERNS = [
    (re.compile(r'analysis server at (\S+)'), '{0}', False),
]


def is_config(raw) -> bool:
    return isinstance(raw, dict) and isinstance(raw.get('pytemlib'), dict)


@dataclass(frozen=True)
class PytemlibConfig:
    transport: str = TRANSPORTS[0]
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    tiled: str = ''
    api_key: str = field(default='', repr=False)
    local_roots: list[str] = field(default_factory=list)
    local_runs: str = ''
    # None means the server default. An empty string turns a log off.
    max_canvas_megapixels: float | None = None
    max_jobs: int | None = None
    trace: str | None = None
    annotations: str | None = None
    placements: str | None = None


@dataclass(frozen=True)
class Config:
    path: Path
    pytemlib: PytemlibConfig


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f'{path.name} is not valid YAML: {exc}']) from exc
    raw = fill_placeholders(raw)
    if isinstance(raw, dict) and not isinstance(raw.get('pytemlib'), dict):
        raise ConfigError([f'{path.name} has no pytemlib: mapping, so it is not a config for {Path(LAUNCHER).name}'])
    values = read_settings(raw, SETTINGS)
    problems = []
    # Report unknown keys so a typo is not silently ignored.
    known = {setting.path.split('.', 1)[1] for setting in SETTINGS}
    unknown = sorted(set(raw['pytemlib']) - known) + sorted(f'(top level) {k}' for k in set(raw) - {'pytemlib'})
    if unknown:
        problems.append(f"unknown key(s) {', '.join(unknown)}; expected pytemlib: with {', '.join(sorted(known))}")
    transport = values['pytemlib.transport'] or TRANSPORTS[0]
    if transport not in TRANSPORTS:
        problems.append(f"pytemlib.transport must be one of {', '.join(TRANSPORTS)}, not {transport!r}")
    port = values['pytemlib.port'] if values['pytemlib.port'] is not None else DEFAULT_PORT
    if not 1 <= port <= 65535:
        problems.append(f'pytemlib.port must be between 1 and 65535, not {port}')
    megapixels = values['pytemlib.max_canvas_megapixels']
    if megapixels in (None, ''):
        megapixels = None
    else:
        try:
            megapixels = float(megapixels)
        except ValueError:
            megapixels = math.nan
        if not (math.isfinite(megapixels) and megapixels > 0):
            problems.append(f"pytemlib.max_canvas_megapixels must be a number above 0, not {values['pytemlib.max_canvas_megapixels']!r}")
    max_jobs = values['pytemlib.max_jobs']
    if max_jobs is not None and max_jobs < 1:
        problems.append('pytemlib.max_jobs must be at least 1')
    if problems:
        raise ConfigError(problems)
    return Config(
        path=path,
        pytemlib=PytemlibConfig(
            transport=transport,
            host=values['pytemlib.host'] or DEFAULT_HOST,
            port=port,
            tiled=values['pytemlib.tiled'] or '',
            api_key=values['pytemlib.api_key'] or '',
            local_roots=values['pytemlib.local_roots'],
            local_runs=values['pytemlib.local_runs'] or '',
            max_canvas_megapixels=megapixels,
            max_jobs=max_jobs,
            trace=values['pytemlib.trace'],
            annotations=values['pytemlib.annotations'],
            placements=values['pytemlib.placements'],
        ),
    )


def build_command(config: Config) -> list[str]:
    """The server command. The Tiled API key is passed in the environment instead."""
    server = config.pytemlib
    args = ['-m', SERVER_MODULE, '--transport', server.transport, '--host', server.host, '--port', str(server.port)]
    if server.tiled:
        args += ['--tiled', server.tiled]
    for root in server.local_roots:
        args += ['--local-root', root]
    if server.local_runs:
        args += ['--local-runs', server.local_runs]
    if server.max_canvas_megapixels is not None:
        args += ['--max-canvas-megapixels', f'{server.max_canvas_megapixels:g}']
    if server.max_jobs is not None:
        args += ['--max-jobs', str(server.max_jobs)]
    for flag, value in (('--trace', server.trace), ('--annotations', server.annotations), ('--placements', server.placements)):
        if value is not None:
            args.append(f'{flag}={value}')  # joined with = so an empty value still belongs to its flag
    return python_command(*args)


def server_env(config: Config) -> dict[str, str]:
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    if config.pytemlib.api_key:
        # Passing the key by environment keeps it out of the printed command and the process list.
        env[API_KEY_VARIABLE] = config.pytemlib.api_key
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='pyTEMlib server YAML config to start from.')
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    stop_on_termination()  # killing the launcher must also stop the server
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1
    server = config.pytemlib
    # One state file per port so cleanup only touches this server.
    state_name = f'run_pytemlib_{server.port}'
    if args.cleanup:
        with ProcessManager(name=state_name):
            pass
        print('Stopped processes left by an earlier run.')
        return 0

    command = build_command(config)

    print('Starting pyTEMlib analysis MCP server')
    print(f'  config:  {config.path}')
    print(f'  http:    http://{server.host}:{server.port}/mcp')
    print(f'  tiled:   {server.tiled or "(none: local files)"}')
    print(f'  command: {" ".join(command)}')
    if not is_loopback_host(server.host):
        print(f'warning: the pyTEMlib analysis server has no login and binds {server.host}, reachable from other machines '
              '(docs/Operation/remote_access.md).', file=sys.stderr)

    try:
        with ProcessManager(name=state_name) as manager:
            managed: ManagedProcess = manager.start_process(
                key='pytemlib_mcp',
                label='pyTEMlib MCP Server',
                command=command,
                env=server_env(config),
                stdout=None,
                stderr=None,
            )
            print('pyTEMlib MCP server started. Press Ctrl+C to terminate.')

            code = wait_for_exit(managed)
            print(f'pyTEMlib MCP server exited with code {code}', file=sys.stderr)
            return code or 1

    except KeyboardInterrupt:
        print('\nShutting down pyTEMlib MCP server...')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
