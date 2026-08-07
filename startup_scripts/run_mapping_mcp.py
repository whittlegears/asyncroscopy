#!/usr/bin/env python
"""Start the standalone mapping/stitching MCP server from an explicit YAML config.

Unlike run_mcp.py, this does not touch Tango at all - it can run on a
different machine than the instrument stack. See configs/mapping_mcp.yaml.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'mapping_mcp.yaml'


@dataclass(frozen=True)
class MappingConfig:
    name: str
    transport: str
    http_host: str
    http_port: int
    output_root: str
    quiet: bool


@dataclass(frozen=True)
class Config:
    path: Path
    mapping: MappingConfig


def _require(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Config section '{where}' is missing required key '{key}'")
    return mapping[key]


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    mapping = _require(raw, 'mapping', '(top level)')
    return Config(
        path=path,
        mapping=MappingConfig(
            name=_require(mapping, 'name', 'mapping'),
            transport=_require(mapping, 'transport', 'mapping'),
            http_host=_require(mapping, 'http_host', 'mapping'),
            http_port=int(_require(mapping, 'http_port', 'mapping')),
            output_root=_require(mapping, 'output_root', 'mapping'),
            quiet=bool(_require(mapping, 'quiet', 'mapping')),
        ),
    )


def build_command(config: Config) -> list[str]:
    command = [
        'uv',
        'run',
        'python',
        '-m',
        'asyncroscopy.mcp.mapping_mcp_server',
        '--name',
        config.mapping.name,
        '--transport',
        config.mapping.transport,
        '--http-host',
        config.mapping.http_host,
        '--http-port',
        str(config.mapping.http_port),
        '--output-root',
        config.mapping.output_root,
    ]
    if config.mapping.quiet:
        command.append('--quiet')
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='Mapping MCP YAML config to start from.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1

    command = build_command(config)
    print(f'Starting mapping MCP server {config.mapping.name}')
    print(f'  config: {config.path}')
    print(f'  http:   http://{config.mapping.http_host}:{config.mapping.http_port}/mcp')
    print(f'  command: {" ".join(command)}')
    return subprocess.run(command, cwd=PROJECT_DIR).returncode


if __name__ == '__main__':
    raise SystemExit(main())
