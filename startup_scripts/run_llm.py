#!/usr/bin/env python
"""Start the asyncroscopy agent MCP server from an explicit YAML config."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'

TITLE = 'Asyncroscopy Agent Startup'
LAUNCHER = 'startup_scripts/run_llm.py'
SETTINGS = [
    Setting('agent.name', required=True, section='Agent MCP server', label='Name'),
    Setting('agent.transport', required=True, section='Agent MCP server', choices=('streamable-http',)),
    Setting('agent.http_host', required=True, section='Agent MCP server', label='HTTP host'),
    Setting('agent.http_port', 'int', required=True, section='Agent MCP server', label='HTTP port'),
    Setting('agent.mcp_urls', 'list', [], section='Agent MCP server', label='Tool MCP URLs'),
    Setting('agent.model', default='gemma4:31b', section='Model'),
    Setting('agent.model_provider', default='ollama', section='Model', label='Provider'),
    Setting('agent.max_steps', 'int', 10, section='Model', label='Max steps'),
    Setting('agent.use_init_chat_model', 'bool', False, section='Model', label='Use init_chat_model'),
    Setting('agent.startup_agents', 'yaml', [], section='Agents', label='Startup agents'),
]


def is_config(raw) -> bool:
    return isinstance(raw, dict) and isinstance(raw.get('agent'), dict)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    transport: str
    http_host: str
    http_port: int
    mcp_urls: list[str] = field(default_factory=list)
    model: str = "gemma4:31b"
    model_provider: str = "ollama"
    max_steps: int = 10
    use_init_chat_model: bool = False
    # Kept as plain dicts so this module never imports the agent extras, which
    # keeps --help and config errors working without them installed.
    startup_agents: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    path: Path
    agent: AgentConfig


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError([f'{path.name} is not valid YAML: {exc}']) from exc
    values = read_settings(raw, SETTINGS)
    problems = []
    # Every key lives under agent:, so a typo there would otherwise be silently ignored.
    known = {setting.path.split('.', 1)[1] for setting in SETTINGS}
    unknown = sorted(set(raw.get('agent') or {}) - known) + sorted(f'(top level) {k}' for k in set(raw) - {'agent'})
    if unknown:
        problems.append(f"unknown key(s) {', '.join(unknown)}; expected agent: with {', '.join(sorted(known))}")
    if values['agent.max_steps'] is not None and values['agent.max_steps'] < 1:
        problems.append('agent.max_steps must be at least 1')
    startup_agents = values['agent.startup_agents'] or []
    if not isinstance(startup_agents, list) or not all(isinstance(e, dict) and e.get('name') for e in startup_agents):
        problems.append('agent.startup_agents must be a list of agents, each with a name')
    if problems:
        raise ConfigError(problems)
    return Config(
        path=path,
        agent=AgentConfig(
            name=values['agent.name'],
            transport=values['agent.transport'],
            http_host=values['agent.http_host'],
            http_port=values['agent.http_port'],
            mcp_urls=[str(url) for url in values['agent.mcp_urls']],
            model=values['agent.model'],
            model_provider=values['agent.model_provider'],
            max_steps=values['agent.max_steps'],
            use_init_chat_model=values['agent.use_init_chat_model'],
            startup_agents=[dict(entry) for entry in startup_agents],
        ),
    )


def build_command(config: Config) -> list[str]:
    agent = config.agent
    command = [
        'uv',
        'run',
        '--extra',
        'agent',
        '--extra',
        'ollama',
        'python',
        '-m',
        'asyncroscopy.mcp.agent_mcp_server',
        '--name',
        agent.name,
        '--transport',
        agent.transport,
        '--http-host',
        agent.http_host,
        '--http-port',
        str(agent.http_port),
        '--mcp-urls-json',
        json.dumps(agent.mcp_urls),
        '--model',
        agent.model,
        '--model-provider',
        agent.model_provider,
        '--max-steps',
        str(agent.max_steps),
        '--startup-agents-json',
        json.dumps(agent.startup_agents),
    ]
    if agent.use_init_chat_model:
        command.append('--use-init-chat-model')
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=DEFAULT_CONFIG_PATH, metavar='PATH', help='Agent YAML config to start from.')
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help='Query the swarm in this terminal instead of serving it over MCP.',
    )
    add_lifecycle_args(parser)
    return parser.parse_args(argv)


def run_interactive(config: Config) -> int:
    """Run the swarm in this process and prompt it from stdin, without serving MCP."""
    from asyncroscopy.mcp.agent import Agent, AgentSwarm

    swarm = AgentSwarm(
        model=config.agent.model,
        model_provider=config.agent.model_provider,
        max_steps=config.agent.max_steps,
        startup_agents=[Agent(**entry) for entry in config.agent.startup_agents],
        use_init_chat_model=config.agent.use_init_chat_model,
    )
    asyncio.run(swarm.start(config.agent.mcp_urls))

    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        prompt = input("Agent Prompt (or 'exit'): ")
        if prompt.lower() == 'exit':
            break
        print(f"Response: {asyncio.run(swarm.query(prompt))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1
    # Its own state file, so a cleanup never touches the server stack or another agent.
    state_name = f"run_llm_{config.agent.name}_{config.agent.http_port}"
    if args.cleanup:
        with ProcessManager(name=state_name):
            pass
        print('Stopped processes left by an earlier run.')
        return 0

    if args.interactive:
        return run_interactive(config)

    command = build_command(config)

    print(f'Starting agent MCP server {config.agent.name}')
    print(f'  config:  {config.path}')
    print(f'  http:    http://{config.agent.http_host}:{config.agent.http_port}/mcp')
    print(f'  tools:   {", ".join(config.agent.mcp_urls) or "(none)"}')
    print(f'  command: {" ".join(command)}')

    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}

    try:
        with ProcessManager(name=state_name) as manager:
            managed: ManagedProcess = manager.start_process(
                key="agent_mcp",
                label=f"Agent MCP Server ({config.agent.name})",
                command=command,
                env=env,
                stdout=None,
                stderr=None,
            )
            print("Agent MCP server started. Press Ctrl+C to terminate.")

            # Loop with a timeout so Python catches SIGINT (Ctrl+C) on Windows
            while managed.running:
                try:
                    managed.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass

    except KeyboardInterrupt:
        print("\nShutting down agent MCP server...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())