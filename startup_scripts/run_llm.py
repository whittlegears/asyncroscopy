#!/usr/bin/env python
"""Start the asyncroscopy agent MCP server from an explicit YAML config."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.utils.process_manager import ManagedProcess, ProcessManager, python_command, stop_on_termination
from asyncroscopy.core.names import endpoint_urls
from asyncroscopy.mcp.agent_tools import DEFAULT_PROFILE_NAME, PROFILES, profile_data
from asyncroscopy.core.net import is_loopback_host
from startup_scripts.config_schema import ConfigError, Setting, fill_placeholders, read_settings
from startup_scripts.lifecycle import add_lifecycle_args, install_shutdown_handlers, wait_for_exit

DEFAULT_CONFIG_PATH = PROJECT_DIR / 'configs' / 'gemma-llm.yaml'
default_profile = profile_data(PROFILES / f'{DEFAULT_PROFILE_NAME}.yaml')

AGENT_EXTRAS = ('agent', 'ollama')
AGENT_EXTRA_MODULES = ('langgraph', 'langchain_mcp_adapters', 'langchain_ollama')


def agent_extras_installed() -> bool:
    """True when the agent extras are importable here, so the child need not sync."""
    return all(importlib.util.find_spec(name) is not None for name in AGENT_EXTRA_MODULES)


TITLE = 'Asyncroscopy Agent Startup'
LAUNCHER = 'startup_scripts/run_llm.py'
SETTINGS = [
    Setting('agent.name', required=True, section='Agent MCP server', label='Name'),
    Setting('agent.transport', required=True, section='Agent MCP server', choices=('streamable-http',)),
    Setting('agent.http_host', required=True, section='Agent MCP server', label='HTTP host'),
    Setting('agent.http_port', 'int', required=True, section='Agent MCP server', label='HTTP port'),
    Setting('agent.mcp_urls', 'list', [], section='Agent MCP server', label='Tool MCP URLs'),
    Setting('agent.urls', 'yaml', {}, section='Agent MCP server', label='Endpoint URLs'),
    Setting('agent.allowed_hosts', 'list', [], section='Agent MCP server', label='Allowed hosts',
            help='Names besides the HTTP host and localhost that requests may give as Host or Origin, such as a proxy\'s.'),
    Setting('agent.model', default=default_profile['model'], section='Model'),
    Setting('agent.model_provider', default=default_profile['model_provider'], section='Model', label='Provider'),
    Setting('agent.max_steps', 'int', 10, section='Model', label='Max steps'),
    Setting('agent.use_init_chat_model', 'bool', False, section='Model', label='Use init_chat_model'),
    Setting('agent.startup_agents', 'yaml', [], section='Agents', label='Startup agents'),
    Setting('agent.workflows', 'bool', True, section='Workflows', label='Run workflows here',
            help='Workflow tools (list_workflows, run_workflow, ...) in the agent process, the one place workflows run. '
                 'On a simulated stack only, unless the next setting is on.'),
    Setting('agent.workflows_hardware', 'bool', False, section='Workflows', label='Also on the real instrument',
            help='Run workflows on a stack that is not simulated. Needs HTTP host 127.0.0.1.'),
    Setting('agent.workflow_instruments', 'list', [], section='Workflows', label='Other instruments',
            help='Instrument endpoint URLs workflows may also run on (instrument_url), such as the instrument a twin is paired with.'),
    Setting('agent.endpoint_role', default='agent', section='Agent MCP server', label='Endpoint role', choices=('agent', 'operator'),
            help='operator: only the workflow review, approve and reject tools, for a person (no swarm, no model). Needs HTTP host 127.0.0.1.'),
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
    model: str = default_profile["model"]
    model_provider: str = default_profile["model_provider"]
    max_steps: int = 10
    use_init_chat_model: bool = False
    startup_agents: list[dict] = field(default_factory=list)
    workflows: bool = True
    workflows_hardware: bool = False
    endpoint_role: str = "agent"
    allowed_hosts: list[str] = field(default_factory=list)
    urls: dict[str, str | None] = field(default_factory=endpoint_urls)
    workflow_instruments: list[str] = field(default_factory=list)


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
    raw = fill_placeholders(raw)
    values = read_settings(raw, SETTINGS)
    problems = []
    known = {setting.path.split('.', 1)[1] for setting in SETTINGS}
    unknown = sorted(set(raw.get('agent') or {}) - known) + sorted(f'(top level) {k}' for k in set(raw) - {'agent'})
    if unknown:
        problems.append(f"unknown key(s) {', '.join(unknown)}; expected agent: with {', '.join(sorted(known))}")
    if values['agent.max_steps'] is not None and values['agent.max_steps'] < 1:
        problems.append('agent.max_steps must be at least 1')
    if values['agent.workflows_hardware'] and not is_loopback_host(str(values['agent.http_host'] or '')):
        problems.append('agent.workflows_hardware needs agent.http_host 127.0.0.1: the agent server has no login, so on another '
                        'address anyone who reaches the port would run workflows on the real instrument')
    if values['agent.endpoint_role'] not in ('agent', 'operator'):
        problems.append(f"agent.endpoint_role must be agent or operator, not {values['agent.endpoint_role']!r}")
    elif values['agent.endpoint_role'] == 'operator' and not is_loopback_host(str(values['agent.http_host'] or '')):
        problems.append('agent.endpoint_role operator needs agent.http_host 127.0.0.1: it approves workflows for the real instrument '
                        'and has no login')
    try:
        urls = endpoint_urls(values['agent.urls'], 'agent.urls')
    except ValueError as exc:
        problems.append(str(exc))
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
            workflows=bool(values['agent.workflows']) or bool(values['agent.workflows_hardware']),
            workflows_hardware=bool(values['agent.workflows_hardware']),
            endpoint_role=str(values['agent.endpoint_role']),
            allowed_hosts=[str(host) for host in values['agent.allowed_hosts'] or []],
            urls=urls,
            workflow_instruments=[str(url) for url in values['agent.workflow_instruments'] or []],
        ),
    )


def build_command(config: Config) -> list[str]:
    agent = config.agent
    command = python_command(
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
        extras=AGENT_EXTRAS,
        # Skip uv's sync when the extras are installed, so this child cannot race other launchers re-syncing the venv.
        sync=not agent_extras_installed(),
    )
    if agent.endpoint_role != 'agent':
        command += ['--endpoint-role', agent.endpoint_role]
    if agent.use_init_chat_model:
        command.append('--use-init-chat-model')
    if agent.allowed_hosts:
        command += ['--allowed-hosts-json', json.dumps(agent.allowed_hosts)]
    if any(agent.urls.values()):
        command += ['--urls-json', json.dumps(agent.urls)]
    if agent.workflow_instruments:
        command += ['--workflow-instruments-json', json.dumps(agent.workflow_instruments)]
    if agent.workflows_hardware:
        command.append('--workflows-hardware')
    elif not agent.workflows:
        command.append('--no-workflows')
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
    from asyncroscopy.agents.graphs.swarm import Agent, AgentSwarm

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
    stop_on_termination()  # `kill <launcher>` must also stop the server
    args = parse_args(argv)
    install_shutdown_handlers(args.parent_pid)
    try:
        config = load_config(args.yaml)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f'Config error: {exc}', file=sys.stderr)
        return 1
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

            code = wait_for_exit(managed)
            print(f"Agent MCP server exited with code {code}", file=sys.stderr)
            return code or 1

    except KeyboardInterrupt:
        print("\nShutting down agent MCP server...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())