"""Supervisor/worker swarm graph.

A supervisor node asks the model which worker should act next and what its
isolated sub-task is; each worker is a ReAct agent restricted to a subset of
the MCP tools. This is the graph the LLM Tango device runs when more than one
agent has been spawned.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agent.config import Agent
from asyncroscopy.agent.state import SupervisorState

FINISH = "FINISH"
SUPERVISOR = "Supervisor"


def extract_json(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def parse_routing_decision(content: str, valid_options: Sequence[str], fallback: str) -> tuple[str, str]:
    """Parse a supervisor response's ``{"next": ..., "task": ...}`` decision.

    Falls back to ``fallback`` on any parse error or an invalid ``next`` value.
    """
    try:
        decision = json.loads(extract_json(content))
        next_agent = decision.get("next", fallback)
        subtask = decision.get("task", "")
        return (next_agent if next_agent in valid_options else fallback), subtask
    except Exception as exc:  # noqa: BLE001 - any malformed output means "fall back"
        print(f"[SUPERVISOR ERROR]: {exc}")
        return fallback, ""


def build_supervisor_prompt(agents: Sequence[Agent], options: Sequence[str]) -> str:
    roster = "\n".join(f"- {a.name}: {a.description or a.system_prompt}" for a in agents)
    return (
        "You are the Swarm Supervisor. "
        f"Below are the available agents and what each is for:\n{roster}\n\n"
        "Based on the conversation, decide which agent should act next to progress the user's request. "
        "Only output FINISH if the user's request has been fully and concretely answered — "
        "not if an agent asked a question, refused, said it lacks the ability, or otherwise failed to "
        "complete the task; in that case, route to a different, more suitable agent instead.\n"
        "Respond with JSON containing two keys:\n"
        f"1. 'next': One of {list(options)}\n"
        "2. 'task': The exact, isolated sub-task that ONLY this specific agent should perform right now. "
        "Do NOT include steps intended for other agents.\n\n"
        "Example output:\n"
        '{"next": "image", "task": "Acquire a scanned HAADF image."}'
    )


def build_supervisor_graph(
    model: Any,
    agents: Sequence[Agent],
    build_worker: Callable[[Agent], Any],
    run_worker: Callable[[Any, list, str], Awaitable[str]],
) -> Any:
    """Compile the supervisor graph.

    Args:
        model: chat model used by the supervisor for routing decisions.
        agents: worker definitions (names must be unique and not ``FINISH``).
        build_worker: ``Agent -> compiled ReAct graph`` (see ``graphs.react``).
        run_worker: ``(executor, messages, label) -> final text`` coroutine
            (see ``streaming.stream_agent``).
    """
    if not agents:
        raise ValueError("build_supervisor_graph needs at least one agent.")
    agent_names = [a.name for a in agents]
    if len(set(agent_names)) != len(agent_names) or FINISH in agent_names or SUPERVISOR in agent_names:
        raise ValueError(f"Agent names must be unique and not {FINISH!r}/{SUPERVISOR!r}: {agent_names}")
    options = agent_names + [FINISH]
    system_prompt = SystemMessage(content=build_supervisor_prompt(agents, options))

    builder = StateGraph(SupervisorState)

    def create_agent_node(agent: Agent):
        executor = build_worker(agent)

        async def node(state: SupervisorState):
            task = state.get("current_task") or "Execute assigned tool."
            print(f"\n[{agent.name}] assigned task: '{task}'")
            content = await run_worker(executor, [HumanMessage(content=task)], agent.name)
            print(f"[{agent.name}] finished.\n")
            return {"messages": [HumanMessage(content=f"[{agent.name}]: {content}", name=agent.name)]}

        return node

    for agent in agents:
        builder.add_node(agent.name, create_agent_node(agent))

    async def supervisor_node(state: SupervisorState):
        print("\n[Supervisor] Evaluating routing...")
        has_delegated = len(state["messages"]) > 1
        if not has_delegated:
            valid_options, fallback = agent_names, agent_names[0]
        else:
            valid_options, fallback = options, FINISH
        response = await model.ainvoke([system_prompt] + list(state["messages"]))
        content = response.content if isinstance(response.content, str) else str(response.content)
        next_agent, subtask = parse_routing_decision(content, valid_options, fallback)
        if next_agent == FINISH:
            print("[Supervisor] Decision: FINISH\n")
        return {"next_agent": next_agent, "current_task": subtask}

    builder.add_node(SUPERVISOR, supervisor_node)
    builder.add_edge(START, SUPERVISOR)
    for name in agent_names:
        builder.add_edge(name, SUPERVISOR)

    def route(state: SupervisorState) -> str:
        return state["next_agent"]

    mapping = {name: name for name in agent_names}
    mapping[FINISH] = END
    builder.add_conditional_edges(SUPERVISOR, route, mapping)
    return builder.compile()


async def run_supervisor(graph: Any, prompt: str, max_steps: int = 10) -> str:
    """Stream the swarm for ``prompt`` and return the last worker's response."""
    print(f"\n{'=' * 50}\n[NEW REQUEST]: {prompt}\n{'=' * 50}")
    last_response: str | None = None
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": max_steps},
    ):
        for node_name, state_update in chunk.items():
            if node_name != SUPERVISOR and state_update and "messages" in state_update:
                last_response = state_update["messages"][-1].content
    if last_response is None:
        return "Swarm Error: No agent produced a response before routing finished."
    return last_response
