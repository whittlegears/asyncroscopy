"""Code-defined LangGraph agents for asyncroscopy.

``graphs.mapping`` is a fixed workflow (plan -> reserve -> approval -> run ->
stitch -> score -> report) where the language model only writes the plan
and the report; ``graphs.general`` is a tool-using agent; ``graphs.team`` is a
planner with an operator and an analyst agent, each holding only its role's
tools. ``skill_middleware`` gives every tool-using agent the skills its
request and its progress call for; ``claims`` checks a final answer against
the run's tool calls and gives the model one chance to fix it; ``memory`` keeps Markdown notes it reads
(INSTRUMENT.md, LESSONS.md, experiment pages); ``learning`` drafts skills from
runs that met their goal, for an operator to promote (``python -m
asyncroscopy.agents.skills``). ``goals`` and ``graphs.goal`` check a goal from
the servers' replies and retry with feedback within a budget. The graphs are
listed in langgraph.json; docs/Operation/agents.md describes all of this.

Configuration comes from environment variables, so Studio, scripts and the agent MCP server share it
(``asyncroscopy.settings``; every key is in docs/Operation/settings.md).

Read by the stack rather than the agents (listed so every ASYNCROSCOPY_* setting
is in one place):

============================ ====================================== ===================================
Variable                     Meaning                                Default
============================ ====================================== ===================================
ASYNCROSCOPY_TILED_URI       Tiled server the DATA device runs      set by run_servers.py from the YAML
ASYNCROSCOPY_ACQUISITION_DIR directory DATA writes and Tiled serves set by run_servers.py from the YAML
ASYNCROSCOPY_TILED_API_KEY   Tiled API key (DATA, segmentation)     secret
============================ ====================================== ===================================
"""

from .config import AgentConfig

__all__ = ["AgentConfig"]
