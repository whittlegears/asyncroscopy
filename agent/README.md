# The agent's skills and workflows

This folder holds what people edit to change what the agent knows and does. The engine that reads it is the `asyncroscopy` package; nothing here needs Python packaging to work.

```
agent/
├── skills/<name>/SKILL.md    an instruction manual for one kind of task
└── workflows/<name>.py       a fixed procedure: the same steps every time
```

- A **skill** is text. It tells the agent how to do one kind of task on this kind of instrument: which tool or workflow to use, what the replies mean and what to report. It is put in the model's prompt and never runs anything itself.
- A **workflow** is a Python module that wires blocks into a LangGraph `StateGraph`. It runs its steps exactly, checked against the live endpoint first. A skill says when to run it and where its answer is.

The folder is found through one setting, `ASYNCROSCOPY_AGENT_DIR` (`settings.agent_dir`). Unset, it is this folder in a checkout, or the copy built into an installed wheel. Point it elsewhere to use another set of skills and workflows.

## Add or edit a skill

Copy `docs/Operation/templates/SKILL.md` to `skills/<name>/SKILL.md`, then edit its front matter and steps. Check it with:

```bash
uv run python -m asyncroscopy.agents.skills lint skills/<name>/SKILL.md
uv run python -m asyncroscopy.agents.skills list
```

The front matter:

```markdown
---
name: tilt-series
description: One line saying when to use it; matched against requests.
version: 1.0.0
enabled: true                 # false switches it off
keywords: [tilt series, tomography, tilt]  # phrases a request would contain; the strongest match signal
min_asyncroscopy: 0.1.0      # optional: refuse to load on older installs
envelope_schema: 1.0.0       # optional: the job envelope major version it was written for
requires: [Mapper*_reserve]   # tool patterns the endpoint must offer; otherwise the skill is not offered
unless: [Tomo*_reserve]       # optional: not offered where these are, because another skill does the job there
tools: [run_job, wait_for_job] # tools it uses: an agent holding none of them is not given it
acts: true                    # following it moves or exposes the instrument; false for one that only reads data
new_map: false                # optional: true for a procedure that makes a new map (put first for "map ...")
matched: true                 # optional: false keeps it out of request matching; another skill sends the agent to it
---
```

Keep it short and concrete: the decision rules (what to do when the request says X), the order of tool calls, and when to stop and ask a person. Write it so each `##` section reads on its own, with a heading that says what it is about and the words an error or a reply would contain: before every model call the agent searches every skill as passages (`asyncroscopy/skills/index.py`) with the request and the latest tool replies. Long experiments go through `run_job` and `wait_for_job`.

## Switch a skill off

Set `enabled: false` in its `SKILL.md`. The agent is no longer given it, and `list_skills` and `python -m asyncroscopy.agents.skills list` show it as disabled. Set it back to `true` to switch it on. The order in which switches apply:

1. `ASYNCROSCOPY_SKILLS` turns every skill on or off (off by default). Off, no skill is read at all.
2. `ASYNCROSCOPY_SKILLS_OFF` hides the skills it names (comma-separated names or globs) for a run, whatever their `enabled` says.
3. Each skill's own `enabled`.

A client's Skills page can also switch a skill off on one endpoint without editing it (`switch_skill`, below).

## Add or edit a workflow

Copy `workflows/map_grid.py` to `workflows/<name>.py`. Build the `StateGraph` in a `build(use)` function, then end with `WORKFLOW = Workflow(...)`:

```python
graph.add_node("tile", use.block("acquire_tile", tiles="{plan.tiles}"))       # a node runs a block
graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=400))
```

Write the nodes first, then the edges, and give a call with many inputs one input per line. Every `*.py` here is loaded by its file path, so a workflow imports what it needs from `asyncroscopy.workflows`, never from a sibling file. Check and run it with:

```bash
uv run python -m asyncroscopy.workflows lint <name>
uv run python -m asyncroscopy.workflows view --open
```

A workflow in this folder is shipped: it is reviewed in the repository like any other code and needs no approval. A workflow written elsewhere is proposed (`python -m asyncroscopy.workflows propose file.py`, or the `propose_workflow` tool) and stays a draft, run on a digital twin only, until a person approves it (`asyncroscopy/workflows/library.py`).

## Other skill folders

- `ASYNCROSCOPY_SKILLS_PATH`: extra folders of skills (several separated by `;` on Windows, `:` elsewhere), so a lab can keep its own outside the repository. A skill there replaces one here with the same name.
- The lab's own skills: written from a client's Skills page (MCP `save_skill`, `delete_skill`, `switch_skill`) into `$ASYNCROSCOPY_LAB_SKILLS` (default `outputs/lab_skills`), read like any other skill folder and checked by the promotion linter first. A lab skill with a shipped skill's name replaces it, and any skill can be switched off there without deleting it (`off.txt`). See `asyncroscopy/skills/lab.py`.
- Learned skills: after a goal run the checker found met, the agent drafts a skill from the calls that worked into `<agent memory>/skills/` (`$ASYNCROSCOPY_AGENT_MEMORY`, default `outputs/agent_memory`). Drafts are never given to an agent. An operator reviews and promotes them with `python -m asyncroscopy.agents.skills list | show <name> | promote <name>`. A promoted learned skill never replaces a shipped or lab skill of the same name, is given only to an agent offered every tool it lists, and only describes calls: every call still goes through the endpoint's safety policy.

## Using skills

- Agents: automatic (`asyncroscopy/agents/skill_middleware.py`); see `docs/Operation/agents.md`.
- MCP: every asyncroscopy MCP endpoint offers `list_skills`, `load_skill(name)` and `skills_for(request)` (the skills matching a request, written out for another harness's prompt).
- Python: `from asyncroscopy.skills import list_skills, load_skill, select_skills, skills_text`; passages with `from asyncroscopy.skills.index import search, passages_text`.
