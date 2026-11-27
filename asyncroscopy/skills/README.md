# Skills

A skill is an instruction manual for one kind of task: how to do it on this kind of instrument, which tool or workflow to use, what the replies mean and what to report. It is context, retrieved for a request and put in the prompt, and it never runs anything itself. A procedure whose calls are the same every time belongs in a LangGraph workflow (`asyncroscopy/workflows/graphs/`); its skill says when to run that workflow and where the answer is in its outputs. Agents do not have to look skills up: the agent graphs match each request against the skills' keywords, names and descriptions and put the one or two procedures that apply into the model's prompt. Then, before every model call, they search every skill as passages (its `##` sections, and a long section's list items) with the request and the latest tool replies, and add the best few as advice for that step (`index.py`). That keeps prompts short for small local models. Write a skill so each section reads on its own: a heading that says what it is about, and the words an error or a reply would contain.

## Using them

- Agents: automatic (`asyncroscopy/agents/skill_middleware.py`); see `docs/Operation/agents.md`.
- MCP: every asyncroscopy MCP endpoint offers `list_skills`, `load_skill(name)` and `skills_for(request)` (the skills matching a request, written out for another harness's prompt) for other clients.
- The lab's own skills: written from a client's Skills page (MCP `save_skill`, `delete_skill`, `switch_skill`) into `$ASYNCROSCOPY_LAB_SKILLS` (default `outputs/lab_skills`), read like any other skill folder, and checked by the promotion linter first. A lab skill with a shipped skill's name replaces it; any skill can be switched off there without deleting it (`off.txt`). See `lab.py`.
- Python: `from asyncroscopy.skills import list_skills, load_skill, select_skills, skills_text`; passages with `from asyncroscopy.skills.index import search, passages_text`.

## Writing one

Make a folder with a `SKILL.md`:

```markdown
---
name: tilt-series
description: One line saying when to use it; matched against requests.
version: 1.0.0
keywords: [tilt series, tomography, tilt]  # phrases a request would contain; the strongest match signal
min_asyncroscopy: 0.1.0      # optional: refuse to load on older installs
envelope_schema: 1.0.0       # optional: the job envelope major version it was written for
requires: [Mapper*_reserve]   # tool patterns the endpoint must offer; otherwise the skill is not offered
unless: [Tomo*_reserve]       # optional: not offered where these are, because another skill does the job there
tools: [run_job, wait_for_job] # tools it uses: an agent holding none of them is not given it
new_map: false                # optional: true for a procedure that makes a new map (put first for "map ...")
---

# Steps the agent should follow
...
```

Keep it short and concrete: the decision rules (what to do when the request says X), the order of tool calls, and when to stop and ask a person. Long experiments go through `run_job` and `wait_for_job`.

Put lab-specific skills outside the repository and point `ASYNCROSCOPY_SKILLS_PATH` at their folder (several folders separated by `;` on Windows, `:` elsewhere). An outside skill replaces a packaged one with the same name, so a lab can adapt a skill without editing the repository.

## Learned skills

After a goal run the checker found met, the agent drafts a skill from the calls that worked into `<agent memory>/skills/` (`$ASYNCROSCOPY_AGENT_MEMORY`, default `outputs/agent_memory`). Drafts are never given to an agent. An operator reviews and promotes them:

```bash
uv run python -m asyncroscopy.agents.skills list
uv run python -m asyncroscopy.agents.skills show <name>
uv run python -m asyncroscopy.agents.skills promote <name>
```

A promoted learned skill never replaces a packaged or lab skill of the same name, is given only to an agent offered every tool it lists, and only describes calls: every call still goes through the endpoint's safety policy. Promotion re-checks it (length, no text about getting around safety, no operator-only tools, no hard-coded data keys).
