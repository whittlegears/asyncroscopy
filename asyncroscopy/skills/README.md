# Skills

A skill teaches an agent how to combine instrument tools for a kind of experiment, without hard-coding the experiment. Agents do not have to look skills up: the agent graphs match each request (and, as the task goes on, each failed call, held call, or data key that comes back) against the skills' keywords, names and descriptions, and put the one or two that apply into the model's prompt. That keeps prompts short for small local models.

## Using them

- Agents: automatic (`asyncroscopy/agents/skill_middleware.py`); see `docs/Operation/agents.md`.
- MCP: every asyncroscopy MCP endpoint offers `list_skills`, `load_skill(name)` and `skills_for(request)` (the skills matching a request, written out for another harness's prompt) for other clients.
- Python: `from asyncroscopy.skills import list_skills, load_skill, select_skills, skills_text`.

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
tools: [run_job, wait_for_job] # tools it uses: an agent holding none of them is not given it
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

A promoted learned skill never replaces a packaged or lab skill of the same name, is given only to an agent offered every tool it lists, and only describes calls: every call still goes through the approval gate. Promotion re-checks it (length, no text about getting around approvals or safety, no operator-only tools, no hard-coded data keys).
