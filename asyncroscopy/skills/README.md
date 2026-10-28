# Skills

A skill teaches an agent how to combine instrument tools for a kind of experiment, without hard-coding the experiment. Agents list skills (name and one-line description) and load the full text only when a request needs it, which keeps prompts short for small local models.

## Using them

- MCP: every asyncroscopy MCP endpoint offers `list_skills` and `load_skill(name)`.
- Python: `from asyncroscopy.skills import list_skills, load_skill`.
- The `general` LangGraph agent lists skills in its system prompt and loads them itself.

## Writing one

Make a folder with a `SKILL.md`:

```markdown
---
name: tilt-series
description: One line saying when to use it; this is all the agent sees before loading.
version: 1.0.0
min_asyncroscopy: 0.1.0      # optional: refuse to load on older installs
envelope_schema: 1.0.0       # optional: the job envelope major version it was written for
tools: [Mapper_reserve, ...] # tools it uses (documentation for readers and reviewers)
---

# Steps the agent should follow
...
```

Keep it short and concrete: the decision rules (what to do when the request says X), the order of tool calls, and when to stop and ask a person.

Put lab-specific skills outside the repository and point `ASYNCROSCOPY_SKILLS_PATH` at their folder (several folders separated by `;` on Windows, `:` elsewhere). A skill in this package wins over an outside one with the same name.

Agents cannot write or change skills; add them by pull request (or to your own folder).
