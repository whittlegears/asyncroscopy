# Skills

Skills are Markdown procedures the agent can consult while working, modelled on
the Hermes agent's skills library. They live in the repository under
`skills/<name>/SKILL.md`, so they are versioned and reviewed like code.

## File format

```markdown
---
name: haadf-imaging            # required, equals the folder name
description: Acquire a scanned HAADF image and confirm it was saved to Tiled.
version: 1.0.0
tags: [imaging, stem]
triggers: ["take an image", "haadf"]
tools: ["*_acquire_scanned_image", "get_data_from_key"]
---

# HAADF imaging

1. Call `<DeviceClass>_acquire_scanned_image` with `{"detector_list": ["haadf"]}`.
2. ...
```

`name` and `description` are required. `tags` and `triggers` improve search;
`tools` documents which MCP tools (glob patterns) the procedure uses. The full
schema is in `skills/README.md`.

## Registry and search

```python
from asyncroscopy.agent.skills import SkillRegistry, render_skills_index

registry = SkillRegistry.from_dirs(["skills"])
registry.list()                     # all skills, sorted
registry.search("eds composition")  # (skill, score) pairs, best first
registry.get("eds-spectrum").body
print(render_skills_index(registry))  # names + descriptions for a system prompt
```

Skills are indexed in SQLite FTS5 (porter stemming, bm25 ranking). If the local
SQLite lacks FTS5, a substring search is used instead. Malformed files are
skipped with a warning and listed in `registry.errors`.

## Progressive disclosure

`build_react_graph(model, tools, registry)` appends only the skills index
(names and descriptions) to the system prompt and adds four tools:

| tool | what the agent gets |
|------|---------------------|
| `skills_list` | JSON list of every skill's metadata |
| `skill_search(query, limit)` | best-matching skills for a keyword query |
| `skill_view(name)` | the full body of one skill |
| `skill_save(name, description, body, tags, triggers, tools, overwrite)` | writes a new `SKILL.md` and re-indexes |

The model reads a body only when it decides a skill is relevant, which keeps
prompts short while still giving it detailed, repo-controlled instructions.

## Agent-authored skills

`skill_save` lets the agent record a procedure that worked (the notebook shows
an example). Files land in the first configured skills directory. Review the
diff before committing, as you would for any generated code. Names are
validated (`^[a-z0-9][a-z0-9_-]{0,63}$`) so a skill can never be written outside
the skills folder.

## On the LLM Tango device

The device loads the registry at start-up from its `skills_dirs` property,
exposes the list through the `skills` attribute and re-scans on `ReloadSkills()`.
Every worker agent spawned on the device is skills-aware.
