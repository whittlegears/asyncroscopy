# Skills

Skills are small Markdown procedures the agent can look up while it works. They
follow the Hermes-agent layout: one folder per skill, each holding a `SKILL.md`
file with a YAML frontmatter header and a Markdown body.

```
skills/
├── README.md
├── haadf-imaging/SKILL.md
├── eds-spectrum/SKILL.md
└── ...
```

`asyncroscopy.agent.skills.SkillRegistry` scans these folders, indexes them in
SQLite (FTS5) and exposes four tools to the agent:

| Tool | Purpose |
|------|---------|
| `skills_list` | names, versions and descriptions of every skill |
| `skill_search(query)` | keyword search over names, tags, triggers and bodies |
| `skill_view(name)` | the full body of one skill |
| `skill_save(name, description, body, ...)` | write a new `SKILL.md` and re-index |

Only the names and descriptions go into the system prompt (see
`render_skills_index`); the agent calls `skill_view` when it needs the details.
This keeps prompts short while still giving the model deep, repo-versioned
context on demand.

## Format

```markdown
---
name: haadf-imaging            # required; must equal the folder name; ^[a-z0-9][a-z0-9_-]{0,63}$
description: One line shown in the agent's system prompt.   # required
version: 1.0.0                 # optional (default 1.0.0)
tags: [imaging, stem]          # optional keywords for search
triggers: ["take an image"]    # optional phrases that should make the agent load this skill
tools: ["*_acquire_scanned_image", "get_data_from_key"]   # optional globs over MCP tool names
---

# Title

Short numbered steps naming the exact MCP tools and argument keys, expected
results, and how to verify. Keep it to what a colleague would need.
```

MCP tool names are `<DeviceClass>_<command>` (for example
`DigitalTwin_acquire_scanned_image`); use globs such as `*_acquire_spectrum` so a
skill works for every microscope class.

## Writing or updating a skill

1. Create `skills/<name>/SKILL.md` (or let the agent call `skill_save`).
2. Run `uv run pytest tests/test_agent_skills.py` to make sure it parses.
3. Commit it like any other file. Skills are reviewed in pull requests.

Malformed files are skipped with a warning and listed in `SkillRegistry.errors`.
