"""Post-run skill reflection: decide whether a finished run is worth a skill draft.

Everything here is stdlib-only so it can be tested without the agent
dependencies. The langgraph backend supplies the model call; this module
supplies the trigger, the prompt material, and the tool-call extraction.
"""

trace_line_cap = 80
trace_char_cap = 300

propose_skill_tool = {
    "type": "function",
    "function": {
        "name": "propose_skill",
        "description": (
            "Save a reusable skill draft for the operator to review. The draft "
            "changes nothing until the operator approves it in the GUI. To "
            "improve an existing skill, pass that skill's exact id as the name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name, or the exact id of an existing skill to revise",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Full SKILL.md text: YAML frontmatter with name and "
                        "description, then step-by-step instructions"
                    ),
                },
            },
            "required": ["name", "content"],
        },
    },
}


def should_reflect(tool_steps: int, min_tool_steps: int, has_service: bool) -> bool:
    if not has_service or min_tool_steps < 1:
        return False
    return tool_steps >= min_tool_steps


def record_trace(trace: list[str], line: str) -> None:
    if len(trace) >= trace_line_cap:
        if len(trace) == trace_line_cap:
            trace.append("... trace truncated ...")
        return
    text = str(line)
    if len(text) > trace_char_cap:
        text = text[:trace_char_cap] + "…"
    trace.append(text)


def reflection_system_prompt() -> str:
    return (
        "You are reviewing a finished agent run on a scientific instrument to "
        "decide whether it taught a reusable procedure. Most runs teach nothing "
        "new — in that case reply with exactly NO_SKILL and nothing else. Only "
        "when the run required a non-obvious sequence of tool calls, or "
        "recovered from a failure in a way worth remembering, call "
        "propose_skill once with a complete SKILL.md: YAML frontmatter "
        "(--- name, description ---) followed by concrete step-by-step "
        "instructions naming the actual tools used. To improve an existing "
        "skill from the list you are given, pass its exact id as the name. "
        "Never invent tools or parameter values that do not appear in the run. "
        "Your proposal changes nothing until the operator reviews and approves "
        "it."
    )


def reflection_task_message(
    prompt: str, answer: str, trace: list[str], skills: list[dict]
) -> str:
    roster = (
        "\n".join(f"- {s['id']}: {s.get('description') or s.get('name', '')}" for s in skills)
        or "none"
    )
    trace_text = "\n".join(trace) or "no tool calls recorded"
    return (
        f"Task the operator gave the agent:\n{prompt}\n\n"
        f"Tool calls the run made:\n{trace_text}\n\n"
        f"Final answer returned:\n{answer}\n\n"
        f"Existing skills (use the exact id to revise one):\n{roster}"
    )


def proposals_from_tool_calls(tool_calls: list[dict]) -> list[tuple[str, str]]:
    proposals = []
    for call in tool_calls or []:
        if call.get("name") != "propose_skill":
            continue
        args = call.get("args") or {}
        name = str(args.get("name", "")).strip()
        content = str(args.get("content", "")).strip()
        if name and content:
            proposals.append((name, content))
    return proposals
