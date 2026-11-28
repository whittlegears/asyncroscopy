---
name: write_workflow
description: Write a new deterministic LangGraph workflow for a multi-step task the user gave when no workflow fits, and propose it for a scientist to approve. The workflow is Python that wires blocks into a StateGraph; it then runs the steps exactly, far faster than calling tools one by one.
version: 1.0.0
enabled: true
min_asyncroscopy: 0.1.0
keywords: [workflow, langgraph, graph, automate, procedure, series, repeat, loop, every, each, survey, sequence, propose]
requires: [propose_workflow]
acts: false
tools: [list_workflows, get_workflow, list_blocks, propose_workflow]
---

# Writing a workflow

A workflow is a deterministic LangGraph graph: Python that wires blocks into a `StateGraph`. You are the one who designs it; a scientist approves it before it runs on the instrument. Write one when the user's task has many instrument steps, a loop, or steps that must happen at the same time, and `list_workflows` has nothing that fits. Never for a single action (one image, one spectrum, one setting), and never for a task nobody gave you.

## 1. Check what exists

1. `list_workflows` once. If one fits, run it with `run_workflow` instead; if one nearly fits, read it with `get_workflow` and start from its source.
2. `list_blocks` once: the blocks, their inputs and outputs, the rules, and two whole examples (a loop; steps at the same time with a question for the person). Use only those blocks. Any tool of the endpoint is a step through `run_tool`.

## 2. Design the graph before writing it

Write the plan in a few lines first, for yourself:

- the nodes in order, each a block with its inputs;
- where values come from: a run input `"{inputs.name}"` or an earlier node's output `"{node.output}"`;
- every loop: the condition that ends it and `max_visits` (every loop needs one);
- steps that can run at the same time (two edges out of one node, joined with `add_edge(["b", "c"], d)`);
- where a person must decide (`ask_person`, then branch on `node.answer`);
- how the instrument is left at the end (back to the start position and field of view).

Make run inputs of what the user may change next time (a size, a count, a field of view, a detector), each with a type, a default, a range and a one-line doc, in SI units (metres, seconds, eV). Take defaults from the instrument's current readings or the user's words, not from memory.

## 3. Write and propose

Write the whole module in the exact shape of the `list_blocks` examples: the same imports, `INPUTS` as plain data, `def build(use)` that returns the graph, then `WORKFLOW = Workflow(name=..., summary=..., inputs=INPUTS, budget=..., build=build)`. No other imports, functions, loops or calls are accepted.

- `name`: lower_case_with_underscores, saying what it does (`eds_at_each_particle`).
- `summary`: one sentence a scientist reads to decide whether to approve it.
- `budget`: `max_seconds` and `max_tool_calls` a little above what the steps need.

Call `propose_workflow` with `user_request` set to the user's own words, quoted, and `source` set to the whole module.

## 4. When it is refused

The reply's `problems` name each fault with its line. Fix exactly those and send the whole module again. Do not drop a step the user asked for to make it pass; if a block cannot do something, say so to the user. Three refusals in a row: stop and show the user the problems.

## 5. Tell the user

Once it is kept, say in a few lines: its name, what it does step by step, its inputs, and that it is a draft that runs on the digital twin only until a scientist approves it (`next` in the reply says how). Do not run it for the real task before that, and never try to approve it yourself.
