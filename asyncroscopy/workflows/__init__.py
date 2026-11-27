"""Workflows are fixed procedures written as LangGraph graphs built from blocks.

A block is code we ship in ``blocks.py`` and ``shipped_blocks.py``. It does one thing and declares its inputs,
outputs and tools. A workflow is a module under ``graphs/`` that builds a ``StateGraph`` whose nodes are blocks.
``graphs/map_grid.py`` is an example. ``runner.py`` checks the graph against the live endpoint and then runs it with
checkpoints, a stop switch and a cleanup that always runs. Blocks reach the instrument only through MCP tools, so
the endpoint's risk policy, limits and trace apply to every call.

    python -m asyncroscopy.workflows blocks
    python -m asyncroscopy.workflows lint map_grid
    python -m asyncroscopy.workflows run map_grid --input size_m=6e-7

An agent can also write a workflow and send it as text. ``authoring.py`` checks that the text only wires blocks
together before any of it runs. ``library.py`` keeps it as a draft until a person approves it with
``python -m asyncroscopy.workflows approve NAME``. Only shipped and approved workflows run on an instrument.
"""

from .blocks import Block, BlockContext, BlockFailed, Stopped, all_blocks, describe_blocks, get_block, register_block
from .graph import Steps, Workflow, WorkflowState, all_workflows, get_workflow, register_workflow
from .library import Library
from .checks import lint

__all__ = [
    "Block", "BlockContext", "BlockFailed", "Library", "Steps", "Stopped", "Workflow", "WorkflowState", "all_blocks", "all_workflows",
    "describe_blocks", "get_block", "get_workflow", "lint", "register_block", "register_workflow",
]
