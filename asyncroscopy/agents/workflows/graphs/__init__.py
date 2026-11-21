"""The shipped workflows, one module each.

``asyncroscopy.agents.workflows.graph.all_workflows`` loads every module here. To add a workflow, copy
``map_grid.py``. Build the ``StateGraph`` in a ``build(use)`` function, then end with ``WORKFLOW = Workflow(...)``:

    graph.add_node("tile", use.block("acquire_tile", tiles="{plan.tiles}"))       # a node runs a block
    graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=400))

Write the nodes first, then the edges, and give a call with many inputs one input per line.
"""
