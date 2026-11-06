"""The MCP server's native tools, one component per concern. Each module has ``register(server, cfg)``,
which builds the component and adds its interceptors; ``natives()`` gives the tools it offers.

gate (approval_status), jobs (run_job, wait_for_job), data_tools (list_data, view_data, previews),
discovery (find_tools, call_tool, endpoint_info, action_space), skills_tools, trace_tools (trace_since),
estop (stop_all).
"""
