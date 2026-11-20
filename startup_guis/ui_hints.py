"""Optional display text for the startup window.

Nothing here is required: a script, section, key or entry with no hint is shown by its
YAML name ("http_port" becomes "Http port"). Which keys exist, their types and their rules
come from the YAML and from the script's SETTINGS, never from here.
"""

from __future__ import annotations

import re

# Opened when no config is named: the digital twin, so a fresh window never targets real hardware.
DEFAULT_CONFIG = "DigitalTwin.yaml"

TITLES = {
    "run_servers": "Asyncroscopy Server Startup",
    "run_mcp": "Asyncroscopy MCP Startup",
    "run_llm": "Asyncroscopy Agent Startup",
    "run_segmentation": "Asyncroscopy Segmentation Startup",
}

# Section titles by top-level YAML key; "" holds the top-level values that are not mappings.
SECTIONS = {
    "": "General",
    "tango": "Database",
    "tiled": "Data server",
    "mcp": "MCP server",
    "agent": "Agent MCP server",
}

LABELS = {
    "instrument.file": "Instrument file",
    "instrument.class_name": "Instrument class",
    "instrument.hardware_host": "Hardware host",
    "instrument.hardware_port": "Hardware port",
    "instrument.timeout_seconds": "Hardware timeout (s)",
    "tango.host": "Tango host",
    "tango.port": "Tango port",
    "tango.reset_database_file": "Delete tango_database.db before start",
    "tiled.host": "Tiled host",
    "tiled.port": "Tiled port",
    "tiled.acquisition_dir": "Acquisition dir",
    "tiled.autostart": "Start Tiled HTTP server",
    "tiled.register_on_startup": "Register acquisition dir on startup (slow for large folders)",
    "device_timeout_seconds": "Device startup timeout (s)",
    "mcp.http_host": "HTTP host",
    "mcp.http_port": "HTTP port",
    "mcp.quiet": "Quiet mode",
    "mcp.tool_log": "Tool call log (JSONL)",
    "mcp.data_device_address": "DATA device",
    "agent.http_host": "HTTP host",
    "agent.http_port": "HTTP port",
    "agent.mcp_urls": "Tool MCP URLs",
    "agent.model_provider": "Provider",
    "agent.use_init_chat_model": "Use init_chat_model",
    "data_device_address": "DATA device",
}

HELP = {
    "instrument.class_name": "Blank: the class the file serves.",
    "instrument.hardware_host": "Blank for digital twins.",
    "mcp.tool_log": "Blank disables it.",
    "mcp.include_only_functions": "Blank: every non-blocked command.",
}

# Keys of an entry (e.g. a device) worth showing beside its name, first one present wins.
ENTRY_SUMMARY_KEYS = ("class_name", "module_name")

# Lines a script prints, shown as a badge beside the title: (regex, badge text, is_error).
STATUS_PATTERNS = {
    "run_mcp": [
        (re.compile(r"MCP ready: (\d+) tool"), "{0} tools", False),
        (re.compile(r"MCP ERROR: .*in use"), "port in use", True),
        (re.compile(r"MCP ERROR: "), "server error", True),
    ],
}


def humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def title(script_name: str) -> str:
    return TITLES.get(script_name) or f"Asyncroscopy {humanize(script_name.removeprefix('run_'))} Startup"


def section_title(key: str) -> str:
    return SECTIONS.get(key) or humanize(key)


def label(path: str) -> str:
    return LABELS.get(path) or humanize(path.rsplit(".", 1)[-1])


def entry_summary(entry) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ENTRY_SUMMARY_KEYS:
        if entry.get(key):
            text = str(entry[key])
            return text.rsplit(".", 1)[-1] if key == "module_name" else text
    return ""
