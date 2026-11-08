#!/usr/bin/env python
"""Server startup window; every field, device and check comes from startup_scripts/run_servers.py."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from startup_guis.script_gui import ScriptGui, main  # noqa: E402
from startup_scripts import run_servers  # noqa: E402


class ServerGui(ScriptGui):
    def __init__(self, config_path=None):
        super().__init__(run_servers, config_path)


if __name__ == '__main__':
    sys.exit(main(run_servers))
