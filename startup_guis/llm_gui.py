#!/usr/bin/env python
"""Agent startup window: the shared window (script_gui.py) opened on gemma-llm.yaml."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from startup_guis.script_gui import main  # noqa: E402
from startup_guis.shared import CONFIG_DIR  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(CONFIG_DIR / 'gemma-llm.yaml'))
