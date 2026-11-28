"""Builds the package with the repository's ``agent/`` folder copied into it as ``asyncroscopy/agent``.

Everything else about the build is in ``pyproject.toml``. ``asyncroscopy.settings.shipped_agent_dir`` finds the
copy in an installed wheel and the repository's own folder in a checkout or an editable install.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithAgent(build_py):
    def run(self):
        super().run()
        if not self.editable_mode:
            shutil.copytree(Path(__file__).resolve().parent / "agent", Path(self.build_lib) / "asyncroscopy" / "agent",
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


setup(cmdclass={"build_py": BuildWithAgent})
