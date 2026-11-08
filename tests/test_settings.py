"""One Settings: every ASYNCROSCOPY_* variable is read in asyncroscopy/settings.py, at call time, and documented once."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from asyncroscopy import settings
from asyncroscopy.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
ENV_READ = re.compile(r"""os\.environ\.get\(\s*['"]ASYNCROSCOPY_|os\.getenv\(\s*['"]ASYNCROSCOPY_|os\.environ\[['"]ASYNCROSCOPY_\w*['"]\](?!\s*=[^=])""")
# Owned by another branch while this one was written; its one read goes through settings once that lands.
NOT_YET_MOVED: set[str] = set()


def python_files():
    for folder in ("asyncroscopy", "benchmarks", "startup_scripts", "startup_guis", "tests"):
        yield from (ROOT / folder).rglob("*.py")


def test_no_code_outside_settings_reads_an_asyncroscopy_variable():
    readers = []
    for path in python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel not in ("asyncroscopy/settings.py", "tests/test_settings.py") and rel not in NOT_YET_MOVED and ENV_READ.search(path.read_text(encoding="utf-8")):
            readers.append(rel)
    assert not readers


def test_no_library_module_reads_the_environment_at_import():
    code = (
        "import os, importlib\n"
        "class Spy(dict):\n"
        "    def get(self, k, d=None):\n"
        "        assert not str(k).startswith('ASYNCROSCOPY_'), k\n"
        "        return dict.get(self, k, d)\n"
        "    def __getitem__(self, k):\n"
        "        assert not str(k).startswith('ASYNCROSCOPY_'), k\n"
        "        return dict.__getitem__(self, k)\n"
        "os.environ = Spy(os.environ)\n"
        "for name in ['asyncroscopy.data.data', 'asyncroscopy.data.tiled_client', 'asyncroscopy.data.integrity', 'asyncroscopy.skills.registry',\n"
        "             'asyncroscopy.plugins', 'asyncroscopy.settings', 'asyncroscopy.trace.writer', 'asyncroscopy.safety.approvals']:\n"
        "    importlib.import_module(name)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=240, check=False)
    assert result.returncode == 0, result.stderr


def test_every_variable_is_in_the_table_once_and_the_table_names_no_other():
    table = (ROOT / "docs/Operation/settings.md").read_text(encoding="utf-8")
    documented = re.findall(r"^\| `([A-Z_]+)` \|", table, re.M)
    assert len(documented) == len(set(documented))
    read = set(re.findall(r'"((?:ASYNCROSCOPY|TILED)_[A-Z_]+)"', (ROOT / "asyncroscopy/settings.py").read_text(encoding="utf-8")))
    read.add("ASYNCROSCOPY_OUTPUTS")
    assert read == set(documented)


def test_paths_left_unset_fall_under_the_outputs_folder(tmp_path):
    s = Settings.from_env({"ASYNCROSCOPY_OUTPUTS": str(tmp_path)})
    assert s.outputs_root == tmp_path.resolve() and s.log_dir == tmp_path.resolve() / "logs"
    assert s.integrity_log == tmp_path.resolve() / "integrity" / "manifest.jsonl" and s.agent_memory == tmp_path.resolve() / "agent_memory"
    assert s.task_trace == str(tmp_path.resolve() / "trace" / "agent_tasks.jsonl")


def test_a_variable_is_read_when_settings_are_asked_for(monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_POLL_S", "2.5")
    assert settings.get().poll_s == 2.5
    monkeypatch.setenv("ASYNCROSCOPY_POLL_S", "3")
    assert settings.get().poll_s == 3.0


def test_a_number_that_is_not_one_names_its_variable():
    with pytest.raises(ValueError, match="ASYNCROSCOPY_LOG_ROTATE_BYTES"):
        Settings.from_env({"ASYNCROSCOPY_LOG_ROTATE_BYTES": "lots"})


def test_empty_and_unset_keep_their_old_meanings():
    assert Settings.from_env({}).tiled_api_key is None and Settings.from_env({"ASYNCROSCOPY_TILED_API_KEY": ""}).tiled_api_key == ""
    assert Settings.from_env({"ASYNCROSCOPY_TASK_TRACE": ""}).task_trace == ""
    assert Settings.from_env({"ASYNCROSCOPY_CLAIM_CHECK": "off"}).claim_check is False and Settings.from_env({}).claim_check is True


def test_secrets_stay_out_of_the_repr():
    text = repr(Settings.from_env({"ASYNCROSCOPY_MCP_TOKEN": "hunter2", "ASYNCROSCOPY_TILED_API_KEY": "key123", "ASYNCROSCOPY_INTEGRITY_KEY": "aa"}))
    assert "hunter2" not in text and "key123" not in text
