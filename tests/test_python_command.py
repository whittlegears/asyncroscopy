"""How launchers build child commands: uv when it can be found, else this interpreter."""

import sys
from pathlib import Path

from asyncroscopy.utils import process_manager
from asyncroscopy.utils.process_manager import find_uv, python_command


def test_find_uv_prefers_uv_env_var(monkeypatch, tmp_path):
    uv = tmp_path / "uv.exe"
    uv.write_text("")
    monkeypatch.setenv("UV", str(uv))
    monkeypatch.setattr(process_manager.shutil, "which", lambda name: None)

    assert find_uv() == str(uv)


def test_find_uv_ignores_stale_uv_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("UV", str(tmp_path / "missing" / "uv"))
    monkeypatch.setattr(process_manager.shutil, "which", lambda name: "/usr/bin/uv")

    assert find_uv() == "/usr/bin/uv"


def test_find_uv_none_when_absent(monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(process_manager.shutil, "which", lambda name: None)

    assert find_uv() is None


def test_python_command_with_uv_skips_sync(monkeypatch):
    monkeypatch.setattr(process_manager, "find_uv", lambda: "C:/tools/uv.exe")

    assert python_command("-m", "pkg.mod", "arg") == [
        "C:/tools/uv.exe", "run", "--no-sync", "python", "-u", "-m", "pkg.mod", "arg",
    ]
    # Extras are irrelevant without a sync, so they are left out.
    assert "--extra" not in python_command("-m", "pkg.mod", extras=("agent",))


def test_python_command_with_uv_syncs_extras_on_request(monkeypatch):
    monkeypatch.setattr(process_manager, "find_uv", lambda: "uv")

    assert python_command("-m", "pkg.mod", extras=("agent", "ollama"), sync=True) == [
        "uv", "run", "--extra", "agent", "--extra", "ollama", "python", "-u", "-m", "pkg.mod",
    ]


def test_python_command_without_uv_uses_this_interpreter(monkeypatch, capsys):
    monkeypatch.setattr(process_manager, "find_uv", lambda: None)
    monkeypatch.setattr(process_manager, "running_in_project_venv", lambda: True)

    assert python_command("script.py", "--yaml", "x.yaml") == [sys.executable, "-u", "script.py", "--yaml", "x.yaml"]
    assert capsys.readouterr().err == ""


def test_python_command_without_uv_warns_outside_project_venv(monkeypatch, capsys):
    monkeypatch.setattr(process_manager, "find_uv", lambda: None)
    monkeypatch.setattr(process_manager, "running_in_project_venv", lambda: False)
    monkeypatch.setattr(process_manager, "_warned_no_uv", False)

    assert python_command("-m", "pkg.mod")[0] == sys.executable
    assert "uv not found" in capsys.readouterr().err


def test_running_in_project_venv_matches_sys_prefix(monkeypatch):
    monkeypatch.setattr(process_manager.sys, "prefix", str(process_manager.PROJECT_DIR / ".venv"))
    assert process_manager.running_in_project_venv()
    monkeypatch.setattr(process_manager.sys, "prefix", str(Path("/somewhere/else")))
    assert not process_manager.running_in_project_venv()
