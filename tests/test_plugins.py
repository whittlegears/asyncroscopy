"""A team's own profile or tracker, defined outside the package, is found in a fresh device process.

The launcher runs each device in its own process, which imports only the device's module; a class
registered with ``@register_profile`` in the team's file is found there only through
``ASYNCROSCOPY_PLUGINS`` (asyncroscopy/plugins.py), which the stack YAML's ``plugins:`` sets.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROFILE = textwrap.dedent(
    """
    from asyncroscopy.jobs.mapping.profile import InstrumentProfile, register_profile
    from asyncroscopy.routines.drift import LinearTracker, register_tracker

    @register_profile
    class TeamProfile(InstrumentProfile):
        name = "team_afm"

    @register_tracker
    class TeamTracker(LinearTracker):
        name = "team_tracker"
    """
)

CHECK = textwrap.dedent(
    """
    from asyncroscopy.jobs.mapping.profile import get_profile
    from asyncroscopy.routines.drift import get_tracker
    print(get_profile("team_afm").__name__, get_tracker("team_tracker").__name__)
    """
)


def _fresh_python(code: str, plugins: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ASYNCROSCOPY_PLUGINS"}
    env["PYTHONPATH"] = str(ROOT)
    if plugins is not None:
        env["ASYNCROSCOPY_PLUGINS"] = plugins
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=120)


def test_a_profile_in_a_team_file_is_found_through_the_env_var_in_a_fresh_process(tmp_path):
    team = tmp_path / "team_profile.py"
    team.write_text(PROFILE, encoding="utf-8")

    without = _fresh_python(CHECK, None)
    assert without.returncode != 0
    assert "unknown instrument profile 'team_afm'" in without.stderr
    assert "ASYNCROSCOPY_PLUGINS" in without.stderr  # says how to fix it

    found = _fresh_python(CHECK, f"some_missing_module_xyz{os.pathsep}{team}")
    assert found.returncode == 0, found.stderr
    assert found.stdout.split() == ["TeamProfile", "TeamTracker"]
    assert "some_missing_module_xyz" in found.stderr  # the broken entry is reported, the good one still loads


def test_a_profile_in_a_team_module_is_found_by_module_name(tmp_path):
    (tmp_path / "team_pkg").mkdir()
    (tmp_path / "team_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "team_pkg" / "profiles.py").write_text(PROFILE, encoding="utf-8")
    code = f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + CHECK
    found = _fresh_python(code, "team_pkg.profiles")
    assert found.returncode == 0, found.stderr
    assert found.stdout.split() == ["TeamProfile", "TeamTracker"]


def test_a_failed_plugin_is_named_in_the_unknown_profile_error(tmp_path):
    broken = tmp_path / "broken_profile.py"
    broken.write_text("raise RuntimeError('typo in my profile')\n", encoding="utf-8")
    result = _fresh_python(CHECK, str(broken))
    assert result.returncode != 0
    assert "typo in my profile" in result.stderr.splitlines()[-1]


def test_plugin_specs_split_on_commas_and_pathsep():
    from asyncroscopy.plugins import plugin_specs

    assert plugin_specs(f"a.b, c.d{os.pathsep}a.b,,") == ["a.b", "c.d"]


def _run_servers():
    spec = importlib.util.spec_from_file_location("launcher_run_servers", ROOT / "startup_scripts" / "run_servers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launcher_run_servers"] = module
    spec.loader.exec_module(module)
    return module


def test_the_stack_yaml_plugins_list_reaches_every_device_through_the_env_var(tmp_path):
    run_servers = _run_servers()
    team = tmp_path / "team_profile.py"
    team.write_text(PROFILE, encoding="utf-8")
    text = (ROOT / "configs" / "DigitalTwin.yaml").read_text(encoding="utf-8")
    path = tmp_path / "stack.yaml"
    path.write_text(text + f"\nplugins:\n  - {team.as_posix()}\n  - json\n", encoding="utf-8")
    config = run_servers.load_config(path)
    assert config.plugins == (str(team.resolve()), "json")
    assert run_servers.plugin_environment(config.plugins, "mine.mod").split(os.pathsep) == ["mine.mod", str(team.resolve()), "json"]

    path.write_text(text + "\nplugins:\n  - no_such_team_module_xyz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_team_module_xyz"):
        run_servers.load_config(path)
    path.write_text(text + f"\nplugins:\n  - {(tmp_path / 'missing.py').as_posix()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no such file"):
        run_servers.load_config(path)


def test_a_tuner_strategy_in_a_team_file_is_found_through_the_env_var(tmp_path):
    team = tmp_path / "team_strategy.py"
    team.write_text(textwrap.dedent("""
        from asyncroscopy.instruments.electron_microscope.tuner import Strategy, register_strategy

        @register_strategy
        class TeamFocus(Strategy):
            name = "team_focus"
    """), encoding="utf-8")
    code = "from asyncroscopy.instruments.electron_microscope.tuner import strategies; print('team_focus' in strategies())"
    assert _fresh_python(code, None).stdout.strip() == "False"
    found = _fresh_python(code, str(team))
    assert found.stdout.strip() == "True", found.stderr
