import pytest
import yaml

from asyncroscopy.utils import process_manager
from startup_scripts import run_llm, run_mcp, run_pytemlib, run_servers
from startup_scripts.config_schema import fill_placeholders

FAKE_UV = "/opt/uv/bin/uv"


@pytest.fixture
def fake_uv(monkeypatch):
    monkeypatch.setattr(process_manager, "find_uv", lambda: FAKE_UV)
    return FAKE_UV


class FakeDataProxy:
    def __init__(self):
        self.timeout_millis = None
        self.stop_called = False
        self.register_called = False

    def set_timeout_millis(self, timeout_millis):
        self.timeout_millis = timeout_millis

    def stop_tiled_server(self):
        self.stop_called = True

    def reconcile(self):
        self.register_called = True
        return '{"registered": ["a.h5"], "already": 3, "failed": []}'


def test_get_data_proxy_sets_extended_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    assert run_servers.get_data_proxy() is proxy
    assert proxy.timeout_millis == run_servers.TILED_COMMAND_TIMEOUT_MILLIS


def test_stop_tiled_server_uses_extended_data_proxy_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    run_servers.stop_tiled_server()

    assert proxy.timeout_millis == run_servers.TILED_COMMAND_TIMEOUT_MILLIS
    assert proxy.stop_called is True


def test_reconcile_tiled_uses_startup_registration_timeout(monkeypatch):
    proxy = FakeDataProxy()
    monkeypatch.setattr(run_servers.tango, "DeviceProxy", lambda _: proxy)

    result = run_servers.reconcile_tiled()

    assert proxy.timeout_millis == run_servers.TILED_STARTUP_REGISTRATION_TIMEOUT_MILLIS
    assert proxy.register_called is True
    assert result == {"registered": ["a.h5"], "already": 3, "failed": []}


def test_the_default_tiled_key_is_flagged_only_off_this_machine():
    assert run_servers.tiled_key_warning("127.0.0.1", "secret") is None
    assert run_servers.tiled_key_warning("localhost", "secret") is None
    assert run_servers.tiled_key_warning("0.0.0.0", "a-real-key") is None
    assert "default API key" in run_servers.tiled_key_warning("0.0.0.0", "secret")


def test_load_spectra300_config_starts_servers_only():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "Spectra300.yaml")
    aperture = next(device for device in config.support_devices if device.key == "aperture")
    stage = next(device for device in config.support_devices if device.key == "stage")

    assert config.tango_host == "192.0.2.10"
    assert config.tiled.host == "127.0.0.1"
    assert config.tiled.register_on_startup is False
    assert config.instrument.class_name == "AutoScriptMicroscope"
    assert config.instrument.module_name == "asyncroscopy.instruments.electron_microscope.auto_script"
    assert aperture.class_name == "AutoScriptAPERTURE"
    assert aperture.module_name == "asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript"
    assert aperture.properties["hardware_host"] == ["192.0.2.10"]
    assert aperture.properties["hardware_port"] == ["9095"]
    assert stage.class_name == "AutoScriptSTAGE"
    assert stage.module_name == "asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript"
    assert stage.properties["hardware_host"] == ["192.0.2.10"]
    assert stage.properties["hardware_port"] == ["9095"]
    assert config.reset_database_file is False
    assert not hasattr(config, "mcp")


def test_build_devices_adds_selected_instrument():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "Test.yaml")

    devices = run_servers.build_devices(config)
    stage = next(device for device in devices if device.key == "stage")

    assert stage.class_name == "TestStage"
    assert stage.module_name == "asyncroscopy.instruments.electron_microscope.hardware.TestStage"
    assert devices[-1].key == "instrument"
    assert devices[-1].class_name == "DigitalTwin"
    assert devices[-1].module_name == "asyncroscopy.instruments.electron_microscope.digital_twin"
    assert devices[-1].device_name == "asyncroscopy/instrument/default"


def test_load_tilt_twin_config_includes_simulation_properties():
    config = run_servers.load_config(
        run_servers.PROJECT_DIR / "configs" / "digital_twin_tilt.yaml"
    )
    stage = next(device for device in config.support_devices if device.key == "stage")
    properties = run_servers.instrument_properties(config)

    assert config.instrument.class_name == "DigitalTwinTilt"
    assert config.instrument.module_name.endswith("digital_twin_tilt")
    assert stage.class_name == "DigitalTwinSTAGE"
    specimen = properties["specimen"][0]
    assert "silicon_lamella" in specimen and "silicon_lattice_parameter_angstrom" in specimen and "5.431" in specimen
    assert "lattice_parameter_gradient_x_percent" in specimen and "multislice_supercell_z" in specimen
    assert properties["potential_sampling_angstrom"] == ["0.08"]
    assert properties["rotation_center_z_nm"] == ["125"]
    assert properties["randomness_scale"] == ["1.0"]
    assert properties["stage_device_address"] == ["asyncroscopy/stage/default"]


def test_load_particle_twin_config_uses_renamed_module():
    config = run_servers.load_config(
        run_servers.PROJECT_DIR / "configs" / "digital_twin_particles.yaml"
    )
    properties = run_servers.instrument_properties(config)

    assert config.instrument.class_name == "DigitalTwinParticles"
    assert config.instrument.module_name.endswith("digital_twin_particles")
    assert "world_imsize" in properties["specimen"][0] and "particles" in properties["specimen"][0]
    assert properties["data_device_address"] == ["asyncroscopy/data/default"]
    assert properties["stage_device_address"] == ["asyncroscopy/stage/default"]


def test_load_mcp_config():
    config = run_mcp.load_config(run_mcp.PROJECT_DIR / "configs" / "mcp.yaml")

    assert config.mcp.name == "Spectra300_MCP"
    assert config.tango_host == "localhost"  # the instrument's address is written only in configs/Spectra300.yaml
    assert config.tango_port == 9094
    assert config.mcp.http_host == "127.0.0.1"  # This is a hardware endpoint. Other machines reach it only with allow_remote_bind.
    assert config.mcp.allow_remote_bind is False
    assert config.mcp.http_port == 8000
    assert config.mcp.blocked_classes == ["DataBase", "DServer"]
    assert "*" not in config.mcp.needs_approval
    assert config.mcp.operator_port is None
    assert "start_tiled_server" in config.mcp.needs_approval["DATA"]
    assert config.mcp.urls["analysis"] == "http://127.0.0.1:8003/mcp" and config.mcp.urls["operator"] is None
    assert config.mcp.endpoint_role == "agent" and config.mcp.trace == "outputs/trace/mcp_agent.jsonl"


def test_run_mcp_builds_server_command(fake_uv):
    config = run_mcp.Config(
        path=run_mcp.PROJECT_DIR / "configs" / "mcp.yaml",
        tango_host="localhost",
        tango_port=9094,
        mcp=run_mcp.MCPConfig(
            name="Spectra300_MCP",
            transport="streamable-http",
            http_host="127.0.0.1",
            http_port=8123,
            data_device_address="asyncroscopy/data/default",
            quiet=True,
            blocked_classes=["DataBase"],
            needs_approval={"DATA": ["stop_tiled_server"]},
        ),
    )

    command = run_mcp.build_command(config)

    assert command[:6] == [fake_uv, "run", "--no-sync", "python", "-u", "-m"]
    assert command[6] == "asyncroscopy.mcp.mcp_server"
    assert "--class-name" not in command
    assert command[command.index("--name") + 1] == "Spectra300_MCP"
    assert command[command.index("--http-port") + 1] == "8123"
    assert "--quiet" in command
    assert command[command.index("--blocked-classes-json") + 1] == '["DataBase"]'
    assert command[command.index("--needs-approval-json") + 1] == (
        '{"DATA": ["stop_tiled_server"]}'
    )
    assert "--search-packages-json" not in command


def test_load_agent_config():
    config = run_llm.load_config(run_llm.PROJECT_DIR / "configs" / "gemma-llm.yaml")

    assert config.agent.name == "Asyncroscopy_Agent_MCP"
    assert config.agent.transport == "streamable-http"
    assert config.agent.http_port == 8002
    assert config.agent.model == "gemma4:31b"
    assert config.agent.mcp_urls == ["http://127.0.0.1:8000/mcp"]
    assert [entry["name"] for entry in config.agent.startup_agents] == ["base", "image"]
    assert config.agent.startup_agents[0]["tools"] == ["list_devices"]


def test_run_llm_builds_server_command(fake_uv, monkeypatch):
    monkeypatch.setattr(run_llm, "agent_extras_installed", lambda: False)
    config = run_llm.Config(
        path=run_llm.PROJECT_DIR / "configs" / "gemma-llm.yaml",
        agent=run_llm.AgentConfig(
            name="Agent_MCP",
            transport="streamable-http",
            http_host="127.0.0.1",
            http_port=8124,
            mcp_urls=["http://127.0.0.1:8000/mcp"],
            startup_agents=[{"name": "base", "system_prompt": "hi", "tools": ["list_devices"]}],
        ),
    )

    command = run_llm.build_command(config)

    # The extras are missing and uv installs them, so this child syncs.
    assert command[:6] == [fake_uv, "run", "--extra", "agent", "--extra", "ollama"]
    assert "--no-sync" not in command
    assert command[command.index("-m") + 1] == "asyncroscopy.mcp.agent_mcp_server"
    assert command[command.index("--http-port") + 1] == "8124"
    assert command[command.index("--mcp-urls-json") + 1] == '["http://127.0.0.1:8000/mcp"]'
    assert "--use-init-chat-model" not in command


def test_run_llm_command_passes_init_chat_model_flag():
    config = run_llm.Config(
        path=run_llm.PROJECT_DIR / "configs" / "gemma-llm.yaml",
        agent=run_llm.AgentConfig(
            name="Agent_MCP",
            transport="streamable-http",
            http_host="127.0.0.1",
            http_port=8124,
            mcp_urls=[],
            use_init_chat_model=True,
        ),
    )

    assert "--use-init-chat-model" in run_llm.build_command(config)


def test_run_llm_passes_the_workflow_switches_and_keeps_hardware_on_loopback(tmp_path):
    def command(**agent):
        return run_llm.build_command(run_llm.Config(path=tmp_path / "a.yaml", agent=run_llm.AgentConfig(
            name="Agent_MCP", transport="streamable-http", http_host="127.0.0.1", http_port=8124, **agent)))

    from asyncroscopy.mcp.agent_mcp_server import parse_args

    base = ["--name", "A", "--transport", "streamable-http", "--http-host", "127.0.0.1", "--http-port", "8124"]
    # Workflows are on by default; the agent server is the one place they run.
    assert not {"--no-workflows", "--workflows-hardware"} & set(command())
    assert parse_args(base).workflows and not parse_args(base).workflows_hardware
    assert "--no-workflows" in command(workflows=False) and not parse_args(base + ["--no-workflows"]).workflows
    assert "--workflows-hardware" in command(workflows_hardware=True) and "--no-workflows" not in command(workflows=False, workflows_hardware=True)
    path = tmp_path / "plain.yaml"
    path.write_text("agent: {name: A, transport: streamable-http, http_host: 127.0.0.1, http_port: 8002}", encoding="utf-8")
    assert run_llm.load_config(path).agent.workflows
    path = tmp_path / "agent.yaml"
    path.write_text("agent: {name: A, transport: streamable-http, http_host: 0.0.0.0, http_port: 8002, workflows_hardware: true}", encoding="utf-8")
    with pytest.raises(run_llm.ConfigError, match="127.0.0.1"):
        run_llm.load_config(path)


def test_run_llm_starts_the_workflow_operator_endpoint_on_loopback_only(tmp_path):
    from asyncroscopy.mcp.agent_mcp_server import parse_args

    config = run_llm.load_config(run_llm.PROJECT_DIR / "configs" / "workflow_operator.yaml")
    assert (config.agent.endpoint_role, config.agent.http_host, config.agent.http_port) == ("operator", "127.0.0.1", 8004)
    command = run_llm.build_command(config)
    assert command[command.index("--endpoint-role") + 1] == "operator"
    assert parse_args(command[command.index("--name"):]).endpoint_role == "operator"
    assert "--endpoint-role" not in run_llm.build_command(run_llm.load_config(run_llm.PROJECT_DIR / "configs" / "gemma-llm.yaml"))
    for text, problem in (("http_host: 0.0.0.0, endpoint_role: operator", "127.0.0.1"), ("http_host: 127.0.0.1, endpoint_role: approver", "agent or operator")):
        path = tmp_path / "operator.yaml"
        path.write_text(f"agent: {{name: W, transport: streamable-http, {text}, http_port: 8004}}", encoding="utf-8")
        with pytest.raises(run_llm.ConfigError, match=problem):
            run_llm.load_config(path)


def test_run_llm_skips_sync_when_extras_installed(fake_uv, monkeypatch):
    monkeypatch.setattr(run_llm, "agent_extras_installed", lambda: True)
    config = run_llm.load_config(run_llm.PROJECT_DIR / "configs" / "gemma-llm.yaml")

    command = run_llm.build_command(config)

    assert command[:5] == [fake_uv, "run", "--no-sync", "python", "-u"]
    assert "--extra" not in command


def test_device_server_command_uses_python_command(fake_uv):
    device = run_servers.DeviceConfig(key="stage", class_name="STAGE", module_name="pkg.stage")

    assert device.command == [fake_uv, "run", "--no-sync", "python", "-u", "-m", "pkg.stage", "stage_instance"]


def test_instrument_kept_outside_the_repository_is_given_by_module_name(tmp_path, monkeypatch):
    (tmp_path / "mylab").mkdir()
    (tmp_path / "mylab" / "__init__.py").write_text("")
    (tmp_path / "mylab" / "optiscope.py").write_text("class OptiScope:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = run_servers._instrument_config({"module_name": "mylab.optiscope", "class_name": "OptiScope"})
    assert config.module_name == "mylab.optiscope" and config.class_name == "OptiScope"
    with pytest.raises(KeyError, match="class_name"):
        run_servers._instrument_config({"module_name": "mylab.optiscope"})
    with pytest.raises(ValueError, match="PYTHONPATH"):  # a typo is caught at load, not after the device timeout
        run_servers._instrument_config({"module_name": "mylab.optiscop", "class_name": "OptiScope"})
    with pytest.raises(ValueError, match="PYTHONPATH"):
        run_servers._instrument_config({"module_name": "nolab.optiscope", "class_name": "OptiScope"})
    outside = run_servers.PROJECT_DIR.parent / "mylab" / "scope.py"
    with pytest.raises(ValueError, match="module_name"):
        run_servers.python_file_to_module(outside)


# Another stack's ports and database file are not this launcher's to clear.

TWIN = run_servers.PROJECT_DIR / "configs" / "digital_twin_mapping.yaml"
RAMAN = run_servers.PROJECT_DIR / "configs" / "yaml_sim_raman.yaml"


def launcher(ppid, config, *extra):
    return run_servers.ProcessInfo(ppid, ("python", "startup_scripts/run_servers.py", "--yaml", str(config), *extra), str(run_servers.PROJECT_DIR))


def twin_running():
    """The STEM twin's stack as process listings show it: uv wrapper, launcher, Tango database, Tiled."""
    return {
        100: run_servers.ProcessInfo(1, ("uv", "run", "startup_scripts/run_servers.py", "--yaml", "configs/digital_twin_mapping.yaml"), str(run_servers.PROJECT_DIR)),
        101: launcher(100, "configs/digital_twin_mapping.yaml"),
        102: run_servers.ProcessInfo(101, ("python", "-m", "tango.databaseds.database", "2")),
        103: run_servers.ProcessInfo(101, ("python", "-m", "tiled", "serve", "catalog")),
    }


def me(snapshot, config, *extra):
    snapshot[900] = launcher(1, config, *extra)
    return {900}


def test_a_second_non_isolated_stack_would_share_the_twins_database_file():
    snapshot = twin_running()
    mine = run_servers.stack_identity(RAMAN, None)
    conflicts = run_servers.stack_conflicts(snapshot, mine, me(snapshot, "configs/yaml_sim_raman.yaml"), {9194: set(), 9191: set()}, share_database_file=True)
    assert len(conflicts) == 1 and "same Tango database file" in conflicts[0][0] and conflicts[0][1] == 100  # 100 is the uv wrapper. Stopping it stops the tree.
    assert "digital_twin_mapping.yaml" in conflicts[0][0]


def test_an_isolated_stack_on_its_own_ports_is_left_alone_by_the_twin_and_leaves_it_alone():
    snapshot = twin_running()
    isolated = run_servers.PROJECT_DIR / "outputs" / "stacks" / "raman"
    mine = run_servers.stack_identity(RAMAN, isolated)
    my_pids = me(snapshot, "configs/yaml_sim_raman.yaml", "--isolated", "outputs/stacks/raman")
    assert run_servers.stack_conflicts(snapshot, mine, my_pids, {9194: set(), 9191: set()}, share_database_file=False) == []


def test_an_isolated_stack_on_the_twins_ports_is_refused_naming_the_twin():
    snapshot = twin_running()
    mine = run_servers.stack_identity(RAMAN, run_servers.PROJECT_DIR / "outputs" / "stacks" / "raman")
    my_pids = me(snapshot, "configs/yaml_sim_raman.yaml", "--isolated", "outputs/stacks/raman")
    conflicts = run_servers.stack_conflicts(snapshot, mine, my_pids, {9094: {102}, 9091: {103}}, share_database_file=False)
    assert [pid for _, pid in conflicts] == [100]  # one entry for the twin, not one per port
    assert "listening on ports 9091, 9094" in conflicts[0][0]
    text = run_servers.explain_conflicts(conflicts, RAMAN)
    assert "--isolated outputs/stacks/<name>" in text and "--replace" in text and "second_stack.md" in text


def test_restarting_the_same_config_still_clears_its_own_leftovers():
    snapshot = twin_running()
    mine = run_servers.stack_identity(TWIN, None)
    my_pids = me(snapshot, "configs/digital_twin_mapping.yaml")
    assert run_servers.stack_conflicts(snapshot, mine, my_pids, {9094: {102}, 9091: {103}}, share_database_file=True) == []
    # The orphans of a crashed run, whose launcher is gone, are cleared too.
    orphans = {7: run_servers.ProcessInfo(1, ("python", "-m", "tango.databaseds.database", "2")), 8: run_servers.ProcessInfo(1, ("python", "-m", "tiled", "serve"))}
    assert run_servers.stack_conflicts(orphans, mine, me(orphans, "configs/digital_twin_mapping.yaml"), {9094: {7}, 9091: {8}}, share_database_file=True) == []


def test_a_foreign_program_on_the_port_is_named_not_killed():
    snapshot = {55: run_servers.ProcessInfo(1, ("C:/Program Files/Other/server.exe", "--port", "9091"))}
    mine = run_servers.stack_identity(TWIN, None)
    conflicts = run_servers.stack_conflicts(snapshot, mine, me(snapshot, "configs/digital_twin_mapping.yaml"), {9091: {55}}, share_database_file=True)
    assert conflicts == [("port 9091 is used by server.exe (pid 55), which is not an asyncroscopy stack", 55)]


def test_launcher_identity_reads_yaml_and_isolated_in_either_form(tmp_path):
    info = run_servers.ProcessInfo(1, ("uv", "run", "startup_scripts/run_servers.py", "--yaml=configs/a.yaml", "--isolated", "outputs/stacks/a"), str(tmp_path))
    assert run_servers.launcher_identity(info) == run_servers.stack_identity(tmp_path / "configs" / "a.yaml", tmp_path / "outputs" / "stacks" / "a")
    assert run_servers.launcher_identity(run_servers.ProcessInfo(1, ("python", "-m", "tiled", "serve"))) is None
    # Without --yaml a launcher refuses to start, so it holds no stack.
    assert run_servers.launcher_identity(run_servers.ProcessInfo(1, ("python", "startup_scripts/run_servers.py"))) is None


def test_a_launcher_stops_its_servers_on_sigterm_once():
    import signal

    before = signal.getsignal(signal.SIGTERM)
    try:
        process_manager.stop_on_termination()
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
        assert handler(signal.SIGTERM, None) is None  # a second copy during cleanup is ignored
    finally:
        signal.signal(signal.SIGTERM, before)


def test_the_tango_database_can_bind_an_interface_other_than_its_client_host(tmp_path):
    """In a pod, tango.host is a service name every client resolves; the database binds 0.0.0.0."""
    twin = run_servers.PROJECT_DIR / "configs" / "digital_twin_mapping.yaml"
    assert run_servers.load_config(twin).tango_bind_host == ""
    path = tmp_path / "pod.yaml"
    source = twin.read_text(encoding="utf-8")
    assert "  host: localhost\n  port: 9094\n" in source
    path.write_text(source.replace("  host: localhost\n  port: 9094\n", "  host: twin-tango\n  bind_host: 0.0.0.0\n  port: 9094\n"), encoding="utf-8")
    config = run_servers.load_config(path)
    assert (config.tango_host, config.tango_bind_host) == ("twin-tango", "0.0.0.0")


def test_run_mcp_fails_when_its_server_stops_on_its_own(monkeypatch):
    """A server that exits by itself (port taken, Tango unreachable) makes the launcher exit non-zero."""

    class Exited:
        returncode = 3

        def poll(self):
            return 3

        def wait(self, timeout=None):
            return 3

    class FakeManager:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def start_process(self, **kwargs):
            return process_manager.ManagedProcess(key="k", label="l", process=Exited())

    monkeypatch.setattr(run_mcp, "ProcessManager", FakeManager)
    monkeypatch.setattr(run_mcp, "stop_on_termination", lambda: None)
    assert run_mcp.main(["--yaml", str(run_mcp.PROJECT_DIR / "configs" / "mcp_dt.yaml")]) == 3


def test_a_config_extends_another_and_overrides_or_removes_what_it_sets(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "simulated: true\n"
        "instrument:\n  class_name: DigitalTwin\n  properties:\n    seed: 1\n    specimen: {source: procedural, support_texture: 0.3}\n    drift: 2.0\n"
        "devices:\n  tuner: {class_name: Tuner, module_name: asyncroscopy.jobs.tuning.tuner}\n  data: {module_name: asyncroscopy.data.data}\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "instrument:\n  properties:\n    seed: 2\n    specimen: {source: crystal}\n    drift: null\n"
        "devices:\n  tuner: null\n",
        encoding="utf-8",
    )
    raw = run_servers.read_config(child)
    assert raw["simulated"] is True and raw["instrument"]["class_name"] == "DigitalTwin"
    assert raw["instrument"]["properties"] == {"seed": 2, "specimen": {"source": "crystal"}}  # a property is replaced whole
    assert list(raw["devices"]) == ["data"]

    base.write_text("extends: child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circle"):
        run_servers.read_config(child)


def test_the_twin_variants_extend_the_mapping_twin():
    for name in ("digital_twin_crystal.yaml", "digital_twin_recorded.yaml", "digital_twin_large_map.yaml"):
        config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / name)
        assert config.instrument.class_name == "DigitalTwin"
        assert {d.key for d in config.support_devices} >= {"data", "mapper", "stage"}


def test_load_pytemlib_config():
    config = run_pytemlib.load_config(run_pytemlib.PROJECT_DIR / "configs" / "pytemlib.yaml")

    assert (config.pytemlib.transport, config.pytemlib.host, config.pytemlib.port) == ("streamable-http", "127.0.0.1", 8003)
    assert config.pytemlib.tiled == "http://localhost:9091"
    assert config.pytemlib.trace is None and config.pytemlib.max_jobs is None  # left out means the server default


def pytemlib_config(**values):
    return run_pytemlib.Config(path=run_pytemlib.DEFAULT_CONFIG_PATH, pytemlib=run_pytemlib.PytemlibConfig(**values))


def test_run_pytemlib_passes_only_what_the_config_sets(fake_uv):
    command = run_pytemlib.build_command(pytemlib_config())

    assert command == [fake_uv, "run", "--no-sync", "python", "-u", "-m", "pytemlib_mcp.analysis_server",
                       "--transport", "streamable-http", "--host", "127.0.0.1", "--port", "8003"]


def test_run_pytemlib_command_is_read_back_by_the_server(fake_uv, monkeypatch):
    """Every flag the launcher writes is accepted by the server parser with the config value."""
    from pytemlib_mcp.analysis_server import environment
    from pytemlib_mcp.analysis_server.__main__ import parser

    config = pytemlib_config(
        transport="sse", host="0.0.0.0", port=8103, tiled="http://localhost:9191", api_key="k3y", local_roots=["outputs", "data"],
        max_canvas_megapixels=250.5, max_jobs=3, trace="outputs/t.jsonl", annotations="", placements="-odd name.jsonl",
    )
    command = run_pytemlib.build_command(config)
    args = parser(environment.get()).parse_args(command[command.index("pytemlib_mcp.analysis_server") + 1:])

    assert (args.transport, args.host, args.port, args.tiled) == ("sse", "0.0.0.0", 8103, "http://localhost:9191")
    assert (args.local_root, args.max_canvas_megapixels, args.max_jobs) == (["outputs", "data"], 250.5, 3)
    assert (args.trace, args.annotations, args.placements) == ("outputs/t.jsonl", "", "-odd name.jsonl")
    # The key is passed by environment variable and not on the command line.
    assert "k3y" not in command and args.api_key is None
    monkeypatch.delenv("ASYNCROSCOPY_TILED_API_KEY", raising=False)
    assert run_pytemlib.server_env(config)["ASYNCROSCOPY_TILED_API_KEY"] == "k3y"
    assert "ASYNCROSCOPY_TILED_API_KEY" not in run_pytemlib.server_env(pytemlib_config())


def test_run_pytemlib_defaults_and_settings_match_the_server_parser():
    from pytemlib_mcp.analysis_server import environment
    from pytemlib_mcp.analysis_server.__main__ import parser

    server = parser(environment.get())
    defaults = server.parse_args([])
    assert (run_pytemlib.DEFAULT_HOST, run_pytemlib.DEFAULT_PORT, run_pytemlib.TRANSPORTS[0]) == (defaults.host, defaults.port, defaults.transport)
    options = {a.dest for a in server._actions if a.option_strings and a.dest != "help"}
    declared = {s.path.split(".", 1)[1] for s in run_pytemlib.SETTINGS}
    assert {"local_root" if name == "local_roots" else name for name in declared} == options


def test_run_pytemlib_starts_only_the_analysis_server_and_refuses_the_old_workflow_keys(tmp_path):
    # Workflows run in the agent MCP server only (run_llm.py, agent.workflows).
    command = run_pytemlib.build_command(pytemlib_config(local_runs="outputs/maps"))
    assert "pytemlib_mcp.analysis_server" in command and not [a for a in command if "workflows" in a]
    path = tmp_path / "p.yaml"
    path.write_text("pytemlib: {workflows: true}", encoding="utf-8")
    with pytest.raises(run_pytemlib.ConfigError, match="unknown key"):
        run_pytemlib.load_config(path)


def test_the_launchers_need_a_yaml_and_never_pick_the_instrument_themselves(capsys):
    assert run_mcp.main([]) == 1
    assert "--yaml is required" in capsys.readouterr().err
    assert run_servers.main([]) == 1
    assert "--yaml is required" in capsys.readouterr().out
    assert run_servers.DEFAULT_CONFIG_PATH.name == "DigitalTwin.yaml"  # only what the GUI opens first
    spectra = yaml.safe_load((run_mcp.PROJECT_DIR / "configs" / "Spectra300.yaml").read_text(encoding="utf-8"))
    assert run_mcp.HOST_PRESETS == {"Digital twin": "localhost", "Spectra300": fill_placeholders(spectra)["tango"]["host"]}
