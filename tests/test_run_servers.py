import pytest

from asyncroscopy.utils import process_manager
from startup_scripts import run_mcp, run_servers, run_llm

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

    assert config.tango_host == "10.46.217.241"
    assert config.tiled.host == "10.46.217.241"
    assert config.tiled.register_on_startup is False
    assert config.instrument.class_name == "AutoScriptMicroscope"
    assert config.instrument.module_name == "asyncroscopy.instruments.electron_microscope.auto_script"
    assert aperture.class_name == "AutoScriptAPERTURE"
    assert aperture.module_name == "asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript"
    assert aperture.properties["hardware_host"] == ["10.46.217.241"]
    assert aperture.properties["hardware_port"] == ["9095"]
    assert stage.class_name == "AutoScriptSTAGE"
    assert stage.module_name == "asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript"
    assert stage.properties["hardware_host"] == ["10.46.217.241"]
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
    assert stage.class_name == "TestStage"
    assert properties["silicon_lattice_parameter_angstrom"] == ["5.431"]
    assert properties["lattice_parameter_gradient_x_percent"] == ["2.0"]
    assert properties["crystal_rotation_gradient_x_deg"] == ["4.0"]
    assert properties["crystal_tilt_gradient_x_mrad"] == ["10.0"]
    assert properties["potential_sampling_angstrom"] == ["0.08"]
    assert properties["multislice_supercell_z"] == ["8"]
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
    assert properties["world_imsize"] == ["128"]
    assert properties["data_device_address"] == ["asyncroscopy/data/default"]
    assert properties["stage_device_address"] == ["asyncroscopy/stage/default"]


def test_load_mcp_config():
    config = run_mcp.load_config(run_mcp.PROJECT_DIR / "configs" / "mcp.yaml")

    assert config.mcp.name == "Spectra300_MCP"
    assert config.tango_host == "10.46.217.241"
    assert config.tango_port == 9094
    assert config.mcp.http_host == "127.0.0.1"  # a hardware endpoint: other machines only with allow_remote_bind
    assert config.mcp.allow_remote_bind is False
    assert config.mcp.http_port == 8000
    assert config.mcp.blocked_classes == ["DataBase", "DServer"]
    assert config.mcp.blocked_functions == {"*": ["Init", "Kill", "RestartServer"]}


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
            blocked_functions={"*": ["Init"], "DATA": ["stop_tiled_server"]},
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
    assert command[command.index("--blocked-functions-json") + 1] == (
        '{"*": ["Init"], "DATA": ["stop_tiled_server"]}'
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

    # Extras missing: uv installs them, so this child syncs.
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


# --- another stack's ports and database file are not this launcher's to clear ---

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
    assert len(conflicts) == 1 and "same Tango database file" in conflicts[0][0] and conflicts[0][1] == 100  # the uv wrapper: stopping it stops the tree
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
    # A crashed run's orphans (launcher gone) are cleared too.
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
    default = run_servers.launcher_identity(run_servers.ProcessInfo(1, ("python", "startup_scripts/run_servers.py")))
    assert default == run_servers.stack_identity(run_servers.DEFAULT_CONFIG_PATH, None)


def test_an_isolated_stack_keeps_its_own_approval_mode_file(tmp_path):
    approvals = run_servers.DeviceConfig(key="approvals", class_name="Approvals", module_name="asyncroscopy.safety.approvals")
    stage = run_servers.DeviceConfig(key="stage", class_name="TestStage", module_name="asyncroscopy.instruments.electron_microscope.hardware.TestStage")
    run_servers.isolate_approvals([approvals, stage], tmp_path)
    assert approvals.properties["mode_file"] == [str(tmp_path / "approvals" / "mode.json")] and "mode_file" not in stage.properties
    configured = run_servers.DeviceConfig(key="approvals", class_name="Approvals", module_name="asyncroscopy.safety.approvals", properties={"mode_file": ["mine.json"]})
    run_servers.isolate_approvals([configured], tmp_path)
    assert configured.properties["mode_file"] == ["mine.json"]
    shared = run_servers.DeviceConfig(key="approvals", class_name="Approvals", module_name="asyncroscopy.safety.approvals")
    run_servers.isolate_approvals([shared], None)
    assert "mode_file" not in shared.properties


def test_the_raman_stack_does_not_share_the_twins_approval_mode():
    config = run_servers.load_config(run_servers.PROJECT_DIR / "configs" / "yaml_sim_raman.yaml")
    approvals = next(d for d in config.support_devices if d.module_name == "asyncroscopy.safety.approvals")
    assert approvals.properties["mode_file"] != ["outputs/approvals/mode.json"]


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
