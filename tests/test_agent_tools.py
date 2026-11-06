"""endpoint_info.agent_tools is exactly the tool set the general agent gives its model."""

import asyncio
import fnmatch
from types import SimpleNamespace

import pytest

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.graphs import general
from asyncroscopy.agents.skill_runs import RUN_TOOLS
from asyncroscopy.mcp.agent_tools import for_model
from asyncroscopy.mcp.mcp_server import MCPServer

NATIVE = ["list_data", "view_data", "list_devices", "list_skills", "load_skill", "endpoint_info", "action_space", "run_job", "wait_for_job", "approval_status"]
TWIN = {
    "DigitalTwin": ["get_stage", "move_stage", "get_parameters", "acquire_scanned_image", "acquire_spectrum", "set_fov", "get_fov",
                    "set_defocus", "place_beam", "blank_beam", "unblank_beam", "set_beam_current", "get_detectors", "State", "Status"],
    "Mapper": ["reserve", "start", "job_status", "results", "release", "abort", "State"],
    "Approvals": ["get_mode", "set_mode"],
}
RAMAN = {"Raman": ["acquire", "set_laser_power", "get_laser_power", "move_stage", "State", "Status"]}
ANALYSIS = ["assemble_map", "score_map", "map_info", "get_region", "list_pois", "add_poi", "list_scorers",
            "analysis_job_status", "analysis_job_results", "calculate_probe_size"]


class EndpointConfig(AgentConfig):
    """An AgentConfig whose endpoint and analysis server offer fixed tool names."""

    def __init__(self, own, info):
        super().__init__(analysis_url=info["analysis_url"] or "")
        self.own, self.endpoint = sorted(own), info

    async def info(self):
        return self.endpoint

    async def tools(self, include_analysis=True, allow=None):
        names = self.own + (ANALYSIS if include_analysis else [])
        return [SimpleNamespace(name=n) for n in names if any(fnmatch.fnmatch(n, p) for p in allow or ["*"])]

    def chat_model(self):
        return "model"


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

    def make(devices, analysis_url):
        server = MCPServer("t", "localhost", 1, blocked_classes=[], blocked_functions={}, data_device_address="d",
                           verbose=False, analysis_url=analysis_url)
        server.tools = {prefix: dict.fromkeys(commands) for prefix, commands in devices.items()}
        server.native_names = set(NATIVE)
        return server

    return make


def general_agent_tools(server, monkeypatch, tmp_path):
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_MEMORY", str(tmp_path / "memory"))
    monkeypatch.delenv("ASYNCROSCOPY_AGENT_TOOLS", raising=False)
    given = {}
    monkeypatch.setattr(general, "create_agent", lambda model, tools, **_: given.setdefault("tools", tools))
    config = EndpointConfig(server.tool_names(), server.discovery.endpoint_info())
    asyncio.run(general.make_graph({"configurable": {"agent_config": config, "claim_check": False}}))
    return sorted(t.name for t in given["tools"]), config


@pytest.mark.parametrize("devices,analysis_url", [
    (TWIN, "http://127.0.0.1:8003/mcp"),
    (TWIN, ""),
    (RAMAN, ""),
])
def test_endpoint_info_agent_tools_are_the_general_agents_tools(endpoint, monkeypatch, tmp_path, devices, analysis_url):
    server = endpoint(devices, analysis_url)
    picked, config = general_agent_tools(server, monkeypatch, tmp_path)
    patterns = server.discovery.endpoint_info()["agent_tools"]
    offered = config.own + ANALYSIS
    # run_skill and wait_for_skill are the agent's own (they send a skill's listed calls): offered once a skill's run list can be sent.
    assert [n for n in picked if n not in RUN_TOOLS] == sorted(n for n in offered if for_model(n, patterns))
    assert not set(RUN_TOOLS) & set(picked)  # no run list here has every tool it calls (place_beam_at, find_features)
    assert "endpoint_info" not in picked and "list_skills" not in picked and "load_skill" in picked


def test_a_stem_gets_the_curated_set_and_the_analysis_waits_only_without_an_analysis_server(endpoint, monkeypatch, tmp_path):
    with_analysis, _ = general_agent_tools(endpoint(TWIN, "http://127.0.0.1:8003/mcp"), monkeypatch, tmp_path)
    assert "DigitalTwin_get_stage" in with_analysis and "assemble_map" in with_analysis
    assert "DigitalTwin_set_beam_current" not in with_analysis and "Mapper_reserve" not in with_analysis
    assert "analysis_job_status" not in with_analysis and "calculate_probe_size" not in with_analysis
    without, _ = general_agent_tools(endpoint(TWIN, ""), monkeypatch, tmp_path)
    assert set(without) - set(with_analysis) == {"analysis_job_status", "analysis_job_results"}


def test_another_instrument_gets_its_own_tools_but_not_state_or_analysis(endpoint, monkeypatch, tmp_path):
    picked, _ = general_agent_tools(endpoint(RAMAN, ""), monkeypatch, tmp_path)
    assert "Raman_set_laser_power" in picked and "Raman_State" not in picked and "assemble_map" not in picked


# An AFM YAML twin mapped through the yaml profile: the Mapper must not make it look like a STEM.
AFM_MAPPED = {
    "AFMTwin": ["approach_tip", "retract_tip", "move_xy", "get_status", "acquire_topography", "acquire_force_curve",
                "get_setpoint_v", "set_setpoint_v", "get_map_profile", "reserve", "start", "abort", "job_status", "results",
                "release", "State", "Status"],
    "Mapper": ["reserve", "start", "job_status", "results", "release", "abort", "get_reservation", "State"],
}


def test_a_mapper_on_another_instrument_keeps_its_own_tools_and_gets_the_map_tools(endpoint, monkeypatch, tmp_path):
    from asyncroscopy.mcp.agent_tools import is_stem_endpoint

    server = endpoint(AFM_MAPPED, "http://127.0.0.1:8003/mcp")
    assert not is_stem_endpoint(server.tool_names())
    picked, _ = general_agent_tools(server, monkeypatch, tmp_path)
    for tool in ("AFMTwin_retract_tip", "AFMTwin_approach_tip", "AFMTwin_move_xy", "AFMTwin_abort", "Mapper_abort"):
        assert tool in picked, tool
    assert "assemble_map" in picked and "score_map" in picked  # its maps are stitched and scored too
    assert "AFMTwin_State" not in picked and "analysis_job_status" not in picked
    without, _ = general_agent_tools(endpoint(AFM_MAPPED, ""), monkeypatch, tmp_path)
    assert set(without) - set(picked) == {"analysis_job_status", "analysis_job_results"}


def test_the_stem_twins_curated_set_is_unchanged(endpoint):
    from asyncroscopy.agents.profile import load_profile
    from asyncroscopy.mcp.agent_tools import is_stem_endpoint

    server = endpoint(TWIN, "http://127.0.0.1:8003/mcp")
    assert is_stem_endpoint(server.tool_names())
    tools = load_profile("default").tools  # the curated lists are the default agent profile's data
    assert server.discovery.endpoint_info()["agent_tools"] == [*tools.stem, *tools.generic, "load_skill"]
    assert "*_move_stage" in tools.stem and "run_job" in tools.generic


def test_the_team_operator_on_a_mapped_afm_gets_the_afms_own_tools(tmp_path, monkeypatch):
    from asyncroscopy.agents.graphs.team import TeamGraph

    monkeypatch.setenv("ASYNCROSCOPY_AGENT_MEMORY", str(tmp_path / "memory"))
    names = [f"{prefix}_{c}" for prefix, cs in AFM_MAPPED.items() for c in cs] + NATIVE
    config = EndpointConfig(names, {"analysis_url": "http://127.0.0.1:8003/mcp"})
    roles = asyncio.run(TeamGraph(config).tools_by_role())
    operator = {t.name for t in roles["operator"]}
    assert {"AFMTwin_retract_tip", "AFMTwin_approach_tip", "AFMTwin_move_xy", "AFMTwin_abort", "Mapper_abort"} <= operator
