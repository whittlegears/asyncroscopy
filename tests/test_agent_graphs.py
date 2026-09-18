"""Graph tests: require the real LangGraph/LangChain packages (uv sync --extra agent)."""

import asyncio
import json
from pathlib import Path

import pytest

from tests._agent_deps import real_agent_deps_installed

pytestmark = pytest.mark.skipif(not real_agent_deps_installed(), reason="requires `uv sync --extra agent`")

if real_agent_deps_installed():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from asyncroscopy.agent.config import Agent
    from asyncroscopy.agent.graphs.react import build_react_graph
    from asyncroscopy.agent.graphs.supervisor import build_supervisor_graph, extract_json, parse_routing_decision, run_supervisor
    from asyncroscopy.agent.graphs.workflows import get_workflow_builder, list_workflows
    from asyncroscopy.agent.graphs.workflows.image_eds_survey import build_image_eds_survey_graph, deterministic_summary
    from asyncroscopy.agent.skills import SkillRegistry
    from asyncroscopy.agent.skills.tools import make_skill_tools
    from asyncroscopy.agent.streaming import stream_agent
    from asyncroscopy.agent.tools import filter_tools, find_tool, invoke_tool_strict, load_mcp_tools, tool_result_text
    from tests._fakes import FakeMicroscope, ScriptedChatModel

PROJECT_SKILLS = Path(__file__).resolve().parents[1] / "skills"


@pytest.fixture
def registry(tmp_path):
    base = tmp_path / "skills"
    (base / "eds-spectrum").mkdir(parents=True)
    (base / "eds-spectrum" / "SKILL.md").write_text(
        "---\nname: eds-spectrum\ndescription: Acquire an EDS spectrum.\n---\nUse detector_name='eds'. SECRET-BODY-TOKEN\n",
        encoding="utf-8",
    )
    return SkillRegistry.from_dirs(base)


class TestSurveyWorkflow:
    def test_graph_structure(self):
        graph = build_image_eds_survey_graph(FakeMicroscope().tools())
        nodes = set(graph.get_graph().nodes)
        assert {"acquire_image", "acquire_spectrum", "read_back", "summarize"} <= nodes
        mermaid = graph.get_graph().draw_mermaid()
        assert "acquire_image" in mermaid and "summarize" in mermaid

    def test_registered_workflow(self):
        assert "image_eds_survey" in list_workflows()
        assert get_workflow_builder("image_eds_survey") is build_image_eds_survey_graph
        with pytest.raises(KeyError, match="Unknown workflow"):
            get_workflow_builder("nope")

    def test_happy_path_without_model(self):
        fake = FakeMicroscope()
        graph = build_image_eds_survey_graph(fake.tools())
        result = asyncio.run(graph.ainvoke({"detector": "haadf", "spectrum_detector": "eds"}))
        assert result["image_key"] == "stem_image_HAADF_1.h5"
        assert result["spectrum_key"] == "spectrum_eds_1.h5"
        assert result["image_info"]["datasets"][0]["shape"] == [64, 64]
        assert "Si 0.61, O 0.39" in result["summary"]
        assert "(64, 64)" in result["summary"]
        assert "errors" not in result or not result["errors"]
        assert [c[0] for c in fake.calls] == ["image", "spectrum", "read", "read"]

    def test_image_failure_routes_to_summary(self):
        fake = FakeMicroscope(fail_image=True)
        graph = build_image_eds_survey_graph(fake.tools())
        result = asyncio.run(graph.ainvoke({}))
        assert result["errors"] and "acquire_image" in result["errors"][0]
        assert "scan generator offline" in result["summary"]
        assert "spectrum_key" not in result
        assert [c[0] for c in fake.calls] == ["image"]

    def test_spectrum_failure_keeps_image(self):
        fake = FakeMicroscope(fail_spectrum=True)
        graph = build_image_eds_survey_graph(fake.tools())
        result = asyncio.run(graph.ainvoke({}))
        assert result["image_key"] == "stem_image_HAADF_1.h5"
        assert any("acquire_spectrum" in e for e in result["errors"])
        assert "stem_image_HAADF_1.h5" in result["summary"]

    def test_model_rephrases_summary(self):
        model = ScriptedChatModel(responses=[AIMessage(content="Rephrased summary.")])
        graph = build_image_eds_survey_graph(FakeMicroscope().tools(), model=model)
        result = asyncio.run(graph.ainvoke({}))
        assert result["summary"] == "Rephrased summary."
        assert "Si 0.61" in model.prompts[0][-1].content

    def test_missing_tool_raises_early(self):
        with pytest.raises(KeyError, match="No tool matches"):
            build_image_eds_survey_graph(FakeMicroscope().tools()[2:])

    def test_deterministic_summary_empty(self):
        assert deterministic_summary({}) == "Nothing was acquired."


class TestReactGraph:
    def test_system_prompt_contains_skills_index(self, registry):
        model = ScriptedChatModel(responses=[AIMessage(content="done")])
        graph = build_react_graph(model, FakeMicroscope().tools(), registry)
        result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}))
        assert result["messages"][-1].content == "done"
        system = model.prompts[0][0]
        assert "eds-spectrum" in system.content and "Acquire an EDS spectrum." in system.content
        assert "SECRET-BODY-TOKEN" not in system.content
        assert {"skills_list", "skill_search", "skill_view", "skill_save"} <= {t["function"]["name"] if isinstance(t, dict) else t.name for t in model.bound_tools}

    def test_agent_loads_skill_then_answers(self, registry):
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "skill_view", "args": {"name": "eds-spectrum"}, "id": "c1"}]),
                AIMessage(content="I read the skill."),
            ]
        )
        graph = build_react_graph(model, FakeMicroscope().tools(), registry)
        result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="how do I take eds?")]}))
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1 and "SECRET-BODY-TOKEN" in tool_messages[0].content
        assert result["messages"][-1].content == "I read the skill."

    def test_streaming_returns_final_answer(self, registry):
        fake = FakeMicroscope()
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "DigitalTwin_acquire_spectrum", "args": {"detector_name": "eds"}, "id": "c1"}]),
                AIMessage(content="Spectrum saved."),
            ]
        )
        graph = build_react_graph(model, fake.tools(), registry)
        chunks: list[str] = []
        answer = asyncio.run(stream_agent(graph, [HumanMessage(content="eds please")], label="t", sink=chunks.append))
        assert answer == "Spectrum saved."
        log = "".join(chunks)
        assert "[EXECUTING TOOL]: DigitalTwin_acquire_spectrum" in log
        assert fake.calls[0] == ("spectrum", {"detector_name": "eds"})

    def test_skill_tools_names(self, registry):
        assert [t.name for t in make_skill_tools(registry)] == ["skills_list", "skill_search", "skill_view", "skill_save"]


class TestSupervisorGraph:
    def test_routing_helpers(self):
        assert extract_json('```json\n{"next": "a"}\n```') == '{"next": "a"}'
        assert parse_routing_decision('{"next": "eds", "task": "go"}', ["eds", "FINISH"], "FINISH") == ("eds", "go")
        assert parse_routing_decision("garbage", ["eds", "FINISH"], "FINISH") == ("FINISH", "")

    def test_routes_then_finishes(self):
        agents = [Agent(name="image", system_prompt="images", tools=["*image*"]), Agent(name="eds", system_prompt="eds", tools=["*spectrum*"])]
        supervisor = ScriptedChatModel(
            responses=[
                AIMessage(content='{"next": "eds", "task": "acquire a spectrum"}'),
                AIMessage(content='{"next": "FINISH", "task": ""}'),
            ]
        )
        built = []

        def build_worker(agent):
            built.append(agent.name)
            return f"executor-{agent.name}"

        async def run_worker(executor, messages, label):
            return f"{executor} did: {messages[0].content}"

        graph = build_supervisor_graph(supervisor, agents, build_worker, run_worker)
        assert {"Supervisor", "image", "eds"} <= set(graph.get_graph().nodes)
        answer = asyncio.run(run_supervisor(graph, "survey", max_steps=10))
        assert answer == "[eds]: executor-eds did: acquire a spectrum"
        assert sorted(built) == ["eds", "image"]

    def test_rejects_duplicate_names(self):
        agents = [Agent(name="a", system_prompt="", tools=["*"]), Agent(name="a", system_prompt="", tools=["*"])]
        with pytest.raises(ValueError):
            build_supervisor_graph(ScriptedChatModel(responses=[]), agents, lambda a: None, None)


class TestMcpToolAdapter:
    def _server(self):
        from fastmcp import FastMCP
        from fastmcp.tools import ToolResult
        from fastmcp.utilities.types import Image

        server = FastMCP("fake")

        @server.tool()
        def DigitalTwin_acquire_scanned_image(detector_list: list[str]) -> ToolResult:
            """Acquire image"""
            key = f"stem_image_{detector_list[0].upper()}_1.h5"
            return ToolResult(content=[key, Image(data=b"\x89PNG", format="png")], structured_content={"result": key})

        @server.tool()
        def get_data_from_key(key: str, max_values: int = 64) -> dict:
            """Describe data"""
            return {"key": key, "datasets": [{"name": "spectrum", "shape": [2], "attrs": {"elements": ["Si", "O"]}, "preview": [0.6, 0.4]}]}

        @server.tool()
        def boom() -> str:
            """Fails"""
            raise RuntimeError("kaboom")

        return server

    def test_tools_round_trip_in_memory(self):
        tools = asyncio.run(load_mcp_tools(self._server()))
        assert [t.name for t in tools] == ["DigitalTwin_acquire_scanned_image", "get_data_from_key", "boom"]
        image_tool = find_tool(tools, "*_acquire_scanned_image")
        assert image_tool.args == {"detector_list": {"items": {"type": "string"}, "type": "array"}}

        content = asyncio.run(image_tool.ainvoke({"detector_list": ["haadf"]}))
        assert tool_result_text(content) == "stem_image_HAADF_1.h5"

        message = asyncio.run(image_tool.ainvoke({"name": image_tool.name, "args": {"detector_list": ["haadf"]}, "id": "c1", "type": "tool_call"}))
        assert isinstance(message, ToolMessage) and message.content == "stem_image_HAADF_1.h5"
        assert message.artifact and message.artifact[0]["type"] == "image"

        data = json.loads(asyncio.run(tools[1].ainvoke({"key": "k"})))
        assert data["datasets"][0]["attrs"]["elements"] == ["Si", "O"]

        assert "kaboom" in asyncio.run(tools[2].ainvoke({}))  # error string for the agent
        with pytest.raises(Exception, match="kaboom"):
            asyncio.run(invoke_tool_strict(tools[2], {}))  # exception for workflows

        assert tools[0].invoke({"detector_list": ["bf"]}) == "stem_image_BF_1.h5"  # sync path

    def test_survey_workflow_over_mcp_tools(self):
        from fastmcp import FastMCP

        server = FastMCP("twin")

        @server.tool()
        def DigitalTwin_acquire_scanned_image(detector_list: list[str]) -> str:
            """img"""
            return "stem_image_HAADF_2.h5"

        @server.tool()
        def DigitalTwin_acquire_spectrum(detector_name: str) -> str:
            """spec"""
            return "spectrum_eds_2.h5"

        @server.tool()
        def get_data_from_key(key: str, max_values: int = 64) -> dict:
            """data"""
            if key.startswith("spectrum"):
                return {"datasets": [{"name": "spectrum", "shape": [3], "attrs": {"elements": ["Si", "O", "C"]}, "preview": [0.5, 0.3, 0.2]}]}
            return {"datasets": [{"name": "image/HAADF", "shape": [128, 128], "dtype": "float32", "attrs": {}, "preview": [0.0]}]}

        tools = asyncio.run(load_mcp_tools(server))
        graph = build_image_eds_survey_graph(tools)
        result = asyncio.run(graph.ainvoke({}))
        assert result["image_key"] == "stem_image_HAADF_2.h5"
        assert "Si 0.50, O 0.30, C 0.20" in result["summary"]
        assert "(128, 128)" in result["summary"]

    def test_filter_tools(self):
        tools = FakeMicroscope().tools()
        assert filter_tools(tools, ["*"]) is tools
        assert [t.name for t in filter_tools(tools, ["*_acquire_*"])] == ["DigitalTwin_acquire_scanned_image", "DigitalTwin_acquire_spectrum"]


class TestStudioFactories:
    def test_factories_build_without_mcp(self, monkeypatch, tmp_path):
        import asyncroscopy.agent.studio as studio

        async def failing(*args, **kwargs):
            raise ConnectionError("no server")

        monkeypatch.setattr(studio, "load_mcp_tools", failing)
        monkeypatch.setattr(studio, "_model", lambda settings: ScriptedChatModel(responses=[]))
        config = {"configurable": {"mcp_url": "http://127.0.0.1:1/mcp", "skills_dirs": str(PROJECT_SKILLS)}}

        with pytest.warns(UserWarning):
            react = asyncio.run(studio.make_react_graph(config))
        assert "tools" in set(react.get_graph().nodes)

        with pytest.warns(UserWarning):
            survey = asyncio.run(studio.make_image_eds_survey_graph(config))
        assert {"acquire_image", "summarize"} <= set(survey.get_graph().nodes)
        result = asyncio.run(survey.ainvoke({}))
        assert result["errors"] and "MCP server not connected" in result["errors"][0]

        with pytest.warns(UserWarning):
            supervisor = asyncio.run(studio.make_supervisor_graph(config))
        assert {"Supervisor", "imaging", "spectroscopy"} <= set(supervisor.get_graph().nodes)

    def test_settings_from_config_overrides_env(self, monkeypatch):
        import asyncroscopy.agent.studio as studio

        monkeypatch.setenv("ASYNCROSCOPY_AGENT_PROVIDER", "openai")
        monkeypatch.setenv("ASYNCROSCOPY_AGENT_MODEL", "gpt-4o-mini")
        settings = studio.settings_from_config({"configurable": {"model": "gpt-4.1", "temperature": 0.5}})
        assert settings.provider == "openai"
        assert settings.model == "gpt-4.1"
        assert settings.temperature == 0.5
