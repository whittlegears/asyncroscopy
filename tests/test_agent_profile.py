"""Agent profiles: the model-specific settings live in agents/profiles/*.yaml, read in one place."""

import pytest

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.profile import MIDDLEWARE, load_profile


def test_the_default_profile_keeps_todays_model_and_middleware():
    profile = load_profile("default")
    assert profile.model == "qwen3:8b" and profile.num_ctx == 8192 and profile.middleware == MIDDLEWARE
    config = AgentConfig()
    assert config.model == "qwen3:8b" and config.model_provider == "ollama" and config.middleware == list(MIDDLEWARE)
    assert AgentConfig(model="other:1b").model == "other:1b"


def test_from_env_picks_a_profile_and_a_model(monkeypatch, tmp_path):
    lean = tmp_path / "lean.yaml"
    lean.write_text("model: tiny:1b\nnum_ctx: 4096\nmiddleware: [job_wait]\ntools: {stem: ['*_move_stage']}\n", encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_PROFILE", str(lean))
    monkeypatch.delenv("ASYNCROSCOPY_MODEL", raising=False)
    config = AgentConfig.from_env()
    assert (config.model, config.num_ctx, config.middleware) == ("tiny:1b", 4096, ["job_wait"])
    assert config.tool_lists().stem == ("*_move_stage",)
    monkeypatch.setenv("ASYNCROSCOPY_MODEL", "chosen:7b")
    assert AgentConfig.from_env().model == "chosen:7b"


def test_a_profile_with_unknown_middleware_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: x\nmiddleware: [skils]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown middleware"):
        load_profile(str(bad))


def test_the_middleware_is_built_in_the_profiles_order():
    from asyncroscopy.agents.middleware import build_middleware

    made = build_middleware(["job_wait", "empty_reply"], prompt="p", tool_names={"run_job"}, offered={"run_job"}, tools=[])
    assert [type(m).__name__ for m in made] == ["JobWaitMiddleware", "EmptyReplyMiddleware"]
    with pytest.raises(ValueError):
        build_middleware(["nope"], prompt="p", tool_names=set(), offered=set(), tools=[])
