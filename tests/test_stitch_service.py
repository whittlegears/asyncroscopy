"""The Mapper's analysis goes through a StitchService: another backend needs no MCP and no analysis server."""

import subprocess
import sys

import pytest

from asyncroscopy.jobs.mapping.pipeline import Pipeline
from asyncroscopy.jobs.mapping.stitch_mcp import McpStitchService
from asyncroscopy.jobs.mapping.stitch_service import STITCH_SERVICES, StitchService, register_stitch_service


def test_importing_the_mapper_loads_neither_mcp_nor_the_mcp_adapter():
    code = (
        "import sys; import asyncroscopy.jobs.mapping.mapper; "
        "print([m for m in ('fastmcp', 'mcp', 'asyncroscopy.jobs.mapping.stitch_mcp') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_the_mcp_backend_is_found_by_name_and_keeps_the_protocol():
    assert STITCH_SERVICES.get("mcp") is McpStitchService
    assert isinstance(McpStitchService("http://127.0.0.1:9/mcp"), StitchService)


def test_a_pipeline_runs_on_a_registered_backend_with_no_mcp():
    @register_stitch_service
    class InMemory:
        name = "test_in_memory"

        def __init__(self, url):
            self.url, self.closed = url, False

        def list_pois(self, map, kind, limit):
            return {"pois": [{"id": "a", "x_m": 0.0, "y_m": 0.0, "score": 2.0}, {"id": "b", "x_m": 1e-9, "y_m": 0.0, "score": 1.0}]}

        def close(self):
            self.closed = True

    try:
        saved = []
        pipeline = Pipeline("nowhere", 0.0, lambda job: False, __import__("threading").RLock(), saved.append, print, backend="test_in_memory")
        job = {"resolved_plan": {"revisit": {"map": "m", "fov_m": 1e-7, "top_k": 3}}}
        assert pipeline.select(job) == {"phase": "select", "regions": 1}  # the two are closer than half a field: one image
        service = pipeline.service()
        assert service.url == "nowhere" and saved
        pipeline.close()
        assert service.closed
    finally:
        STITCH_SERVICES.unregister("test_in_memory")


def test_an_unknown_backend_is_refused_by_name():
    pipeline = Pipeline("nowhere", 0.0, lambda job: False, __import__("threading").RLock(), lambda job: None, print, backend="nonesuch")
    with pytest.raises(ValueError, match="unknown stitch service 'nonesuch'"):
        pipeline.service()
