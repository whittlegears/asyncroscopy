"""What the Mapper asks of whatever stitches and explores its maps: the ``StitchService`` protocol.

The pipeline (pipeline.py) knows only this protocol, so ``jobs/`` runs with any backend. The Mapper's ``stitch_service``
property names one, built with the ``analysis_url`` it is given, the first time it is needed:

* ``mcp`` (the default, stitch_mcp.py) calls the analysis MCP server, and is the only code here that knows about MCP;
* another backend registers a class with ``@register_stitch_service``, ``ASYNCROSCOPY_PLUGINS`` or the
  ``asyncroscopy.stitch_services`` entry point.

Every method returns the JSON-like reply of the operation of the same name on the analysis server
(docs/Operation/agentic_mapping.md), and may carry ``{"error": {"code", "message"}}``. A method raises when the backend
cannot be reached; the pipeline retries with backoff and, if it stays down, finishes the job with the tiles safe.
``assemble_map`` and ``score_map`` start a background job and reply with its ``job_id``; the three ``analysis_job_*``
methods follow it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from asyncroscopy.registry import Registry


@runtime_checkable
class StitchService(Protocol):
    """Stitch a finished run into a map, score it, and read and annotate the map."""

    def assemble_map(self, run: str, stage_sigma_m: float, max_drift_m_per_s: float, eds_pois: bool) -> dict:
        """Start stitching run ``run``, its search bounded by the stage error and the drift."""
        ...

    def score_map(self, map: str, top_k: int, tag: str) -> dict:
        """Start scoring a map for its ``top_k`` best regions; ``tag`` makes a retry replace what it added before."""
        ...

    def analysis_job_status(self, job_id: str, wait_s: float) -> dict:
        """A background job's state ("state": ..., "last_error": ...), waiting up to ``wait_s`` seconds for it to change."""
        ...

    def analysis_job_results(self, job_id: str) -> dict:
        """A finished job's "result_keys" and "summary"."""
        ...

    def analysis_job_abort(self, job_id: str) -> dict:
        """Stop a background job."""
        ...

    def map_info(self, map: str) -> dict:
        """What a stitched map covers."""
        ...

    def list_pois(self, map: str, kind: str, limit: int) -> dict:
        """A map's points of interest of one kind: {"pois": [...]}."""
        ...

    def add_poi(self, map: str, x_m: float, y_m: float, kind: str, reason: str, links: list[str], id: str) -> dict:
        """Add a point of interest (idempotent on ``id``)."""
        ...

    def find_features(self, image: str, n: int, min_size_m: float | None = None, max_size_m: float | None = None) -> dict:
        """The ``n`` most distinct features of an image, with its "shape" and "pixel_size_m"."""
        ...

    def place_image(self, image: str, parent: str, centre_xy_m: list[float], window_m: float) -> dict:
        """Place a finer image on its parent, searching ``window_m`` around ``centre_xy_m``."""
        ...

    def close(self) -> None:
        """Release any connection."""
        ...


STITCH_SERVICES: Registry = Registry("asyncroscopy.stitch_services", {"mcp": "asyncroscopy.jobs.mapping.stitch_mcp"}, kind="stitch service")
register_stitch_service = STITCH_SERVICES.register
