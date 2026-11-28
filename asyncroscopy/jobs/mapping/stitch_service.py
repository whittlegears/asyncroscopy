"""What the Mapper asks of whatever stitches and explores its maps: the ``StitchService`` protocol.

The pipeline (pipeline.py) knows only this protocol, so ``jobs/`` runs with any backend. The Mapper's ``stitch_service``
property names one, built with the ``analysis_url`` it is given, the first time it is needed:

* ``mcp`` (the default, stitch_mcp.py) calls the analysis MCP server, and is the only code here that knows about MCP;
* another backend registers a class with ``@register_stitch_service``, ``ASYNCROSCOPY_PLUGINS`` or the
  ``asyncroscopy.stitch_services`` entry point.

``call(tool, arguments)`` returns the JSON-like reply of that tool of the analysis server (docs/Operation/agentic_mapping.md),
which may carry ``{"error": {"code", "message"}}``, and raises when the backend cannot be reached. Stitching and scoring go
through asyncroscopy.core.map_analysis over this call; the pipeline also calls ``list_pois``, ``add_poi``, ``map_info``,
``find_features`` and ``place_image``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from asyncroscopy.registry import Registry


@runtime_checkable
class StitchService(Protocol):
    """Call the analysis server's tools for a Mapper."""

    def call(self, tool: str, arguments: dict) -> dict:
        """The reply of ``tool`` called with ``arguments``."""
        ...

    def close(self) -> None:
        """Release any connection."""
        ...


STITCH_SERVICES: Registry = Registry("asyncroscopy.stitch_services", {"mcp": "asyncroscopy.jobs.mapping.stitch_mcp"}, kind="stitch service")
register_stitch_service = STITCH_SERVICES.register
