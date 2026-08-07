"""Grid stitching and zoomable sample-map generation for asyncroscopy.

The pipeline runs across two independent MCP servers:

1. The instrument server (``asyncroscopy.mcp.mcp_server``) exposes granular
   acquisition tools (``move_stage``, ``acquire_scanned_image``,
   ``get_data_from_key``, ...) plus the convenience command
   ``ElectronMicroscope.acquire_map_grid`` for planning a stage grid. This
   module's :func:`plan_grid` backs that command and has no image-processing
   dependencies, so importing it does not pull opencv/pillow into the Tango
   device process.
2. A standalone ``asyncroscopy.mcp.mapping_mcp_server`` (separately started,
   separately connected in the GUI) does the actual stitching and export:
   :class:`~asyncroscopy.mapping.grid_stitcher.GridStitcher`,
   :func:`~asyncroscopy.mapping.map_export.export_map`, and
   :func:`~asyncroscopy.mapping.overview.register_overview` for placing
   lower-magnification "zoom out" acquisitions into the same map.

Only the dependency-free helpers are re-exported at package level; import the
cv2-dependent stitching/export/registration APIs from their submodules
directly (``asyncroscopy.mapping.grid_stitcher``,
``asyncroscopy.mapping.map_export``, ``asyncroscopy.mapping.montage``,
``asyncroscopy.mapping.overview``) so instrument-side code never accidentally
imports opencv.
"""

from .grid_math import plan_grid, parse_map_grid_result

__all__ = [
    "plan_grid",
    "parse_map_grid_result",
]
