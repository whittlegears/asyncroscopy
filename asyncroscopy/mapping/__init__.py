"""Instrument-side pieces of sample mapping.

asyncroscopy does not stitch. The grid scan runs on the instrument
(``ElectronMicroscope.acquire_map_grid``), and the pyTEMlib analysis MCP
server (``python -m pyTEMlib.analysis_server``) stitches, scores and keeps
points of interest and notes. The LangGraph workflow
``asyncroscopy.agent_workflows.sample_map`` joins them.

- :mod:`.calibration` measures which way each stage axis moves the image
  before a grid, and plans the grid in image directions.
- :mod:`.run_file` turns a grid manifest into the map run pyTEMlib reads, with
  tile centres converted into its image frame (columns +x, rows +y).

Neither imports Tango or the analysis server.
"""
