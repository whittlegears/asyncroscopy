"""The Instrument Trace, online: follow every trace file of a stack at once, in order.

``asyncroscopy.trace.stream`` is the Python API (``read_trace`` for a finished run,
``follow_trace`` / ``TraceStream`` to follow one live); ``python -m asyncroscopy.trace.tail``
is the command line; the operator MCP endpoint's ``trace_since`` tool serves the GUI.
See docs/Operation/trace.md.
"""

from asyncroscopy.trace.stream import TraceStream, follow_trace, read_trace

__all__ = ["TraceStream", "follow_trace", "read_trace"]
