"""``command``: tango.server.command that also tells agents what the command does.

Tango does not carry a command's Python docstring to clients. The MCP server
builds each tool's description from the command's ``doc_in`` and ``doc_out``,
so a command declared with only a docstring reaches an agent as
":return: (not documented)". This decorator copies the docstring's first
paragraph into ``doc_out`` when ``doc_out`` is not given:

    from asyncroscopy.utils.commands import command

    @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.")
    def indent(self, load_n):
        \"\"\"Make one indent at the current position. Returns JSON with hardness (GPa).\"\"\"

When ``doc_in`` does not start with ":param <name>:", the Python parameter's
name is added, so the agent's tool argument is called ``load_n`` rather than
``arg``. It accepts everything tango.server.command accepts, with or without
arguments.
"""

from __future__ import annotations

import inspect

from tango.server import command as tango_command


def summary(func) -> str:
    """First paragraph of a function's docstring, on one line."""
    doc = inspect.getdoc(func) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def command(f=None, **kwargs):
    def decorate(func):
        if not kwargs.get("doc_out"):
            kwargs["doc_out"] = summary(func)
        # The MCP server names the tool's argument after ":param <name>:" (otherwise "arg");
        # take the name from the Python signature when doc_in does not give one.
        params = [p for p in inspect.signature(func).parameters.values() if p.name != "self"]
        doc_in = kwargs.get("doc_in") or ""
        if params and ":param " not in doc_in:
            kwargs["doc_in"] = f":param {params[0].name}: {doc_in}".rstrip()
        return tango_command(func, **kwargs)

    return decorate(f) if f is not None else decorate
