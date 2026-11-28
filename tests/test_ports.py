import socket

from asyncroscopy.mcp.ports import check_port_free, report_busy_port


def test_a_free_port_is_reported_free(capsys) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert check_port_free("127.0.0.1", port) is None
    assert report_busy_port("127.0.0.1", port, "agent MCP server") is False
    assert capsys.readouterr().out == ""


def test_a_taken_port_prints_the_line_the_startup_guis_parse(capsys) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        assert report_busy_port("127.0.0.1", port, "agent MCP server") is True
    out = capsys.readouterr().out
    assert out.startswith(f"MCP ERROR: http://127.0.0.1:{port} is already in use")
    assert "another agent MCP server" in out
