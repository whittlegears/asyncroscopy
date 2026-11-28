"""Images a tool returns (segment_sam, find_atoms_blobnet): kept for the chat, shown to a model that reads images,
carried by query_agent's answer, and never counted as reply text."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from asyncroscopy.agents.goals import content_text, trace_from_messages
from asyncroscopy.agents.mcp_tools import McpServer, collect_images
from asyncroscopy.mcp.agent_mcp_server import with_images

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


class ImageServer(McpServer):
    """An McpServer whose one tool replies with numbers and a picture, without a network."""

    def __init__(self, images: bool) -> None:
        super().__init__("http://unused", images=images)

    async def list_tools(self):
        return [SimpleNamespace(name="segment_sam", description="Segment an image.",
                                inputSchema={"type": "object", "properties": {"dataset": {"type": "string"}}})]

    def _client(self):  # only the transport is replaced
        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def call_tool(self, name, arguments, raise_on_error=False):
                return SimpleNamespace(is_error=False, structured_content=None,
                                       content=[SimpleNamespace(type="text", text='{"total": 2}'),
                                                SimpleNamespace(type="image", data=PNG, mimeType="image/png")])

        return Client()


def reply(images: bool) -> ToolMessage:
    tool = asyncio.run(ImageServer(images).langchain_tools())[0]
    return asyncio.run(tool.ainvoke({"type": "tool_call", "id": "c1", "name": "segment_sam", "args": {"dataset": "k"}}))


def test_a_model_that_does_not_read_images_gets_the_numbers_and_is_told_the_user_has_the_picture():
    message = reply(images=False)
    assert isinstance(message.content, str) and '"total": 2' in message.content and "image" in message.content
    assert PNG not in message.content
    assert message.artifact == {"images": [{"mime_type": "image/png", "base64": PNG}]}  # kept for the chat


def test_a_model_that_reads_images_is_shown_the_picture_beside_the_numbers():
    message = reply(images=True)
    assert message.content[0]["type"] == "text" and '"total": 2' in message.content[0]["text"]
    assert message.content[1] == {"type": "image", "base64": PNG, "mime_type": "image/png"}
    assert message.artifact["images"][0]["base64"] == PNG


def test_the_picture_is_not_counted_as_reply_text():
    message = reply(images=True)
    assert content_text(message.content) == message.content[0]["text"]
    call = AIMessage(content="", tool_calls=[{"id": "c1", "name": "segment_sam", "args": {"dataset": "k"}}])
    traced = trace_from_messages([call, message])
    assert PNG not in traced[0].result and '"total": 2' in traced[0].result


def test_the_agent_servers_answer_carries_the_pictures_its_tools_returned():
    with collect_images() as images:
        reply(images=False)
    assert images == [{"tool": "segment_sam", "mime_type": "image/png", "base64": PNG}]
    answer = with_images("Two particles.", images)
    kinds = [type(block).__name__ for block in answer.content]
    assert kinds == ["TextContent", "ImageContent"] and answer.content[0].text.startswith("Two particles.")
    assert answer.content[1].data == PNG and answer.content[1].mime_type == "image/png"
    assert with_images("No pictures.", []) == "No pictures."
