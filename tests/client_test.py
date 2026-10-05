"""
scripts/test_client.py

Minimal MCP client that connects to mcp_server/server.py over stdio and
calls all 4 tools with real questions from golden_set.json — this is
the actual protocol path (client -> transport -> server -> tool),
not just calling the Python functions directly like the unit tests do.

Run with:
    python scripts/test_client.py

Requires the `mcp` package to be installed (the client lives in the
same SDK as the server).
"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Spawns `python -m mcp_server.server` as a subprocess and talks to it
# over stdin/stdout — exactly how Claude Desktop or any other real MCP
# client would connect to this server.
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server.server"],
)


async def main() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== Tools exposed by the server ===")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n=== list_documents() ===")
            result = await session.call_tool("list_documents", {})
            print(_first_text(result))

            print("\n=== search_policy('how many vacation days do I get?') ===")
            result = await session.call_tool(
                "search_policy", {"query": "how many vacation days do I get?", "top_k": 3}
            )
            print(_first_text(result))

            print("\n=== cite_source('how many vacation days do I get?') ===")
            result = await session.call_tool(
                "cite_source", {"query": "how many vacation days do I get?"}
            )
            print(_first_text(result))

            print("\n=== cite_source('how many days of parental leave?') [trick question] ===")
            result = await session.call_tool(
                "cite_source", {"query": "how many days of parental leave does the company offer?"}
            )
            print(_first_text(result))

            print("\n=== summarize_document('vacation_policy.md') ===")
            result = await session.call_tool(
                "summarize_document", {"doc_id": "vacation_policy.md"}
            )
            print(_first_text(result))


def _first_text(call_tool_result) -> str:
    """Tool results come back as a list of content blocks; grab the
    first text block and pretty-print it if it's JSON."""
    for block in call_tool_result.content:
        if block.type == "text":
            try:
                return json.dumps(json.loads(block.text), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return block.text
    return "(no text content returned)"


if __name__ == "__main__":
    asyncio.run(main())