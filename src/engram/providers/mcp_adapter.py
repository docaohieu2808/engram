"""MCP (Model Context Protocol) server adapter for memory providers."""

from __future__ import annotations

import json
import logging
from typing import Any

from engram.providers.base import MemoryProvider, ProviderResult

logger = logging.getLogger("engram.providers.mcp")


class McpAdapter(MemoryProvider):
    """Connects to an MCP server and calls a search tool."""

    def __init__(
        self,
        name: str,
        command: str,
        tool_name: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, provider_type="mcp", **kwargs)
        self.command = command
        self.tool_name = tool_name
        self.args = args or []
        self.env = env or {}
        self._client = None
        self._session = None
        self._transport = None

    async def _ensure_client(self):
        """Lazy-init MCP client connection via stdio transport."""
        if self._session is not None:
            return

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parts = self.command.split()
        server_params = StdioServerParameters(
            command=parts[0],
            args=parts[1:] + self.args,
            env=self.env or None,
        )

        transport = stdio_client(server_params)
        try:
            read_stream, write_stream = await transport.__aenter__()
            self._transport = transport
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            self._session = session
            await self._session.initialize()
        except Exception:
            # Clean up partially-initialized resources
            if self._session:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception as cleanup_exc:
                    logger.debug("mcp_adapter: session cleanup error: %s", cleanup_exc)
                self._session = None
            if self._transport:
                try:
                    await self._transport.__aexit__(None, None, None)
                except Exception as cleanup_exc:
                    logger.debug("mcp_adapter: transport cleanup error: %s", cleanup_exc)
                self._transport = None
            raise

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        await self._ensure_client()

        result = await self._session.call_tool(
            self.tool_name,
            arguments={"query": query, "limit": limit},
        )

        if self.debug:
            logger.debug("[%s] MCP result: %s", self.name, str(result)[:500])

        results = []
        for item in result.content:
            text = item.text if hasattr(item, "text") else str(item)
            if not text:
                continue

            # Try to parse as JSON for structured results
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for i, entry in enumerate(parsed[:limit]):
                        content = entry.get("content", str(entry)) if isinstance(entry, dict) else str(entry)
                        results.append(ProviderResult(
                            content=content,
                            score=1.0 - (i * 0.1),
                            source=self.name,
                        ))
                    return results
            except (json.JSONDecodeError, TypeError):
                pass

            results.append(ProviderResult(
                content=text,
                score=1.0,
                source=self.name,
            ))

        return results[:limit]

    async def health(self) -> bool:
        try:
            await self._ensure_client()
            # List tools to verify connection
            tools = await self._session.list_tools()
            return any(t.name == self.tool_name for t in tools.tools)
        except Exception:
            return False

    async def close(self):
        """Clean up MCP session and transport."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport:
            await self._transport.__aexit__(None, None, None)
            self._transport = None
        self._client = None
