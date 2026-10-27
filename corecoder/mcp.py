"""MCP client supporting both Stdio and SSE transports.

Servers are configured in ./mcp.json (project root) or ~/.corecoder/mcp.json:

    {
      "mcpServers": {
        "amap-amap-sse": { "url": "https://mcp.amap.com/sse?key=xxx", "type": "sse" },
        "fs": { "command": "npx", "args": ["-y", "some-fs-server", "/tmp"], "type": "stdio" }
      }
    }

Stdio servers are subprocesses speaking JSON-RPC 2.0 over stdin/stdout.
SSE servers are remote endpoints speaking JSON-RPC 2.0 over HTTP POST + SSE GET.
Startup handshakes (`initialize`), pulls `tools/list`, and every remote tool
joins the agent as `mcp__<server>__<tool>`. A server that hangs or dies fails
that one call as an ordinary error string; it never kills the loop.
"""

import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from . import __version__
from .tools.base import Tool

log = logging.getLogger(__name__)

# ==========================================================
# 智能配置文件路径：优先项目根目录，备选全局用户目录
# ==========================================================
PROJECT_ROOT_CONFIG = Path(__file__).resolve().parent.parent / "mcp.json"
GLOBAL_CONFIG = Path.home() / ".corecoder" / "mcp.json"

def get_config_path() -> Path:
    if PROJECT_ROOT_CONFIG.exists():
        return PROJECT_ROOT_CONFIG
    return GLOBAL_CONFIG

PROTOCOL_VERSION = "2025-06-18"
INIT_TIMEOUT = 15  # seconds for initialize + tools/list at startup
CALL_TIMEOUT = 60  # seconds for one tools/call


class MCPError(RuntimeError):
    """Transport or protocol failure talking to one server."""


class BaseMCPClient:
    """Abstract base for MCP clients, handling the common JSON-RPC handshake and tool calling."""

    def __init__(self, name: str):
        self.name = name
        self.call_timeout = CALL_TIMEOUT
        self.tools: list['MCPTool'] = []
        self._next_id = 0
        self._id_lock = threading.Lock()

    def _get_next_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        raise NotImplementedError

    def _notify(self, method: str):
        raise NotImplementedError

    def initialize(self):
        """Perform the standard MCP handshake and fetch tools."""
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "corecoder", "version": __version__},
            }, INIT_TIMEOUT)
            self._notify("notifications/initialized")
            listed = self._request("tools/list", {}, INIT_TIMEOUT)
            self.tools = [MCPTool(self, t) for t in listed.get("tools", [])]
            log.info("✅ MCP server %r (%s) connected! Loaded %d tools.",
                     self.name, self.__class__.__name__, len(self.tools))
        except BaseException:
            self.close()  # a half-started server must not leak
            raise

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Run one remote tool and return its text content."""
        result = self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}, self.call_timeout
        )
        text = "\n".join(
            part.get("text", "")
            for part in result.get("content", [])
            if part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(text or f"{tool_name} reported an error")
        return text or json.dumps(result)

    def close(self):
        """Shut the server down. Safe to call twice."""
        raise NotImplementedError


class StdioMCPClient(BaseMCPClient):
    """One stdio server process: handshake, list, call, shut down.
    (Your original brilliant implementation, slightly refactored to inherit BaseMCPClient)
    """

    def __init__(self, name: str, command: str, args: list = (), env: dict | None = None):
        super().__init__(name)
        self._proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={**os.environ, **(env or {})},
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._dead: MCPError | None = None
        self._responses: dict[int, dict] = {}
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()

        threading.Thread(target=self._read_loop, daemon=True).start()
        self.initialize()

    def _notify(self, method: str):
        try:
            self._write({"jsonrpc": "2.0", "method": method, "params": {}})
        except MCPError:
            pass

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._cond:
            if self._dead is not None:
                raise self._dead
            req_id = self._get_next_id()

        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout

        with self._cond:
            while req_id not in self._responses:
                if self._dead is not None:
                    raise self._dead
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPError(f"MCP server {self.name!r} gave no answer to {method} in {timeout:g}s")
                self._cond.wait(left)
            msg = self._responses.pop(req_id)

        if "error" in msg:
            raise MCPError(f"MCP server {self.name!r} rejected {method}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result", {})

    def _write(self, msg: dict):
        try:
            with self._write_lock:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
        except OSError as e:
            raise MCPError(f"MCP server {self.name!r} is not writable: {e}") from e

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP server %r sent a non-JSON line, skipped", self.name)
                    continue
                if "id" not in msg:
                    continue
                with self._cond:
                    self._responses[msg["id"]] = msg
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        with self._cond:
            if self._dead is None:
                self._dead = MCPError(f"MCP server {self.name!r} exited")
            self._cond.notify_all()

    def close(self):
        if self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class SSEMCPClient(BaseMCPClient):
    """One remote SSE server: maintains a GET stream for responses, uses POST for requests."""

    def __init__(self, name: str, url: str):
        super().__init__(name)
        self.sse_url = url
        self.post_url: Optional[str] = None
        self._dead: MCPError | None = None
        self._responses: dict[int, dict] = {}
        self._cond = threading.Condition()

        # httpx client for HTTP requests
        self._client = httpx.Client(timeout=httpx.Timeout(self.call_timeout + 10.0))

        # Daemon thread to listen to the SSE stream
        self._sse_thread = threading.Thread(target=self._sse_listen_loop, daemon=True)
        self._sse_thread.start()

        # Block until the background thread receives the POST endpoint URL
        deadline = time.monotonic() + INIT_TIMEOUT
        while self.post_url is None and self._dead is None:
            if time.monotonic() > deadline:
                raise MCPError(f"SSE server {self.name!r} failed to provide endpoint URL in {INIT_TIMEOUT}s")
            time.sleep(0.1)

        if self._dead:
            raise self._dead

        self.initialize()

    def _sse_listen_loop(self):
        """Background thread: keeps the GET connection alive and parses SSE events."""
        try:
            with self._client.stream("GET", self.sse_url, headers={"Accept": "text/event-stream"}) as response:
                event_type = None
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()

                        # 1. Server tells us where to POST future requests
                        if event_type == "endpoint":
                            if data.startswith("http"):
                                self.post_url = data
                            else:
                                # Handle relative paths like /messages?sessionId=xxx
                                base_url = self.sse_url.rsplit('/', 1)[0]
                                self.post_url = f"{base_url}{data}" if data.startswith("/") else f"{base_url}/{data}"

                        # 2. Server sends a JSON-RPC response to our previous POST
                        elif event_type == "message":
                            try:
                                msg = json.loads(data)
                                if "id" in msg:
                                    with self._cond:
                                        self._responses[msg["id"]] = msg
                                        self._cond.notify_all()
                            except json.JSONDecodeError:
                                log.warning("SSE server %r sent invalid JSON: %s", self.name, data)

        except httpx.HTTPError as e:
            with self._cond:
                self._dead = MCPError(f"SSE server {self.name!r} connection error: {e}")
                self._cond.notify_all()
        except Exception as e:
            with self._cond:
                self._dead = MCPError(f"SSE server {self.name!r} unexpected error: {e}")
                self._cond.notify_all()

    def _notify(self, method: str):
        try:
            self._http_post({"jsonrpc": "2.0", "method": method, "params": {}})
        except MCPError:
            pass

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._cond:
            if self._dead is not None:
                raise self._dead
            req_id = self._get_next_id()

        # 1. Send request via HTTP POST
        self._http_post({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

        # 2. Wait for the response to arrive via the SSE GET stream
        deadline = time.monotonic() + timeout
        with self._cond:
            while req_id not in self._responses:
                if self._dead is not None:
                    raise self._dead
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPError(f"SSE server {self.name!r} gave no answer to {method} in {timeout:g}s")
                self._cond.wait(left)
            msg = self._responses.pop(req_id)

        if "error" in msg:
            raise MCPError(f"SSE server {self.name!r} rejected {method}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result", {})

    def _http_post(self, payload: dict):
        try:
            resp = self._client.post(self.post_url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise MCPError(f"SSE server {self.name!r} POST failed: {e}") from e

    def close(self):
        if self._client:
            self._client.close()


class MCPTool(Tool):
    """A tool living in an MCP server, registered under a mcp__ name so it can
    never collide with a built-in. Side effects are unknown, so it stays out of
    Permission.READ_ONLY and the consent gate asks first, like any mutating tool.
    """

    def __init__(self, client: BaseMCPClient, spec: dict):
        self._client = client
        self._remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description") or ""
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> str:
        return self._client.call_tool(self._remote_name, kwargs)


_live_clients: list[BaseMCPClient] = []


def load_mcp_tools(path: Optional[Path] = None) -> list[Tool]:
    """Start the configured servers and return their tools.

    Automatically detects 'sse' or 'stdio' based on the config.
    """
    if path is None:
        path = get_config_path()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.info(f"No MCP config found at {path}, skipping.")
        return []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring %s: %s", path, e)
        return []

    tools: list[Tool] = []
    servers = data.get("mcpServers") or {}

    if not servers:
        return []

    log.info(f"🚀 Loading {len(servers)} MCP servers from {path}...")

    for name, spec in servers.items():
        try:
            # Default to stdio for backward compatibility if 'type' is missing
            transport_type = spec.get("type", "stdio").lower()

            if transport_type == "sse":
                if "url" not in spec:
                    raise MCPError("SSE config missing 'url'")
                client = SSEMCPClient(name, spec["url"])
            else:
                if "command" not in spec:
                    raise MCPError("Stdio config missing 'command'")
                client = StdioMCPClient(name, spec["command"], spec.get("args", []), spec.get("env"))

        except (MCPError, OSError, KeyError, TypeError) as e:
            log.warning("⚠️ MCP server %r skipped: %s", name, e)
            continue

        _live_clients.append(client)
        tools.extend(client.tools)

    return tools


@atexit.register
def _shutdown():
    for client in _live_clients:
        try:
            client.close()
        except Exception:
            pass