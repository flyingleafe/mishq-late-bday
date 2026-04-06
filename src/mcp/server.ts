import { randomUUID } from "node:crypto";
import { createSuttaRetriever } from "../retriever/index.ts";

type JsonRpcId = string | number | null;

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: JsonRpcId;
  method: string;
  params?: unknown;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: JsonRpcId;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

interface ToolArguments {
  user_query: string;
}

interface Session {
  id: string;
  initializedAt: string;
}

const SERVER_INFO = {
  name: "sutta-mcp-server",
  version: "0.2.0",
} as const;

const MCP_PROTOCOL_VERSION = "2025-03-26";
const MCP_SESSION_HEADER = "mcp-session-id";

const GET_SUTTA_TOOL = {
  name: "get-sutta-tool",
  description:
    "Use this tool whenever a user contemplates on a philosophical question about life, love or career. It accepts `user_query: str`, finds the most appropriate Buddhist sutta from the Pali Canon, and returns that sutta.",
  inputSchema: {
    type: "object",
    properties: {
      user_query: {
        type: "string",
        description: "The user's philosophical question or contemplation.",
      },
    },
    required: ["user_query"],
    additionalProperties: false,
  },
} as const;

const retriever = createSuttaRetriever();
const sessions = new Map<string, Session>();

export function startMcpServer() {
  const config = getServerConfig();
  const server = Bun.serve({
    hostname: config.host,
    port: config.port,
    fetch: (request) => handleHttpRequest(request, config.path),
  });

  console.error(
    `MCP HTTP server listening on http://${server.hostname}:${server.port}${config.path}`,
  );

  return server;
}

function getServerConfig() {
  return {
    host: process.env.HOST ?? "127.0.0.1",
    port: Number(process.env.PORT ?? "3000"),
    path: process.env.MCP_PATH ?? "/mcp",
  };
}

async function handleHttpRequest(
  request: Request,
  mcpPath: string,
): Promise<Response> {
  const url = new URL(request.url);

  if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/playground")) {
    return htmlResponse(renderPlaygroundHtml(mcpPath));
  }

  if (request.method === "GET" && url.pathname === "/healthz") {
    return Response.json({ ok: true, service: SERVER_INFO.name });
  }

  if (request.method === "GET") {
    if (url.pathname !== mcpPath) {
      return new Response("Not found", { status: 404 });
    }

    return new Response("SSE is not enabled on this MCP server.", {
      status: 405,
      headers: {
        Allow: "POST, DELETE",
      },
    });
  }

  if (request.method === "DELETE") {
    return handleDelete(request, mcpPath);
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", {
      status: 405,
      headers: {
        Allow: "POST, GET, DELETE",
      },
    });
  }

  if (url.pathname !== mcpPath) {
    return new Response("Not found", { status: 404 });
  }

  const originValidation = validateOrigin(request);
  if (originValidation) {
    return originValidation;
  }

  const accept = request.headers.get("accept") ?? "";
  const acceptsJson = accept.includes("application/json");
  const acceptsEventStream = accept.includes("text/event-stream");

  if (!acceptsJson || !acceptsEventStream) {
    return errorHttpResponse(
      406,
      -32000,
      "Accept header must include application/json and text/event-stream.",
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorHttpResponse(400, -32700, "Invalid JSON body.");
  }

  if (Array.isArray(body)) {
    return errorHttpResponse(
      400,
      -32600,
      "Batch requests are not supported by this server.",
    );
  }

  if (!body || typeof body !== "object") {
    return errorHttpResponse(400, -32600, "JSON-RPC body must be an object.");
  }

  const message = body as JsonRpcRequest;
  const result = await handleMessage(message, request);

  if (!result) {
    return new Response(null, { status: 202 });
  }

  return result;
}

async function handleMessage(
  message: JsonRpcRequest,
  request: Request,
): Promise<Response | null> {
  if (!("id" in message)) {
    return null;
  }

  try {
    switch (message.method) {
      case "initialize":
        return jsonResponse(
          success(message.id ?? null, {
            protocolVersion: MCP_PROTOCOL_VERSION,
            capabilities: {
              tools: {},
            },
            serverInfo: SERVER_INFO,
          }),
          {
            [MCP_SESSION_HEADER]: createSession().id,
          },
        );
      case "ping":
        requireSession(request);
        return jsonResponse(success(message.id ?? null, {}));
      case "tools/list":
        requireSession(request);
        return jsonResponse(
          success(message.id ?? null, {
            tools: [GET_SUTTA_TOOL],
          }),
        );
      case "tools/call":
        requireSession(request);
        return jsonResponse(
          success(message.id ?? null, await callTool(message.params)),
        );
      default:
        requireSession(request);
        return jsonResponse(
          failure(message.id ?? null, -32601, `Method not found: ${message.method}`),
        );
    }
  } catch (error) {
    const messageText =
      error instanceof Error ? error.message : "Unknown server error";
    const status = messageText.startsWith("Missing or invalid session")
      ? 400
      : messageText.startsWith("Unknown session")
        ? 404
        : 500;
    return jsonResponse(failure(message.id ?? null, -32000, messageText), status);
  }
}

async function callTool(params: unknown) {
  const parsed = parseToolCall(params);

  if (parsed.name !== GET_SUTTA_TOOL.name) {
    throw new Error(`Unknown tool: ${parsed.name}`);
  }

  const sutta = await retriever.retrieve(parsed.arguments.user_query);

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(sutta, null, 2),
      },
    ],
    structuredContent: {
      sutta,
    },
  };
}

function parseToolCall(params: unknown): {
  name: string;
  arguments: ToolArguments;
} {
  if (!params || typeof params !== "object") {
    throw new Error("Tool call params must be an object.");
  }

  const candidate = params as {
    name?: unknown;
    arguments?: unknown;
  };

  if (typeof candidate.name !== "string" || !candidate.name.trim()) {
    throw new Error("Tool call is missing a valid name.");
  }

  if (!candidate.arguments || typeof candidate.arguments !== "object") {
    throw new Error("Tool call is missing arguments.");
  }

  const args = candidate.arguments as Record<string, unknown>;
  const userQuery = args.user_query;

  if (typeof userQuery !== "string" || !userQuery.trim()) {
    throw new Error("`user_query` must be a non-empty string.");
  }

  return {
    name: candidate.name,
    arguments: {
      user_query: userQuery.trim(),
    },
  };
}

function validateOrigin(request: Request): Response | null {
  const origin = request.headers.get("origin");

  if (!origin) {
    return null;
  }

  const url = new URL(request.url);
  const allowedOrigins = new Set([
    `http://${url.host}`,
    `https://${url.host}`,
  ]);

  if (!allowedOrigins.has(origin)) {
    return errorHttpResponse(403, -32000, `Origin not allowed: ${origin}`);
  }

  return null;
}

function requireSession(request: Request): Session {
  const sessionId = request.headers.get(MCP_SESSION_HEADER);

  if (!sessionId) {
    throw new Error("Missing or invalid session. Send initialize first.");
  }

  const session = sessions.get(sessionId);

  if (!session) {
    throw new Error("Unknown session.");
  }

  return session;
}

function createSession(): Session {
  const session: Session = {
    id: randomUUID(),
    initializedAt: new Date().toISOString(),
  };

  sessions.set(session.id, session);
  return session;
}

function handleDelete(request: Request, mcpPath: string): Response {
  const url = new URL(request.url);
  if (url.pathname !== mcpPath) {
    return new Response("Not found", { status: 404 });
  }

  const sessionId = request.headers.get(MCP_SESSION_HEADER);
  if (!sessionId) {
    return new Response(null, { status: 400 });
  }

  if (!sessions.has(sessionId)) {
    return new Response(null, { status: 404 });
  }

  sessions.delete(sessionId);
  return new Response(null, { status: 204 });
}

function jsonResponse(
  body: JsonRpcResponse,
  statusOrHeaders: number | Record<string, string> = 200,
): Response {
  const status = typeof statusOrHeaders === "number" ? statusOrHeaders : 200;
  const extraHeaders =
    typeof statusOrHeaders === "number" ? {} : statusOrHeaders;

  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      ...extraHeaders,
    },
  });
}

function htmlResponse(html: string): Response {
  return new Response(html, {
    headers: {
      "content-type": "text/html; charset=utf-8",
    },
  });
}

function renderPlaygroundHtml(mcpPath: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sutta MCP Playground</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f5efe1;
        --panel: #fffaf0;
        --panel-strong: #fff;
        --text: #2d2418;
        --muted: #6a5a43;
        --border: #d8c9aa;
        --accent: #9a3412;
        --accent-soft: #fed7aa;
        --success: #166534;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background:
          radial-gradient(circle at top left, #fff7ed 0, transparent 28%),
          linear-gradient(180deg, #f8f1e4 0%, var(--bg) 100%);
        color: var(--text);
      }
      .wrap {
        max-width: 960px;
        margin: 0 auto;
        padding: 32px 20px 64px;
      }
      .hero {
        margin-bottom: 24px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: clamp(2rem, 5vw, 3.5rem);
        line-height: 1;
        letter-spacing: -0.03em;
      }
      .sub {
        margin: 0;
        color: var(--muted);
        max-width: 700px;
      }
      .grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
      }
      @media (min-width: 880px) {
        .grid {
          grid-template-columns: minmax(0, 360px) minmax(0, 1fr);
        }
      }
      .card {
        background: color-mix(in srgb, var(--panel) 92%, white);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 12px 30px rgba(90, 62, 24, 0.08);
      }
      .card h2 {
        margin: 0 0 14px;
        font-size: 1.1rem;
      }
      label {
        display: block;
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: var(--muted);
      }
      input, textarea {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel-strong);
        padding: 12px 14px;
        font: inherit;
        color: var(--text);
      }
      textarea {
        min-height: 140px;
        resize: vertical;
      }
      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
      }
      button {
        border: 0;
        border-radius: 999px;
        padding: 11px 16px;
        background: var(--accent);
        color: white;
        font: inherit;
        cursor: pointer;
      }
      button.secondary {
        background: #7c6a4d;
      }
      button.ghost {
        background: var(--accent-soft);
        color: var(--accent);
      }
      .meta {
        display: grid;
        gap: 10px;
        margin-bottom: 14px;
      }
      .meta-item {
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,0.5);
        border: 1px solid var(--border);
      }
      .meta-key {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      .meta-value {
        margin-top: 4px;
        word-break: break-all;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 13px;
        line-height: 1.5;
      }
      .status {
        color: var(--success);
        min-height: 24px;
        margin-bottom: 12px;
      }
      .hint {
        margin-top: 12px;
        color: var(--muted);
        font-size: 0.92rem;
      }
      code {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="hero">
        <h1>Sutta MCP Playground</h1>
        <p class="sub">Browser-friendly MCP testing for <code>${mcpPath}</code>. Initialize a session, inspect tools, and call <code>get-sutta-tool</code> without hand-writing JSON-RPC requests.</p>
      </div>
      <div class="grid">
        <section class="card">
          <h2>Session</h2>
          <div class="meta">
            <div class="meta-item">
              <div class="meta-key">Endpoint</div>
              <div class="meta-value" id="endpoint"></div>
            </div>
            <div class="meta-item">
              <div class="meta-key">Session ID</div>
              <div class="meta-value" id="sessionId">Not initialized</div>
            </div>
          </div>
          <div class="actions">
            <button id="initBtn">Initialize</button>
            <button id="listBtn" class="secondary">List Tools</button>
            <button id="resetBtn" class="ghost">Reset Session</button>
          </div>
          <p class="hint">Use <code>/healthz</code> for a plain browser health check. Use this page for actual MCP calls.</p>
        </section>
        <section class="card">
          <h2>Call Tool</h2>
          <label for="query">user_query</label>
          <textarea id="query">How should I approach heartbreak?</textarea>
          <div class="actions">
            <button id="callBtn">Call get-sutta-tool</button>
          </div>
        </section>
        <section class="card">
          <h2>Status</h2>
          <div class="status" id="status">Ready.</div>
          <pre id="response"></pre>
        </section>
      </div>
    </div>
    <script>
      const endpoint = ${JSON.stringify(mcpPath)};
      const endpointUrl = new URL(endpoint, window.location.origin).toString();
      let sessionId = null;
      let nextId = 1;

      const endpointEl = document.getElementById("endpoint");
      const sessionEl = document.getElementById("sessionId");
      const statusEl = document.getElementById("status");
      const responseEl = document.getElementById("response");
      const queryEl = document.getElementById("query");

      endpointEl.textContent = endpointUrl;

      function setStatus(text, isError = false) {
        statusEl.textContent = text;
        statusEl.style.color = isError ? "#b91c1c" : "#166534";
      }

      function setResponse(data) {
        responseEl.textContent =
          typeof data === "string" ? data : JSON.stringify(data, null, 2);
      }

      function updateSession(newSessionId) {
        sessionId = newSessionId;
        sessionEl.textContent = newSessionId || "Not initialized";
      }

      async function sendRpc(method, params, options = {}) {
        const headers = {
          "Content-Type": "application/json",
          "Accept": "application/json, text/event-stream",
        };

        if (sessionId && !options.skipSession) {
          headers["mcp-session-id"] = sessionId;
        }

        const response = await fetch(endpointUrl, {
          method: "POST",
          headers,
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: nextId++,
            method,
            params,
          }),
        });

        const payload = await response.json().catch(() => null);
        const maybeSessionId = response.headers.get("mcp-session-id");
        if (maybeSessionId) {
          updateSession(maybeSessionId);
        }

        if (!response.ok) {
          throw new Error(payload?.error?.message || "Request failed");
        }

        return payload;
      }

      document.getElementById("initBtn").addEventListener("click", async () => {
        setStatus("Initializing session...");
        try {
          const payload = await sendRpc("initialize", {
            protocolVersion: "2025-03-26",
            capabilities: {},
            clientInfo: {
              name: "browser-playground",
              version: "0.1.0"
            }
          }, { skipSession: true });
          setResponse(payload);
          setStatus("Session initialized.");
        } catch (error) {
          setStatus(error.message, true);
        }
      });

      document.getElementById("listBtn").addEventListener("click", async () => {
        setStatus("Listing tools...");
        try {
          const payload = await sendRpc("tools/list");
          setResponse(payload);
          setStatus("Tools loaded.");
        } catch (error) {
          setStatus(error.message, true);
        }
      });

      document.getElementById("callBtn").addEventListener("click", async () => {
        setStatus("Calling get-sutta-tool...");
        try {
          const payload = await sendRpc("tools/call", {
            name: "get-sutta-tool",
            arguments: {
              user_query: queryEl.value
            }
          });
          setResponse(payload);
          setStatus("Tool call completed.");
        } catch (error) {
          setStatus(error.message, true);
        }
      });

      document.getElementById("resetBtn").addEventListener("click", async () => {
        if (!sessionId) {
          setStatus("No session to reset.");
          return;
        }

        setStatus("Resetting session...");
        try {
          await fetch(endpointUrl, {
            method: "DELETE",
            headers: {
              "mcp-session-id": sessionId
            }
          });
          updateSession(null);
          setResponse("");
          setStatus("Session cleared.");
        } catch (error) {
          setStatus(error.message, true);
        }
      });
    </script>
  </body>
</html>`;
}

function errorHttpResponse(
  status: number,
  code: number,
  message: string,
): Response {
  return jsonResponse(failure(null, code, message), status);
}

function success(id: JsonRpcId, result: unknown): JsonRpcResponse {
  return {
    jsonrpc: "2.0",
    id,
    result,
  };
}

function failure(
  id: JsonRpcId,
  code: number,
  message: string,
  data?: unknown,
): JsonRpcResponse {
  return {
    jsonrpc: "2.0",
    id,
    error: {
      code,
      message,
      data,
    },
  };
}
