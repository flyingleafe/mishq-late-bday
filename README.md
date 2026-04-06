# mishq-late-bday

To install dependencies:

```bash
bun install
```

To run the HTTP MCP server:

```bash
bun run index.ts
```

By default it listens on:

```text
http://127.0.0.1:3000/mcp
```

You can override that with `HOST`, `PORT`, and `MCP_PATH`.

Browser-friendly pages:

- `http://127.0.0.1:3000/playground` for interactive testing
- `http://127.0.0.1:3000/healthz` for a simple health check

The server exposes one MCP tool over Streamable HTTP:

- `get-sutta-tool`

It accepts:

- `user_query: string`

The tool is described for philosophical questions about life, love, or career. Internally it calls a retriever module at `src/retriever/index.ts`. That retriever is currently a stub and is intended to be replaced later.

Example initialize request:

```bash
curl -i \
  -X POST http://127.0.0.1:3000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": { "name": "curl", "version": "0.0.1" }
    }
  }'
```

Then use the returned `mcp-session-id` header in later calls:

```bash
curl -s \
  -X POST http://127.0.0.1:3000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'mcp-session-id: YOUR_SESSION_ID' \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get-sutta-tool",
      "arguments": {
        "user_query": "How should I approach heartbreak?"
      }
    }
  }'
```

To run it in Docker together with `pgvector`:

```bash
docker compose up --build
```

That starts:

- MCP server at `http://127.0.0.1:3000/mcp`
- Postgres with pgvector at `127.0.0.1:5433`

The app container already receives:

- `HOST=0.0.0.0`
- `PORT=3000`
- `MCP_PATH=/mcp`
- `DATABASE_URL=postgresql://sutta:suttapass@pgvector:5432/suttas`

So when the retriever is implemented later, it can use `DATABASE_URL` directly without changing the container wiring.

For the existing SuttaCentral smoke test:

```bash
bun run smoke
```
