import { serve } from "@hono/node-server";
import app from "./src/query/server.ts";

const PORT = parseInt(process.env.PORT || "8000", 10);

console.log(`Starting merged server on port ${PORT}...`);
console.log(`  Query UI:     http://localhost:${PORT}/`);
console.log(`  Search API:   http://localhost:${PORT}/search?q=...`);
console.log(`  Stream API:   http://localhost:${PORT}/stream?q=...`);
console.log(`  MCP:          http://localhost:${PORT}/mcp`);
console.log(`  Playground:   http://localhost:${PORT}/playground`);
console.log(`  API docs:    http://localhost:${PORT}/docs`);

serve({
  port: PORT,
  fetch: app.fetch,
});
