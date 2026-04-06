import { startMcpServer } from "./src/mcp/server.ts";

try {
  startMcpServer();
} catch (error) {
  console.error("Failed to start MCP server:", error);
  process.exit(1);
}
