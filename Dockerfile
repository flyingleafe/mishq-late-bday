FROM oven/bun:1.3.11 AS base

WORKDIR /app

COPY package.json bun.lock tsconfig.json ./
RUN bun install --frozen-lockfile

COPY index.ts ./
COPY MESSAGE.md ./
COPY index.html ./
COPY data ./data
COPY src ./src

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=3000
ENV MCP_PATH=/mcp

EXPOSE 3000

CMD ["bun", "run", "index.ts"]
