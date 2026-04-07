import { Hono } from "hono";
import { cors } from "hono/cors";
import { Scalar } from "@scalar/hono-api-reference";
import { SQL } from "bun";
import { pipeline } from "@xenova/transformers";
import { readFileSync } from "fs";
import { join } from "path";

const DB_URL = process.env.COCOINDEX_DATABASE_URL ?? process.env.DATABASE_URL;
const DATA_DIR = join(process.cwd(), "data", "texts");
const MODEL_NAME = process.env.EMBED_MODEL || "Xenova/all-MiniLM-L6-v2";
const LLM_MODEL = "google/gemini-2.5-flash";
const NUM_SUBQUERIES = 5;
const CANDIDATES_PER_SUBQUERY = 30;

let db: SQL;

function getDb() {
  if (!DB_URL) {
    throw new Error("COCOINDEX_DATABASE_URL or DATABASE_URL must be set");
  }
  if (!db) {
    db = new SQL(DB_URL);
  }
  return db;
}

let systemMessageCache: string | null = null;

function getSystemMessage(): string {
  if (systemMessageCache === null) {
    try {
      const msgPath = join(process.cwd(), "MESSAGE.md");
      systemMessageCache = readFileSync(msgPath, "utf-8");
    } catch {
      systemMessageCache = "Hello! This is a Buddhist sutta search engine. How can I help you today?";
    }
  }
  return systemMessageCache;
}

async function isSystemQuery(query: string): Promise<boolean> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) return false;

  const prompt = `You are a classifier. Given a user query to a Buddhist sutta search engine, determine if the user is:
1. Asking about the website/system itself (e.g. "what is this", "who made you", "how does this work")
2. Expressing frustration or complaints about the search results (e.g. "this is garbage", "why are results so bad")
3. Writing a meta-message to the developer (e.g. "happy birthday", "tell Alex I said hi")
4. Or just greeting/short meta comments (e.g. "hi", "hello", "ok")

If ANY of the above is true, respond with exactly: YES
If the query is a genuine search for Buddhist sutta/dhamma content (even if poorly phrased), respond with exactly: NO

User query: "${query}"

Response:`;

  try {
    const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/flyingleafe/mishq",
        "X-Title": "SuttaQuery",
      },
      body: JSON.stringify({
        model: LLM_MODEL,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 5,
        temperature: 0,
      }),
    });

    if (!resp.ok) return false;

    const data = await resp.json() as { choices?: { message?: { content?: string } }[] };
    const content = data.choices?.[0]?.message?.content?.trim().toUpperCase();
    return content === "YES";
  } catch {
    return false;
  }
}

let extractorPromise: ReturnType<typeof pipeline> | null = null;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_NAME, {
      pooling: "mean",
      normalize: true,
    });
  }
  return extractorPromise;
}

async function getEmbeddings(texts: string | string[]): Promise<number[][]> {
  const extractor = await getExtractor();
  const textsArr = Array.isArray(texts) ? texts : [texts];
  const output = await extractor(textsArr, { pooling: "mean", normalize: true });
  const result = (output as any).tolist();
  return Array.isArray(result) ? result : [[]];
}

async function generateSubqueries(query: string): Promise<string[]> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not set");

  const prompt = `Given a Buddhist sutta search query, generate ${NUM_SUBQUERIES} thematically correlated subqueries that explore different angles, aspects, or phrasings of the original query. Each subquery should target semantically distinct aspects while remaining related to the original.

Return a JSON object with a "subqueries" key containing an array of exactly ${NUM_SUBQUERIES} strings.

Original query: "${query}"

Output format:
{"subqueries": ["subquery 1", "subquery 2", "subquery 3", "subquery 4", "subquery 5"]}`;

  for (let attempt = 0; attempt < 2; attempt++) {
    const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/flyingleafe/mishq",
        "X-Title": "SuttaQuery",
      },
      body: JSON.stringify({
        model: LLM_MODEL,
        messages: [{ role: "user", content: prompt }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "subqueries",
            schema: {
              type: "object",
              properties: {
                subqueries: {
                  type: "array",
                  items: { type: "string" },
                  minItems: NUM_SUBQUERIES,
                  maxItems: NUM_SUBQUERIES,
                },
              },
              required: ["subqueries"],
            },
          },
        },
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`OpenRouter error ${resp.status}: ${text}`);
    }

    const data = await resp.json() as { choices: { message: { content: string } }[] };
    const content = data.choices[0]?.message?.content;
    if (!content) throw new Error("Empty response from OpenRouter");

    try {
      const parsed = JSON.parse(content) as { subqueries?: string[] };
      if (!parsed.subqueries || !Array.isArray(parsed.subqueries)) {
        throw new Error(`Invalid subquery response`);
      }
      return parsed.subqueries.slice(0, NUM_SUBQUERIES);
    } catch (e: any) {
      if (attempt === 0) {
        const trimmed = content.trim();
        const fixed = trimmed.endsWith("}") ? trimmed : trimmed + "\"}";
        try {
          const parsed = JSON.parse(fixed) as { subqueries?: string[] };
          if (parsed.subqueries && Array.isArray(parsed.subqueries)) {
            return parsed.subqueries.slice(0, NUM_SUBQUERIES);
          }
        } catch {}
        continue;
      }
      throw new Error(`JSON parse error after retry: ${e.message} — raw: ${content.slice(0, 200)}`);
    }
  }
  return [];
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function getSuttaText(uid: string): Promise<string> {
  try {
    const filePath = join(DATA_DIR, `${uid}.json`);
    const content = readFileSync(filePath, "utf-8");
    const data = JSON.parse(content) as { text?: string };
    return data.text || "";
  } catch {
    return "";
  }
}

export interface ScoredChunk {
  chunk_uid: string;
  sutta_uid: string;
  sutta_title: string;
  chunk_text: string;
  score: number;
  avgDistance: number;
}

export interface SearchResult {
  sutta_uid: string;
  sutta_title: string;
  chunk: {
    chunk_uid: string;
    chunk_text: string;
    score: number;
  };
  sutta_text: string;
}

export interface SearchResponse {
  query: string;
  top: number;
  subqueries: string[];
  timing_ms: number;
  results: SearchResult[];
  is_system_message?: boolean;
  message?: string;
}

export async function* streamSearchResults(
  query: string,
  topK: number
): AsyncGenerator<{ type: "subqueries"; data: string[] } | { type: "subquery-progress"; data: { subquery: string; index: number; total: number; chunks: ScoredChunk[] } } | { type: "done"; data: { total_ms: number } }> {
  const t0 = Date.now();
  const subqueries = await generateSubqueries(query);
  yield { type: "subqueries", data: subqueries };

  const allEmbeds = await getEmbeddings([query, ...subqueries]);
  const queryEmb = allEmbeds[0];
  const subEmbeds = allEmbeds.slice(1);

  if (!queryEmb) throw new Error("Failed to compute query embedding");

  const candidateMap = new Map<string, { chunk: ScoredChunk; distances: number[]; embedding: number[] }>();

  for (let i = 0; i < subEmbeds.length; i++) {
    const emb = subEmbeds[i];
    if (!emb) continue;
    const subquery = subqueries[i];

    const rows = await getDb()`
      SELECT chunk_uid, sutta_uid, sutta_title, chunk_text, embedding
      FROM sutta_chunks
      ORDER BY embedding <=> ${JSON.stringify(emb)}::vector
      LIMIT ${CANDIDATES_PER_SUBQUERY}
    `;

    const subChunks: ScoredChunk[] = [];

    for (const row of rows as any[]) {
      const rowEmb: number[] = Array.isArray(row.embedding)
        ? row.embedding
        : JSON.parse(row.embedding as string);
      const distance = 1 - cosineSim(emb, rowEmb);

      if (!candidateMap.has(row.chunk_uid)) {
        const scored: ScoredChunk = {
          chunk_uid: row.chunk_uid,
          sutta_uid: row.sutta_uid,
          sutta_title: row.sutta_title,
          chunk_text: row.chunk_text,
          score: 0,
          avgDistance: 0,
        };
        candidateMap.set(row.chunk_uid, { chunk: scored, distances: [], embedding: rowEmb });
      }
      candidateMap.get(row.chunk_uid)!.distances.push(distance);
    }

    for (const [_, entry] of candidateMap) {
      const avgDist = entry.distances.reduce((a, b) => a + b, 0) / entry.distances.length;
      entry.chunk.avgDistance = avgDist;
      const querySim = cosineSim(queryEmb, entry.embedding);
      entry.chunk.score = Math.round(((1 - avgDist + querySim) / 2) * 10000) / 10000;
    }

    const sorted = Array.from(candidateMap.values())
      .map(e => e.chunk)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK * 3);

    subChunks.push(...sorted);

    yield {
      type: "subquery-progress",
      data: { subquery: subquery || `subquery-${i}`, index: i + 1, total: subEmbeds.length, chunks: subChunks },
    };
  }

  yield { type: "done", data: { total_ms: Date.now() - t0 } };
}

export async function searchSuttas(
  query: string,
  top: number,
): Promise<SearchResponse> {
  const normalizedQuery = query.trim();
  const normalizedTop = Math.min(50, Math.max(1, Math.trunc(top)));

  if (!normalizedQuery) {
    throw new Error("Query must be a non-empty string.");
  }

  if (await isSystemQuery(normalizedQuery)) {
    return {
      query: normalizedQuery,
      top: normalizedTop,
      is_system_message: true,
      message: getSystemMessage(),
      subqueries: [],
      timing_ms: 0,
      results: [],
    };
  }

  const t0 = Date.now();
  const chunks: ScoredChunk[] = [];
  const subqueries: string[] = [];

  for await (const event of streamSearchResults(normalizedQuery, normalizedTop)) {
    if (event.type === "subqueries") {
      subqueries.push(...event.data);
    } else if (event.type === "subquery-progress") {
      chunks.length = 0;
      chunks.push(...event.data.chunks);
    }
  }

  const results: SearchResult[] = [];
  const seen = new Set<string>();

  for (const chunk of chunks) {
    if (seen.has(chunk.sutta_uid)) continue;
    seen.add(chunk.sutta_uid);
    const sutta_text = await getSuttaText(chunk.sutta_uid);
    results.push({
      sutta_uid: chunk.sutta_uid,
      sutta_title: chunk.sutta_title,
      chunk: {
        chunk_uid: chunk.chunk_uid,
        chunk_text: chunk.chunk_text,
        score: chunk.score,
      },
      sutta_text,
    });
    if (results.length >= normalizedTop) break;
  }

  return {
    query: normalizedQuery,
    top: normalizedTop,
    results,
    subqueries,
    timing_ms: Date.now() - t0,
  };
}

const app = new Hono();

app.use("*", cors());

app.get("/", (c) => {
  const html = readFileSync(join(process.cwd(), "index.html"), "utf-8");
  return c.html(html);
});

app.get("/js/:file", (c) => {
  const file = c.req.param("file");
  const filePath = join(process.cwd(), "public", "js", file);
  try {
    const content = readFileSync(filePath);
    return new Response(content, {
      headers: { "Content-Type": "application/javascript" },
    });
  } catch {
    return c.json({ error: "Not found" }, 404);
  }
});

app.get("/health", (c) => c.json({ status: "ok" }));

const OPENAPI_SPEC = {
  openapi: "3.1.0",
  info: {
    title: "Sutta Query API",
    description: "Semantic search over Buddhist suttas with query reinterpretation via Gemini Flash.",
    version: "1.0.0",
  },
  servers: [{ url: "/", description: "Current server" }],
  paths: {
    "/search": {
      get: {
        summary: "Search suttas (blocking)",
        description: "Returns full results after all subqueries complete. For streaming, use /stream instead.",
        parameters: [
          { name: "q", in: "query", required: true, schema: { type: "string" }, description: "Search query" },
          { name: "top", in: "query", required: false, schema: { type: "integer", default: 5, minimum: 1, maximum: 50 }, description: "Number of results to return" },
        ],
        responses: {
          "200": {
            description: "Search results",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    query: { type: "string" },
                    top: { type: "integer" },
                    subqueries: { type: "array", items: { type: "string" } },
                    timing_ms: { type: "integer" },
                    results: {
                      type: "array",
                      items: { $ref: "#/components/schemas/SearchResult" },
                    },
                  },
                },
              },
            },
          },
          "400": { description: "Missing q parameter" },
          "500": { description: "Server error" },
        },
      },
    },
    "/stream": {
      get: {
        summary: "Search suttas (streaming SSE)",
        description: "Streams results via Server-Sent Events. Events: subqueries → progress (per subquery) → done.",
        parameters: [
          { name: "q", in: "query", required: true, schema: { type: "string" }, description: "Search query" },
          { name: "top", in: "query", required: false, schema: { type: "integer", default: 5, minimum: 1, maximum: 50 }, description: "Number of results to return" },
        ],
        responses: {
          "200": {
            description: "SSE stream",
            content: { "text/event-stream": { schema: { type: "string" } } },
          },
          "400": { description: "Missing q parameter" },
        },
      },
    },
  },
  components: {
    schemas: {
      SearchResult: {
        type: "object",
        properties: {
          sutta_uid: { type: "string", example: "sn46.54" },
          sutta_title: { type: "string", example: "Full of Love" },
          chunk: {
            type: "object",
            properties: {
              chunk_uid: { type: "string", example: "sn46.54.c3" },
              chunk_text: { type: "string" },
              score: { type: "number", format: "float" },
            },
          },
          sutta_text: { type: "string" },
        },
      },
    },
  },
};

app.get("/openapi.json", (c) => c.json(OPENAPI_SPEC));

app.use("/docs", Scalar({ url: "/openapi.json" }));

app.get("/search", async (c) => {
  const q = c.req.query("q") || "";
  const top = Math.min(50, Math.max(1, parseInt(c.req.query("top") || "5", 10)));

  if (!q.trim()) {
    return c.json({ error: "q parameter is required" }, 400);
  }

  try {
    return c.json(await searchSuttas(q, top));
  } catch (err: any) {
    console.error("Search error:", err);
    return c.json({ error: err.message }, 500);
  }
});

app.get("/stream", async (c) => {
  const q = c.req.query("q") || "";
  const top = Math.min(50, Math.max(1, parseInt(c.req.query("top") || "5", 10)));

  if (!q.trim()) {
    return c.json({ error: "q parameter is required" }, 400);
  }

  if (await isSystemQuery(q)) {
    c.header("Content-Type", "text/event-stream");
    c.header("Cache-Control", "no-cache");
    c.header("Connection", "keep-alive");
    c.header("Transfer-Encoding", "chunked");
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`event: system-message\ndata: ${JSON.stringify({ message: getSystemMessage() })}\n\n`));
        controller.close();
      },
    });
    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  }

  c.header("Content-Type", "text/event-stream");
  c.header("Cache-Control", "no-cache");
  c.header("Connection", "keep-alive");
  c.header("Transfer-Encoding", "chunked");

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (event: string, data: any) => {
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      };

      try {
        const chunks: ScoredChunk[] = [];
        const subqueries: string[] = [];
        let lastFlush = Date.now();

        for await (const event of streamSearchResults(q, top)) {
          if (event.type === "subqueries") {
            subqueries.push(...event.data);
            send("subqueries", { subqueries: event.data });
          } else if (event.type === "subquery-progress") {
            chunks.length = 0;
            chunks.push(...event.data.chunks);

            const now = Date.now();
            if (now - lastFlush > 200 || event.data.index === event.data.total) {
              const results: any[] = [];
              const seen = new Set<string>();
              for (const chunk of chunks) {
                if (seen.has(chunk.sutta_uid)) continue;
                seen.add(chunk.sutta_uid);
                results.push({
                  sutta_uid: chunk.sutta_uid,
                  sutta_title: chunk.sutta_title,
                  chunk: { chunk_uid: chunk.chunk_uid, chunk_text: chunk.chunk_text, score: chunk.score },
                });
                if (results.length >= top) break;
              }
              send("progress", {
                subquery_index: event.data.index,
                subquery_total: event.data.total,
                subquery_text: event.data.subquery,
                results_so_far: results,
                is_complete: event.data.index === event.data.total,
              });
              lastFlush = now;
            }
          } else if (event.type === "done") {
            const results: any[] = [];
            const seen = new Set<string>();
            for (const chunk of chunks) {
              if (seen.has(chunk.sutta_uid)) continue;
              seen.add(chunk.sutta_uid);
              const sutta_text = await getSuttaText(chunk.sutta_uid);
              results.push({
                sutta_uid: chunk.sutta_uid,
                sutta_title: chunk.sutta_title,
                chunk: { chunk_uid: chunk.chunk_uid, chunk_text: chunk.chunk_text, score: chunk.score },
                sutta_text,
              });
              if (results.length >= top) break;
            }
            send("done", { results, total_ms: event.data.total_ms });
            controller.close();
          }
        }
      } catch (err: any) {
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`));
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
});

const PORT = parseInt(process.env.PORT || "8000", 10);

console.log(`Starting Sutta Query Server on port ${PORT}...`);
console.log(`Embed model: ${MODEL_NAME}`);
console.log(`LLM model: ${LLM_MODEL}`);
console.log(`DB: ${DB_URL ?? "<unset>"}`);

export default {
  port: PORT,
  fetch: app.fetch,
};
