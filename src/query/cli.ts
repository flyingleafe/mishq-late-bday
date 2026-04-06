import { SQL } from "bun";
import { pipeline } from "@xenova/transformers";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

const DB_URL = process.env.COCOINDEX_DATABASE_URL!;
const DATA_DIR = join(process.cwd(), "data", "texts");
const MODEL_NAME = process.env.EMBED_MODEL || "Xenova/all-MiniLM-L6-v2";
const LLM_MODEL = "google/gemini-2.5-flash";
const NUM_SUBQUERIES = 5;
const CANDIDATES_PER_SUBQUERY = 30;
const SERVER_URL = process.env.SERVER_URL || "http://localhost:8000";

let db: SQL;

function getDb() {
  if (!db) {
    db = new SQL(DB_URL);
  }
  return db;
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
  const filePath = join(DATA_DIR, `${uid}.json`);
  if (!existsSync(filePath)) return "";
  try {
    const content = readFileSync(filePath, "utf-8");
    const data = JSON.parse(content) as { text?: string };
    return data.text || "";
  } catch {
    return "";
  }
}

interface ChunkCandidate {
  chunk_uid: string;
  sutta_uid: string;
  sutta_title: string;
  chunk_text: string;
  embedding: number[];
}

async function searchChunksWithSubqueries(
  query: string,
  topK: number
): Promise<{ chunk: any; avgDistance: number; subqueries: string[] }[]> {
  const subqueries = await generateSubqueries(query);
  const allEmbeds = await getEmbeddings([query, ...subqueries]);
  const queryEmb = allEmbeds[0];
  const subEmbeds = allEmbeds.slice(1);

  if (!queryEmb) throw new Error("Failed to compute query embedding");

  const candidateMap = new Map<string, { chunk: ChunkCandidate; distances: number[] }>();

  for (const emb of subEmbeds) {
    if (!emb) continue;

    const rows = await getDb()`
      SELECT chunk_uid, sutta_uid, sutta_title, chunk_text, embedding
      FROM sutta_chunks
      ORDER BY embedding <=> ${JSON.stringify(emb)}::vector
      LIMIT ${CANDIDATES_PER_SUBQUERY}
    `;

    for (const row of rows as any[]) {
      const rowEmb: number[] = Array.isArray(row.embedding)
        ? row.embedding
        : JSON.parse(row.embedding as string);
      const distance = 1 - cosineSim(emb, rowEmb);

      if (!candidateMap.has(row.chunk_uid)) {
        candidateMap.set(row.chunk_uid, {
          chunk: {
            chunk_uid: row.chunk_uid,
            sutta_uid: row.sutta_uid,
            sutta_title: row.sutta_title,
            chunk_text: row.chunk_text,
            embedding: rowEmb,
          },
          distances: [],
        });
      }
      candidateMap.get(row.chunk_uid)!.distances.push(distance);
    }
  }

  const scored = Array.from(candidateMap.values()).map(({ chunk, distances }) => {
    const avgDistance = distances.reduce((a, b) => a + b, 0) / distances.length;
    const querySim = cosineSim(queryEmb, chunk.embedding);
    return {
      chunk,
      avgDistance,
      querySim,
      combinedScore: (1 - avgDistance + querySim) / 2,
    };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);

  return scored.slice(0, topK * 3).map((s) => ({
    chunk: {
      chunk_uid: s.chunk.chunk_uid,
      sutta_uid: s.chunk.sutta_uid,
      sutta_title: s.chunk.sutta_title,
      chunk_text: s.chunk.chunk_text,
      score: Math.round(s.combinedScore * 10000) / 10000,
    },
    avgDistance: s.avgDistance,
    subqueries,
  }));
}

function formatResult(r: any, index: number): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  lines.push(`📖 ${r.sutta_title} (${r.sutta_uid})`);
  lines.push(`🔍 Score: ${(r.chunk.score * 100).toFixed(1)}%`);
  lines.push(`📌 Chunk: ${r.chunk.chunk_uid}`);
  lines.push("");
  lines.push(r.chunk.chunk_text);
  if (r.sutta_text) {
    lines.push("");
    lines.push(`💬 Full sutta (${r.sutta_text.length} chars):`);
    const preview = r.sutta_text.slice(0, 300).replace(/\n/g, " ").trim();
    lines.push(`   ${preview}...`);
  }
  return lines.join("\n");
}

function formatProgress(results: any[], subqueryIndex: number, subqueryTotal: number, subqueryText: string): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`⏳ Subquery [${subqueryIndex}/${subqueryTotal}]: "${subqueryText}"`);
  lines.push(`📊 Top ${results.length} candidates so far:`);
  for (const r of results.slice(0, 5)) {
    lines.push(`   • ${r.sutta_title} (${r.sutta_uid}) — ${(r.chunk.score * 100).toFixed(1)}%`);
  }
  return lines.join("\n");
}

async function streamSearch(query: string, topK: number): Promise<void> {
  const url = `${SERVER_URL}/stream?q=${encodeURIComponent(query)}&top=${topK}`;

  const resp = await fetch(url);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error ${resp.status}: ${text}`);
  }

  if (!resp.body) throw new Error("No response body");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  const buffer = { value: "" };

  let subqueries: string[] = [];
  let progressResults: any[] = [];
  let done = false;

  const readChunk = (chunk: string) => {
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (!line.startsWith("event: ") && !line.startsWith("data: ")) continue;
      if (line.startsWith("event: ")) continue;
      const data = line.slice(6).trim();
      if (!data) continue;

      const parsed = JSON.parse(data);
      if (parsed.subqueries) {
        subqueries = parsed.subqueries;
        process.stdout.write(`\n📝 Generated ${subqueries.length} subqueries:\n`);
        for (const sq of subqueries) {
          process.stdout.write(`   → "${sq}"\n`);
        }
        process.stdout.write("\n🔍 Searching...\n");
      } else if (parsed.results_so_far !== undefined) {
        progressResults = parsed.results_so_far;
        const txt = formatProgress(
          progressResults,
          parsed.subquery_index,
          parsed.subquery_total,
          parsed.subquery_text
        );
        process.stdout.write(`\r${txt}`);
        if (parsed.is_complete) {
          process.stdout.write(" ✅\n");
        }
      } else if (parsed.results) {
        done = true;
        for (let i = 0; i < parsed.results.length; i++) {
          process.stdout.write(formatResult(parsed.results[i], i + 1));
        }
        process.stdout.write(`\n✅ Done in ${parsed.total_ms}ms — ${parsed.results.length} result(s)\n`);
      } else if (parsed.error) {
        throw new Error(parsed.error);
      }
    }
  };

  while (!done) {
    const { done: readerDone, value } = await reader.read();
    if (readerDone) break;
    buffer.value += decoder.decode(value, { stream: true });
    const lines = buffer.value.split("\n\n");
    buffer.value = lines.pop() || "";
    for (const chunk of lines) {
      readChunk(chunk);
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  let topK = 5;
  let interactive = false;
  let noStream = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-n" && args[i + 1]) {
      topK = parseInt(args[i + 1], 10);
      args.splice(i, 2);
      i--;
    } else if (args[i] === "-i") {
      interactive = true;
      args.splice(i, 1);
      i--;
    } else if (args[i] === "--no-stream") {
      noStream = true;
      args.splice(i, 1);
      i--;
    }
  }

  if (args.length === 0 && !interactive) {
    console.log(`
Sutta Query CLI — search suttas by semantic similarity with streaming results

Usage:
  bun run src/query/cli.ts <query> [-n N]        Stream results (default)
  bun run src/query/cli.ts <query> [-n N] --no-stream  Non-streaming (slower, full results at end)
  bun run src/query/cli.ts -i                      Interactive mode (streaming)
  bun run src/query/cli.ts -h                     Show this help

Examples:
  bun run src/query/cli.ts "what is the nature of suffering"
  bun run src/query/cli.ts "how to cultivate compassion" -n 10
  bun run src/query/cli.ts -i
`);
    return;
  }

  const doSearch = async (query: string) => {
    process.stdout.write(`\n🔍 Searching: "${query}"\n`);

    if (noStream) {
      process.stdout.write("⏳ Generating subqueries via Gemini Flash...\n");
      const t0 = Date.now();
      const scoredChunks = await searchChunksWithSubqueries(query, topK);
      const llMs = Date.now() - t0;

      const seen = new Set<string>();
      const results: any[] = [];
      for (const sc of scoredChunks) {
        if (seen.has(sc.chunk.sutta_uid)) continue;
        seen.add(sc.chunk.sutta_uid);
        const sutta_text = await getSuttaText(sc.chunk.sutta_uid);
        results.push({
          sutta_uid: sc.chunk.sutta_uid,
          sutta_title: sc.chunk.sutta_title,
          chunk: { chunk_uid: sc.chunk.chunk_uid, chunk_text: sc.chunk.chunk_text, score: sc.chunk.score },
          sutta_text,
        });
      }

      const totalMs = Date.now() - t0;
      process.stdout.write(`✅ Found ${results.length} results in ${totalMs}ms (LLM+embed: ${llMs}ms)\n`);
      process.stdout.write(`📝 Subqueries: ${scoredChunks[0]?.subqueries.join(" | ")}\n`);
      for (let i = 0; i < results.length; i++) {
        process.stdout.write(formatResult(results[i], i + 1));
      }
      process.stdout.write("\n");
    } else {
      await streamSearch(query, topK);
    }
  };

  if (interactive) {
    const readline = await import("readline");
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: true });
    const prompt = () => new Promise<string>((res) => rl.question("\n❓ Query: ", res));
    process.stdout.write("\n🪷 Sutta Semantic Search — Interactive Mode (streaming)\n");
    process.stdout.write("Type a query and press Enter. Ctrl+C or Ctrl+D to exit.\n");
    while (true) {
      try {
        const q = await prompt();
        if (!q.trim()) continue;
        await doSearch(q);
      } catch {
        break;
      }
    }
    rl.close();
  } else {
    await doSearch(args.join(" "));
  }
}

main();
