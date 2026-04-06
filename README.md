# mishq — Buddhist Sutta Semantic Search

Semantic search over Buddhist suttas using subquery expansion (Gemini Flash) + sentence embeddings (all-MiniLM-L6-v2).

## Quick Start

```bash
bun install
bun run query:server
```

Open `http://localhost:8000` in your browser.

---

## Full Local Setup

### 1. Prerequisites

- **Bun** ≥ 1.3 — `curl -fsSL https://bun.sh/install | bash`
- **PostgreSQL 17** with `pgvector` extension
- **pg_dump client 17** (must match server version)
- **An OpenRouter API key** — get one at [openrouter.ai](https://openrouter.ai)

### 2. Load the Database Dump

The repo includes a pre-built database dump with pre-computed embeddings:

```
suttas_dump.sql   (281 MB)
```

Create the database and restore:

```bash
# Create the database
createdb -h localhost -p 5433 -U postgres suttas

# Grant access to sutta user
psql -h localhost -p 5433 -U postgres -c "ALTER DATABASE suttas OWNER TO sutta;"

# Load the dump
PGPASSWORD=suttapass pg_restore -h localhost -p 5433 -U sutta -d suttas ./suttas_dump.sql --no-owner --no-acl
```

Or with the default postgres superuser:

```bash
psql -h localhost -p 5433 -U postgres -c "CREATE DATABASE suttas;"
psql -h localhost -p 5433 -U postgres -d suttas -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -h localhost -p 5433 -U postgres -d suttas -f ./suttas_dump.sql
```

The dump includes:
- `sutta_chunks` — suttas split into ~200-char overlapping chunks, each with a 384-dim embedding
- `suttas` — sutta metadata
- `cocoindex_setup_metadata` — indexer bookkeeping

### 3. Environment Variables

Create a `.env` file in the project root:

```
COCOINDEX_DATABASE_URL=postgresql://sutta:suttapass@localhost:5433/suttas
OPENROUTER_API_KEY=sk-or-v1-...
EMBED_MODEL=Xenova/all-MiniLM-L6-v2
PORT=8000
```

- `OPENROUTER_API_KEY` is **required** — the search uses Gemini Flash to generate subqueries
- `EMBED_MODEL` defaults to `Xenova/all-MiniLM-L6-v2` (384-dim, must match the dumped embeddings)

### 4. Fetch Sutta Text Files

The search server reads full sutta text from local JSON files, not from the database. You need the `data/texts/` directory:

**Option A: Use the existing texts** (already in the repo if present)

**Option B: Crawl fresh texts from SuttaCentral**

```bash
bun run src/crawler/main.ts ./data 3
```

This fetches all English suttas from SuttaCentral and saves them to `data/texts/*.json`. Takes ~15–20 minutes. The crawler uses the SuttaCentral public API — no API key needed.

**Option C: Copy texts from the colleague's setup**

Copy the `data/texts/` directory as-is.

### 5. Run the Server

```bash
bun run query:server
```

Server starts on `http://localhost:8000`:

| Route | Description |
|---|---|
| `/` | Search UI |
| `/stream?q=...&top=N` | SSE stream of results |
| `/search?q=...&top=N` | Blocking JSON results |
| `/openapi.json` | OpenAPI spec |
| `/docs` | Scalar API docs |
| `/health` | Health check |

---

## Project Structure

```
src/
  query/
    server.ts      — HTTP server (Hono + Bun)
    cli.ts         — CLI search client
    highlight.ts   — Viterbi-based chunk highlighting
  crawler/
    main.ts        — Crawl entry point
    crawl.ts       — SuttaCentral crawler
  api/
    client.ts      — SuttaCentral API client
  types/
    suttacentral.ts
data/
  texts/
    *.json         — Full sutta texts (crawled from SuttaCentral)
suttas_dump.sql    — Full DB dump with embeddings
```

---

## Database Schema

```sql
-- Chunked suttas with 384-dim embeddings (used for vector search)
CREATE TABLE sutta_chunks (
  chunk_uid   text PRIMARY KEY,   -- e.g. "sn46.54.c3"
  sutta_uid   text,
  sutta_title text,
  chunk_text  text,
  embedding   public.vector(384)
);

-- Sutta metadata
CREATE TABLE suttas (
  uid            text PRIMARY KEY,
  title          text,
  text_snippet   text,
  teaching_essence text text,
  life_situations   jsonb
);
```

## Troubleshooting

**`pgvector extension not found`**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**`Server crashes on startup — embedding dimension mismatch`**
Set `EMBED_MODEL` to match the dumped embeddings. The dump uses `Xenova/all-MiniLM-L6-v2` (384 dimensions). If you re-index with a different model you must re-dump.

**`OPENROUTER_API_KEY not set`**
The server will fail at subquery generation. The search pipeline requires Gemini Flash for query reformulation.
