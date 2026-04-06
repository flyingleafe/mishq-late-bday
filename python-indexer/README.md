# Sutta Wisdom Index – CocoIndex Pipeline

Reads 6 400+ sutta JSON files, uses **Gemini 2.5 Flash** (thinking) to produce
a wisdom interpretation per sutta, embeds each one with
**gemini-embedding-2-flash** (768 dim), and stores everything in
**PostgreSQL + pgvector** with an HNSW cosine-similarity index.

## Prerequisites

1. **Docker** – the pgvector container (started once):

   ```bash
   docker run -d --name sutta-pgvector \
     -e POSTGRES_PASSWORD=suttapass \
     -e POSTGRES_USER=sutta \
     -e POSTGRES_DB=suttas \
     -p 5433:5432 \
     pgvector/pgvector:pg17
   ```

2. **Gemini API key** – set in the repo-root `.env` file:

   ```
   GEMINI_API_KEY=<your key>
   COCOINDEX_DATABASE_URL=postgresql://sutta:suttapass@localhost:5433/suttas
   ```

## Running

```bash
cd python-indexer

# First run – setup schema + index all suttas (~6 400 LLM calls, ~2–4 h)
uv run python indexer.py

# Subsequent runs – only re-index changed files (seconds)
uv run python indexer.py

# Schema only (no data)
uv run python indexer.py --setup

# Wipe everything and start fresh
uv run python indexer.py --drop
uv run python indexer.py
```

CocoIndex is fully incremental: it hashes each file and skips unchanged suttas,
so re-runs are near-instant once the initial index is built.

## What ends up in Postgres

Table `suttas`:

| Column            | Type          | Description                                          |
|-------------------|---------------|------------------------------------------------------|
| `uid`             | text (PK)     | Sutta identifier, e.g. `mn1`, `dn15`                |
| `title`           | text          | English title                                         |
| `text_snippet`    | text          | First 600 chars of the original Pali text            |
| `teaching_essence`| text          | 3–5 sentence modern-language summary of the teaching |
| `life_situations` | jsonb         | List of concrete life situations this sutta speaks to|
| `themes`          | jsonb         | Buddhist themes, e.g. `["attachment","impermanence"]`|
| `search_document` | text          | Rich 200–350 word passage for semantic search        |
| `embedding`       | vector(768)   | RETRIEVAL_DOCUMENT embedding of search_document      |

Index: `suttas__embedding__vector_cosine_ops` (HNSW, cosine similarity)

## Querying from the web app

Embed the user's query with `gemini-embedding-2-flash` using task type
`RETRIEVAL_QUERY`, then run:

```sql
SELECT uid, title, teaching_essence, life_situations, themes,
       1 - (embedding <=> $1::vector) AS score
FROM suttas
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

where `$1` is the 768-float query embedding.
