# How It Works

This is a **semantic search engine** over Buddhist suttas — it finds passages that match the *meaning* of your query, not just keywords.

## The Pipeline

### 1. Subquery Expansion
Your query goes to **Gemini Flash** (via OpenRouter), which generates 5 related subqueries — different angles on the same question. For example, "how to handle anger" might expand to "mindfulness of anger", "letting go of irritation", "the Buddhist view of rage", etc.

### 2. Embedding
Every query (the original + 5 subqueries) is converted into a **384-dimensional vector** using the `all-MiniLM-L6-v2` model. This is a sentence embedding model — it turns text into a point in semantic space. Similar meanings land close together.

### 3. Vector Search
The pre-computed embeddings live in **PostgreSQL with pgvector**. Each sutta is split into overlapping ~200-character chunks, each with its own embedding. Subquery embeddings are compared against all chunks using **cosine similarity**, and the top candidates are retrieved.

### 4. Ranking
A chunk that appears across multiple subquery results gets a higher score. The final ranking blends:
- How consistently the chunk matched across subqueries
- How close the chunk embedding is to the original query embedding

### 5. Highlighting
The matching chunk is aligned back to the **full sutta text** using a Viterbi algorithm (beam search), accounting for whitespace differences and HTML entities. The matched region is wrapped in `<mark>` tags and displayed.

## Data

- **Sutta texts**: crawled from [SuttaCentral](https://suttacentral.net) and stored as JSON in `data/texts/`
- **Embeddings**: pre-computed with `all-MiniLM-L6-v2` and stored in the `sutta_chunks` table
- **Database**: PostgreSQL 17 with `pgvector` extension for efficient similarity search

## The UI

The search page sends queries to the `/stream` endpoint, which streams results via **Server-Sent Events (SSE)**. The frontend renders results progressively as each subquery completes.

The interface is deliberately minimal — the suttas speak for themselves.

## Why Subqueries?

A single query like "dealing with grief" might not exactly match any chunk's wording. But "coping with loss", "mindfulness of sorrow", and "impermanence of attachment" probably do. Subquery expansion bridges the gap between how a user thinks and how the text is written.
