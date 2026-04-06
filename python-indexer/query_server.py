"""
Sutta Query Server
==================

FastAPI server for semantic search over sutta chunks.
Returns both matching chunks AND the full sutta text.

Run:
    cd python-indexer
    uv run uvicorn query_server:app --port 8000
"""

import json
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from sentence_transformers import SentenceTransformer
import psycopg

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# App with lifespan
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    print("Model ready.")
    yield


app = FastAPI(title="Sutta Query API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChunkMatch(BaseModel):
    chunk_uid: str
    chunk_text: str
    score: float


class SearchResult(BaseModel):
    sutta_uid: str
    sutta_title: str
    chunk: ChunkMatch
    sutta_text: str = Field(description="Full sutta text from JSON file")


class SearchResponse(BaseModel):
    query: str
    top: int
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "texts")


def get_sutta_text(uid: str) -> str:
    path = Path(_DATA_DIR) / f"{uid}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("text", "")
    return ""


def search_chunks(query: str, top_k: int) -> list[dict]:
    model = get_model()
    emb = model.encode(query, normalize_embeddings=True).tolist()

    conn = psycopg.connect(os.environ["COCOINDEX_DATABASE_URL"])
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_uid, sutta_uid, sutta_title, chunk_text,
                   1 - (embedding <=> %s::vector) AS score
            FROM sutta_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (emb, emb, top_k),
        )
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "chunk_uid": r[0],
            "sutta_uid": r[1],
            "sutta_title": r[2],
            "chunk_text": r[3],
            "score": round(r[4], 4),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/search", response_model=SearchResponse)
def search(
    q: Annotated[str, Query(description="Natural language search query")],
    top: Annotated[int, Query(ge=1, le=50, description="Number of results")] = 5,
) -> SearchResponse:
    chunks = search_chunks(q, top_k=top)

    results = []
    seen_suttas = set()
    for chunk in chunks:
        uid = chunk["sutta_uid"]
        if uid in seen_suttas:
            continue
        seen_suttas.add(uid)

        results.append(
            SearchResult(
                sutta_uid=uid,
                sutta_title=chunk["sutta_title"],
                chunk=ChunkMatch(
                    chunk_uid=chunk["chunk_uid"],
                    chunk_text=chunk["chunk_text"],
                    score=chunk["score"],
                ),
                sutta_text=get_sutta_text(uid),
            )
        )

    return SearchResponse(query=q, top=top, results=results)


@app.get("/health")
def health():
    return {"status": "ok"}
