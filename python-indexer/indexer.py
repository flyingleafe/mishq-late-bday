"""
CocoIndex pipeline – Sutta Wisdom Finder
=========================================

Reads all sutta JSON files from  ../data/texts/,  uses an LLM to produce a
"wisdom interpretation" (which life situations / human struggles does this
teaching speak to?), then embeds that interpretation with a local Ollama
embedding model and stores everything in PostgreSQL + pgvector.

Required environment variables (place in the repo-root  .env  file):
    OPENROUTER_API_KEY        – your OpenRouter API key (for LLM interpretation)
    COCOINDEX_DATABASE_URL    – postgres URL, e.g.
                                postgresql://sutta:suttapass@localhost:5433/suttas

Prerequisites:
    Ollama running locally with nomic-embed-text:
        ollama pull nomic-embed-text
        ollama serve

Run:
    cd python-indexer
    uv run python indexer.py            # build / update index
    uv run python indexer.py --setup    # only create schema, don't index yet
    uv run python indexer.py --drop     # wipe all CocoIndex tables (destructive)
"""

import json
import os
import argparse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (one level up from this file)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cocoindex
from cocoindex.typing import Vector
from sentence_transformers import SentenceTransformer
import numpy as np
import typing
import httpx

from cocoindex.setting import Settings
from cocoindex.sources import LocalFile
from cocoindex.targets import Postgres

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters of sutta text sent to the interpretation LLM (some suttas are
# 300 K chars; 12 K gives the LLM enough to grasp the full teaching cheaply)
MAX_TEXT_CHARS = 8_000

# Characters stored as a human-readable snippet in the DB
SNIPPET_CHARS = 600

# Cheap thinking model for semantic interpretation
INTERPRET_MODEL = "google/gemini-2.5-flash"

# Sentence-transformers model for fast CPU embedding
EMBED_MODEL = "all-MiniLM-L6-v2"

# Full output dimension of all-MiniLM-L6-v2
EMBED_DIM = 384

# Absolute path to the sutta texts directory
_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "texts")

# ---------------------------------------------------------------------------
# Schema: wisdom interpretation produced by the LLM
# ---------------------------------------------------------------------------


@dataclass
class SuttaWisdom:
    """
    Structured interpretation of a sutta – produced by the LLM and used as the
    primary source for semantic search.  Every field is written in language a
    modern person would naturally use when searching for guidance.
    """

    life_situations: list[str]
    """5–10 concrete, first-person life situations or emotional states this
    sutta speaks to.  Written as a seeker would describe their own experience,
    e.g. "I feel my efforts are never recognised by others" or
    "I can't stop replaying a painful memory"."""

    themes: list[str]
    """3–6 core Buddhist themes, e.g. 'attachment', 'impermanence', 'craving',
    'compassion', 'right speech', 'emptiness'."""

    teaching_essence: str
    """The heart of this teaching in 3–5 warm, jargon-free sentences that
    someone with no background in Buddhism would immediately understand and
    find comforting or clarifying."""

    search_document: str
    """A rich, flowing paragraph (200–350 words) describing – from the
    seeker's perspective – which human experiences, painful emotions,
    existential questions and life crossroads this sutta is most relevant for.
    Imagine you are a wise, compassionate guide helping a distressed person
    find the right teaching.  Be concrete, evocative and kind.  Avoid
    academic phrasing.  This text will be embedded and used for
    vector-similarity search."""


# ---------------------------------------------------------------------------
# CocoIndex settings  (loaded from COCOINDEX_DATABASE_URL etc.)
# ---------------------------------------------------------------------------


@cocoindex.settings
def cocoindex_settings() -> Settings:
    return Settings.from_env()


# ---------------------------------------------------------------------------
# Wisdom-extraction instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """
You are a compassionate Buddhist scholar helping modern seekers find the right sutta for their struggles.

Given a Buddhist sutta, produce a structured interpretation:
• life_situations – 5–10 first-person emotional situations (concrete, human, not academic)
• themes – 3–6 core Buddhist themes
• teaching_essence – 3–5 warm, jargon-free sentences capturing the heart of the teaching
• search_document – 100–150 words from seeker's perspective about their life struggles and emotional pain

Example searches: "I feel unappreciated", "afraid of dying", "can't stop comparing", "chasing happiness", "lost someone"

Be concrete, evocative, and compassionate. Wisdom matters more than scholarship.
""".strip()

# ---------------------------------------------------------------------------
# Pure-Python helper transforms
# ---------------------------------------------------------------------------


@cocoindex.op.function()
def parse_sutta_text(content: str) -> str | None:
    """
    Extract and return the truncated sutta text ready for LLM interpretation.
    Returns None for corrupt or effectively-empty files (CocoIndex will then
    skip all downstream transforms for that row).
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    title: str = data.get("title") or ""
    raw_text: str = data.get("text") or ""

    if len(raw_text) < 80:
        # Very short stubs – still index them using the title as context
        if not title:
            return None
        return f"Sutta title: {title}\n(Full text not available in this edition.)"

    return raw_text[:MAX_TEXT_CHARS]


@cocoindex.op.function()
def parse_sutta_uid(content: str) -> str | None:
    """Extract the uid field (primary key) from the sutta JSON."""
    try:
        data = json.loads(content)
        return data.get("uid") or None
    except (json.JSONDecodeError, TypeError):
        return None


@cocoindex.op.function()
def parse_sutta_title(content: str) -> str | None:
    """Extract the title field from the sutta JSON."""
    try:
        data = json.loads(content)
        return data.get("title") or None
    except (json.JSONDecodeError, TypeError):
        return None


@cocoindex.op.function()
def parse_sutta_snippet(content: str) -> str | None:
    """Extract a short readable snippet of the sutta text for display."""
    try:
        data = json.loads(content)
        text: str = data.get("text") or data.get("title") or ""
        return text[:SNIPPET_CHARS] if text else None
    except (json.JSONDecodeError, TypeError):
        return None


@cocoindex.op.function()
def format_embed_input(wisdom: SuttaWisdom) -> str:
    """
    Combine all wisdom fields into one richly descriptive passage that
    maximises semantic search quality when embedded as RETRIEVAL_DOCUMENT.
    """
    situations_block = "\n".join(f"• {s}" for s in wisdom.life_situations)
    themes_block = ", ".join(wisdom.themes)
    return (
        f"Themes: {themes_block}\n\n"
        f"Life situations this teaching speaks to:\n{situations_block}\n\n"
        f"Core teaching: {wisdom.teaching_essence}\n\n"
        f"{wisdom.search_document}"
    )


# ---------------------------------------------------------------------------
# Local embedding with sentence-transformers (avoids Ollama HTTP overhead)
# ---------------------------------------------------------------------------

_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


# Type alias for 384-dim normalized float32 vectors (required by CocoIndex for pgvector)
EmbedVector = Vector[np.float32, typing.Literal[384]]


@cocoindex.op.function()
def embed_text_st(text: str) -> EmbedVector | None:
    if not text:
        return None
    model = _get_st_model()
    embedding: np.ndarray = model.encode(text, normalize_embeddings=True)
    return embedding


# ---------------------------------------------------------------------------
# OpenRouter LLM call with structured outputs (guarantees valid JSON)
# ---------------------------------------------------------------------------

_WISDOM_SCHEMA = {
    "name": "wisdom",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "life_situations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "5–10 first-person life situations this sutta speaks to",
            },
            "themes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3–6 core Buddhist themes",
            },
            "teaching_essence": {
                "type": "string",
                "description": "3–5 warm, jargon-free sentences capturing the heart of the teaching",
            },
            "search_document": {
                "type": "string",
                "description": "100–150 words from seeker's perspective about relevant life struggles",
            },
        },
        "required": [
            "life_situations",
            "themes",
            "teaching_essence",
            "search_document",
        ],
        "additionalProperties": False,
    },
}


@cocoindex.op.function()
def extract_wisdom(text: str) -> SuttaWisdom | None:
    if not text:
        return None
    api_key = os.environ["OPENROUTER_API_KEY"]
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/flyingleafe/mishq",
                "X-Title": "SuttaWisdomIndex",
            },
            json={
                "model": INTERPRET_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{_INSTRUCTION}\n\nSutta text:\n{text}",
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _WISDOM_SCHEMA,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return SuttaWisdom(
            life_situations=parsed["life_situations"],
            themes=parsed["themes"],
            teaching_essence=parsed["teaching_essence"],
            search_document=parsed["search_document"],
        )


# ---------------------------------------------------------------------------
# Flow definition
# ---------------------------------------------------------------------------


@cocoindex.flow_def(name="SuttaWisdomIndex")
def sutta_wisdom_flow(flow: cocoindex.FlowBuilder, scope: cocoindex.DataScope) -> None:
    """
    Full indexing pipeline:
      LocalFile(*.json) → parse → LLM wisdom → embed → Postgres + pgvector
    """
    files = flow.add_source(
        LocalFile(
            path=_DATA_DIR,
            included_patterns=["*.json"],
        )
    )

    collector = scope.add_collector()

    with files.row() as file:
        file["uid"] = file["content"].transform(parse_sutta_uid)
        file["title"] = file["content"].transform(parse_sutta_title)
        file["snippet"] = file["content"].transform(parse_sutta_snippet)
        file["text_for_llm"] = file["content"].transform(parse_sutta_text)

        file["wisdom"] = file["text_for_llm"].transform(extract_wisdom)

        file["embed_text"] = file["wisdom"].transform(format_embed_input)

        file["embedding"] = file["embed_text"].transform(embed_text_st)

        collector.collect(
            uid=file["uid"],
            title=file["title"],
            text_snippet=file["snippet"],
            teaching_essence=file["wisdom"]["teaching_essence"],
            life_situations=file["wisdom"]["life_situations"],
            themes=file["wisdom"]["themes"],
            search_document=file["wisdom"]["search_document"],
            embedding=file["embedding"],
        )

    collector.export(
        "suttas",
        Postgres(table_name="suttas"),
        primary_key_fields=["uid"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sutta Wisdom Index builder")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create Postgres schema only (no data processing)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all CocoIndex-managed tables (destructive!)",
    )
    args = parser.parse_args()

    cocoindex.init()

    if args.drop:
        print("⚠  Dropping all CocoIndex tables …")
        sutta_wisdom_flow.drop(report_to_stdout=True)
    elif args.setup:
        print("Setting up Postgres schema …")
        sutta_wisdom_flow.setup(report_to_stdout=True)
        print("Done.  Run without --setup to build the index.")
    else:
        print(f"Building Sutta Wisdom Index …")
        print(f"  Data dir : {_DATA_DIR}")
        print(f"  LLM model: {INTERPRET_MODEL}")
        print(f"  Embed    : {EMBED_MODEL}  dim={EMBED_DIM}")
        print()
        sutta_wisdom_flow.setup(report_to_stdout=True)
        stats = sutta_wisdom_flow.update(full_reprocess=False, print_stats=True)
        print(f"\n✓ Index update complete.  Stats: {stats}")
