"""
CocoIndex pipeline – Sutta Wisdom Finder
=========================================

Reads all sutta JSON files from  ../data/texts/,  uses a Gemini thinking
model to produce a "wisdom interpretation" (which life situations / human
struggles does this teaching speak to?), then embeds that interpretation
with gemini-embedding-001 and stores everything in PostgreSQL + pgvector.

Required environment variables (place in the repo-root  .env  file):
    GEMINI_API_KEY            – your Google AI API key
    COCOINDEX_DATABASE_URL    – postgres URL, e.g.
                                postgresql://sutta:suttapass@localhost:5433/suttas

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
from cocoindex import LlmApiType, LlmSpec
from cocoindex.functions import EmbedText, ExtractByLlm
from cocoindex.index import VectorIndexDef, VectorSimilarityMetric
from cocoindex.setting import Settings
from cocoindex.sources import LocalFile
from cocoindex.targets import Postgres

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters of sutta text sent to the interpretation LLM (some suttas are
# 300 K chars; 12 K gives the LLM enough to grasp the full teaching cheaply)
MAX_TEXT_CHARS = 12_000

# Characters stored as a human-readable snippet in the DB
SNIPPET_CHARS = 600

# Cheap thinking model for semantic interpretation
INTERPRET_MODEL = "gemini-2.5-flash"

# Fast embedding model for vector search
EMBED_MODEL = "gemini-embedding-001"

# Output dimension — capped at 1536 so pgvector HNSW/IVFFlat (<= 2000) works
EMBED_DIM = 1536

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
You are a compassionate Buddhist scholar helping modern people find the right
sutta for their personal struggles and life situations.

Given the text of a Buddhist sutta, produce a structured wisdom interpretation
that will let someone find this sutta by searching with plain everyday language
such as:
  "I feel like my talents go unappreciated"
  "I'm afraid of dying"
  "I can't stop comparing myself to others"
  "I keep chasing happiness but never find it"
  "I lost someone I love and I can't cope"

Guidelines:
• life_situations – list 5–10 specific first-person emotional situations or
  life problems. Be concrete and human, not academic.
• themes – list 3–6 core Buddhist themes covered.
• teaching_essence – 3–5 warm, jargon-free sentences capturing the heart of
  the teaching.
• search_document – 200–350 words from the seeker's perspective, describing
  which human experiences, inner struggles, painful emotions and existential
  questions this sutta most speaks to.  Imagine guiding a distressed person
  who has never heard of Buddhism.  Be evocative, concrete, and compassionate.

Think carefully before answering.  Wisdom matters more than scholarship.
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
# Flow definition
# ---------------------------------------------------------------------------


@cocoindex.flow_def(name="SuttaWisdomIndex")
def sutta_wisdom_flow(flow: cocoindex.FlowBuilder, scope: cocoindex.DataScope) -> None:
    """
    Full indexing pipeline:
      LocalFile(*.json) → parse → LLM wisdom → embed → Postgres + pgvector
    """
    # Register the Gemini API key as a transient auth entry so it's available
    # to both the LLM interpretation step and the embedding step.
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    gemini_key_ref = cocoindex.add_transient_auth_entry(gemini_api_key)

    # LLM spec: cheap thinking model for semantic interpretation
    interpret_llm = LlmSpec(
        api_type=LlmApiType.GEMINI,
        model=INTERPRET_MODEL,
        api_key=gemini_key_ref,
    )

    # --- Source ---
    files = flow.add_source(
        LocalFile(
            path=_DATA_DIR,
            included_patterns=["*.json"],
        )
    )

    collector = scope.add_collector()

    with files.row() as file:
        # Extract individual fields from the JSON
        file["uid"] = file["content"].transform(parse_sutta_uid)
        file["title"] = file["content"].transform(parse_sutta_title)
        file["snippet"] = file["content"].transform(parse_sutta_snippet)
        file["text_for_llm"] = file["content"].transform(parse_sutta_text)

        # --- LLM wisdom interpretation ---
        # When text_for_llm is None, ExtractByLlm propagates None → row skipped
        file["wisdom"] = file["text_for_llm"].transform(
            ExtractByLlm(
                llm_spec=interpret_llm,
                output_type=SuttaWisdom,
                instruction=_INSTRUCTION,
            )
        )

        # --- Combine wisdom fields into a single embeddable document ---
        file["embed_text"] = file["wisdom"].transform(format_embed_input)

        # --- Vector embedding (RETRIEVAL_DOCUMENT task for asymmetric search) ---
        file["embedding"] = file["embed_text"].transform(
            EmbedText(
                api_type=LlmApiType.GEMINI,
                model=EMBED_MODEL,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimension=EMBED_DIM,
                expected_output_dimension=EMBED_DIM,
                api_key=gemini_key_ref,
            )
        )

        # --- Collect output row ---
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

    # --- Export to Postgres with a cosine-similarity HNSW vector index ---
    collector.export(
        "suttas",
        Postgres(table_name="suttas"),
        primary_key_fields=["uid"],
        vector_indexes=[
            VectorIndexDef(
                field_name="embedding",
                metric=VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
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
