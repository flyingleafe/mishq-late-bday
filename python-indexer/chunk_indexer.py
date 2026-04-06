"""
Chunk Indexer – Fast Sutta Text Chunk Index
===========================================

Step 1: chunk all suttas, embed them, write to chunks.jsonl
Step 2: index chunks.jsonl with CocoIndex

Run:
    cd python-indexer
    uv run python chunk_indexer.py        # chunk + embed + write
    uv run python chunk_indexer.py --index # index chunks.jsonl into postgres
    uv run python chunk_indexer.py --drop  # wipe tables
"""

import json
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cocoindex
from cocoindex.typing import Vector
from sentence_transformers import SentenceTransformer
import numpy as np
import typing

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 80
_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "texts")
_CHUNKS_FILE = str(Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl")

_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


EmbedVector = Vector[np.float32, typing.Literal[384]]


def split_into_chunks(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < MIN_CHUNK_CHARS:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append(para)
        else:
            sentences = para.replace(". ", ".\n").split("\n")
            current = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current) + len(sent) + 1 <= MAX_CHUNK_CHARS:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)
    return chunks


BATCH_SIZE = 64


def build_chunk_corpus():
    print(f"Loading model {EMBED_MODEL} …")
    model = _get_st_model()
    print("Model loaded.")

    chunks_path = Path(_CHUNKS_FILE)

    import glob

    existing_uids = set()
    if chunks_path.exists():
        with open(chunks_path) as f:
            for line in f:
                d = json.loads(line)
                existing_uids.add(d["sutta_uid"])
        mode = "a"
        print(f"Resuming: {len(existing_uids)} suttas already processed, appending …")
    else:
        mode = "w"
        print("Starting fresh …")

    files = glob.glob(f"{_DATA_DIR}/*.json")
    print(f"Processing {len(files)} sutta files …")

    total_chunks = 0
    batch = []
    metadata = []

    with open(chunks_path, mode) as out:
        for fi, filepath in enumerate(files):
            with open(filepath) as f:
                data = json.load(f)
            uid = data.get("uid", "")
            title = data.get("title", "")
            text = data.get("text", "")
            if not text or uid in existing_uids:
                continue

            chunk_texts = split_into_chunks(text)
            for ci, chunk_text in enumerate(chunk_texts):
                batch.append(chunk_text)
                metadata.append((f"{uid}.c{ci}", uid, title, ci))
                total_chunks += 1

                if len(batch) >= BATCH_SIZE:
                    embeddings = model.encode(batch, normalize_embeddings=True)
                    for i, (m, emb) in enumerate(zip(metadata, embeddings)):
                        record = {
                            "chunk_uid": m[0],
                            "sutta_uid": m[1],
                            "sutta_title": m[2],
                            "chunk_index": m[3],
                            "chunk_text": batch[i],
                            "embedding": emb.tolist(),
                        }
                        out.write(json.dumps(record) + "\n")
                    batch.clear()
                    metadata.clear()

            if (fi + 1) % 500 == 0:
                print(
                    f"  {fi + 1}/{len(files)} files, {total_chunks} chunks written this run"
                )

        # flush remaining
        if batch:
            embeddings = model.encode(batch, normalize_embeddings=True)
            for i, (m, emb) in enumerate(zip(metadata, embeddings)):
                record = {
                    "chunk_uid": m[0],
                    "sutta_uid": m[1],
                    "sutta_title": m[2],
                    "chunk_index": m[3],
                    "chunk_text": batch[i],
                    "embedding": emb.tolist(),
                }
                out.write(json.dumps(record) + "\n")

    print(f"Done.  {total_chunks} new chunks appended to {_CHUNKS_FILE}")
    return total_chunks


from cocoindex.setting import Settings


@cocoindex.settings
def cocoindex_settings() -> Settings:
    return Settings.from_env()


from cocoindex.sources import LocalFile
from cocoindex.targets import Postgres


@cocoindex.op.function()
def embed_from_list(emb_list: list[float]) -> EmbedVector | None:
    if not emb_list:
        return None
    return np.array(emb_list, dtype=np.float32)


import json as _json


@cocoindex.op.function()
def parse_chunk(content: str) -> cocoindex.Json | None:
    try:
        return _json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


@cocoindex.op.function()
def get_chunk_uid(d: cocoindex.Json) -> str | None:
    return d.get("chunk_uid") if d else None


@cocoindex.op.function()
def get_sutta_uid(d: cocoindex.Json) -> str | None:
    return d.get("sutta_uid") if d else None


@cocoindex.op.function()
def get_sutta_title(d: cocoindex.Json) -> str | None:
    return d.get("sutta_title") if d else None


@cocoindex.op.function()
def get_chunk_text(d: cocoindex.Json) -> str | None:
    return d.get("chunk_text") if d else None


@cocoindex.op.function()
def get_embedding(d: cocoindex.Json) -> list[float] | None:
    return d.get("embedding") if d else None


@cocoindex.flow_def(name="SuttaChunkIndex")
def sutta_chunk_flow(flow: cocoindex.FlowBuilder, scope: cocoindex.DataScope) -> None:
    files = flow.add_source(
        LocalFile(
            path=str(Path(__file__).resolve().parent.parent / "data"),
            included_patterns=["chunks.jsonl"],
        )
    )

    collector = scope.add_collector()

    with files.row() as file:
        raw = file["content"].transform(parse_chunk)
        file["chunk_uid"] = raw.transform(get_chunk_uid)
        file["sutta_uid"] = raw.transform(get_sutta_uid)
        file["sutta_title"] = raw.transform(get_sutta_title)
        file["chunk_text"] = raw.transform(get_chunk_text)
        file["embedding"] = raw.transform(get_embedding).transform(embed_from_list)

        collector.collect(
            chunk_uid=file["chunk_uid"],
            sutta_uid=file["sutta_uid"],
            sutta_title=file["sutta_title"],
            chunk_text=file["chunk_text"],
            embedding=file["embedding"],
        )

    collector.export(
        "sutta_chunks",
        Postgres(table_name="sutta_chunks"),
        primary_key_fields=["chunk_uid"],
    )


def index_chunks_bulk() -> int:
    import psycopg

    chunks_file = Path(_CHUNKS_FILE)
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Run without --index first.")
        return 0

    print("Bulk loading chunks into postgres …")
    conn = psycopg.connect(os.environ["COCOINDEX_DATABASE_URL"])
    cur = conn.cursor()

    total = 0
    with open(chunks_file) as f:
        for line in f:
            record = json.loads(line)
            cur.execute(
                """
                INSERT INTO sutta_chunks (chunk_uid, sutta_uid, sutta_title, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_uid) DO UPDATE
                SET sutta_uid = EXCLUDED.sutta_uid,
                    sutta_title = EXCLUDED.sutta_title,
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding
                """,
                (
                    record["chunk_uid"],
                    record["sutta_uid"],
                    record["sutta_title"],
                    record["chunk_text"],
                    record["embedding"],
                ),
            )
            total += 1
            if total % 5000 == 0:
                print(f"  {total} chunks loaded …")
                conn.commit()

    conn.commit()
    cur.close()
    conn.close()
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sutta Chunk Index builder")
    parser.add_argument("--drop", action="store_true", help="Drop all CocoIndex tables")
    parser.add_argument("--setup", action="store_true", help="Create schema only")
    parser.add_argument(
        "--index", action="store_true", help="Index chunks.jsonl into postgres"
    )
    args = parser.parse_args()

    cocoindex.init()

    if args.drop:
        print("Dropping all CocoIndex tables …")
        sutta_chunk_flow.drop(report_to_stdout=True)
    elif args.setup:
        print("Setting up Postgres schema …")
        sutta_chunk_flow.setup(report_to_stdout=True)
    elif args.index:
        total = index_chunks_bulk()
        print(f"\nIndexed {total} chunks into postgres.")
    else:
        total = build_chunk_corpus()
        print(f"\nChunk corpus built: {total} chunks.")
        print("Run with --index to load into postgres, or --setup to create schema.")
