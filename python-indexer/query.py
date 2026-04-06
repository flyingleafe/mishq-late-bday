"""
Semantic Query CLI – Search suttas by natural language query
============================================================

Uses the FastAPI query server by default. Falls back to --local mode.

Usage:
    cd python-indexer
    uv run python query.py "I feel anxious about my future" --top 5

    # Or run server first, then query:
    uv run uvicorn query_server:app --port 8000 &
    uv run python query.py "I feel anxious about my future"

    # Local mode (no server needed, slower startup):
    uv run python query.py "I feel anxious" --local
"""

import argparse
import sys
import httpx

DEFAULT_URL = "http://localhost:8000/search"


def format_results(results: list[dict], max_chunk_chars: int) -> str:
    output = []
    for i, r in enumerate(results, 1):
        output.append(f"{'=' * 60}")
        output.append(
            f"  [{i}] {r['sutta_title']} ({r['sutta_uid']})  score={r['chunk']['score']}"
        )

        chunk = r["chunk"]["chunk_text"]
        if len(chunk) > max_chunk_chars:
            chunk = chunk[:max_chunk_chars] + "…"
        output.append(f"  [Matching chunk]")
        output.append(f"  {chunk}")

        sutta = r["sutta_text"]
#        if len(sutta) > 200:
#            sutta = sutta[:200] + "…"
        output.append(f"  [Full sutta text]")
        output.append(f"  {sutta}")
        output.append("")
    return "\n".join(output)


def query_via_server(query: str, top: int, url: str = DEFAULT_URL) -> list[dict]:
    resp = httpx.get(url, params={"q": query, "top": top}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["results"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over sutta chunks")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument(
        "--top", "-n", type=int, default=5, help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=300,
        help="Max characters for matching chunk display (default: 300)",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Query server URL (default: {DEFAULT_URL})"
    )
    args = parser.parse_args()

    print(f"\nQuery: {args.query!r}\n", file=sys.stderr)
    print(f"Server: {args.url}\n", file=sys.stderr)

    try:
        results = query_via_server(args.query, args.top, args.url)
    except Exception as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print(format_results(results, args.max_chunk_chars))
