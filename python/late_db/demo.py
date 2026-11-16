"""Run Rust retrieval over illustrative vectors or a saved ColVision index."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import LateInteractionEngine


def synthetic_index() -> Dict[str, Any]:
    """Return hand-authored vectors; these are not model embeddings."""
    examples = [
        ("invoice", "Invoice with item prices and a total", [[1, 0, 0, 0], [0.8, 0.2, 0, 0]]),
        ("chart", "Quarterly revenue bar chart", [[0, 1, 0, 0], [0.2, 0.8, 0, 0]]),
        ("diagram", "Labeled water cycle diagram", [[0, 0, 1, 0], [0, 0, 0.8, 0.2]]),
        ("satellite", "Satellite view of forest and farmland", [[0, 0, 0, 1], [0, 0, 0.2, 0.8]]),
    ]
    return {
        "schema_version": 1,
        "model_id": "synthetic-hand-authored-v1",
        "embedding_dim": 4,
        "image_embeddings": [
            {"image_id": image_id, "vectors": vectors, "record": {"title": title}}
            for image_id, title, vectors in examples
        ],
        "query_embeddings": [
            {
                "query": {"query_id": image_id, "text": "Find: " + title.lower()},
                "vectors": vectors,
            }
            for image_id, title, vectors in examples
        ],
    }


def validate_index(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("embedding index must be a JSON object")
    schema_version = payload.get("schema_version", payload.get("version"))
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("embedding index must have schema_version 1")
    dimension = payload.get("embedding_dim")
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("embedding_dim must be a positive integer")
    if not isinstance(payload.get("model_id"), str) or not payload["model_id"].strip():
        raise ValueError("model_id must be a nonempty string")
    for collection, metadata_key, identifier_key in [
        ("image_embeddings", "record", "image_id"),
        ("query_embeddings", "query", "query_id"),
    ]:
        entries = payload.get(collection)
        if not isinstance(entries, list) or not entries:
            raise ValueError(collection + " must be a nonempty list")
        identifiers = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get(metadata_key), dict):
                raise ValueError(collection + " entries must contain " + metadata_key + " objects")
            metadata = entry if collection == "image_embeddings" else entry[metadata_key]
            identifier = metadata.get(identifier_key)
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(identifier_key + " must be a nonempty string")
            if identifier in identifiers:
                raise ValueError("duplicate " + identifier_key + ": " + identifier)
            identifiers.add(identifier)
            vectors = entry.get("vectors")
            if not isinstance(vectors, list) or not vectors:
                raise ValueError(identifier + ": vectors must be a nonempty list")
            if collection == "query_embeddings":
                if not isinstance(metadata.get("text"), str) or not metadata["text"].strip():
                    raise ValueError(identifier + ": query text must be a nonempty string")
    return payload


def build_engine(payload: Dict[str, Any]) -> LateInteractionEngine:
    document_tokens = sum(len(entry["vectors"]) for entry in payload["image_embeddings"])
    query_tokens = max(len(entry["vectors"]) for entry in payload["query_embeddings"])
    engine = LateInteractionEngine(
        payload["embedding_dim"],
        max_resident_tokens=max(250000, document_tokens),
        max_query_tokens=max(512, query_tokens),
    )
    for entry in payload["query_embeddings"]:
        engine.search(entry["vectors"], top_k=1)
    for document_id, entry in enumerate(payload["image_embeddings"]):
        engine.add_document(document_id, entry["vectors"], modality="image")
    return engine


def print_results(
    engine: LateInteractionEngine,
    payload: Dict[str, Any],
    query_entry: Dict[str, Any],
    top_k: int,
) -> None:
    query = query_entry["query"]
    print("\nQuery [{}]: {}".format(query["query_id"], query["text"]))
    print("Rank  MaxSim score  Document")
    for rank, (document_id, score) in enumerate(
        engine.search(query_entry["vectors"], top_k=top_k), start=1
    ):
        entry = payload["image_embeddings"][document_id]
        record = entry["record"]
        title = record.get("title") or record.get("slice") or entry["image_id"]
        print("{:>4}  {:>12.6f}  {} | {}".format(rank, score, entry["image_id"], title))
        if record.get("local_path"):
            print("                    image: {}".format(record["local_path"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, help="saved schema-version-1 ColVision JSON index")
    parser.add_argument("--query-id", help="cached query ID (default: first query)")
    parser.add_argument("--top-k", type=int, default=3, help="number of results (default: 3)")
    parser.add_argument("--interactive", action="store_true", help="choose cached queries repeatedly")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        if args.embeddings is None:
            payload = synthetic_index()
        else:
            with args.embeddings.open(encoding="utf-8") as index_file:
                payload = json.load(index_file)
        payload = validate_index(payload)
        query_entries = payload["query_embeddings"]
        queries_by_id = {entry["query"]["query_id"]: entry for entry in query_entries}
        selected_id = args.query_id or query_entries[0]["query"]["query_id"]
        if selected_id not in queries_by_id:
            raise ValueError("unknown query ID {!r}; choose: {}".format(selected_id, ", ".join(queries_by_id)))
        engine = build_engine(payload)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))
    if args.embeddings is None:
        print("SYNTHETIC DEMO: hand-authored vectors, not model retrieval quality.")
    else:
        print("Saved embedding replay: {}".format(args.embeddings))
    print("Model: {}".format(payload["model_id"]))
    print("Rust engine: {} documents, {} tokens, dimension {}".format(
        engine.document_count, engine.token_count, engine.embedding_dim
    ))
    print_results(engine, payload, queries_by_id[selected_id], args.top_k)
    if args.interactive:
        while True:
            print("\nCached queries (enter a number or ID; q quits):")
            for position, entry in enumerate(query_entries, start=1):
                query = entry["query"]
                print("  {}. [{}] {}".format(position, query["query_id"], query["text"]))
            try:
                selection = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if selection.lower() in {"q", "quit", "exit"}:
                break
            entry = queries_by_id.get(selection)
            if entry is None and selection.isdecimal():
                position = int(selection) - 1
                if 0 <= position < len(query_entries):
                    entry = query_entries[position]
            if entry is None:
                print("Unknown cached query; select one from the list.")
                continue
            print_results(engine, payload, entry, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
