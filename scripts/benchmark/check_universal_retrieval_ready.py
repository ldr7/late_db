#!/usr/bin/env python3
"""Check whether the universal retrieval benchmark can run locally."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REQUIRED_PACKAGES = [
    "PIL",
    "torch",
    "transformers",
    "colpali_engine",
]


def package_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_records(manifest: Path) -> list[dict]:
    with manifest.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/universal_retrieval/manifest.jsonl"),
    )
    args = parser.parse_args()

    records = load_records(args.manifest)
    missing_images = [
        record["local_path"]
        for record in records
        if "TODO:" in record["source_url"] or not Path(record["local_path"]).exists()
    ]
    todo_queries = [
        query["query_id"]
        for record in records
        for query in record["queries"]
        if "TODO:" in query["text"]
    ]
    missing_packages = [pkg for pkg in REQUIRED_PACKAGES if not package_status(pkg)]

    print(f"manifest_records={len(records)}")
    print(f"missing_or_placeholder_images={len(missing_images)}")
    print(f"placeholder_queries={len(todo_queries)}")
    print(f"missing_packages={','.join(missing_packages) if missing_packages else 'none'}")

    if missing_images:
        print("first_missing_image=" + missing_images[0])
    if todo_queries:
        print("first_placeholder_query=" + todo_queries[0])

    if missing_images or todo_queries or missing_packages:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
