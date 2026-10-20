#!/usr/bin/env python3
"""Create a hand-curation manifest scaffold for the universal retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_plan(path: Path) -> dict:
    return json.loads(path.read_text())


def make_record(slice_spec: dict, index: int) -> dict:
    image_id = f"{slice_spec['slice']}_{index:02d}"
    query_base = f"{image_id}_q"
    return {
        "image_id": image_id,
        "slice": slice_spec["slice"],
        "source": "TODO: choose one of " + ", ".join(slice_spec["sources"]),
        "license": "TODO: record license or dataset terms",
        "local_path": f"data/universal_retrieval/images/{slice_spec['slice']}/{image_id}.png",
        "source_url": "TODO: dataset locator, DOI, or image URL",
        "notes": "TODO: describe why this image belongs in the benchmark",
        "queries": [
            {
                "query_id": f"{query_base}_direct",
                "kind": "direct",
                "text": "TODO: direct retrieval query",
                "positive_image_ids": [image_id],
                "hard_negative_image_ids": [],
                "tags": slice_spec["tags"],
            },
            {
                "query_id": f"{query_base}_fine",
                "kind": "fine_grained",
                "text": "TODO: fine-grained text/layout/visual query",
                "positive_image_ids": [image_id],
                "hard_negative_image_ids": [],
                "tags": slice_spec["tags"],
            },
            {
                "query_id": f"{query_base}_negative",
                "kind": "hard_negative",
                "text": "TODO: near-miss query for a visually similar wrong image",
                "positive_image_ids": [],
                "hard_negative_image_ids": [image_id],
                "tags": slice_spec["tags"] + ["near_duplicate"],
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("benchmarks/universal_retrieval/seed_plan.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/universal_retrieval/manifest.jsonl"),
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for slice_spec in plan["slices"]:
        for index in range(1, slice_spec["target"] + 1):
            records.append(make_record(slice_spec, index))

    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
