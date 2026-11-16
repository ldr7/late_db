#!/usr/bin/env python3
"""Validate manifest shape without requiring third-party packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_RECORD_FIELDS = {
    "image_id",
    "slice",
    "source",
    "license",
    "local_path",
    "source_url",
    "queries",
}

REQUIRED_QUERY_FIELDS = {
    "query_id",
    "kind",
    "text",
    "positive_image_ids",
    "hard_negative_image_ids",
    "tags",
}


def validate_record(record: dict, line_no: int) -> list[str]:
    errors = []
    missing = REQUIRED_RECORD_FIELDS - record.keys()
    if missing:
        errors.append(f"line {line_no}: missing record fields {sorted(missing)}")
    if not isinstance(record.get("queries"), list) or not record.get("queries"):
        errors.append(f"line {line_no}: queries must be a non-empty list")
        return errors
    for query in record["queries"]:
        missing_query = REQUIRED_QUERY_FIELDS - query.keys()
        if missing_query:
            errors.append(
                f"line {line_no}: query {query.get('query_id')} missing fields {sorted(missing_query)}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    errors = []
    count = 0
    with args.manifest.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            errors.extend(validate_record(json.loads(line), line_no))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"validated {count} records")


if __name__ == "__main__":
    main()
