#!/usr/bin/env python3
"""Fetch a 12-image public smoke dataset from Hugging Face datasets."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def save_image(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, bytes):
        image = Image.open(io.BytesIO(value)).convert("RGB")
    else:
        raise TypeError(f"unsupported image type: {type(value)!r}")
    image.save(path)


def make_query(image_id: str, kind: str, text: str, tags: list[str]) -> dict:
    return {
        "query_id": f"{image_id}_q_{kind}",
        "kind": "direct" if kind == "direct" else "fine_grained",
        "text": text,
        "positive_image_ids": [image_id],
        "hard_negative_image_ids": [],
        "tags": tags,
    }


def make_record(
    image_id: str,
    slice_name: str,
    source: str,
    source_url: str,
    license_text: str,
    local_path: Path,
    queries: list[dict],
    notes: str,
) -> dict:
    return {
        "image_id": image_id,
        "slice": slice_name,
        "source": source,
        "license": license_text,
        "local_path": local_path.as_posix(),
        "source_url": source_url,
        "notes": notes,
        "queries": queries,
    }


def pick_text(words: list[str], max_words: int = 8) -> str:
    usable = [word for word in words if any(ch.isalpha() for ch in word)]
    return " ".join(usable[:max_words])


def add_funsd(records: list[dict], image_dir: Path, cache_dir: Path) -> None:
    ds = load_dataset("nielsr/funsd", cache_dir=cache_dir)["train"]
    for out_idx, row_idx in enumerate([0, 1], start=1):
        row = ds[row_idx]
        image_id = f"public_funsd_{out_idx:02d}"
        path = image_dir / "forms_ocr_docs" / f"{image_id}.png"
        save_image(row["image"], path)
        snippet = pick_text(row["words"])
        records.append(
            make_record(
                image_id=image_id,
                slice_name="forms_ocr_docs",
                source="nielsr/funsd",
                source_url="https://huggingface.co/datasets/nielsr/funsd",
                license_text="dataset terms on Hugging Face",
                local_path=path,
                notes=f"FUNSD row {row_idx}, id={row['id']}",
                queries=[
                    make_query(
                        image_id,
                        "direct",
                        f"Find the scanned form containing the text {snippet!r}.",
                        ["forms_ocr_docs", "ocr", "tiny_text"],
                    ),
                    make_query(
                        image_id,
                        "fine",
                        "Find the noisy scanned form with labeled fields and boxed document layout.",
                        ["forms_ocr_docs", "layout_sensitive", "key_value_layout"],
                    ),
                ],
            )
        )


def add_chartqa(records: list[dict], image_dir: Path, cache_dir: Path) -> None:
    ds = load_dataset("ahmed-masry/ChartQA", cache_dir=cache_dir)["train"]
    for out_idx, row_idx in enumerate([0, 1], start=1):
        row = ds[row_idx]
        image_id = f"public_chartqa_{out_idx:02d}"
        path = image_dir / "charts_tables" / f"{image_id}.png"
        save_image(row["image"], path)
        records.append(
            make_record(
                image_id=image_id,
                slice_name="charts_tables",
                source="ahmed-masry/ChartQA",
                source_url="https://huggingface.co/datasets/ahmed-masry/ChartQA",
                license_text="GPL-3.0 per dataset card",
                local_path=path,
                notes=f"ChartQA row {row_idx}, imgname={row['imgname']}, label={row['label']}",
                queries=[
                    make_query(
                        image_id,
                        "direct",
                        row["query"],
                        ["charts_tables", "chart", "numeric_labels"],
                    ),
                    make_query(
                        image_id,
                        "fine",
                        "Find the chart image with axes, plotted values, and a question-answer visual pattern.",
                        ["charts_tables", "legend_binding", "layout_sensitive"],
                    ),
                ],
            )
        )


def add_ai2d(records: list[dict], image_dir: Path, cache_dir: Path) -> None:
    ds = load_dataset("lmms-lab/ai2d", cache_dir=cache_dir)["test"]
    for out_idx, row_idx in enumerate([0, 1, 2, 3], start=1):
        row = ds[row_idx]
        image_id = f"public_ai2d_{out_idx:02d}"
        path = image_dir / "diagrams" / f"{image_id}.png"
        save_image(row["image"], path)
        records.append(
            make_record(
                image_id=image_id,
                slice_name="diagrams",
                source="lmms-lab/ai2d",
                source_url="https://huggingface.co/datasets/lmms-lab/ai2d",
                license_text="dataset terms on Hugging Face",
                local_path=path,
                notes=f"AI2D row {row_idx}, answer={row['answer']}",
                queries=[
                    make_query(
                        image_id,
                        "direct",
                        row["question"],
                        ["diagrams", "science_diagram", "symbolic_structure"],
                    ),
                    make_query(
                        image_id,
                        "fine",
                        "Find the science diagram with labels, arrows, or structured visual explanation.",
                        ["diagrams", "arrows", "line_art"],
                    ),
                ],
            )
        )


def add_eurosat(records: list[dict], image_dir: Path, cache_dir: Path) -> None:
    ds = load_dataset("blanchon/EuroSAT_RGB", cache_dir=cache_dir)["train"]
    label_names = ds.features["label"].names
    picked = []
    seen = set()
    for idx, row in enumerate(ds):
        name = label_names[row["label"]]
        if name not in seen:
            picked.append((idx, row, name))
            seen.add(name)
        if len(picked) == 4:
            break

    for out_idx, (row_idx, row, label_name) in enumerate(picked, start=1):
        image_id = f"public_eurosat_{out_idx:02d}"
        path = image_dir / "remote_sensing" / f"{image_id}.png"
        save_image(row["image"], path)
        readable_label = label_name.replace("_", " ").lower()
        records.append(
            make_record(
                image_id=image_id,
                slice_name="remote_sensing",
                source="blanchon/EuroSAT_RGB",
                source_url="https://huggingface.co/datasets/blanchon/EuroSAT_RGB",
                license_text="dataset terms on Hugging Face",
                local_path=path,
                notes=f"EuroSAT row {row_idx}, filename={row['filename']}, label={label_name}",
                queries=[
                    make_query(
                        image_id,
                        "direct",
                        f"Find the satellite image of {readable_label} land use.",
                        ["remote_sensing", "aerial", "domain_shift"],
                    ),
                    make_query(
                        image_id,
                        "fine",
                        "Find the small RGB aerial scene from a satellite viewpoint.",
                        ["remote_sensing", "small_objects", "aerial"],
                    ),
                ],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/universal_retrieval/images/public_smoke"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/universal_retrieval/public_smoke_manifest.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/universal_retrieval/.hf_cache"),
    )
    args = parser.parse_args()

    records: list[dict] = []
    add_funsd(records, args.image_dir, args.cache_dir)
    add_chartqa(records, args.image_dir, args.cache_dir)
    add_ai2d(records, args.image_dir, args.cache_dir)
    add_eurosat(records, args.image_dir, args.cache_dir)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote {len(records)} public smoke records to {args.manifest}")
    print(f"wrote images under {args.image_dir}")


if __name__ == "__main__":
    main()
