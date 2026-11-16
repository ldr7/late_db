#!/usr/bin/env python3
"""Embed visual documents, search with the Rust engine, and save replayable results."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if limit is None else records[:limit]


def validate_records(records: list[dict], require_images: bool = True) -> None:
    if not records:
        raise ValueError("manifest must contain at least one image")
    image_ids = set()
    query_ids = set()
    query_count = 0
    for record in records:
        image_id = record["image_id"]
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError(f"invalid or duplicate image_id: {image_id!r}")
        image_ids.add(image_id)
        if require_images and not Path(record["local_path"]).is_file():
            raise FileNotFoundError(f"missing image file: {record['local_path']}")
        for query in record["queries"]:
            query_id = query["query_id"]
            if not isinstance(query_id, str) or not query_id or query_id in query_ids:
                raise ValueError(f"invalid or duplicate query_id: {query_id!r}")
            query_ids.add(query_id)
            if not isinstance(query["text"], str) or not query["text"].strip() or "TODO:" in query["text"]:
                raise ValueError(f"empty or placeholder query: {query_id}")
            if not isinstance(query["positive_image_ids"], list):
                raise ValueError(f"positive_image_ids must be a list: {query_id}")
            query_count += bool(query["positive_image_ids"])
    if not query_count:
        raise ValueError("manifest must contain at least one query with a positive image")
    if not flatten_queries(records):
        raise ValueError("no queries have positive images in the selected corpus")


def flatten_queries(records: list[dict]) -> list[dict]:
    image_ids = {record["image_id"] for record in records}
    return [
        query
        for record in records
        for query in record["queries"]
        if image_ids.intersection(query["positive_image_ids"])
    ]


def resolve_model_classes(model_id: str):
    from colpali_engine import models

    lowered = model_id.lower()
    if "colsmol" in lowered:
        return models.ColIdefics3, models.ColIdefics3Processor
    if "colqwen2.5" in lowered or "colqwen2_5" in lowered:
        return models.ColQwen2_5, models.ColQwen2_5_Processor
    if "colqwen2" in lowered:
        return models.ColQwen2, models.ColQwen2Processor
    if "colpali" in lowered:
        return models.ColPali, models.ColPaliProcessor
    raise ValueError(f"unsupported model id: {model_id}")


def resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_records(records: list[dict], model_id: str, requested_device: str) -> tuple[dict, dict]:
    queries = flatten_queries(records)
    if not queries:
        raise ValueError("no queries have positive images in the selected corpus")

    import torch
    from PIL import Image

    device = resolve_device(torch, requested_device)
    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_cls, processor_cls = resolve_model_classes(model_id)
    started = time.perf_counter()
    model = model_cls.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    processor = processor_cls.from_pretrained(model_id)
    timings = {"load_seconds": time.perf_counter() - started}

    image_entries = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        print(f"embedding image {index}/{len(records)}: {record['image_id']}", file=sys.stderr, flush=True)
        with Image.open(record["local_path"]) as source:
            batch = processor.process_images([source.convert("RGB")]).to(device)
        with torch.inference_mode():
            vectors = model(**batch)[0].detach().to(device="cpu", dtype=torch.float32).tolist()
        image_entries.append({"image_id": record["image_id"], "record": record, "vectors": vectors})
    timings["image_embedding_seconds_total"] = time.perf_counter() - started

    query_entries = []
    started = time.perf_counter()
    for index, query in enumerate(queries, start=1):
        print(f"embedding query {index}/{len(queries)}: {query['query_id']}", file=sys.stderr, flush=True)
        batch = processor.process_queries([query["text"]]).to(device)
        with torch.inference_mode():
            vectors = model(**batch)[0].detach().to(device="cpu", dtype=torch.float32).tolist()
        query_entries.append({"query": query, "vectors": vectors})
    timings["query_embedding_seconds_total"] = time.perf_counter() - started
    return {
        "schema_version": 1,
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "embedding_dim": len(image_entries[0]["vectors"][0]),
        "image_embeddings": image_entries,
        "query_embeddings": query_entries,
    }, timings


def validate_vectors(vectors: list, embedding_dim: int, label: str) -> None:
    if not isinstance(vectors, list) or not vectors:
        raise ValueError(f"empty or invalid vectors: {label}")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != embedding_dim:
            raise ValueError(f"embedding dimension mismatch: {label}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            raise ValueError(f"non-finite or invalid embedding value: {label}")


def validate_embedding_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("embedding cache must be a JSON object")
    if cache.get("schema_version", cache.get("version")) != 1:
        raise ValueError("unsupported embedding cache version")
    if not isinstance(cache.get("model_id"), str) or not cache["model_id"]:
        raise ValueError("embedding cache requires a model_id")
    embedding_dim = cache.get("embedding_dim")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError("embedding_dim must be a positive integer")
    images = cache.get("image_embeddings")
    queries = cache.get("query_embeddings")
    if not isinstance(images, list) or not images or not isinstance(queries, list) or not queries:
        raise ValueError("embedding cache requires non-empty image_embeddings and query_embeddings")
    image_ids = set()
    for entry in images:
        image_id = entry["image_id"]
        if image_id in image_ids or entry["record"]["image_id"] != image_id:
            raise ValueError(f"duplicate or mismatched cached image_id: {image_id}")
        image_ids.add(image_id)
        validate_vectors(entry["vectors"], embedding_dim, image_id)
    query_ids = set()
    for entry in queries:
        query = entry["query"]
        query_id = query["query_id"]
        if query_id in query_ids:
            raise ValueError(f"duplicate cached query_id: {query_id}")
        query_ids.add(query_id)
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError(f"empty or invalid cached query text: {query_id}")
        if not image_ids.intersection(query["positive_image_ids"]):
            raise ValueError(f"query has no positive image in cache: {query_id}")
        validate_vectors(entry["vectors"], embedding_dim, query_id)


def load_embedding_cache(path: Path, limit: int | None = None) -> dict:
    cache = json.loads(path.read_text())
    validate_embedding_cache(cache)
    if limit is not None:
        cache["image_embeddings"] = cache["image_embeddings"][:limit]
        image_ids = {entry["image_id"] for entry in cache["image_embeddings"]}
        cache["query_embeddings"] = [
            entry for entry in cache["query_embeddings"]
            if image_ids.intersection(entry["query"]["positive_image_ids"])
        ]
        validate_embedding_cache(cache)
    return cache


def build_engine(cache: dict):
    from late_db import LateInteractionEngine

    images = cache["image_embeddings"]
    engine = LateInteractionEngine(
        cache["embedding_dim"],
        max_resident_tokens=sum(len(entry["vectors"]) for entry in images),
        max_query_tokens=max(len(entry["vectors"]) for entry in cache["query_embeddings"]),
    )
    for document_id, entry in enumerate(images):
        engine.add_document(document_id, entry["vectors"], modality="image")
    return engine


def reference_maxsim(query_vectors: list, image_vectors: list) -> float:
    return sum(
        max(sum(query_value * image_value for query_value, image_value in zip(query_vector, image_vector)) for image_vector in image_vectors)
        for query_vector in query_vectors
    )


def verify_scores(query_vectors: list, image_entries: list[dict], ranked: list[tuple]) -> float:
    try:
        import torch
    except ImportError:
        torch = None
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32) if torch is not None else None
    max_error = 0.0
    for document_id, score in ranked:
        image_vectors = image_entries[document_id]["vectors"]
        if torch is None:
            reference = reference_maxsim(query_vectors, image_vectors)
        else:
            image_tensor = torch.tensor(image_vectors, dtype=torch.float32)
            reference = float((query_tensor @ image_tensor.T).max(dim=1).values.sum())
        max_error = max(max_error, abs(score - reference))
        if not math.isclose(score, reference, rel_tol=2e-4, abs_tol=1e-4):
            raise AssertionError(f"native/reference score mismatch for document {document_id}: {score} != {reference}")
    return max_error


def evaluate(cache: dict, verify_native: bool = False) -> tuple[list[dict], dict]:
    engine = build_engine(cache)
    images = cache["image_embeddings"]
    image_ids = [entry["image_id"] for entry in images]
    image_id_set = set(image_ids)
    results = []
    recall_at_1 = 0
    recall_at_5 = 0
    mrr_total = 0.0
    ndcg_total = 0.0
    score_seconds = 0.0
    max_error = 0.0
    for entry in cache["query_embeddings"]:
        query = entry["query"]
        started = time.perf_counter()
        ranked = engine.search(entry["vectors"], top_k=len(images))
        score_seconds += time.perf_counter() - started
        if len(ranked) != len(images) or any(not math.isfinite(score) for _, score in ranked):
            raise ValueError("native search returned missing or non-finite scores")
        if verify_native:
            max_error = max(max_error, verify_scores(entry["vectors"], images, ranked))
        ranked_ids = [image_ids[document_id] for document_id, _ in ranked]
        positives = image_id_set.intersection(query["positive_image_ids"])
        recall_at_1 += int(bool(positives.intersection(ranked_ids[:1])))
        recall_at_5 += int(bool(positives.intersection(ranked_ids[:5])))
        rank = next((index for index, image_id in enumerate(ranked_ids, start=1) if image_id in positives), None)
        mrr_total += 1.0 / rank if rank else 0.0
        dcg = sum(1.0 / math.log2(index + 2) for index, image_id in enumerate(ranked_ids[:5]) if image_id in positives)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(5, len(positives))))
        ndcg_total += dcg / ideal
        results.append({
            "query_id": query["query_id"],
            "query_text": query["text"],
            "positive_image_ids": sorted(positives),
            "top_5": [{"image_id": image_ids[document_id], "score": score} for document_id, score in ranked[:5]],
        })
    query_count = len(results)
    metrics = {
        "model_id": cache["model_id"],
        "device": cache.get("device", "cached"),
        "scoring_backend": "late_db_rust_exact_maxsim",
        "records": len(images),
        "queries": query_count,
        "embedding_dim": cache["embedding_dim"],
        "score_seconds": score_seconds,
        "recall_at_1": recall_at_1 / query_count,
        "recall_at_5": recall_at_5 / query_count,
        "mrr": mrr_total / query_count,
        "ndcg_at_5": ndcg_total / query_count,
        "image_vector_counts": {entry["image_id"]: len(entry["vectors"]) for entry in images},
        "query_vector_counts": {entry["query"]["query_id"]: len(entry["vectors"]) for entry in cache["query_embeddings"]},
        "native_verified": verify_native,
    }
    if verify_native:
        metrics["native_reference_max_abs_error"] = max_error
    return results, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/universal_retrieval/public_smoke_manifest.jsonl"))
    parser.add_argument("--model", default="vidore/colSmol-256M")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Embedding batch size; currently only 1 is supported.")
    parser.add_argument("--out", type=Path, default=Path("data/universal_retrieval/runs/local_colvision_smoke"))
    parser.add_argument("--embeddings-in", type=Path, help="Replay a JSON embedding cache without loading a model.")
    parser.add_argument("--verify-native", action="store_true", help="Compare every native score against independent MaxSim.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size != 1:
        parser.error("--batch-size must be 1; images are encoded individually to avoid padding artifacts")
    if args.embeddings_in is not None:
        cache = load_embedding_cache(args.embeddings_in, args.limit)
        timings = {"load_seconds": 0.0, "image_embedding_seconds_total": 0.0, "query_embedding_seconds_total": 0.0}
    else:
        records = load_manifest(args.manifest, args.limit)
        validate_records(records)
        cache, timings = embed_records(records, args.model, args.device)
        validate_embedding_cache(cache)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "embeddings.json"
    if args.embeddings_in is None or args.embeddings_in.resolve() != cache_path.resolve() or args.limit is not None:
        cache_path.write_text(json.dumps(cache, separators=(",", ":"), allow_nan=False))
    print("scoring with the late_db Rust engine", file=sys.stderr, flush=True)
    results, metrics = evaluate(cache, args.verify_native)
    metrics.update(timings)
    metrics["embeddings_reused"] = args.embeddings_in is not None
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
