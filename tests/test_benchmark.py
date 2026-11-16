"""Native benchmark integration and replay validation, without downloading models."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", ROOT / "scripts" / "benchmark" / "run_colvision_local.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def sample_cache():
    image_entries = []
    query_entries = []
    for image_id, vectors in [
        ("positive", [[1.0, 0.0], [0.0, 1.0]]),
        ("negative", [[-1.0, 0.0], [0.0, -1.0]]),
    ]:
        query = {
            "query_id": f"{image_id}_query",
            "text": f"Find {image_id}",
            "positive_image_ids": [image_id],
        }
        record = {
            "image_id": image_id,
            "local_path": "not-needed-for-embedding-replay.png",
            "queries": [query],
        }
        image_entries.append({"image_id": image_id, "record": record, "vectors": vectors})
        query_entries.append({"query": query, "vectors": vectors})
    return {
        "schema_version": 1,
        "model_id": "test/known-vectors",
        "embedding_dim": 2,
        "image_embeddings": image_entries,
        "query_embeddings": query_entries,
    }


class BenchmarkTests(unittest.TestCase):
    def test_empty_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one image"):
            RUNNER.validate_records([])

    def test_manifest_without_positive_queries_rejected(self):
        records = [entry["record"] for entry in sample_cache()["image_embeddings"]]
        for record in records:
            record["queries"][0]["positive_image_ids"] = []
        with self.assertRaisesRegex(ValueError, "at least one query"):
            RUNNER.validate_records(records, require_images=False)

    def test_invalid_embedding_cache_rejected(self):
        for vectors in [[], [[1.0]], [[float("nan"), 0.0]], [[True, 0.0]]]:
            cache = sample_cache()
            cache["image_embeddings"][0]["vectors"] = vectors
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                RUNNER.validate_embedding_cache(cache)

    def test_query_without_positive_in_corpus_rejected(self):
        cache = sample_cache()
        cache["query_embeddings"][0]["query"]["positive_image_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "no positive image"):
            RUNNER.validate_embedding_cache(cache)

    def test_reference_preserves_negative_maxima(self):
        self.assertEqual(
            RUNNER.reference_maxsim([[1.0, 1.0]], [[-1.0, 0.0], [0.0, -2.0]]),
            -1.0,
        )

    def test_native_scores_and_metrics(self):
        cache = sample_cache()
        RUNNER.validate_embedding_cache(cache)
        with patch.dict("sys.modules", {"torch": None}):
            results, metrics = RUNNER.evaluate(cache, verify_native=True)
        self.assertEqual([result["top_5"][0]["image_id"] for result in results], ["positive", "negative"])
        self.assertEqual([result["top_5"][0]["score"] for result in results], [2.0, 2.0])
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)
        self.assertTrue(metrics["native_verified"])

    def test_replay_writes_results_without_model_dependencies(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temporary = Path(temporary)
            input_path = temporary / "input.json"
            input_path.write_text(json.dumps(sample_cache()))
            output_path = temporary / "replay"
            with patch.object(RUNNER, "embed_records", side_effect=AssertionError("model loaded during replay")):
                with patch.dict("sys.modules", {"torch": None, "PIL": None, "colpali_engine": None}):
                    RUNNER.main(["--embeddings-in", str(input_path), "--out", str(output_path), "--verify-native"])
            metrics = json.loads((output_path / "metrics.json").read_text())
            self.assertTrue(metrics["embeddings_reused"])
            self.assertEqual(metrics["image_embedding_seconds_total"], 0.0)
            self.assertEqual(len(json.loads((output_path / "results.json").read_text())), 2)
            self.assertEqual(RUNNER.load_embedding_cache(output_path / "embeddings.json")["model_id"], "test/known-vectors")

    def test_limit_keeps_only_queries_with_remaining_positive_images(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps(sample_cache()))
            cache = RUNNER.load_embedding_cache(path, limit=1)
        self.assertEqual(len(cache["image_embeddings"]), 1)
        self.assertEqual(len(cache["query_embeddings"]), 1)
        self.assertEqual(cache["query_embeddings"][0]["query"]["query_id"], "positive_query")

    def test_duplicate_cached_images_rejected(self):
        cache = sample_cache()
        cache["image_embeddings"].append(copy.deepcopy(cache["image_embeddings"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            RUNNER.validate_embedding_cache(cache)


if __name__ == "__main__":
    unittest.main()
