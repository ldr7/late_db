"""Integration tests for Python bindings and the dependency-light demo."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from late_db import LateInteractionEngine


class EngineTests(unittest.TestCase):
    def test_maxsim_uses_per_query_token_maxima(self):
        engine = LateInteractionEngine(2)
        engine.add_document(20, [[1.0, 0.0], [0.0, 1.0]])
        engine.add_document(10, [[0.6, 0.6]], modality="text")
        results = engine.search([[1.0, 0.0], [0.0, 1.0]], top_k=5)
        self.assertEqual([document_id for document_id, _ in results], [20, 10])
        self.assertAlmostEqual(results[0][1], 2.0)
        self.assertAlmostEqual(results[1][1], 1.2, places=6)
        self.assertEqual(engine.document_count, 2)
        self.assertEqual(engine.token_count, 3)
        self.assertEqual(engine.embedding_dim, 2)

    def test_negative_scores_and_ties(self):
        engine = LateInteractionEngine(2)
        engine.add_document(30, [[-2.0, -2.0], [-1.0, -3.0]])
        engine.add_document(20, [[-1.0, -2.0]])
        engine.add_document(10, [[-1.0, -2.0]])
        self.assertEqual(
            engine.search([[1.0, 0.0], [0.0, 1.0]], top_k=3),
            [(10, -3.0), (20, -3.0), (30, -3.0)],
        )
        self.assertEqual(engine.search([[1.0, 0.0]], top_k=1), [(10, -1.0)])

    def test_invalid_shapes_and_values_leave_index_intact(self):
        engine = LateInteractionEngine(2)
        engine.add_document(1, [[1.0, 0.0]])
        for vectors in ([], [[]], [[1.0]], [[1.0, 0.0], [1.0]], [[float("nan"), 0.0]], [[float("inf"), 0.0]]):
            with self.subTest(vectors=vectors):
                with self.assertRaises(ValueError):
                    engine.add_document(2, vectors)
                with self.assertRaises(ValueError):
                    engine.search(vectors)
        self.assertEqual(engine.document_count, 1)
        self.assertEqual(engine.token_count, 1)
        self.assertEqual(engine.search([[1.0, 0.0]]), [(1, 1.0)])

    def test_capacity_and_duplicate_errors_preserve_documents(self):
        engine = LateInteractionEngine(2, max_resident_tokens=1, max_query_tokens=1)
        engine.add_document(7, [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            engine.add_document(7, [[0.0, 1.0]])
        with self.assertRaises(ValueError):
            engine.add_document(8, [[0.0, 1.0]])
        with self.assertRaises(ValueError):
            engine.search([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(engine.search([[1.0, 0.0]]), [(7, 1.0)])


class DemoTests(unittest.TestCase):
    def run_demo(self, *arguments, input_text=None):
        return subprocess.run(
            [sys.executable, "-m", "late_db.demo", *arguments],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_synthetic_demo_is_deterministic_and_ranks_query(self):
        first = self.run_demo("--query-id", "chart", "--top-k", "1")
        second = self.run_demo("--query-id", "chart", "--top-k", "1")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertIn("SYNTHETIC DEMO", first.stdout)
        self.assertIn("4 documents, 8 tokens, dimension 4", first.stdout)
        self.assertIn("1.800000  chart | Quarterly revenue bar chart", first.stdout)
        self.assertNotIn("invoice |", first.stdout)

    def test_invalid_arguments_are_actionable(self):
        for arguments, expected in [
            (("--top-k", "0"), "--top-k must be positive"),
            (("--query-id", "missing"), "unknown query ID"),
            (("--embeddings", "missing-index.json"), "missing-index.json"),
        ]:
            with self.subTest(arguments=arguments):
                result = self.run_demo(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_saved_embedding_replay_and_validation(self):
        payload = {
            "schema_version": 1,
            "model_id": "fixture-model",
            "embedding_dim": 2,
            "image_embeddings": [
                {"image_id": "actual-page", "record": {"slice": "forms", "local_path": "images/page.png"}, "vectors": [[1.0, 0.0]]},
                {"image_id": "other-page", "record": {}, "vectors": [[0.0, 1.0]]},
            ],
            "query_embeddings": [
                {"query": {"query_id": "invoice-query", "text": "Find an invoice"}, "vectors": [[1.0, 0.0]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "embeddings.json"
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self.run_demo("--embeddings", str(index_path), "--query-id", "invoice-query", "--top-k", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("fixture-model", result.stdout)
            self.assertIn("1.000000  actual-page | forms", result.stdout)
            self.assertIn("images/page.png", result.stdout)
            self.assertNotIn("SYNTHETIC", result.stdout)
            payload["query_embeddings"][0]["vectors"] = [[1.0]]
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            invalid = self.run_demo("--embeddings", str(index_path))
            self.assertEqual(invalid.returncode, 2)
            self.assertNotIn("Traceback", invalid.stderr)
            payload["schema_version"] = 2
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            invalid = self.run_demo("--embeddings", str(index_path))
            self.assertIn("schema_version 1", invalid.stderr)

    def test_interactive_cached_queries(self):
        result = self.run_demo("--interactive", "--top-k", "1", input_text="unknown\nchart\nq\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Unknown cached query", result.stdout)
        self.assertIn("Query [chart]", result.stdout)


if __name__ == "__main__":
    unittest.main()
