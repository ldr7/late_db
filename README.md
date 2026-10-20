# late_db

A Rust-first, in-memory late-interaction retrieval engine with Python bindings.
It stores document token vectors and ranks documents using exact MaxSim:

```text
score(query, document) = sum over query tokens of max dot(query_token, document_token)
```

Each document and query can contain multiple vectors. Scores are ordered from
highest to lowest, with ascending document IDs breaking ties. Negative
similarities remain negative. Storage, scoring, and orchestration stay in Rust;
Python provides model embedding and demo commands.

## Quickstart

Run commands from the repository root. Install a Rust toolchain and Python 3.9+
first; Python 3.12 is suitable for the optional model benchmark dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
source ./env.sh
python -m ensurepip --upgrade
python -m pip install 'maturin>=1.10,<2.0'
maturin develop --release
python -m late_db.demo
python -m late_db.demo --query-id chart --top-k 3 --interactive
```

If `.venv` already exists, activate it and skip creating it. The installed
`late-db-demo` command also runs the demo.

The default demo uses four labeled examples with hand-authored vectors. It
exercises the real Rust engine without downloading a model or dataset. Its
scores demonstrate engine behavior; they do not measure model retrieval quality.

## Python API

```python
from late_db import LateInteractionEngine

engine = LateInteractionEngine(embedding_dim=2)
engine.add_document(10, [[1.0, 0.0], [0.0, 1.0]], modality="image")
engine.add_document(20, [[0.5, 0.5]], modality="text")
print(engine.search([[1.0, 0.0], [0.0, 1.0]], top_k=2))
```

This returns `[(10, 2.0), (20, 1.0)]`.

Inputs must contain nonempty, finite vectors with the configured dimension.
Document IDs are unique. Resident-token and query-token limits are configurable
when constructing the engine; invalid inputs and exceeded capacities raise
`ValueError` without partially inserting a document.

## Real Embedding Workflow

Use the repository's existing images and manifest. Embedding generation requires
the optional model dependencies and locally available model weights. With the
existing cache, an offline run avoids downloading model files:

```sh
python -m pip install -r requirements-benchmark.txt
HF_HUB_OFFLINE=1 python scripts/benchmark/run_colvision_local.py \
  --manifest data/universal_retrieval/public_smoke_manifest.jsonl \
  --model vidore/colSmol-256M --device auto \
  --out data/universal_retrieval/runs/native_demo --verify-native
```

The runner writes `embeddings.json`, `results.json`, and `metrics.json`.
Native scores can be checked against an independent Python MaxSim
reference with `--verify-native`. Model embedding is separate from scoring, so a
saved index can be replayed without PyTorch, model weights, or image files:

```sh
python -m late_db.demo \
  --embeddings data/universal_retrieval/runs/native_demo/embeddings.json \
  --interactive
python scripts/benchmark/run_colvision_local.py \
  --embeddings-in data/universal_retrieval/runs/native_demo/embeddings.json \
  --out data/universal_retrieval/runs/native_replay --verify-native
```

The saved index records the model ID, embedding dimension, image metadata, and
cached query vectors. Interactive replay selects among those cached queries;
embedding arbitrary new text requires the original model. Source image paths
are displayed when present; the CLI does not render images.

`scripts/benchmark/colab_colvision_embed.py` exposes the same runner for a future
Colab or other GPU environment. Stage the chosen repository files and model
artifacts there before running it. See `docs/colab_integration.md` for the
platform workflow.

## Validation

After building the Python extension:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
python -m unittest discover -s tests -p 'test_*.py' -v
python -m late_db.demo --query-id chart --top-k 3
```

## Scope

Search is exhaustive and uses scalar CPU scoring. There is no approximate index,
GPU scoring kernel, durable Rust index, or server API. The JSON embedding cache
is a demo and benchmark interchange format. Synthetic demo success does not
establish real retrieval accuracy; existing model results and their limitations
are described in `docs/results_analysis.md`.

Local datasets, model caches, generated runs, and learning-only directories are
ignored by Git. The default demo needs none of those artifacts. Architecture
and module boundaries are documented in `docs/architecture.md`.
