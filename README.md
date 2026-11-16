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

## Install and Run the Draft

The default demo runs locally on CPU. It uses four labeled examples with
hand-authored vectors and the real Rust scoring engine. After installation, it
needs no model, dataset, account, or network access. Its scores demonstrate
engine behavior; they do not measure model retrieval quality.

### Prerequisites

- Git and Python 3.12 with `venv` and `pip` support.
- A current stable Rust toolchain, including Cargo. The crate requires Rust 1.85
  or newer; installing from source compiles a native Python extension.
- Native build tools: Xcode Command Line Tools on macOS, a C compiler/linker on
  Linux, or Visual Studio Build Tools with the C++ workload and the Rust MSVC
  toolchain on Windows.

The commands below install from source. The first installation may download
Python build dependencies and Rust crates. Python 3.12 is the recommended demo
environment, even though package metadata permits Python 3.9 and newer.

### Get the Source

```sh
git clone --branch feat/initial-working-demo https://github.com/ldr7/late_db.git
cd late_db
```

Until the draft PR is merged into `main`, clone its source branch,
`feat/initial-working-demo`, as shown above. After that merge,
`git clone https://github.com/ldr7/late_db.git` is sufficient.

### macOS or Linux

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
late-db-demo
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
late-db-demo
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly for
the `-m pip` commands, then run
`.\.venv\Scripts\python.exe -m late_db.demo`.

### Try a Query

Either installed entry point works while the virtual environment is active:

```sh
python -m late_db.demo --query-id chart --top-k 3
late-db-demo --interactive
```

Expect the `SYNTHETIC DEMO` label, four documents, eight tokens, and the chart
document ranked first for the chart query with score `1.800000`. Interactive
mode offers four cached query choices; enter `q` to quit.

`pip` builds the package through its declared build backend. Demo users do not
need `env.sh`, a repository-local toolchain cache, or a separate Maturin command.

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

## Replay Real Embeddings

A fresh checkout does not contain datasets, model weights, real embedding
caches, or generated result directories: those local artifacts are ignored by
Git. Supply a saved `embeddings.json` produced by the benchmark runner to try
real-model retrieval. Replace the example path with the file you received:

```sh
python -m late_db.demo --embeddings path/to/embeddings.json --interactive
```

This uses the installed Rust engine and needs no PyTorch, model weights, or
image files. The index contains its model ID, embedding dimension, image
metadata, and cached query vectors. Replay selects among those cached queries;
embedding arbitrary new text requires the original model. Source image paths
are displayed when present; the CLI does not render images.

To validate a supplied cache against the independent Python MaxSim reference,
run this from the source checkout:

```sh
python scripts/benchmark/run_colvision_local.py --embeddings-in path/to/embeddings.json --out data/universal_retrieval/runs/native_replay --verify-native
```

The runner writes `embeddings.json`, `results.json`, and `metrics.json` under the
chosen output directory. No optional model dependencies are needed for replay.

## Generate Your Own Embeddings

This optional step requires a prepared JSONL manifest, its referenced local
image files, and model weights. None are bundled with the draft. The manifest
format is documented in
[the benchmark guide](docs/universal_retrieval_benchmark.md) and its
[schema](benchmarks/universal_retrieval/manifest.schema.json).

Install the optional model dependencies, then supply your own paths. The model
directory in this example must contain a compatible ColSmol model and processor:

```sh
python -m pip install -r requirements-benchmark.txt
python scripts/benchmark/run_colvision_local.py --manifest path/to/manifest.jsonl --model path/to/colSmol-256M --device auto --out data/universal_retrieval/runs/native_demo --verify-native
```

`scripts/benchmark/colab_colvision_embed.py` exposes the same runner for a Colab
or other GPU environment. Stage the required source files, images, manifest,
and model artifacts there first. See
[the Colab guide](docs/colab_integration.md) for platform integration.

## Contributor Validation

After the source installation above, run the relevant checks from the
repository root. Reinstall with `python -m pip install .` after changing Rust
or packaged Python code so the installed demo reflects your changes.

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
