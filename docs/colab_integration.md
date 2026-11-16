# Colab Integration

Use a GPU notebook for model embedding and replay the saved index locally.
The same Python commands work in Colab, Kaggle notebooks, or a Linux GPU VM.
Colab-specific notebook syntax below uses `!` and `%cd`.

## Recommended Workflow

1. Select a GPU runtime. Start with `vidore/colSmol-256M` for a small smoke run.
2. Upload this project folder, or clone it after you publish your repository:

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
```

3. Ensure Rust is installed, then install the native package and benchmark
   dependencies. In a new notebook runtime:

```python
!command -v cargo || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal)

import os
from pathlib import Path
os.environ["PATH"] = str(Path.home() / ".cargo" / "bin") + os.pathsep + os.environ["PATH"]
os.environ["HF_HOME"] = str(Path.cwd() / "data" / "universal_retrieval" / ".hf_home")

!python -m pip install .
!python -m pip install -r requirements-benchmark.txt
!python -m late_db.demo
```

4. Supply the public smoke images and manifest. They are intentionally ignored
   by Git, so a fresh clone does not contain them. Upload your existing
   `data/universal_retrieval` folder or regenerate the 12-image sample:

```python
!python scripts/benchmark/fetch_public_smoke_dataset.py
```

5. Run the readiness check against the populated public manifest:

```python
!python scripts/benchmark/check_universal_retrieval_ready.py \
  --manifest data/universal_retrieval/public_smoke_manifest.jsonl
```

6. Embed, search with Rust, verify the native scores, and save artifacts:

```python
!python scripts/benchmark/colab_colvision_embed.py \
  --manifest data/universal_retrieval/public_smoke_manifest.jsonl \
  --model vidore/colSmol-256M \
  --device auto \
  --out data/universal_retrieval/runs/colab_colsmol256m \
  --verify-native
```

`--device auto` selects CUDA, then MPS, then CPU. CUDA uses BF16 where supported
and FP16 on GPUs such as the T4; CPU and MPS use FP32. Images are encoded one at
a time to avoid padded token artifacts. Scoring always uses the Rust CPU engine.

For later model comparisons, switch to:

```python
--model vidore/colqwen2-v1.0
```

or:

```python
--model vidore/colqwen2.5-v0.2
```

Use distinct output directories for each model. The default 64-image
`manifest.jsonl` remains a planning scaffold with missing images and placeholder
queries; use `public_smoke_manifest.jsonl` until a curated replacement exists.

## How To Share Results Back

Each run writes these files under its output directory:

- `embeddings.json`: reusable image/query multivectors and manifest metadata.
- `metrics.json`: retrieval quality, timing, and native verification error.
- `results.json`: per-query top-five results.

Download the run folder to the same path locally. Then, with the native package
installed, replay without Torch or model downloads:

```sh
python -m late_db.demo \
  --embeddings data/universal_retrieval/runs/colab_colsmol256m/embeddings.json \
  --interactive

python scripts/benchmark/run_colvision_local.py \
  --embeddings-in data/universal_retrieval/runs/colab_colsmol256m/embeddings.json \
  --out data/universal_retrieval/runs/colab_colsmol256m_replay \
  --verify-native
```

Keep original images at their manifest paths if you want to open the paths
printed by the CLI. Cached query replay itself needs only the JSON index.
Generated run folders remain ignored. Copy any intentionally published result
summary into `docs/` after reviewing it.

The current public sample includes exact duplicate chart/diagram images under
different IDs and some broad, ambiguous queries. Treat its scores as smoke-test
evidence. The next quality experiment is a deduplicated 30–50-page document
corpus with shared positives, followed by model and single-vector comparisons.
