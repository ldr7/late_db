#!/usr/bin/env python3
"""Colab/GPU entry point: embed images and queries, score with Rust, save a replayable index.

Install the native package with `pip install .` and the model dependencies with
`pip install -r requirements-benchmark.txt`, then run with the same arguments as
run_colvision_local.py, for example:

    python scripts/benchmark/colab_colvision_embed.py \
        --manifest data/universal_retrieval/public_smoke_manifest.jsonl \
        --model vidore/colSmol-256M --device auto \
        --out data/universal_retrieval/runs/colab_colsmol256m --verify-native

This entry point runs both embedding and scoring. Its embeddings.json can be
replayed on a CPU machine with --embeddings-in, without model dependencies.
"""

from run_colvision_local import main


if __name__ == "__main__":
    main()
