# Universal Image Retrieval Benchmark

Goal: test whether late-interaction/multivector image embeddings behave like a
general retrieval primitive across document pages, diagrams, UI, natural images,
and domain-specific imagery.

## First-Pass Dataset

Target size: 64 images across 9 slices. The split is intentionally uneven
because some sources are easier to sample cleanly than others.

Each image should receive 2-3 natural-language queries:

- `direct`: caption-like or obvious semantic query.
- `fine_grained`: small text, structural, positional, or attribute-binding query.
- `hard_negative`: near-miss query used to expose false positives.

| Slice | Target | Preferred sources | Probes |
|---|---:|---|---|
| `forms_ocr_docs` | 8 | FUNSD | noisy scans, key-value fields, tiny text |
| `enterprise_docs` | 6 | ViDoRe V3 | page-image retrieval, layout, OCR-free text |
| `charts_tables` | 8 | ChartQA, PlotQA-like public examples, ViDoRe chart pages | legends, axes, table structure, numbers |
| `diagrams` | 8 | AI2D, patent figures | arrows, labels, symbolic structure |
| `floorplans_civil` | 6 | CubiCasa5K | rooms, doors, stairs, spatial relationships |
| `ui_screens` | 8 | RICO or hand-curated public screenshots | icons, short UI labels, repeated templates |
| `natural_images` | 8 | COCO 2017 val | object attributes, composition, ordinary images |
| `product_catalog` | 8 | Amazon Berkeley Objects | product attributes, materials, fine-grained visual similarity |
| `remote_sensing` | 4 | EuroSAT RGB | aerial viewpoint, small objects, domain shift |

## Model Ladder

Use three tiers so experiments can move forward on an M1 Air and scale later.

| Tier | Model | Why |
|---|---|---|
| Local smoke | `vidore/colSmol-256M` or `vidore/colSmol-500M` | Smallest ColVision late-interaction family; best chance of running locally on M1 Air. |
| Local stretch / Colab | `vidore/colpali-v1.3` | PaliGemma-based visual document retriever; useful reference point if memory allows. |
| Colab main | `vidore/colqwen2-v1.0` or `vidore/colqwen2.5-v0.2` | Stronger open visual retrievers using Qwen-VL backbones and ColBERT-style multivectors. |
| Colab/API comparison | `jinaai/jina-embeddings-v4` | Universal multimodal model supporting both dense and late-interaction modes. |

Single-vector baselines should use CLIP/SigLIP or Jina v4 dense mode. Keep them
in the same manifest so retrieval metrics are directly comparable.

## Manifest Fields

Each image record:

- `image_id`: stable local identifier.
- `slice`: benchmark slice.
- `source`: dataset or URL source.
- `license`: license or usage note.
- `local_path`: relative path under `data/universal_retrieval/images`.
- `source_url`: original URL or dataset locator.
- `queries`: query objects with `query_id`, `kind`, `text`, `positive_image_ids`,
  `hard_negative_image_ids`, and `tags`.

## Metrics

Record:

- `Recall@1`, `Recall@5`, `MRR`, `nDCG@5`.
- embedding wall time per image/query.
- vector count per image/query.
- embedding bytes per image/query.
- exact MaxSim scoring time.
- peak memory for embedding and scoring phases.

## Current Decision

Start by building the 64-item manifest and embedding only 8-16 images with
`colSmol-500M` locally. If local MPS/CPU is too slow, under roughly 1 image per
minute, or memory-bound, move the same manifest to Colab.

Colab guidance:

- T4: enough for a 64-image first pass with ColSmol/ColPali and batch size 1.
- L4: preferred for ColQwen2/ColQwen2.5 and repeated experiments.
- A100: reserve for larger batches, 4B+ model sweeps, or training.
