# Image Scene Graph Generation benchmark

**Status checked:** 2026-11-13. The three fixed 200-image manifests are ready. Three of eight candidates have all three box-grounded panels. Qwen2-VL-72B IQ2_XXS completed VG-150 and Open Images and stopped at 123/200 on PSG. Four other GGUF variants lack complete box-grounded panels. The [native VG50 PCIS track](../qwenVL_Bench/image_SGG_Bench/native-pcis-vg50.md) has scores for all eight configurations and uses a separate tuple-based protocol.

## Goal

Measure whether an image VLM can jointly identify scene entities, ground them in an image, and produce directed `<subject, predicate, object>` relations. This is a separate capability benchmark from GRAB-Lite, which tests graph analysis on plotted data.

The task is **scene graph detection (SGDet)** from pixels. Do not provide ground-truth object labels, boxes, masks, captions, or candidate relations to the model. Do not combine these scores with the existing VQA or plot-reasoning macro.

## Dataset selection and local readiness

| Dataset and split | Role | Local data | Planned panel |
|---|---|---|---:|
| Visual Genome, canonical VG-150 test | Standard closed-vocabulary image SGG | Images and source annotations are under `/mnt/HDD1/Project_Data/public_datasets/visual_genome/`. Canonical VG-150 preprocessing and the fixed manifest are ready at `qwenVL_Bench/image_SGG_Bench/data/vg150/manifest.json`; the recorded panel excludes test rows with out-of-bounds boxes. | 200 images |
| Open Images V7 validation, visual relationships | Diverse relationship detection with a larger object and relation vocabulary | Images, detection labels, relationship labels, metadata, and the fixed manifest are under `/mnt/HDD1/Project_Data/public_datasets/open_images_v7/validation/` and `qwenVL_Bench/image_SGG_Bench/data/openimages_v7/manifest.json`. | 200 images |
| PSG validation in the local Hugging Face mirror | Panoptic scene content and relations; a separate box-grounded adaptation for VLMs | Parquet shards are under `/mnt/HDD1/Project_Data/public_datasets/psg/huggingface_dataset/`. The fixed manifest is `qwenVL_Bench/image_SGG_Bench/data/psg/manifest.json`. The mirror labels the official PSG test split `validation`; record both names. | 200 images |

The panels are fixed, deterministic, and relation-stratified. Greedily select unique images that add the most uncovered predicates; break ties by SHA-256 of `3407|dataset|image_id`. Fill remaining slots with a seeded random selection (`3407`). Save selected IDs, class support, source split, source revision or file hashes, and selection code version. These are **small local panels**, not full-set leaderboard scores.

Use the official VG-150 category mapping and split rather than creating a new vocabulary. Open Images V7 has image boxes and relationship annotations in separate CSVs; join by image ID, map class IDs through the supplied class descriptions, and exclude `RelationLabel="is"` attributes from the pairwise-relation task. Preserve those rows in the source data. PSG includes both thing and stuff nodes and masks. Its VLM panel uses the available `validation` records only, never the training split.

Before scoring, record dataset terms and source provenance. The PSG Hugging Face copy is a third-party mirror, not the upstream release; retain its README and repository revision. The official OpenPSG repository documents the original data and split files. The [Visual Genome project and API](https://visualgenome.org/home) provide the raw scene-graph annotations; [OpenPSG](https://github.com/Jingkang50/OpenPSG) documents VG-150 and PSG protocols; Google's [Open Images V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html) describes its relationship CSV format; the local [PSG mirror card](https://huggingface.co/datasets/wliafe/PSG) documents its Parquet fields and split mapping.

## Model input and output contract

Run every image-capable model serially through its existing local image adapter. Keep the current image decoding and resize policy for that backend; record actual dimensions and any image-token or crop limits. Do not add a detector, segmenter, retrieval step, or external service to the VLM path.

Use one prompt template with a dataset-specific vocabulary section. VG-150 and PSG are closed-vocabulary tracks; the prompt gives their object and predicate names. Open Images is vocabulary-guided using its class descriptions, but predictions must still be grounded in image evidence. Use the same instructions, vocabulary ordering, decoding settings, and output limits for every model on a given panel.

Require JSON only, with normalized `[x1, y1, x2, y2]` boxes:

```json
{
  "objects": [
    {"id": "o1", "label": "person", "box": [0.1, 0.1, 0.5, 0.9]}
  ],
  "relations": [
    {"subject": "o1", "predicate": "holding", "object": "o2"}
  ]
}
```

Ask the model to list its most confident relations first. Cap output at 50 objects and 100 relations. Preserve raw text and parse state. Do not repair malformed JSON with an LLM or silently map unknown labels to a known class. Use only a versioned, deterministic alias table for spelling and approved synonyms.

## Scoring

The primary task is SGDet. Match predicted objects one-to-one to ground truth using the same class and box IoU of at least 0.5. Use deterministic maximum-IoU assignment with stable tie-breaking. A predicted triplet is correct only when both endpoints match and the directed predicate matches. Do not reverse subject and object unless the dataset explicitly defines that relation as symmetric.

For each dataset, report:

- relation Recall@20, @50, and @100;
- mean Recall@20, @50, and @100 over predicates represented in the fixed panel, with per-predicate support;
- triplet precision, recall, and F1 at 50 predictions;
- object recall@50, JSON validity, completion count, and parse failures.

Order predictions by their order in the model response for top-K scoring. This avoids inventing confidence calibration when a model does not provide comparable scores. If an implementation later adds confidence-based ranking, report it as a separate protocol. Report each dataset separately. An optional suite macro is the unweighted mean of dataset-level mR@50, published only when all three panels complete. Do not pool all classes or use micro averaging across different label spaces.

PSG's official task grounds nodes with pixel masks and uses mask-level matching. The shared VLM protocol emits boxes, so its default score must be named **PSG box-adapted SGG** and must not be presented as the official PSG metric. Add an official mask-based PSG score only for a separately declared model output that supplies masks and can be evaluated with the official mask-IoU/PQ protocol. Do not insert an external segmenter into the common VLM score.

## Runner, artifacts, and failure behavior

Create `image_SGG_Bench/` with a data-preparation command, a serial runner, a deterministic evaluator, an aggregator, and a report generator. Keep large source images in `/mnt/HDD1/Project_Data/public_datasets/`. Store generated manifests and raw model output under ignored `image_SGG_Bench/data/` and `image_SGG_Bench/results/`; manifests contain paths and IDs, not copied images. Keep model snapshots offline under `HF_HOME`.

Each item record must include dataset, split, source image ID and hash, manifest hash, model revision, backend, prompt and vocabulary hashes, input dimensions, raw response, parsed graph, per-item score, latency, output tokens when available, peak VRAM, and error details. The report site contains aggregate metrics only. Never publish images, annotations, predictions, or restricted source data in the static site.

Abort preparation before inference if a selected image, annotation, label mapping, or split is missing or ambiguous. Treat a malformed response, truncation, unsupported image input, and runtime failure as explicit item failures. Preserve failures in the denominator; never replace a failed example or silently score only successful outputs. Refuse to aggregate runs whose manifest, prompt, evaluator, vocabulary, or source hashes differ.

## Implementation sequence

1. **Data audit:** verify local files, upstream revisions, terms, image-to-annotation joins, VG-150 preprocessing, and PSG mirror split mapping. Produce a read-only inventory before preparing panels.
2. **Panel preparation:** create deterministic 200-image manifests for each dataset. Validate unique IDs, present images, finite in-bounds boxes, valid category IDs, non-empty relation support, and train/test separation. Save source hashes and per-class support.
3. **Protocol and scorer:** define a versioned prompt, vocabulary maps, strict JSON parser, one-to-one object matcher, and triplet metrics. Independently review the metric implementation against hand-inspected examples from each real dataset before model runs.
4. **Runner integration:** add image adapters for cached, image-capable models already supported by the repository. Run one model at a time, offline, batch size one, and preserve incomplete records.
5. **Aggregation and report:** publish per-dataset metrics, support counts, completion rates, model/backend details, and limitations. Keep the optional three-dataset macro clearly secondary.

## Acceptance criteria

- Rebuilding panels with the same source files and seed yields identical sorted IDs and manifest hashes.
- Every selected sample resolves to one image and its complete, valid annotation record.
- Scoring never reuses a ground-truth node and never counts a relation with a mismatched endpoint or predicate.
- A real model response can be parsed, scored, and retained even when another item fails.
- A completed report identifies the fixed subset, source revisions, protocols, per-class support, and all unsupported or failed items.
- No application tests or GPU runs are added as planning artifacts. Before implementation work, draft and independently accept the required real-data checks under the repository workflow.

## Scope limits

This is a benchmark of direct VLM image-to-graph generation, not a comparison of learned SGG detectors. VG-150 and Open Images scores are subset results. PSG box-adapted scores are not official mask-grounded PSG results. Keep these qualifications visible in tables and report text.
