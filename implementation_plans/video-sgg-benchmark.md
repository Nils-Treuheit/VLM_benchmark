# Video Scene Graph Generation benchmark

**Status:** Proposed. Dataset code is present; video and annotation assets are not yet acquired. No benchmark code or run artifacts have been created.

## Goal

Measure whether a video VLM can generate time-indexed scene graphs from a video, including object localization, directed relations, and changes across time. This is separate from the current video QA and temporal retrieval scores in `video_Bench/`.

The common task is video scene graph detection (VidSGDet): predict objects and relations from video pixels at fixed annotated timestamps. Do not provide ground-truth labels, boxes, masks, captions, or relations to the model.

## Dataset selection and readiness

| Dataset | Role | Local state | Planned panel |
|---|---|---|---:|
| Action Genome, official test videos and frame annotations | Bounding-box human-object relations across time; core video SGG track | `/mnt/HDD1/Project_Data/public_datasets/action_genome/ActionGenome/` currently contains the code repository only. The 480p Charades videos and annotation files are still needed. | 50 video windows |
| PVSG, official held-out videos and panoptic relation annotations | Broader video graphs with masks/tubes and thing/stuff nodes | `/mnt/HDD1/Project_Data/public_datasets/pvsg/OpenPVSG/` currently contains the code repository only. The PVSG archives, `pvsg.json`, videos, and masks are still needed. | 50 video windows |

Do not start model runs until the official assets are acquired, access terms are recorded, and every selected video can be verified locally. Action Genome requires Charades 480p videos and the official annotation files. Its repository notes that the dumped frame collection is about 74 GB; do not dump every frame for this small panel. PVSG's official repository distributes separate video, mask, and annotation archives and includes videos sourced from VidOR, EPIC-KITCHENS, and Ego4D. Honor each source's access requirements. The [Action Genome repository](https://github.com/JingweiJ/ActionGenome) and [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf) describe its annotations; the [PVSG project](https://jingkangyang.com/PVSG/) and [official code/data instructions](https://github.com/LilyDaytoy/OpenPVSG) describe its mask-tube task and assets.

Select only from official held-out videos. Keep all windows from one source video in one split. If a release does not define a usable held-out split, create and freeze a split by video ID before sampling, document that it is local, and do not compare it with official benchmark scores. Greedily select unique windows that add the most uncovered relation labels; break ties by SHA-256 of `3407|dataset|video_id|window_start`. Fill remaining slots with seeded sampling (`3407`). Use at most one window per source video unless the official split is too small; record any exception.

## Video input and prediction contract

Reuse the repository's native-video adapter pattern and the timestamp-preserving frame preparation in `video_Bench/prepare_video_frames.py`. For each selected 8–12 second window, prepare up to 16 evenly spaced source-PTS frames. Include timestamps in the model input. Use the exact same cached frames and time stamps for every model. A model that cannot accept video input is unsupported for this suite; do not substitute a single frame or a montage.

Choose four annotated timestamps inside each window. Include those timestamps among the shared 16 sampled frames; use the remaining slots for uniform temporal context. This gives every model evidence at each scored instant while retaining motion context. Store the frame-sampling policy, source PTS, time base, timestamp, image dimensions, source-video hash, and cache hash.

Require JSON only, with a graph snapshot at each requested timestamp. Boxes use normalized `[x1, y1, x2, y2]` coordinates. Track IDs must be stable within the video window:

```json
{
  "frames": [
    {
      "timestamp": 3.2,
      "objects": [
        {"track_id": "v1", "label": "person", "box": [0.1, 0.1, 0.5, 0.9]}
      ],
      "relations": [
        {"subject": "v1", "predicate": "holding", "object": "v2"}
      ]
    }
  ]
}
```

Use dataset-specific closed vocabularies for Action Genome and PVSG, and the same explicit instructions for every model. Ask for at most 30 objects and 50 relations per timestamp, ordered from most to least confident. Keep raw output and parser state. No captioning stage, external tracker, detector, segmenter, or LLM-based evaluator may augment the VLM's predictions.

## Scoring

For Action Genome, evaluate frame-level SGDet at the four sampled annotated timestamps. Match predicted objects one-to-one to annotated objects by class and box IoU at least 0.5; a relation is correct only when both endpoints and the directed predicate match. Report `R@20`, `R@50`, `mR@20`, `mR@50`, and triplet precision/recall/F1@50. Break out attention, spatial, and contacting relations, with support counts. Average scores first within each video, then across videos so long or densely annotated videos do not dominate.

For PVSG, report the same **box-adapted** graph metrics using the bounding rectangles derived from its ground-truth masks. Name these scores `PVSG-box-adapted`; do not call them official PVSG scores. The official PVSG task uses panoptic masks and temporal mask tubes. Add mask IoU, panoptic quality, or mask-grounded relation recall only as a separate output track when the VLM itself can supply masks. Do not add an external segmenter to the shared model path.

Report frame-level graph scores separately from temporal scores. For PVSG, where ground-truth mask tubes provide persistent identities, match objects per timestamp as above, then calculate association precision/recall/F1 over object pairs linked across adjacent target timestamps: an association is correct when both predictions map to the same ground-truth track identities and preserve those identities in their predicted track IDs. Report relation-tube R@50 and mR@50 over annotated relation intervals; a tube matches when endpoint labels and predicate match, both endpoint boxes meet IoU 0.5 at overlapping target timestamps, and temporal IoU is at least 0.5. For Action Genome, do not invent identity ground truth when the annotation release does not provide persistent object IDs; report frame-level scores and predicate-group results only.

Keep Action Genome and PVSG scorecards separate. An optional two-dataset macro is the unweighted mean of dataset-level mR@50 and is published only when both panels complete. It is a local sampled-panel metric, not a published official leaderboard score.

## Runner, artifacts, and failure behavior

Create `video_SGG_Bench/` with asset verification, annotation preparation, clip-window sampling, frame caching, serial video inference, scoring, aggregation, and report generation. Keep raw videos, masks, annotations, and frame caches under ignored `video_SGG_Bench/data/` or `/mnt/HDD1/Project_Data/public_datasets/`; do not copy them into Git. Save local model snapshots under `HF_HOME`. Keep manifests as relative or absolute source paths with hashes, IDs, and timestamps; never embed image/video bytes.

Each result must preserve dataset/split/video ID, source and annotation hashes, window bounds, four target timestamps, frame-sampling metadata, model revision, adapter/backend, prompt/vocabulary hashes, raw response, parsed graph, per-frame and per-video scores, latency, generated tokens, VRAM peak, and item errors. The static report contains aggregate metrics only. Exclude videos, frames, masks, and item-level annotations from the report site.

Abort preparation if an annotation timestamp cannot be mapped to a decoded source frame, if a video or sidecar hash differs, if frames are duplicated or out of order, or if any sampled window crosses a split boundary. Preserve malformed output, truncation, unsupported video input, and runtime errors as failed items. Do not replace failed windows or omit them from the denominator. Refuse to aggregate runs with different clip manifests, frame caches, prompts, vocabularies, or evaluator hashes.

## Implementation sequence

1. **Access and data gate:** obtain Action Genome and PVSG assets from their documented sources, review and record terms, verify archive contents, and confirm the official held-out splits. If access fails, stop before model inference and report the affected track as blocked.
2. **Annotation adapters:** convert each source format to a shared local schema with video ID, split, timestamps, object labels/boxes, relation labels, and persistent IDs where available. Preserve dataset-specific raw labels and provenance.
3. **Window panel:** sample 50 held-out windows per dataset with at least four usable annotation timestamps. Select by relation coverage, fixed seed, and unique source video. Save a manifest and hashes before running a model.
4. **Frame cache and protocol:** reuse the native-video PTS pipeline with 16 frames maximum. Add a preflight that proves every requested timestamp maps to the same cached frame set across models.
5. **Scorer:** implement frame-level SGDet and PVSG track metrics with deterministic one-to-one object matching. Review category mappings and relation direction against real annotation examples before inference.
6. **Model runner:** adapt only cached, video-capable local models already supported by `video_Bench/`. Run one model at a time, offline, with identical windows and frame caches. Preserve partial runs and per-item errors.
7. **Aggregation and report:** publish separate Action Genome and PVSG results, panel support, temporal metrics, model coverage, and the distinction between box-adapted and official mask metrics.

## Acceptance criteria

- Both datasets have terms, official split information, source hashes, and all selected local assets before inference begins.
- Rebuilding manifests with the same source files and seed yields identical video IDs, window bounds, timestamps, and hashes.
- Every selected prediction timestamp maps to an ordered, verified source-PTS frame in the shared cache.
- Object and relation matches are one-to-one and respect class, direction, and IoU thresholds.
- The report identifies missing/unsupported tracks, failures, per-category support, and non-official adapted metrics.
- No application tests or GPU runs are added as planning artifacts. Before implementation work, draft and independently accept the required real-data checks under the repository workflow.

## Scope limits

This plan evaluates structured graph output from raw video VLM input. It does not train SGG models or compare external detection/tracking pipelines. The current local state is insufficient to run either video track: Action Genome and PVSG code repositories are present, but their data assets are not.
