# Video-oriented VLM benchmark implementation plan

## Scope and cached checkpoints

- `yanziang/InternVideo3-8B-Instruct`
- `MCG-NJU/TimeLens2-8B`
- `llava-hf/LLaVA-NeXT-Video-7B-hf`
- `MCG-NJU/VideoChat3-4B`

These four cached checkpoints are the complete model roster for this project. Do not add Cosmos, OneVision, 34B, or other candidates. The General Transformers comparison is outside this work. Do not gate video work on its status.

## Proposed task set

1. **Video QA, MVBench:** run the prepared seeded 95-question sample across 19 task types, five items per task. Exclude all five `fine_grained_pose` questions because their clips are ROSE Lab-controlled NTU RGB+D assets. Score exact option accuracy by task and as a task macro average. This track includes InternVideo3, LLaVA-NeXT-Video, and VideoChat3. TimeLens2 is not scored on MVBench because its official interface returns temporal intervals rather than answer choices. Use the [MVBench paper](https://arxiv.org/abs/2311.17005) and [project code](https://github.com/OpenGVLab/Ask-Anything). Keep the 95-item scope and exclusions visible in results.
2. **Temporal retrieval, VUE-TR:** include all four models on one common vision-only fixed sample from VUE-TR. Use the released ground-truth annotations, query IDs, and official interval scorer. Select 100 queries with seed 3407, stratified across duration bins and query formats after confirming available counts. Exclude audio-only and vision-plus-audio queries because none of the four adapters accepts audio. The official VUE-TR instructions publish annotations and a scorer, and direct new evaluators to download source videos by YouTube ID. Keep downloaded videos local and publish neither video nor ground-truth annotations. The [official VUE-TR evaluation README](https://github.com/bytedance/vidi/tree/main/VUE_TR) defines the input/output format; [TimeLens2's official evaluation guide](https://github.com/MCG-NJU/TimeLens2/blob/main/evaluation/README.md) lists VUE-TR as a supported benchmark. Use its temporal-retrieval AUC/precision/recall metrics and report duration and query-format strata.
3. **Scope boundary:** this is a two-track video benchmark suite, not one cross-task score. Report MVBench QA and VUE-TR temporal retrieval separately. Do not use TimeLens-Bench, Video-MME, or ROSE/NTU samples in this report.

## Runner and prompt rules

Use native video inputs and model processors. Do not replace clips with a keyframe or create synthetic timestamps. Pin the clip sampling policy where shared: record frame count, FPS, resolution, duration, subtitles/audio setting, and effective context. Keep one active model, one request at a time, and no batching.

For video QA, use the machine-readable sampling policy in `data/video_bench/prepared/video_qa_sampling_policy.json`. It caps input at 16 unique frames and selects positions uniformly across the full clip from valid source timestamps. Preserve source dimensions until each model's documented video processor runs. Record source FPS, duration, dimensions, sampled count and timestamps, processor output dimensions/crops, and effective context per item. Do not add audio or subtitle tracks; leave visually burned-in text unchanged. If usable timestamps cannot be read, mark the item unavailable rather than synthesizing them. The 16-frame cap matches the lowest configured candidate-adapter limit; it is a sampling policy, not a measured runtime limit.

For MVBench, preserve its choice prompt and apply only each model's required chat or video wrapper. Decode the final response and score with a shared parser that supports each item's actual option count, including labels A through E. For VUE-TR, use the same query and interval output contract for all four models. Use each model's required wrapper, but do not expose ground-truth intervals during generation. Convert outputs to seconds and score them with the official VUE-TR evaluator.

## Model fit and local references

- MVBench QA: InternVideo3-8B-Instruct, LLaVA-NeXT-Video-7B, and VideoChat3-4B.
- VUE-TR temporal retrieval: InternVideo3-8B-Instruct, TimeLens2-8B, LLaVA-NeXT-Video-7B, and VideoChat3-4B.
- Adapters: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/internvideo3.py`, `timelens2.py`, `llava_video.py`, and `videochat3.py`.
- Existing model prompts: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/benchmark_vqa.py` and `scripts/common.py`.
- Keep the existing image tasks and scores separate: [Qwen-VL benchmark README](../qwenVL_Bench/README.md).

## Data and implementation order

The MVBench dataset and local clips are prepared. Fetch the VUE-TR public annotation file, scorer, and video ID list from the official repository. Create one reproducible 100-query, vision-only manifest. Download only videos required by that manifest. Preserve each video's source ID, duration, retrieval status, file size, and SHA-256. Keep all video files and source annotations out of Git and report-site assets.

Implement the MVBench QA track for its three supported models and the VUE-TR temporal-retrieval track for all four required models. Use the same fixed manifests and sampling policy across model comparisons. Run one model at a time. Preserve partial results and explicit unsupported/failure states. A model is complete only when it finishes every item in each track assigned to it.

## Report site delivery

After completing both tracks and writing `report.md`, keep a working static report site at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Show distinct MVBench QA and VUE-TR temporal-retrieval scorecards. Show which models participate in each track, full sampling and decoding settings, task or stratum breakdowns, incomplete/unsupported states, and the excluded ROSE/NTU category. Do not combine unlike task scores into one ranking.

## Validation plan draft

Use the real prepared MVBench subset, official VUE-TR annotations, downloaded source videos, actual model outputs, and final report artifacts. Do not use mocks, synthetic clips, or CPU substitutes for GPU model paths. The five ROSE Lab-controlled NTU examples are excluded and do not block either track.

1. **MVBench preparation integrity** (`tests/video/test_mvbench_manifest.py`): verify the preserved 100-record source manifest, active 95-record sample, five-per-task counts, exclusion ledger, active file paths, sizes, hashes, and archive audit. Fail if any ROSE/NTU item enters the active sample.
2. **VUE-TR manifest integrity** (`tests/video/test_vue_tr_manifest.py`): verify the immutable source annotation revision, selected 100 query IDs, seed, vision-only modality, duration/query-format stratification, required YouTube IDs, and local video hashes. Require no video/audio-only-incompatible query or missing asset in the active manifest.
3. **Model output and score consistency** (`tests/video/test_video_scoring.py`): for real completed runs, reconcile every MVBench exact-choice score and each VUE-TR temporal retrieval score against item-level output using the official VUE-TR scorer. Fail on missing items, parse failures, invalid intervals, mismatched aggregates, or video-only models assigned audio queries.
4. **Static report behavior** (`tests/video/test_report_site.py`): serve the completed site locally and inspect it in a browser with real result JSON. Require both scorecards, correct per-track model membership, coverage and settings, excluded task disclosure, incomplete/unsupported states, source links, and no embedded dataset assets. Fail on broken navigation, missing rows, or score mismatches.

The model execution itself is the real benchmark run, not a CPU test substitute: call `release-local` before each CUDA run, use one model at a time, and evaluate the assigned active sample with native-video processors. A model is complete only when every assigned item finishes and reconciles with its scorecard. Report track membership and the 95-item/19-task MVBench scope beside the scores. If a required active clip or GPU is unavailable, preserve that model/track as blocked or incomplete and do not claim a pass.

## Preparation audit (2026-11-03)

- The required video roster is InternVideo3-8B-Instruct, TimeLens2-8B, LLaVA-NeXT-Video-7B, and VideoChat3-4B. The General Transformers comparison is outside this project and does not gate video work.
- All four required checkpoints are present in the local HF cache with complete weight shards and processors. LLaVA-NeXT-Video 34B and Cosmos3-Nano exceed the 30 GiB cap and are outside this roster.
- The read-only HF snapshot audit found complete local weight shards for VideoChat3-4B (8.33 GiB), TimeLens2-8B (16.33 GiB), InternVideo3-8B-Instruct (17.44 GiB), LLaVA-NeXT-Video-7B (13.16 GiB), Cosmos-Reason2-8B (16.33 GiB), Cosmos-Reason2-2B (4.54 GiB), and Cosmos3-Edge (8.50 GiB). LLaVA-NeXT-Video-34B is 64.73 GiB and Cosmos3-Nano is 32.50 GiB; both exceed the 30 GiB cap before runtime overhead and stay excluded.
- Cached revisions: VideoChat3 `37fa901ec5913f84bc31108ebc1e60ad1903634c`; TimeLens2 `5392a2baa0f943e86b602fade8aa00578c59aa88`; InternVideo3 `c4602918b65225650d152db2850fe34e01d21fcd`; LLaVA-NeXT-Video-7B `f417d9edeb69a9eaa88283780332d00ce3c66593`; 34B `4603c5c2f3b2c1eb1c383c21014d4d53d8a34c4f`; Cosmos-Reason2-8B `a9fae2cf89dc64db96b12860417f0eb403013bb9`; 2B `9ce19a195e423419c349abfc86fd07178b230561`; Cosmos3-Edge `344d602b128d1bbdacb43b08d0a3626f46343e29`; Cosmos3-Nano `e59a53c25979a090fa8706c9acc0c254a6e89b92`.
- The official TimeLens2-8B model card identifies the checkpoint as Apache-2.0 and documents interval output. The separate TimeLens-Bench dataset remains quarantined and will not be used.
- The other selected model cards identify Apache-2.0 for VideoChat3 and InternVideo3, and Llama 2 for LLaVA-NeXT-Video. Preserve those model identifiers in results and follow their published terms.
- The active MVBench sample, VUE-TR annotation source, 100-query vision-only manifest, and 100 hash-verified local clips are prepared under git-ignored `data/video_bench/`. The VUE-TR manifest records 15 source-video replacements within the selected duration strata; the final replacement ledger is preserved locally.
- Dataset acquisition, local integrity, and scope states are consolidated in `data/video_bench/prepared/dataset_inventory.json`. Video-MME and other datasets are outside the four-model comparison.
- The original deterministic MVBench manifest at `data/video_bench/prepared/mvbench_manifest_fixed100_unavailable_nturgbd.json` contains 100 questions, five from each of 20 task types, selected with seed 3407. The active manifest keeps 95 records from 19 task types and excludes the five originally selected NTU fine-grained-pose records. Ninety native videos were extracted from the official dataset archives, and five episodic clips were downloaded from the official MVBench video branch into `data/video_bench/prepared/mvbench_assets/`. All 95 active files have recorded byte sizes and SHA-256 hashes; the branch files also retain their repository object identifiers. All 11 source ZIP archives (17.33 GB, 40,568 files) passed a full CRC scan. The archive scan record is `data/video_bench/prepared/mvbench_archive_integrity.json`.
- All five originally selected fine-grained-pose examples require NTU RGB+D videos. The task's 200 source records all reference NTU clips. At the user's direction, all five ROSE Lab-controlled examples are permanently excluded from this run. The active evaluation covers 19 tasks and 95 examples, with no fine-grained-pose score. All five episodic-reasoning MP4s were downloaded from the official OpenGVLab/MVBench `video` branch and verified against repository file sizes and SHA-256 metadata. The [MVBench dataset card](https://huggingface.co/datasets/OpenGVLab/MVBench/blob/video/README.md) limits original-video use to academic research and warns that other agreements may apply. The [TVQA data instructions](https://github.com/jayleicn/TVQA/blob/master/data/README.md) require a signed agreement for the separate 3-fps frame pack, which was not downloaded or used. Keep videos local and do not redistribute them.
- The local model snapshot audit is captured in `data/video_bench/prepared/model_inventory.json`. It records cached immutable revisions, sizes, licenses, run-cap exclusions, and assigned benchmark tracks. Inference is gated only on completing video-specific manifest, runtime, and model-readiness checks.
- The runtime audit found a dependency-compatible candidate in `/mnt/HDD1/Project_Code/robotics_VLM/.venv` (Python 3.10.12, Transformers 5.19.0, PyTorch 2.13.0, PyAV 12.3.0, decord2 3.4.0, and timm 1.0.27) for the referenced video adapters. The benchmark checkout's active `.venv` has Transformers 4.57.1 and lacks decord2 and timm, so it is not selected. The candidate runtime has not loaded a checkpoint or been validated for memory fit or benchmark I/O.
- The nine pinned cache snapshots were rechecked offline: each has `config.json`, video processor/tokenizer metadata, and all local safetensors files. Every indexed shard resolves and recorded weight totals match file sizes. Cosmos3 auxiliary VAE/sound-tokenizer weights are included. No model was loaded during this audit.
- The referenced adapters' own offline checkpoint audits now pass for all eight configured model setups, including required-file and declared InternVideo3 code-hash checks. LLaVA-NeXT-Video 34B has no configured adapter setup and remains excluded. The audit only verifies checkpoint files; it does not establish GPU fit, inference compatibility, or license permission.
- The local license audit for the other cached model families is irrelevant to this fixed roster. The four selected model cards and adapter profiles are documented in `model_inventory.json`; GPU fit still needs a real one-model-at-a-time evaluation run.
- TimeLens-Bench is quarantined from evaluation because its dataset card and upstream license conflict. This does not apply to the separately licensed Apache-2.0 TimeLens2 model checkpoint. Do not use TimeLens-Bench data or annotations.
- The official MVBench dataset card reports a 17.3 GB dataset and an MIT license. It notes that 320 NTU RGB+D videos require manual access through NTU; do not claim complete coverage for examples whose videos cannot be obtained.
- The official VUE-TR repository releases annotations and a scorer, and directs evaluators to download videos by YouTube ID. The VUE-TR README does not grant redistribution rights for video; keep fetched video files local and omit annotations from report artifacts.
- None of the dataset video files or annotations may be committed to Git. Store only source links, permitted manifests, and aggregate results in the report site.
