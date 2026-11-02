# Grounding specialist benchmark implementation plan

## Scope

Create a separate benchmark for two cached checkpoints:

- `nvidia/LocateAnything-3B`, snapshot `c32291ca5e996f5a7a485845b4f57a233936bba0`.
- `GroundingPI/GroundAnything-VLM`, snapshot `0f3cf22027493b4f97f342dae1fb298c38b68d27`.

Both snapshots and their weight shards, processors, tokenizers, and model code were present in `HF_HOME` during planning. These models output grounded boxes or points. Keep this benchmark and its results separate from the general VLM image-QA leaderboard.

## Task set

Use 100 examples per dataset, sampled once with seed 3407 and reused by both models. Save manifests without redistributing restricted image/video assets.

| Task | Dataset and split | Input prompt | Score |
|---|---|---|---|
| Referring-expression grounding | RefCOCOg test | One annotated phrase; request its target box | mean IoU and Acc@IoU 0.5 |
| Open-vocabulary detection | COCO val2017 | Image-specific category list; request all instances per category | official COCO AP where category mapping is supported; also F1@mIoU |
| GUI grounding | ScreenSpot-Pro test | Screenshot plus action instruction; request target point and box where supported | point-in-target accuracy and Acc@IoU 0.5 |
| Text-region detection | HierText validation/test | Request all visible text regions | official detection F1@mIoU, precision, and recall |

Report every task separately. A normalized local macro score may be added for selection, clearly labeled as a local metric. Do not average these localization metrics into Qwen-VL's VQA score. Count malformed, truncated, empty, and duplicate outputs. Treat `None` as a valid no-target response only when the annotation includes a true negative.

## Exact prompting and model setup

Use each model's own processor and chat template. These are grounding models, so avoid free-form captioning or generic VQA prompts. Use the model-card task templates verbatim and change only the target phrase, category list, or GUI instruction supplied by each dataset.

**LocateAnything-3B**

- Load the pinned local snapshot through the documented Transformers or SGLang path with `trust_remote_code=True`, BF16, and the processor/tokenizer from that same snapshot.
- Use `generation_mode="hybrid"` as the accuracy/latency default. Record mode in results. The card recommends up to 8,192 output tokens to avoid truncation on dense scenes.
- Category detection: `Locate all the instances that matches the following description: [CATEGORIES].` Join categories using `</c>`. Referring box: `Locate a single instance that matches the following description: [PHRASE].` GUI point: `Point to: [INSTRUCTION].` OCR: `Detect all the text in box format.`
- Do not include visual prompting in this benchmark. NVIDIA's official [Eagle/Embodied README](https://github.com/NVlabs/Eagle/tree/main/Embodied) states that the released 3B weights do not support visual-prompt inference out of the box.

**GroundAnything-VLM**

- Follow the local [setup and usage README](/home/nils/groundanything-vlm/README.md). It documents a model-specific SGLang integration; a plain upstream SGLang install does not provide its adapter or cache semantics.
- Use a separate serving environment and the GroundAnything source checkout. The documented workflow is `python3 run.py setup serve`, then `python3 run.py serve --config configs/release/vlm_sglang.yaml`. Point the config to the already-cached snapshot or a local symlink; do not fetch a duplicate checkpoint.
- Start the VLM service, not the sibling diffusion model service. The documented endpoint is `http://127.0.0.1:8102/v1`, model ID `groundinganything-vlm`.
- Keep the documented BF16/eager profile for the initial accuracy run. The authors' serving profile pins Torch, Transformers, Triton, SGLang kernels, and FlashInfer; do not install these into the Qwen benchmark environment. RTX 5090 execution still needs local GPU validation.
- Referring box: `Locate the target referred to by the following description: [PHRASE].` Category grounding: `Locate all the instances that match the following categories: [CAT1]</c>[CAT2].` GUI point: `Point to the UI element to click for the following instruction: [INSTRUCTION].` OCR: `OCR task detect all the text in box format.` Use its official client for box/point requests, or its OpenAI-compatible endpoint for other task prompts.

The two outputs use similar coordinate vocabularies but not identical parsing behavior. Preserve special tokens and raw text until each model-specific parser produces a canonical `{label, kind, box_xyxy_norm, point_xy_norm, valid}` record. Preserve GroundAnything's GAM tokens by using `skip_special_tokens=false` and avoiding spaces between coordinate tokens. Apply each model's documented 0–999/0–1000 coordinate conversion; do not share a regex parser by assumption.

## Runtime and measurements

Use the local API service for serial calls. Run only one model at a time and one request at a time for initial accuracy results. Record model revision, snapshot, generation mode, precision, server and package versions, image dimensions, input/output tokens, latency, peak VRAM, parse status, and raw output. Keep throughput or generation-mode sweeps separate from accuracy results.

For GroundAnything-VLM, preserve the model-specific launch configuration and runtime metadata. The local README says the source repo's serving profile controls its custom SGLang engine and that the VLM server runs on port 8102. The HF card's generic SGLang example should not override the custom profile. For LocateAnything, follow its HF card's supported Transformers/SGLang paths and pin the exact snapshot.

## Data, evaluation, and references

The inspected local HF dataset cache did not contain the proposed benchmark data. Before downloading, verify official test/validation splits, license terms, and model evaluation instructions. Prefer the [NVIDIA LocateAnything evaluation guide](https://github.com/NVlabs/Eagle/blob/main/Embodied/evaluation/README.md) and the [GroundAnything evaluation toolkit](https://github.com/groundingpi/GroundAnything). Use shared image manifests for both systems and adapt only the dataset loader, native prompt, output parser, and scorer bridge.

- [LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [GroundAnything-VLM model card](https://huggingface.co/GroundingPI/GroundAnything-VLM)
- Local GroundAnything notes: `/home/nils/groundanything-vlm/README.md`
- Official [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- Official repositories contain task prompts, generation modes, spatial tokens, output parsing, and evaluation procedures.

## Implementation order

1. Audit cached snapshot files, model code, environment requirements, dataset access, and licenses without loading a model onto the GPU.
2. Prepare the shared sample manifests, then implement each model's native prompt path and parser.
3. Run LocateAnything using its documented local route. Release GPU resources before starting GroundAnything-VLM's dedicated SGLang service.
4. Inspect raw predictions, parsing failures, and per-task scores before issuing recommendations. Keep these specialists out of the generic VLM ranking.

## Report site delivery

After completing the grounding benchmark and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Keep localization task metrics in their own scorecard and separate from generic image-QA rankings.
