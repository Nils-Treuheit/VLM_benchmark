# Video-oriented VLM benchmark implementation plan

## Scope and cached checkpoints

- `MCG-NJU/VideoChat3-4B`
- `MCG-NJU/TimeLens2-8B`
- `yanziang/InternVideo3-8B-Instruct`
- `llava-hf/LLaVA-NeXT-Video-7B-hf` and `llava-hf/LLaVA-NeXT-Video-34B-DPO-hf`
- Video-capable Cosmos Reason2 and Cosmos3 variants
- Consider LLaVA-OneVision only if its cached processor supports native video input

The named video checkpoints and both LLaVA-NeXT-Video checkpoints were present in `HF_HOME`. The Qwen-VL image suite stays unchanged. This plan adds a separate video scorecard.

## Proposed task set

1. **MVBench:** run a seeded, fixed 100-question subset, stratified across its 20 temporal task types. Preserve the multiple-choice question and score exact option accuracy per task and as a macro average. MVBench was designed to include tasks that cannot be solved from a single frame. Use the [official paper](https://arxiv.org/abs/2311.17005) and [project code](https://github.com/OpenGVLab/Ask-Anything).
2. **Video-MME:** run a seeded, fixed 100-question subset balanced across short, medium, and long duration groups and visual domains. Use the official no-subtitle setting for the primary comparison and report duration groups separately. Score exact multiple-choice accuracy. The complete test set has 900 videos and 2,700 questions, so a stable local subset limits run time. Follow the [official repository's prompt and evaluator](https://github.com/MME-Benchmarks/Video-MME) and [paper](https://arxiv.org/abs/2405.21075). Its dataset license restricts use to academic research and disallows redistribution; keep only permitted local data and result artifacts.
3. **TimeLens2 temporal grounding:** use the official temporal-grounding evaluation path and fixed subsets of Charades, ActivityNet Captions, and QVHighlights. These cover short actions, event-level localization, and highlight intervals. Start with 100 examples per dataset if the official split and local licenses allow. Report mean temporal IoU and the official recall/IoU thresholds per dataset. Do not combine this with video QA. TimeLens2 predicts one or more evidence intervals and reports seven temporal grounding settings in its [project page](https://mcg-nju.github.io/TimeLens2/), [paper](https://arxiv.org/abs/2607.17423), and [evaluation code](https://github.com/MCG-NJU/TimeLens2/tree/main/evaluation).

## Runner and prompt rules

Use native video inputs and model processors. Do not replace clips with a keyframe or create synthetic timestamps. Pin the clip sampling policy where shared: record frame count, FPS, resolution, duration, subtitles/audio setting, and effective context. Keep one active model, one request at a time, and no batching.

For MVBench and Video-MME, preserve each dataset's choice prompt and apply only each model's required chat or video wrapper. Decode the final response and score with a shared A/B/C/D parser. For TimeLens2, use the annotated sentence or question as the temporal query and preserve its native interval-set response for the official scorer. The robotics adapter's appended raw-JSON interval instruction is a reference, not a replacement for the official evaluation protocol.

## Model fit and local references

- MVBench and Video-MME: VideoChat3, InternVideo3, LLaVA-NeXT-Video 7B, and Cosmos variants that document video QA. Add 34B only after it passes the same memory cap. Check OneVision's actual video processor path before including it.
- TimeLens2: temporal grounding only. The robotics adapter marks it native-video-only and rejects image/text requests.
- Adapters: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/videochat3.py`, `internvideo3.py`, `llava_video.py`, `timelens2.py`, and `cosmos_reason.py`.
- Existing model prompts: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/benchmark_vqa.py` and `scripts/common.py`.
- Keep the existing image tasks and scores separate: [Qwen-VL benchmark README](../qwenVL_Bench/README.md).

## Data and implementation order

The dataset files were not found in the inspected local HF datasets cache. Check official splits, access rules, download requirements, and licenses before fetching data. Prepare each seeded manifest once and use it across all models. Do not include downloaded video files in distributed benchmark artifacts.

Implement MVBench first with VideoChat3 and LLaVA-NeXT-Video 7B. Add Video-MME, then InternVideo3 and compatible Cosmos/OneVision variants. Add TimeLens2's official temporal grounding evaluator as a separately reported task group. Preserve partial results and explicit unsupported/failure states; do not rank a model as complete unless it completed that group's full task set.

## Report site delivery

After completing the video benchmark and writing `report.md`, create its `report-site/` using the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Give video QA and temporal grounding distinct scorecards, and report native-video settings and temporal metrics explicitly.
