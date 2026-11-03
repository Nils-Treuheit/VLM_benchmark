# Video VLM Benchmark Report

**Evaluation status:** complete

**Generated:** 2026-11-03T16:09:15.540762+00:00

## Scope

| Track | Sample | Models | Score |
|---|---:|---:|---|
| MVBench video QA | 95 questions / 19 tasks | 3 | Exact-choice accuracy and task macro accuracy |
| VUE-TR temporal grounding | 100 queries / 100 videos | 4 | Official success, precision, and recall AUC |

MVBench answer-choice accuracy and VUE-TR temporal grounding are separate tasks and are not combined.

## Results

### MVBench

| Model | Accuracy | Task macro accuracy | Coverage | Mean latency | Parse failures | Status |
|---|---:|---:|---:|---:|---:|---|
| InternVideo3-8B-Instruct | 52.6% | 52.6% | 95/95 | 945 ms | 0 | complete |
| LLaVA-NeXT-Video-7B | 45.3% | 45.3% | 95/95 | 304 ms | 0 | complete |
| VideoChat3-4B | 61.1% | 61.1% | 95/95 | 170 ms | 0 | complete |

### VUE-TR

| Model | Success AUC | Precision AUC | Recall AUC | Coverage | Mean latency | Parse failures | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| InternVideo3-8B-Instruct | 9.7% | 53.3% | 12.8% | 100/100 | 410 ms | 0 | complete |
| TimeLens2-8B | 31.0% | 36.2% | 53.2% | 100/100 | 452 ms | 0 | complete |
| LLaVA-NeXT-Video-7B | 0.1% | 14.7% | 4.5% | 100/100 | 3268 ms | 86 | complete |
| VideoChat3-4B | 1.3% | 25.0% | 1.9% | 100/100 | 197 ms | 0 | complete |

## Protocol

Up to 16 shared source-PTS frames per clip. Target positions span the timeline; select the nearest decoded source-PTS frame independently for each timeline target; choose the earlier source presentation timestamp. decode sequentially for clips up to 120 seconds; use a backward keyframe seek and local nearest-frame decode for longer clips.

Original source resolution is retained through decoding; pinned model processors apply each model's configured preprocessing. Profiles: InternVideo3-8B-Instruct: bf16, max 16 frames; TimeLens2-8B: bf16, max 16 frames; LLaVA-NeXT-Video-7B: bf16, max 64 frames; VideoChat3-4B: fp16, max 32 frames, 50176 px/frame.

Audio: ignore; do not pass audio to models. Subtitles: disabled; burned-in text is retained. Generation temperature is 0.0; max new tokens are 64 for QA and 512 for temporal retrieval.

VUE-TR interval parser video-vlm-intervals-v2 accepts numeric endpoint pairs, numeric-string endpoint pairs, and a top-level numeric pair as one interval. counted as parse failures and submitted as empty predictions to the official VUE-TR scorer.

Five NTU RGB+D fine-grained-pose examples were excluded at the user's direction.

100 vision-only queries, one per video, seed 3407; audio and audiovisual queries excluded.

Dataset videos, annotations, queries, and item-level outputs remain local and are not included in the website bundle.

## Sources

- [MVBench](<https://github.com/OpenGVLab/Ask-Anything>)
- [VUE-TR](<https://github.com/bytedance/vidi/tree/main/VUE_TR>)
