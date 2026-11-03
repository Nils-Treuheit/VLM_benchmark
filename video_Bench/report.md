# Video VLM Benchmark Report

**Evaluation status:** not_started  
**Generated:** 2026-11-03T13:13:07.092525+00:00

## Scope

| Track | Sample | Models | Score |
|---|---:|---:|---|
| MVBench video QA | 95 questions / 19 tasks | 3 | Exact-choice accuracy and task macro accuracy |
| VUE-TR temporal grounding | 100 queries / 100 videos | 4 | Official success, precision, and recall AUC |

MVBench answer-choice accuracy and VUE-TR temporal grounding are separate tasks and are not combined.

## Results

### MVBench

| Model | Accuracy | Task macro accuracy | Coverage | Parse failures | Status |
|---|---:|---:|---:|---:|---|
| InternVideo3-8B-Instruct | — | — | 0/95 | 0 | not_run |
| LLaVA-NeXT-Video-7B | — | — | 0/95 | 0 | not_run |
| VideoChat3-4B | — | — | 0/95 | 0 | not_run |

### VUE-TR

| Model | Success AUC | Precision AUC | Recall AUC | Coverage | Parse failures | Status |
|---|---:|---:|---:|---:|---:|---|
| InternVideo3-8B-Instruct | — | — | — | 0/100 | 0 | not_run |
| TimeLens2-8B | — | — | — | 0/100 | 0 | not_run |
| LLaVA-NeXT-Video-7B | — | — | — | 0/100 | 0 | not_run |
| VideoChat3-4B | — | — | — | 0/100 | 0 | not_run |

## Protocol

16 shared source-PTS frames per clip; audio disabled; temperature 0.0.

Five NTU RGB+D fine-grained-pose examples were excluded at the user's direction.

100 vision-only queries, one per video, seed 3407; audio and audiovisual queries excluded.

Dataset videos, annotations, queries, and item-level outputs remain local and are not included in the website bundle.

## Sources

- [MVBench](<https://github.com/OpenGVLab/Ask-Anything>)
- [VUE-TR](<https://github.com/bytedance/vidi/tree/main/VUE_TR>)
