# Native VG50 PCIS results

**Status:** saved PCIS score artifacts are available for all eight cached model
configurations. These are separate from the shared JSON and box-grounded SGDet
track described in the main benchmark README.

The runs use the same fixed 50-image Visual Genome `canonical_test` panel and
the pinned G-Substrate PCIS evaluator at threshold 0.6. The panel has 410
ground-truth relation instances. No full dataset was run. The reported metrics
are official PCIS R@50 and mR@50; strict tuple validity is a separate output
format diagnostic.

| Model | Prompt version | PCIS R@50 | PCIS mR@50 | Strict-valid rows | Cap or inference errors |
|---|---|---:|---:|---:|---:|
| `zmli/G-Substrate-Qwen3-VL-2B` | v3 task-token variant | 22.68% | 8.06% | 42/50 | 1 cap |
| `Qwen/Qwen2.5-VL-7B-Instruct` | v3 | 3.90% | 2.80% | 0/50 | 0 |
| `Qwen/Qwen2-VL-7B-Instruct` | v3 | 0.98% | 0.17% | 35/50 | 1 capped/error row* |
| `bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS` | v3 repair 1 | 0.73% | 0.09% | 28/50 | 0 |
| `unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL` | v3 repair 1 | 2.44% | 1.11% | 0/50 | 0 |
| `unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL` | v3 repair 2 | 4.15% | 1.22% | 0/50 | 1 capped/error row |
| `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL` | v3 repair 2 | 0.24% | 0.02% | 0/50 | 0 |
| `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL` | v3 repair 3 | 4.39% | 3.99% | 0/50 | 0 |

G-Substrate's native score is nonzero and substantially higher than the other
saved runs. Its raw relation tuples were rejected by the separate JSON-and-box
parser, which explains those earlier zeros; they were not native PCIS results.
The official scorer can map tuple text even when the local strict parser
rejects it, so the official score and strict-valid count must both remain
visible.

Qwen2 and Qwen2.5 share one v3 prompt hash. G-Substrate uses a different v3
prompt hash with its required task token. Repairs were tailored to individual
model runs. The models share the panel, scorer, and threshold, but these scores
are not a controlled head-to-head ranking. Repair prompts cap output at ten
tuples, which also limits recall.

\* The Qwen2 run records 4,096 generated tokens and `finish_reason=length`,
but its error text incorrectly says the limit was 1,024 tokens. Its cap and
inference-error flags refer to the same row. The 30B Thinking-1M run's cap and
inference-error flags also refer to the same row; that prediction was blanked
before scoring.

The source run records, official scorer inputs, and score manifests are kept in
the local ignored recovery workspace at
`.agentic_workflow/workspace/image-sgg-recovery/native-pcis/comparisons/`.
All eight run records still carry `awaiting_official_pcis_score` phase labels
despite having paired score artifacts; treat those labels as stale metadata
and retain the score-to-input hash checks in that workspace.

## Matched prompt-screen baselines

A low-effort prompt and output-budget screen compares four cached revisions on
the same frozen ten-image qualification panel. Current-best saved predictions
were sliced to those exact IDs and rescored with the pinned evaluator at
threshold 0.6. An independent audit verified the baseline panel order, image
and prediction joins, model revisions, and provenance hashes. Qwen2.5 and
Qwen3 30B Thinking-1M produce the same aggregate baseline on this small panel;
independently rescoring both inputs reproduced that equality.

| Model | Baseline prompt | Matched-panel R@50 | Matched-panel mR@50 | Cap/error rows |
|---|---|---:|---:|---:|
| `zmli/G-Substrate-Qwen3-VL-2B` | v3 task-token variant | 23.60% | 16.98% | 0/10 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | v3 | 1.12% | 0.23% | 0/10 |
| `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL` | v3 repair 3 | 4.49% | 6.34% | 0/10 |
| `unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL` | v3 repair 2 | 1.12% | 0.23% | 0/10 |

The candidate uses a concise tuple prompt with a 4,096-token greedy output
cap. It requests up to fifty unique visible relations and keeps the same exact
one-line tuple format. None of the four completed screens qualifies for a selective full
50-image rerun:

| Model | Candidate R@50 | Candidate mR@50 | PCIS matched relations, baseline → candidate | Cap/error rows, baseline → candidate | Screen |
|---|---:|---:|---:|---:|---|
| `zmli/G-Substrate-Qwen3-VL-2B` | 12.36% | 3.41% | 21 → 11 | 0 → 1 | Fail: both PCIS metrics fell; one output hit the cap |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 0.00% | 0.00% | 1 → 0 | 0 → 0 | Fail: both PCIS metrics fell |
| `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL` | 1.12% | 0.23% | 4 → 1 | 0 → 6 | Fail: both metrics fell; six outputs hit the cap |
| `unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF:UD-Q4_K_XL` | 0.00% | 0.00% | 1 → 0 | 0 → 9 | Fail: both metrics fell; nine outputs hit the cap |

The PCIS matched-relation counts use the same 89 ground-truth relation
instances in the frozen panel. No candidate meets the screen gate, so no
selective full 50-image rerun was started. The G-Substrate candidate matched
ten fewer PCIS relations and added one cap/error row. Qwen2.5 and Qwen3 30B
Thinking-1M each lost one match. Qwen3 8B Q8 lost three matches and added six
cap/error rows. These ten-image results are screening evidence and do not
replace the 50-image results above.
Baseline and candidate records, exact scorer inputs, scores, and provenance
manifests are retained in the ignored
`.agentic_workflow/workspace/image-sgg-recovery/prompt-tuning-20261112/`
workspace.

These results cover only the VG50 native PCIS panel. They do not complete the
three-panel box-grounded SGDet benchmark, and PSG is not included here.
