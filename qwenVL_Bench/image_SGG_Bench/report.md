# Image SGG benchmark

Fixed 200-image panels; direct image-to-graph generation. Results are subset metrics.
PSG uses box-adapted SGG and is not the official mask-based metric.

## VG-150

Split: `canonical_test`; panel size: 200.

| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | 0.0028 | 0.0028 | 0.0028 | 0.0003 | 0.0003 | 0.0003 | 0.0889 | 0.0028 | 0.0054 | 0.0125 | 22/200 | 200/200 | 178 |
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 2/200 | 200/200 | 198 |
| zmli/G-Substrate-Qwen3-VL-2B | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/200 | 200/200 | 200 |

## Open Images V7 visual relationships

Split: `validation`; panel size: 200.

| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | 0.0074 | 0.0074 | 0.0074 | 0.0617 | 0.0617 | 0.0617 | 0.0195 | 0.0074 | 0.0107 | 0.1057 | 75/200 | 200/200 | 125 |
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1/200 | 200/200 | 199 |
| zmli/G-Substrate-Qwen3-VL-2B | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/200 | 200/200 | 200 |

## PSG box-adapted SGG

Split: `validation (upstream test)`; panel size: 200.

| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | 0.0072 | 0.0072 | 0.0072 | 0.0031 | 0.0031 | 0.0031 | 0.0826 | 0.0072 | 0.0133 | 0.0386 | 48/200 | 200/200 | 152 |
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1/200 | 200/200 | 199 |
| zmli/G-Substrate-Qwen3-VL-2B | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/200 | 200/200 | 200 |

## Optional three-panel macro

Unweighted mean of dataset-level mR@50; shown only for models with all three full panels.

| Model | Macro mR@50 |
|---|---:|
| Qwen/Qwen2-VL-7B-Instruct | 0.0217 |
| Qwen/Qwen2.5-VL-7B-Instruct | 0.0000 |
| zmli/G-Substrate-Qwen3-VL-2B | 0.0000 |

## Protocol and limitations

Each dataset is reported separately. Source hashes, source revisions, class support, and predicate support are recorded in `summary.json`. Malformed output and runtime failures remain in the denominator. The panels are small local subsets, not full-set leaderboard scores. PSG results are box-adapted and must not be compared as official mask scores.
