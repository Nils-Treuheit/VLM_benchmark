# Image SGG benchmark

Fixed 200-image panels; direct image-to-graph generation. Results are subset metrics.
PSG uses box-adapted SGG and is not the official mask-based metric.

## VG-150

Split: `canonical_test`; panel size: 200.

| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | 0.0028 | 0.0028 | 0.0028 | 0.0003 | 0.0003 | 0.0003 | 0.0889 | 0.0028 | 0.0054 | 0.0125 | 22/200 | 200/200 | 178 |
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | 2/200 | 200/200 | 198 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | llama.cpp (IQ2_XXS) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/200 | 200/200 | 200 |
| zmli/G-Substrate-Qwen3-VL-2B | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/200 | 200/200 | 200 |

## Open Images V7 visual relationships

Split: `validation`; panel size: 200.

| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2-VL-7B-Instruct | Transformers (model card) | 0.0074 | 0.0074 | 0.0074 | 0.0617 | 0.0617 | 0.0617 | 0.0195 | 0.0074 | 0.0107 | 0.1057 | 75/200 | 200/200 | 125 |
| Qwen/Qwen2.5-VL-7B-Instruct | Transformers (model card) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1/200 | 200/200 | 199 |
| bartowski/Qwen2-VL-72B-Instruct-GGUF:IQ2_XXS | llama.cpp (IQ2_XXS) | 0.0025 | 0.0025 | 0.0025 | 0.0028 | 0.0028 | 0.0028 | 0.0417 | 0.0025 | 0.0047 | 0.0041 | 2/200 | 200/200 | 198 |
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

Each dataset is reported separately. Class and predicate counts are published in [data.json](data.json). Source hashes and revisions are retained in the local run records. Malformed output and runtime failures remain in the denominator. The panels are small local subsets, not full-set leaderboard scores. PSG results are box-adapted and must not be compared as official mask scores.

Qwen2-VL-72B IQ2_XXS has full VG-150 and Open Images panels. Its PSG run stopped after 123/200 images, so it has no published PSG score or three-panel macro.

The [native VG50 PCIS results](native-pcis-vg50.md) use a separate tuple-based scoring protocol.
