# General Transformers VLM benchmark

Updated: 2026-11-03 10:06:58 +0100
Run ID: 20261103-1010-general-reconciled
Run state: partial; scored model variants: 4; complete full-suite runs: 3; planned candidates not run: 1.
Runtime versions: Transformers 4.44.2 / PyTorch 2.15.0 / CUDA 13.2, Transformers 4.57.1 / PyTorch 2.14.1 / CUDA 13.0, Transformers 5.19.0 / PyTorch 2.13.0+cu130 / CUDA 13.0.
Execution: Local CUDA inference, one model at a time on NVIDIA GeForce RTX 5090 (compute capability 12.0); configured process cap 30.0 GiB.
Sampling seed: 3407; memory ceiling: 30.0 GiB.
The full-suite macro is the unweighted mean of eight local task scores and requires complete coverage on all requested examples.

## Selection guide

Best complete overall: **google/paligemma2-3b-mix-224** at 54.4%. Task leaders below use only complete full-suite models.

## Quality results

| Model | Run status | Preflight | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allenai/Molmo2-8B | complete | passed | 87.0% | 74.2% | 8.0% | 12.0% | 21.2% | 4.0% | 80.0% | 13.8% | 8/8 | 37.5% |
| google/paligemma2-3b-mix-224 | complete | passed | 71.0% | 57.6% | 97.2% | 78.0% | 69.2% | 8.0% | 52.0% | 2.6% | 8/8 | 54.4% |
| microsoft/Florence-2-large-ft | complete | passed | 2.0% | 45.4% | 48.8% | 53.6% | 45.6% | 0.0% | 12.0% | 0.0% | 8/8 | 25.9% |
| OpenGVLab/InternVL3_5-30B-A3B | failed | failed/not run | — | — | — | — | — | — | — | — | 0/8 | — |

## Planned candidates not run

The following candidates are part of the implementation plan but lack a scored run in this report.
| Model | Cache status | Status | Reason |
|---|---|---|---|
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | incomplete | not_run | Both indexed safetensors shards are missing from the local cache; only the configuration and an incomplete weight blob are present. Offline loading is not possible. Benchmark use also remains gated on confirming applicability of Meta’s Llama 3.2 Vision terms. |

## Task leaders

- **AI2D:** allenai/Molmo2-8B (87.0%).
- **TextVQA / OCR:** allenai/Molmo2-8B (74.2%).
- **Truth:** google/paligemma2-3b-mix-224 (97.2%).
- **Counting:** google/paligemma2-3b-mix-224 (78.0%).
- **Description:** google/paligemma2-3b-mix-224 (69.2%).
- **Affordance:** google/paligemma2-3b-mix-224 (8.0%).
- **Negation:** allenai/Molmo2-8B (80.0%).
- **GRAB-Lite:** allenai/Molmo2-8B (13.8%).

## GRAB-Lite group scores

| Model | Function | Properties | Real figures | Series | Transformations |
|---|---:|---:|---:|---:|---:|
| allenai/Molmo2-8B | 13.0% | 19.0% | 16.0% | 15.0% | 6.0% |
| google/paligemma2-3b-mix-224 | 2.0% | 3.0% | 4.0% | 3.0% | 1.0% |
| microsoft/Florence-2-large-ft | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| OpenGVLab/InternVL3_5-30B-A3B | — | — | — | — | — |

## Real-image preflight

Each model must produce a nonempty TextVQA answer and a parseable AI2D answer before any full task begins. Passing examples are reused in the scored task results.
| Model | Result | AI2D format | AI2D parse error | AI2D output tokens | TextVQA response | TextVQA parse error | TextVQA output tokens |
|---|---|---|---|---:|---|---|---:|
| allenai/Molmo2-8B | passed | valid | False | 2 | nonempty | False | 4 |
| google/paligemma2-3b-mix-224 | passed | valid | False | 2 | nonempty | False | 2 |
| microsoft/Florence-2-large-ft | passed | valid | False | 9 | nonempty | False | 8 |
| OpenGVLab/InternVL3_5-30B-A3B | failed/not run | invalid/not run | — | — | empty/not run | — | — |

## Runtime and resources

Generation time excludes preprocessing. Peak allocated VRAM is process-local, not whole-device memory.
| Model | Task | Completed / requested | Mean seconds | Tokens / second | Peak allocated GiB | Parse errors |
|---|---|---:|---:|---:|---:|---:|
| allenai/Molmo2-8B | AI2D | 100/100 | 0.26 | 7.59 | 17.52 | 0 |
| allenai/Molmo2-8B | TextVQA / OCR | 100/100 | 0.25 | 18.34 | 17.52 | 0 |
| allenai/Molmo2-8B | Truth | 25/25 | 0.23 | 31.60 | 17.45 | 0 |
| allenai/Molmo2-8B | Counting | 25/25 | 0.16 | 21.40 | 17.45 | 0 |
| allenai/Molmo2-8B | Description | 25/25 | 0.18 | 25.06 | 17.45 | 0 |
| allenai/Molmo2-8B | Affordance | 25/25 | 0.23 | 16.04 | 17.52 | 0 |
| allenai/Molmo2-8B | Negation | 25/25 | 0.16 | 12.27 | 17.52 | 0 |
| allenai/Molmo2-8B | GRAB-Lite | 500/500 | 0.26 | 13.76 | 17.53 | 0 |
| **allenai/Molmo2-8B peak** | GRAB-Lite | — | — | — | 17.53 | — |
| google/paligemma2-3b-mix-224 | AI2D | 100/100 | 0.05 | 43.02 | 5.74 | 0 |
| google/paligemma2-3b-mix-224 | TextVQA / OCR | 100/100 | 0.07 | 58.15 | 5.73 | 0 |
| google/paligemma2-3b-mix-224 | Truth | 25/25 | 0.04 | 51.03 | 5.73 | 0 |
| google/paligemma2-3b-mix-224 | Counting | 25/25 | 0.04 | 51.86 | 5.73 | 0 |
| google/paligemma2-3b-mix-224 | Description | 25/25 | 0.04 | 52.63 | 5.73 | 0 |
| google/paligemma2-3b-mix-224 | Affordance | 25/25 | 0.05 | 49.50 | 5.73 | 0 |
| google/paligemma2-3b-mix-224 | Negation | 25/25 | 0.04 | 47.07 | 5.74 | 0 |
| google/paligemma2-3b-mix-224 | GRAB-Lite | 500/500 | 0.13 | 37.85 | 5.76 | 17 |
| **google/paligemma2-3b-mix-224 peak** | GRAB-Lite | — | — | — | 5.76 | — |
| microsoft/Florence-2-large-ft | AI2D | 100/100 | 0.07 | 105.45 | 1.74 | 82 |
| microsoft/Florence-2-large-ft | TextVQA / OCR | 100/100 | 0.09 | 150.00 | 1.74 | 0 |
| microsoft/Florence-2-large-ft | Truth | 25/25 | 0.07 | 119.78 | 1.74 | 0 |
| microsoft/Florence-2-large-ft | Counting | 25/25 | 0.07 | 129.23 | 1.74 | 0 |
| microsoft/Florence-2-large-ft | Description | 25/25 | 0.07 | 127.33 | 1.74 | 0 |
| microsoft/Florence-2-large-ft | Affordance | 25/25 | 0.23 | 211.08 | 1.74 | 0 |
| microsoft/Florence-2-large-ft | Negation | 25/25 | 0.08 | 130.16 | 1.74 | 16 |
| microsoft/Florence-2-large-ft | GRAB-Lite | 500/500 | 0.14 | 48.70 | 1.74 | 467 |
| **microsoft/Florence-2-large-ft peak** | AI2D | — | — | — | 1.74 | — |
| OpenGVLab/InternVL3_5-30B-A3B | AI2D | 0/100 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | TextVQA / OCR | 0/100 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | Truth | 0/25 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | Counting | 0/25 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | Description | 0/25 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | Affordance | 0/25 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | Negation | 0/25 | — | — | — | 0 |
| OpenGVLab/InternVL3_5-30B-A3B | GRAB-Lite | 0/500 | — | — | — | 0 |
| **OpenGVLab/InternVL3_5-30B-A3B peak** | — | — | — | — | — | — |

## Load-time memory

Load-time measurements are recorded separately from per-task generation. The sampled process peak uses nvidia-smi and can miss short peaks.
| Model | Allocated GiB | Reserved GiB | Sampled process peak GiB | Samples |
|---|---:|---:|---:|---:|
| allenai/Molmo2-8B | — | — | — | — |
| google/paligemma2-3b-mix-224 | — | — | — | — |
| microsoft/Florence-2-large-ft | — | — | — | — |
| OpenGVLab/InternVL3_5-30B-A3B | — | — | 0.49 | 24 |

## Method

Tasks reuse the prepared examples, prompts, sample indices, and scorers described in `../../qwenVL_Bench/README.md`.
AI2D uses 100 test examples; TextVQA uses 100 validation examples; the five additional VQA panel categories use 25 examples each; GRAB-Lite uses all 500 test examples.
Requests run serially with one image and greedy decoding, with a 256-token generation limit. Molmo2 uses its chat template and bfloat16 weights. PaliGemma uses `answer en`, bfloat16 weights, SDPA, and 224-pixel inputs. Florence uses a separate Transformers 4.44.2 environment, float16 weights, eager attention, and the experimental `<VQA>` wrapper.
Florence's cached processor does not register `<VQA>` and post-processing falls back to plain text. It is scored only if its real-image preflights demonstrate parseable AI2D output and a nonempty TextVQA answer.
For Florence AI2D and negation choice questions, exact generated option text is mapped to the displayed label before shared scoring; raw generated text is retained in per-item results.

## Provenance and limitations

- **allenai/Molmo2-8B:** complete; Transformers 4.57.1; checkpoint weights 32.27 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-1010-general-reconciled/molmo2_8b/report.md`.
- **google/paligemma2-3b-mix-224:** complete; Transformers 4.57.1; checkpoint weights 5.65 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-1010-general-reconciled/paligemma2_mix_224/report.md`.
- **microsoft/Florence-2-large-ft:** complete; Transformers 4.44.2; checkpoint weights 1.44 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-1010-general-reconciled/florence2_large_ft/report.md`.
- **OpenGVLab/InternVL3_5-30B-A3B:** failed; Transformers 5.19.0; checkpoint weights 57.46 GiB; cache status ready. Transformers 5.19 BNB NF4 quantizes Linear4bit modules, but the pinned Qwen3-MoE fused expert tensors remain BF16. A CPU meta-skeleton audit found 30,848,730,112 parameters and only 1,220,542,464 eligible for NF4 in 290 modules (3.96% of parameters). The loader therefore requested 30.15 GiB during allocator warmup with 29.93 GiB free; no inference started.
  Source run: `../runs/20261103-1010-general-reconciled/internvl3_5_30b_a3b/report.md`.

Raw per-item outputs and errors remain in each model's `results.json`. The report site publishes aggregate metrics only.
