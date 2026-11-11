# General Transformers VLM benchmark

Updated: 2026-11-04 21:28:00 +0100
Run ID: 20261103-133909-internvl35-candidates
Run state: complete; scored model variants: 10; complete full-suite runs: 10; planned candidates not run: 0.
Runtime versions: Transformers 4.44.2 / PyTorch 2.15.0 / CUDA 13.2, Transformers 4.57.1 / PyTorch 2.14.1 / CUDA 13.0, Transformers 4.57.1 / PyTorch 2.14.1+cu130 / CUDA 13.0.
Execution: Local CUDA inference, one model at a time on NVIDIA GeForce RTX 5090 (compute capability 12.0); configured process cap 30.0 GiB.
Sampling seed: 3407; memory ceiling: 30.0 GiB.
The full-suite macro is the unweighted mean of eight local task scores and requires complete coverage on all requested examples.

## Edge refinement results

| Whole-device cap | Model | Status | Full-suite macro | Measured context | Probe peak | Scoring peak | Whole-run peak |
|---:|---|---|---:|---:|---:|---:|---:|
| 8 GiB | google/paligemma2-3b-mix-224 | verified | 54.53% | 1,800 tokens | 7.951 GiB | 7.271 GiB | 7.955 GiB |
| 10 GiB | google/paligemma2-3b-mix-224 | verified | 54.53% | 4,418 tokens | 9.961 GiB | 7.244 GiB | 9.961 GiB |
| 12 GiB | google/paligemma2-3b-mix-224 | verified | 54.53% | 6,065 tokens | 11.883 GiB | 7.252 GiB | 11.883 GiB |

The eager-attention run completed all eight shared tasks. The [detailed report](../edge_refinement/8gib/paligemma2-3b-mix-224/retry_eager_1800_20261104/report.md) and raw artifact (local artifact; not published) contain the measured context and sampling evidence. The earlier SDPA failure and first eager probe that exceeded the cap remain in their separate attempt directories.

The 10 GiB run also completed all eight tasks and passed cap verification. See its [detailed report](../edge_refinement/10gib/paligemma2-3b-mix-224/report.md) and raw artifact (local artifact; not published). The 4,418-token measured context reflects the largest observed input-plus-output length that remained under the 10 GiB cap; larger probe attempts are preserved in the artifact.

The 12 GiB run completed all eight tasks and passed cap verification at 6,065 tokens. Its 11.883 GiB whole-run peak remained below the cap. See the [detailed report](../edge_refinement/12gib/google--paligemma2-3b-mix-224/report.md) and raw artifact (local artifact; not published).

## Selection guide

Best complete overall: **OpenGVLab/InternVL3_5-8B** at 59.7%. Task leaders below use only complete full-suite models.

## Quality results

| Model | Run status | Preflight | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allenai/Molmo2-8B | complete | passed | 87.0% | 74.2% | 8.0% | 12.0% | 21.2% | 4.0% | 80.0% | 13.8% | 8/8 | 37.5% |
| google/paligemma2-3b-mix-224 | complete | passed | 71.0% | 57.6% | 97.2% | 78.0% | 69.2% | 8.0% | 52.0% | 2.6% | 8/8 | 54.4% |
| microsoft/Florence-2-large-ft | complete | passed | 2.0% | 45.4% | 48.8% | 53.6% | 45.6% | 0.0% | 12.0% | 0.0% | 8/8 | 25.9% |
| OpenGVLab/InternVL3_5-8B-Instruct | complete | passed | 76.0% | 81.3% | 92.8% | 60.4% | 69.2% | 8.0% | 76.0% | 13.2% | 8/8 | 59.6% |
| OpenGVLab/InternVL3_5-8B-MPO | complete | passed | 76.0% | 80.1% | 92.8% | 69.2% | 65.2% | 8.0% | 72.0% | 12.8% | 8/8 | 59.5% |
| OpenGVLab/InternVL3_5-8B | complete | passed | 78.0% | 78.0% | 92.8% | 73.2% | 64.8% | 8.0% | 68.0% | 14.6% | 8/8 | 59.7% |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | complete | passed | 74.0% | 71.4% | 49.2% | 55.2% | 56.4% | 4.0% | 76.0% | 16.2% | 8/8 | 50.3% |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | complete | passed | 79.0% | 65.9% | 57.2% | 22.4% | 53.2% | 4.0% | 72.0% | 14.8% | 8/8 | 46.1% |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | complete | passed | 79.0% | 66.9% | 57.2% | 4.0% | 45.2% | 0.0% | 60.0% | 13.8% | 8/8 | 40.8% |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | complete | passed | 74.0% | 66.8% | 53.2% | 24.0% | 29.6% | 0.0% | 24.0% | 7.0% | 8/8 | 34.8% |

## Planned candidates not run

All candidates in the current plan have complete scored runs.

## Task leaders

- **AI2D:** allenai/Molmo2-8B (87.0%).
- **TextVQA / OCR:** OpenGVLab/InternVL3_5-8B-Instruct (81.3%).
- **Truth:** google/paligemma2-3b-mix-224 (97.2%).
- **Counting:** google/paligemma2-3b-mix-224 (78.0%).
- **Description:** google/paligemma2-3b-mix-224, OpenGVLab/InternVL3_5-8B-Instruct (69.2%).
- **Affordance:** google/paligemma2-3b-mix-224, OpenGVLab/InternVL3_5-8B-Instruct, OpenGVLab/InternVL3_5-8B-MPO, OpenGVLab/InternVL3_5-8B (8.0%).
- **Negation:** allenai/Molmo2-8B (80.0%).
- **GRAB-Lite:** bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M (16.2%).

## GRAB-Lite group scores

| Model | Function | Properties | Real figures | Series | Transformations |
|---|---:|---:|---:|---:|---:|
| allenai/Molmo2-8B | 13.0% | 19.0% | 16.0% | 15.0% | 6.0% |
| google/paligemma2-3b-mix-224 | 2.0% | 3.0% | 4.0% | 3.0% | 1.0% |
| microsoft/Florence-2-large-ft | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| OpenGVLab/InternVL3_5-8B-Instruct | 11.0% | 22.0% | 20.0% | 5.0% | 8.0% |
| OpenGVLab/InternVL3_5-8B-MPO | 11.0% | 23.0% | 18.0% | 4.0% | 8.0% |
| OpenGVLab/InternVL3_5-8B | 14.0% | 25.0% | 18.0% | 7.0% | 9.0% |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | 9.0% | 31.0% | 20.0% | 10.0% | 11.0% |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | 15.0% | 20.0% | 17.0% | 8.0% | 14.0% |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | 11.0% | 25.0% | 15.0% | 7.0% | 11.0% |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | 6.0% | 8.0% | 9.0% | 7.0% | 5.0% |

## Real-image preflight

Each model must produce a nonempty TextVQA answer and a parseable AI2D answer before any full task begins. Passing examples are reused in the scored task results.
| Model | Result | AI2D format | AI2D parse error | AI2D output tokens | TextVQA response | TextVQA parse error | TextVQA output tokens |
|---|---|---|---|---:|---|---|---:|
| allenai/Molmo2-8B | passed | valid | False | 2 | nonempty | False | 4 |
| google/paligemma2-3b-mix-224 | passed | valid | False | 2 | nonempty | False | 2 |
| microsoft/Florence-2-large-ft | passed | valid | False | 9 | nonempty | False | 8 |
| OpenGVLab/InternVL3_5-8B-Instruct | passed | valid | False | 1 | nonempty | False | 2 |
| OpenGVLab/InternVL3_5-8B-MPO | passed | valid | False | 1 | nonempty | False | 2 |
| OpenGVLab/InternVL3_5-8B | passed | valid | False | 1 | nonempty | False | 2 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | passed | valid | False | 2 | nonempty | False | 4 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | passed | valid | False | 2 | nonempty | False | 4 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | passed | valid | False | 2 | nonempty | False | 4 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | passed | valid | False | 2 | nonempty | False | 4 |

## Runtime and resources

Peak memory is measured per model process. Transformers use PyTorch allocated VRAM; llama.cpp uses sampled nvidia-smi process VRAM.
| Model | Task | Completed / requested | Mean seconds | Tokens / second | Peak measured GiB | Parse errors |
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
| OpenGVLab/InternVL3_5-8B-Instruct | AI2D | 100/100 | 0.36 | 3.80 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | TextVQA / OCR | 100/100 | 0.35 | 9.59 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | Truth | 25/25 | 0.29 | 3.44 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | Counting | 25/25 | 0.29 | 4.03 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | Description | 25/25 | 0.25 | 6.51 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | Affordance | 25/25 | 0.34 | 4.93 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | Negation | 25/25 | 0.30 | 3.72 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-Instruct | GRAB-Lite | 500/500 | 0.43 | 6.10 | 16.77 | 0 |
| **OpenGVLab/InternVL3_5-8B-Instruct peak** | GRAB-Lite | — | — | — | 16.77 | — |
| OpenGVLab/InternVL3_5-8B-MPO | AI2D | 100/100 | 0.29 | 5.49 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | TextVQA / OCR | 100/100 | 0.36 | 9.13 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | Truth | 25/25 | 0.29 | 3.39 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | Counting | 25/25 | 0.29 | 3.72 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | Description | 25/25 | 0.26 | 6.47 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | Affordance | 25/25 | 0.35 | 4.69 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | Negation | 25/25 | 0.30 | 3.73 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B-MPO | GRAB-Lite | 500/500 | 0.44 | 5.95 | 16.77 | 0 |
| **OpenGVLab/InternVL3_5-8B-MPO peak** | GRAB-Lite | — | — | — | 16.77 | — |
| OpenGVLab/InternVL3_5-8B | AI2D | 100/100 | 0.34 | 5.34 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | TextVQA / OCR | 100/100 | 0.37 | 8.88 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | Truth | 25/25 | 0.31 | 3.25 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | Counting | 25/25 | 0.30 | 3.62 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | Description | 25/25 | 0.27 | 6.16 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | Affordance | 25/25 | 0.35 | 4.62 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | Negation | 25/25 | 0.31 | 3.18 | 16.77 | 0 |
| OpenGVLab/InternVL3_5-8B | GRAB-Lite | 500/500 | 0.44 | 5.88 | 16.77 | 0 |
| **OpenGVLab/InternVL3_5-8B peak** | GRAB-Lite | — | — | — | 16.77 | — |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | AI2D | 100/100 | 0.49 | 4.12 | 19.48 | 2 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | TextVQA / OCR | 100/100 | 1.18 | 3.98 | 19.48 | 0 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | Truth | 25/25 | 0.54 | 11.07 | 19.48 | 0 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | Counting | 25/25 | 0.48 | 4.84 | 19.48 | 0 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | Description | 25/25 | 0.45 | 7.93 | 19.48 | 0 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | Affordance | 25/25 | 1.15 | 2.57 | 19.48 | 0 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | Negation | 25/25 | 0.85 | 27.12 | 19.48 | 1 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | GRAB-Lite | 500/500 | 2.18 | 7.29 | 19.48 | 0 |
| **bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M peak** | TextVQA / OCR | — | — | — | 19.48 | — |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | AI2D | 100/100 | 0.47 | 15.44 | 10.74 | 1 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | TextVQA / OCR | 100/100 | 1.09 | 4.62 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | Truth | 25/25 | 0.50 | 9.82 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | Counting | 25/25 | 0.44 | 7.47 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | Description | 25/25 | 0.41 | 8.02 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | Affordance | 25/25 | 1.04 | 2.95 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | Negation | 25/25 | 0.83 | 14.26 | 10.74 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | GRAB-Lite | 500/500 | 3.42 | 51.71 | 10.74 | 3 |
| **bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M peak** | GRAB-Lite | — | — | — | 10.74 | — |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | AI2D | 100/100 | 0.45 | 6.77 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | TextVQA / OCR | 100/100 | 1.11 | 4.59 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | Truth | 25/25 | 0.52 | 9.36 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | Counting | 25/25 | 0.45 | 7.61 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | Description | 25/25 | 0.42 | 8.24 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | Affordance | 25/25 | 1.05 | 2.94 | 16.61 | 0 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | Negation | 25/25 | 0.84 | 14.51 | 16.61 | 1 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | GRAB-Lite | 500/500 | 4.16 | 43.72 | 16.61 | 3 |
| **bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 peak** | AI2D | — | — | — | 16.61 | — |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | AI2D | 100/100 | 0.32 | 7.58 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | TextVQA / OCR | 100/100 | 0.37 | 14.99 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | Truth | 25/25 | 0.37 | 16.18 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | Counting | 25/25 | 0.40 | 18.65 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | Description | 25/25 | 0.39 | 17.66 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | Affordance | 25/25 | 1.02 | 38.49 | 7.61 | 0 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | Negation | 25/25 | 0.76 | 34.42 | 7.61 | 2 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | GRAB-Lite | 500/500 | 1.29 | 39.68 | 7.61 | 3 |
| **unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit peak** | AI2D | — | — | — | 7.61 | — |

## Load-time memory

Load-time measurements are recorded separately from per-task generation. The sampled process peak uses nvidia-smi and can miss short peaks.
| Model | Allocated GiB | Reserved GiB | Sampled process peak GiB | Samples |
|---|---:|---:|---:|---:|
| allenai/Molmo2-8B | — | — | — | — |
| google/paligemma2-3b-mix-224 | — | — | — | — |
| microsoft/Florence-2-large-ft | — | — | — | — |
| OpenGVLab/InternVL3_5-8B-Instruct | 15.89 | 15.89 | 17.55 | 926 |
| OpenGVLab/InternVL3_5-8B-MPO | 15.89 | 15.89 | 17.55 | 897 |
| OpenGVLab/InternVL3_5-8B | 15.89 | 15.92 | 17.65 | 1028 |
| bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M | — | — | 19.48 | 2642 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M | — | — | 10.74 | 3826 |
| bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0 | — | — | 16.61 | 4700 |
| unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit | 9.45 | 9.73 | 10.42 | 1572 |

## Method

Tasks reuse the prepared examples, prompts, sample indices, and scorers described in `../../qwenVL_Bench/README.md`.
AI2D uses 100 test examples; TextVQA uses 100 validation examples; the five additional VQA panel categories use 25 examples each; GRAB-Lite uses all 500 test examples.
Requests run serially with one image and greedy decoding, with a 256-token generation limit. Molmo2 uses its chat template and bfloat16 weights. PaliGemma uses `answer en`, bfloat16 weights, SDPA, and 224-pixel inputs. Florence uses a separate Transformers 4.44.2 environment, float16 weights, eager attention, and the experimental `<VQA>` wrapper.
InternVL3.5 official 8B checkpoints use the official custom chat interface with bfloat16 weights and FlashAttention. The 14B and 30B-A3B GGUF candidates use CUDA-enabled llama.cpp with their matching vision projector and FlashAttention; the logged layer placement confirms full GPU offload.
Llama 3.2 Vision uses the Mllama image-text template with the cached Unsloth bitsandbytes 4-bit checkpoint and bfloat16 compute. SDPA is used because the Transformers 4.57 FlashAttention integration is incompatible with MllamaVisionAttention in this runtime. The user confirmed authorization for this local, non-commercial academic functional comparison.
Florence's cached processor does not register `<VQA>` and post-processing falls back to plain text. It is scored only if its real-image preflights demonstrate parseable AI2D output and a nonempty TextVQA answer.
For Florence AI2D and negation choice questions, exact generated option text is mapped to the displayed label before shared scoring; raw generated text is retained in per-item results.

## Provenance and limitations

- **allenai/Molmo2-8B:** complete; Transformers 4.57.1; checkpoint weights 32.27 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/molmo2_8b/report.md`.
- **google/paligemma2-3b-mix-224:** complete; Transformers 4.57.1; checkpoint weights 5.65 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/paligemma2_mix_224/report.md`.
- **microsoft/Florence-2-large-ft:** complete; Transformers 4.44.2; checkpoint weights 1.44 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/florence2_large_ft/report.md`.
- **OpenGVLab/InternVL3_5-8B-Instruct:** complete; Transformers 4.57.1; checkpoint weights 15.89 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_8b_instruct/report.md`.
- **OpenGVLab/InternVL3_5-8B-MPO:** complete; Transformers 4.57.1; checkpoint weights 15.89 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_8b_mpo/report.md`.
- **OpenGVLab/InternVL3_5-8B:** complete; Transformers 4.57.1; checkpoint weights 15.89 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_8b_cascaderl/report.md`.
- **bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M:** complete; Transformers 4.57.1; checkpoint weights 17.94 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_30b_a3b_q4_k_m_gguf/report.md`.
- **bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M:** complete; Transformers 4.57.1; checkpoint weights 9.04 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_14b_q4_k_m_gguf/report.md`.
- **bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0:** complete; Transformers 4.57.1; checkpoint weights 15.28 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/internvl35_14b_q8_0_gguf/report.md`.
- **unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit:** complete; Transformers 4.57.1; checkpoint weights 6.69 GiB; cache status ready. nonempty TextVQA response and valid AI2D answer format on real shared examples
  Source run: `../runs/20261103-133909-internvl35-candidates/llama3_2_11b_vision_bnb_4bit/report.md`.

Raw per-item outputs and errors remain in each model's `results.json`. The report site publishes aggregate metrics only.
