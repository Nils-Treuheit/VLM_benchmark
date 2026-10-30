# Corrected Qwen VL Benchmark

Generated 2026-10-28 17:44:35 +0100 with 1 fixed examples per task.
All calls are serial, single image, single request, and at most 256 output tokens (1,024 for Thinking).

## Task scores and deltas

| Model | Runner | AI2D | Δ AI2D | TextVQA | Δ TextVQA | Status |
|---|---|---:|---:|---:|---:|---|
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | llama.cpp (UD-Q4_K_XL) | 100.0% | +0.0 pp | 100.0% | +0.0 pp | complete |

## Per task VRAM and speed

| Model | Task | Completed | Peak GPU used | Mean seconds/example | Generated tokens/second | Context | KV cache |
|---|---|---:|---:|---:|---:|---:|---|
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | ai2d | 1/1 | 19.997 GiB | 20.6078 | 0.24262702332549216 | 262144 | q8_0 K and V |
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | textvqa | 1/1 | 19.997 GiB | 0.1369 | 29.220150964151056 | 262144 | q8_0 K and V |

## VRAM and context configuration

| Model | Min / default / max context | Peak resident GPU memory | MTP | Runner |
|---|---|---:|---|---|
| unsloth/Qwen3-VL-2B-Instruct-GGUF:UD-Q4_K_XL | 8K / 128K / 256K | 19.997 GiB | no (startup did not accept draft-mtp) | llama.cpp (UD-Q4_K_XL) |

## Notes

- Unsloth GGUF checkpoints use the llama.cpp server and its Q8_0 K/V cache; MTP is requested on Qwen3-VL models and the server log records whether a model head was found.
- Original Qwen and G-Substrate snapshots follow their model-card Transformers loading path and use the HQQ 8-bit KV cache where supported.
- Each task uses the same deterministic sample IDs across models. Results are small-sample selection evidence, not leaderboard scores.
- VRAM is peak whole-GPU usage sampled during the task, including the desktop baseline.
- The third table reports the largest server context accepted under the 30 GiB fit target; it is not a claim that the dataset filled that context.
- G-Substrate is a full SFT fine-tune of Qwen3-VL-2B-Instruct trained on scene/event/molecular graphs and graph reasoning, with a unified graph schema across modalities.
