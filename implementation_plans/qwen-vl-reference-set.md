# Qwen-VL reference set implementation plan

## Purpose

Include a small, fixed Qwen-VL subset from the completed benchmark so results in the expanded benchmark remain directly comparable with the existing report. Do not rerun every cached Qwen model or quantization in the main benchmark.

## Recommended reference checkpoints

Use the corrected Qwen-VL report at `qwenVL_Bench/results/corrected/report.md` as the source of selection. Include these representative checkpoints:

- `Qwen/Qwen2.5-VL-7B-Instruct` as the previous-family Transformers baseline.
- `Qwen/Qwen3-VL-8B-Instruct` as the current-family Transformers baseline.
- `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL` as a high-quality compact GGUF point.
- `unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL` as a strong MoE point.
- `unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:UD-Q4_K_XL` as the reasoning-oriented MoE comparison, only if it fits the fixed memory cap.

The 30B thinking checkpoint is optional if it cannot fit. Use exact model identities and only include checkpoints with complete results from the existing benchmark. Other candidates can be selected if the local cache or current full-panel results show a better representative, but record the reason and update this plan first.

## Runner and prompts

Reuse the completed Qwen-VL benchmark runners and configuration exactly. Preserve the Qwen family sampling settings, final-answer extraction, GGUF cache configuration, context selection, and task prompts. Do not normalize model-specific answers differently from the existing benchmark.

## References

- Benchmark method, prompts, tasks, and run instructions: [qwenVL_Bench/README.md](../qwenVL_Bench/README.md).
- Corrected model scores and runtime records: [results/corrected/report.md](../qwenVL_Bench/results/corrected/report.md).
- GGUF and Transformers paths: [benchmark_runners.py](../qwenVL_Bench/benchmark_runners.py).

## Report site delivery

After completing the reference-set run and writing its `report.md`, keep its static report page and assets at the benchmark folder root, using the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Link the completed Qwen-VL benchmark for baseline context and avoid implying the small reference set replaces its full report.
