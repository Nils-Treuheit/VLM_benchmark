# LiquidAI/LFM3-VL-2B-GGUF Edge run (8 GiB)

Generated: 2026-11-04 20:24:26 +0100
Status: complete; cap verified: True.
Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; revision: `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`; quantization: Q4_K_M + F16 mmproj.
GPU: NVIDIA GeForce RTX 5090; driver 595.99.02; idle baseline 0.8310546875 GiB.
Measured context: 128000 tokens; configured limit reached: True.
Context image: AI2D index 1717, [1500, 1500]; selection: highest llama.cpp prompt-token count across the fixed AI2D sample (100/100 measured).
Whole-device peaks: load 4.822 GiB; context probe 5.02 GiB; scoring 5.43 GiB; whole run 5.43 GiB.
Cap enforcement: llama.cpp --fit on --fit-target 25266 MiB; whole-device nvidia-smi sampling; llama.cpp fit target 25266 MiB.

## Shared task results

| Task | Completed / requested | Score | Peak whole-device GiB |
|---|---:|---:|---:|
| AI2D | 100/100 | 86.0% | 5.02 |
| TextVQA / OCR | 100/100 | 74.2% | 5.023 |
| Truth | 25/25 | 61.2% | 5.012 |
| Counting | 25/25 | 72.4% | 5.023 |
| Description | 25/25 | 60.4% | 5.023 |
| Affordance | 25/25 | 8.0% | 5.023 |
| Negation | 25/25 | 56.0% | 5.26 |
| GRAB-Lite | 500/500 | 27.6% | 5.428 |

Full-suite macro: 55.7%.
Peak task: GRAB-Lite.

## Verification

```json
{
  "whole_device_cap_gib": 8,
  "context_probe_passed": true,
  "complete_shared_task_suite": true,
  "scoring_peak_within_cap": true,
  "whole_run_peak_within_cap": true,
  "context_fit_adjusted": true,
  "configured_limit_reached": true,
  "whole_device_sampling_verified": true,
  "scoring_sampling_verified": true,
  "task_sampling_verified": true
}
```

Source artifact: `results.json`.
