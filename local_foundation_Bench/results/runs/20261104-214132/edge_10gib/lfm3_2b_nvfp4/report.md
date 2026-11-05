# LiquidAI/LFM3-VL-2B-GGUF:NVFP4 Edge run (10 GiB)

Generated: 2026-11-04 21:41:32 +0100
Status: complete; cap verified: True.
Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; revision: `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`; quantization: NVFP4 feed-forward weights, Q8_0 other weights + F16 mmproj.
GPU: NVIDIA GeForce RTX 5090; driver 595.99.02; idle baseline 0.8447265625 GiB.
Measured context: 128000 tokens; configured limit reached: True.
Context image: AI2D index 1717, [1500, 1500]; selection: highest llama.cpp prompt-token count across the fixed AI2D sample (100/100 measured).
Whole-device peaks: load 5.007 GiB; context probe 5.204 GiB; scoring 5.607 GiB; whole run 5.693 GiB.
Cap enforcement: llama.cpp --fit on --fit-target 23232 MiB; whole-device nvidia-smi sampling; llama.cpp fit target 23232 MiB.

## Shared task results

| Task | Completed / requested | Score | Peak whole-device GiB |
|---|---:|---:|---:|
| AI2D | 100/100 | 92.0% | 5.204 |
| TextVQA / OCR | 100/100 | 78.5% | 5.2 |
| Truth | 25/25 | 54.4% | 5.188 |
| Counting | 25/25 | 75.6% | 5.691 |
| Description | 25/25 | 66.0% | 5.202 |
| Affordance | 25/25 | 4.0% | 5.202 |
| Negation | 25/25 | 64.0% | 5.442 |
| GRAB-Lite | 500/500 | 33.8% | 5.573 |

Full-suite macro: 58.5%.
Peak task: Counting.

## Verification

```json
{
  "whole_device_cap_gib": 10,
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
