# LiquidAI/LFM3-VL-2B-GGUF:NVFP4 Edge run (12 GiB)

Generated: 2026-11-05 03:15:08 +0100
Status: complete; cap verified: True.
Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; revision: `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`; quantization: NVFP4 feed-forward weights, Q8_0 other weights + F16 mmproj.
GPU: NVIDIA GeForce RTX 5090; driver 595.99.02; idle baseline 0.80859375 GiB.
Measured context: 128000 tokens; configured limit reached: True.
Context image: AI2D index 1717, [1500, 1500]; selection: highest llama.cpp prompt-token count across the fixed AI2D sample (100/100 measured).
Whole-device peaks: load 4.971 GiB; context probe 5.164 GiB; scoring 5.554 GiB; whole run 5.554 GiB.
Cap enforcement: llama.cpp --fit on --fit-target 21147 MiB; whole-device nvidia-smi sampling; llama.cpp fit target 21147 MiB.

## Shared task results

| Task | Completed / requested | Score | Peak whole-device GiB |
|---|---:|---:|---:|
| AI2D | 100/100 | 92.0% | 5.164 |
| TextVQA / OCR | 100/100 | 78.5% | 5.168 |
| Truth | 25/25 | 54.4% | 5.168 |
| Counting | 25/25 | 75.6% | 5.168 |
| Description | 25/25 | 66.0% | 5.168 |
| Affordance | 25/25 | 4.0% | 5.168 |
| Negation | 25/25 | 64.0% | 5.404 |
| GRAB-Lite | 500/500 | 33.8% | 5.554 |

Full-suite macro: 58.5%.
Peak task: GRAB-Lite.

## Verification

```json
{
  "whole_device_cap_gib": 12,
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
