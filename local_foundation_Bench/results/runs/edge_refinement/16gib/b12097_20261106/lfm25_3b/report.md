# unsloth/LFM2.5-VL-3B-GGUF Edge run (16 GiB)

Generated: 2026-11-06 11:55:19 +0100
Status: complete; cap verified: True.
Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; revision: `22f063714556bd4daa81c40dc1ff15ca37ae92ea`; quantization: UD-Q4_K_XL + F16 mmproj.
GPU: NVIDIA GeForce RTX 5090; driver 595.99.02; idle baseline 0.7109375 GiB.
Measured context: 32768 tokens; configured limit reached: True.
Context image: AI2D index 1071, [625, 1500]; selection: highest llama.cpp prompt-token count across the fixed AI2D sample (100/100 measured).
Whole-device peaks: load 4.532 GiB; context probe 4.644 GiB; scoring 5.158 GiB; whole run 5.158 GiB.
Cap enforcement: llama.cpp --fit on --fit-target 16951 MiB; whole-device nvidia-smi sampling; llama.cpp fit target 16951 MiB.

## Shared task results

| Task | Completed / requested | Score | Peak whole-device GiB |
|---|---:|---:|---:|
| AI2D | 100/100 | 73.0% | 4.644 |
| TextVQA / OCR | 100/100 | 84.3% | 4.691 |
| Truth | 25/25 | 93.2% | 4.696 |
| Counting | 25/25 | 64.0% | 4.696 |
| Description | 25/25 | 67.6% | 4.677 |
| Affordance | 25/25 | 8.0% | 4.677 |
| Negation | 25/25 | 64.0% | 4.677 |
| GRAB-Lite | 500/500 | 13.4% | 4.781 |

Full-suite macro: 58.4%.
Peak task: GRAB-Lite.

## Verification

The initial runner output reported a 4.781 GiB whole-run peak but omitted the verified scoring-sampler maximum of 5.158 GiB. The whole-run peak is reconciled to 5.158 GiB; the original runner JSON is preserved as `results.runner_raw.json`. The runner aggregation now includes the scoring sampler.


```json
{
  "whole_device_cap_gib": 16,
  "context_probe_passed": true,
  "complete_shared_task_suite": true,
  "scoring_peak_within_cap": true,
  "whole_run_peak_within_cap": true,
  "context_fit_adjusted": false,
  "configured_limit_reached": true,
  "whole_device_sampling_verified": true,
  "scoring_sampling_verified": true,
  "task_sampling_verified": true
}
```

Source artifact: `results.json`.
