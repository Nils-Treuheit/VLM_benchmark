# Selective Edge run · 16.0 GiB whole-device cap

Generated: 2026-11-06 17:57:16 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 0.9 GiB  
Cap enforcement: per-model; see each model record

Only cap-verified complete runs qualify for the root Edge Limited rankings. Scores below are cap-specific.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL | complete | 262144 | 14.97 GiB | 15.75 GiB | 83.0% (100/100) | 78.1% (100/100) | 100.0% (25/25) | 76.4% (25/25) | 58.8% (25/25) | 4.0% (25/25) | 64.0% (25/25) | 15.0% (500/500) |

## Verification

### unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q8_K_XL

Cap verified: `True`. Probe passed: `True`. Complete suite: `True`. Scoring peak within cap: `True`.

Context probe image: AI2D index 1717; actual input 262143 tokens; output 1 tokens; configured limit reached `True`.

