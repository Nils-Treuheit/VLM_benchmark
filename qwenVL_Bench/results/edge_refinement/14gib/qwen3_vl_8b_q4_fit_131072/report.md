# Selective Edge run · 14.0 GiB whole-device cap

Generated: 2026-11-05 08:54:43 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 0.825 GiB  
Cap enforcement: per-model; see each model record

Only cap-verified complete runs qualify for the root Edge Limited rankings. Scores below are cap-specific.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | unverified_cap | None | 14.27 GiB | 14.67 GiB | 82.0% (100/100) | 77.1% (100/100) | 97.2% (25/25) | 76.4% (25/25) | 58.8% (25/25) | 4.0% (25/25) | 72.0% (25/25) | 13.8% (500/500) |

## Verification

### unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL

Cap verified: `False`. Probe passed: `False`. Complete suite: `True`. Scoring peak within cap: `False`.

Context probe image: AI2D index 1717; actual input None tokens; output None tokens; configured limit reached `False`.

