# Selective Edge run · 14.0 GiB whole-device cap

Generated: 2026-11-05 10:12:14 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 1.009 GiB  
Cap enforcement: per-model; see each model record

Only cap-verified complete runs qualify for the root Edge Limited rankings. Scores below are cap-specific.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL | complete | 65536 | 12.90 GiB | 13.36 GiB | 82.0% (100/100) | 80.0% (100/100) | 97.2% (25/25) | 72.8% (25/25) | 58.8% (25/25) | 4.0% (25/25) | 76.0% (25/25) | 14.0% (500/500) |

## Verification

### unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL

Cap verified: `True`. Probe passed: `True`. Complete suite: `True`. Scoring peak within cap: `True`.

Context probe image: AI2D index 1717; actual input 65535 tokens; output 1 tokens; configured limit reached `True`.

