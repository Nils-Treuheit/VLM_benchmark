# Selective Edge run · 12.0 GiB whole-device cap

Generated: 2026-11-05 01:52:31 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 0.825 GiB  
Cap enforcement: torch per-process CUDA memory fraction capped to the whole-device limit minus measured baseline and 0.5 GiB reserve; cap qualification uses sampled whole-device nvidia-smi peaks

Only cap-verified complete runs qualify for the root Edge Limited rankings. Scores below are cap-specific.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/Cosmos3-Edge | complete | 37888 | 11.89 GiB | 11.61 GiB | 63.0% (100/100) | 74.1% (100/100) | 90.4% (25/25) | 73.6% (25/25) | 53.2% (25/25) | 16.0% (25/25) | 44.0% (25/25) | 6.4% (500/500) |

## Verification

### nvidia/Cosmos3-Edge

Cap verified: `True`. Probe passed: `True`. Complete suite: `True`. Scoring peak within cap: `True`.

Context probe image: AI2D index 1717; actual input 37880 tokens; output 8 tokens; configured limit reached `True`.

