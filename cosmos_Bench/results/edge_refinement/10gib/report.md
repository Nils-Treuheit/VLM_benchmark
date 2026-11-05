# Selective Edge run · 10.0 GiB whole-device cap

Generated: 2026-11-04 22:45:30 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 0.811 GiB  
Cap enforcement: torch per-process CUDA memory fraction capped to the whole-device limit minus measured baseline and 0.5 GiB reserve; cap qualification uses sampled whole-device nvidia-smi peaks

Only cap-verified complete runs qualify for the root Edge Limited rankings. Scores below are cap-specific.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/Cosmos3-Edge | failed | — | — | — | — (0/0) | — (0/0) | — (0/0) | — (0/0) | — (0/0) | — (0/0) | — (0/0) | — (0/0) |

## Verification

### nvidia/Cosmos3-Edge

Cap verified: `False`. Probe passed: `False`. Complete suite: `False`. Scoring peak within cap: `False`.

Context probe image: AI2D index unknown; actual input unknown tokens; output unknown tokens; configured limit reached `False`.

