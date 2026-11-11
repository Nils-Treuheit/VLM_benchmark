# Selective Edge run · 16.0 GiB whole-device cap

Generated: 2026-11-05 15:59:07 +0100  
GPU: NVIDIA GeForce RTX 5090 · driver 595.99.02  
Measured idle baseline: 1.005 GiB  
Status: verified at selected 58,368-token context; macro 52.59%; whole-device peak 15.907 GiB.

The completed full score run is reused. Its configured ceiling was 61,440 tokens, but all scored requests used no more than 12618 total tokens. No scored request exceeded the selected 58,368-token measured context.

| Model | Status | Context probe | Probe peak | Score peak | AI2D | TextVQA/OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvidia/Cosmos3-Edge | verified | 58,368 | 15.446 GiB | 15.907 GiB | 63.0% (100/100) | 74.1% (100/100) | 90.4% (25/25) | 73.6% (25/25) | 53.2% (25/25) | 16.0% (25/25) | 44.0% (25/25) | 6.4% (500/500) |

## Verification

- Selected original context-probe attempt: 58,368 tokens, 58,360 input plus 8 output, 31 samples, no sampling error. This is reused evidence, not a fresh lower-context probe process.
- Scored examples: 825/825 complete; largest actual input-plus-output context: 12618 tokens.
- Score-run peak: 15.907 GiB. Selected probe peak: 15.446 GiB. Combined verified peak: 15.907 GiB.
- The original broad context search included higher attempts that exceeded 16 GiB; those attempts remain in the original source artifact and are excluded from this selected-context qualification.
- Full scored artifact: `../results.json` (local artifact; not published).
