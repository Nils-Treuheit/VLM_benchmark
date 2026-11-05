# Cosmos VLM benchmark

**Updated:** 2026-11-01T05:32:21+01:00  
**Status:** 5 complete; 0 incomplete; 0 pending; 0 unsupported  
**Runtime:** Python 3.13.13, PyTorch 2.11.0+cu130, Transformers 4.57.6, qwen-vl-utils None, HQQ None  
**Execution:** offline cached Transformers and managed vLLM CUDA 13, serial models and requests; 30.0 GiB per-process target

## Summary

5 checkpoints completed all eight shared task groups. Each completed model scored 825 examples across the eight task groups. All configured checkpoints completed.

The overall score is the unweighted macro average of AI2D, TextVQA, the five additional VQA panel categories, and GRAB-Lite. It is a local comparison score, not an official leaderboard metric. Among complete models, `nvidia/Cosmos-Reason1-7B` has the highest overall macro score. Models with incomplete coverage are excluded from comparisons.

## Edge refinement

| Whole-device cap | Model | Status | Full-suite macro | Measured context | Probe peak | Scoring peak | Whole-run peak |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 GiB | nvidia/Cosmos3-Edge | verified | 52.59% | 23,552 tokens | 9.82 GiB | 9.82 GiB | 9.82 GiB |
| 12 GiB | nvidia/Cosmos3-Edge | verified | 52.59% | 37,888 tokens | 11.89 GiB | 11.61 GiB | 11.89 GiB |

The 10 and 12 GiB verified runs each completed all 825 shared examples. See the [10 GiB report](results/edge_refinement/10gib/context23552_retry_20261104/report.md), [10 GiB raw artifact](results/edge_refinement/10gib/context23552_retry_20261104/results.json), [12 GiB report](results/edge_refinement/12gib/context37888_retry_20261105/report.md), and [12 GiB raw artifact](results/edge_refinement/12gib/context37888_retry_20261105/results.json). Failed over-cap attempts remain preserved and excluded: the 10 GiB attempt in `results/edge_refinement/10gib/uv_retry_20261104/`, and the 12 GiB probes in `results/edge_refinement/12gib/contextsearch_20261105/` and `results/edge_refinement/12gib/context44032_retry_20261105/`.

| Model | Overall macro | VQA panel macro | Completed examples | Peak whole-GPU memory | Status |
|---|---:|---:|---:|---:|---|
| `nvidia/Cosmos-Reason1-7B` | 59.6% | 64.5% | 825/825 | 22.89 GiB | Complete |
| `nvidia/Cosmos3-Nano` | 57.0% | 61.2% | 825/825 | 31.07 GiB | Complete |
| `nvidia/Cosmos-Reason2-8B` | 54.7% | 58.3% | 825/825 | 23.82 GiB | Complete |
| `nvidia/Cosmos3-Edge` | 52.6% | 58.6% | 825/825 | 9.40 GiB | Complete |
| `nvidia/Cosmos-Reason2-2B` | 50.0% | 54.5% | 825/825 | 10.69 GiB | Complete |

## Quality results

Scores use the shared fixed sample IDs selected with seed 3407. Every completed model has full coverage for each listed task.

| Model | AI2D (100) | TextVQA (100) | Truth (25) | Counting (25) | Description (25) | Affordance (25) | Negation (25) | GRAB-Lite (500) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nvidia/Cosmos-Reason1-7B` | 79.0% | 76.8% | 87.6% | 82.8% | 52.0% | 12.0% | 76.0% | 10.2% |
| `nvidia/Cosmos3-Nano` | 80.0% | 79.3% | 90.4% | 67.6% | 61.6% | 4.0% | 64.0% | 9.0% |
| `nvidia/Cosmos-Reason2-8B` | 75.0% | 78.2% | 94.4% | 63.2% | 46.0% | 0.0% | 68.0% | 12.8% |
| `nvidia/Cosmos3-Edge` | 63.0% | 74.1% | 90.4% | 73.6% | 53.2% | 16.0% | 44.0% | 6.4% |
| `nvidia/Cosmos-Reason2-2B` | 66.0% | 70.6% | 92.8% | 55.6% | 52.0% | 12.0% | 44.0% | 7.2% |

### Best complete model by task

| Task | Highest score | Model |
|---|---:|---|
| AI2D test | 80.0% | `nvidia/Cosmos3-Nano` |
| TextVQA validation | 79.3% | `nvidia/Cosmos3-Nano` |
| VQA truth | 94.4% | `nvidia/Cosmos-Reason2-8B` |
| VQA counting | 82.8% | `nvidia/Cosmos-Reason1-7B` |
| VQA description | 61.6% | `nvidia/Cosmos3-Nano` |
| VQA affordance | 16.0% | `nvidia/Cosmos3-Edge` |
| VQA negation | 76.0% | `nvidia/Cosmos-Reason1-7B` |
| GRAB-Lite graph analysis | 12.8% | `nvidia/Cosmos-Reason2-8B` |

## Runtime and resources

Latency is mean generation-call time per example. Throughput is generated tokens divided by generation-call time. Peak memory is whole-device `nvidia-smi` usage sampled during each task, including desktop use; it is not process-only VRAM. The benchmark used a 30 GiB per-process allocation target.

| Model | Task | Completed | Mean latency (s) | Output tokens/s | Peak whole-GPU (GiB) |
|---|---|---:|---:|---:|---:|
| `nvidia/Cosmos-Reason1-7B` | AI2D test | 100/100 | 0.174 | 11.5 | 18.45 |
| `nvidia/Cosmos-Reason1-7B` | TextVQA validation | 100/100 | 0.264 | 18.9 | 18.45 |
| `nvidia/Cosmos-Reason1-7B` | VQA truth | 25/25 | 0.149 | 13.4 | 17.46 |
| `nvidia/Cosmos-Reason1-7B` | VQA counting | 25/25 | 0.148 | 14.0 | 17.46 |
| `nvidia/Cosmos-Reason1-7B` | VQA description | 25/25 | 0.158 | 18.2 | 17.46 |
| `nvidia/Cosmos-Reason1-7B` | VQA affordance | 25/25 | 0.250 | 16.0 | 17.74 |
| `nvidia/Cosmos-Reason1-7B` | VQA negation | 25/25 | 0.220 | 9.1 | 18.42 |
| `nvidia/Cosmos-Reason1-7B` | GRAB-Lite graph analysis | 500/500 | 1.613 | 2.5 | 22.89 |
| `nvidia/Cosmos3-Nano` | AI2D test | 100/100 | 0.087 | 23.0 | 27.81 |
| `nvidia/Cosmos3-Nano` | TextVQA validation | 100/100 | 0.173 | 25.5 | 27.82 |
| `nvidia/Cosmos3-Nano` | VQA truth | 25/25 | 0.070 | 28.5 | 27.82 |
| `nvidia/Cosmos3-Nano` | VQA counting | 25/25 | 0.070 | 29.8 | 27.82 |
| `nvidia/Cosmos3-Nano` | VQA description | 25/25 | 0.081 | 38.4 | 27.82 |
| `nvidia/Cosmos3-Nano` | VQA affordance | 25/25 | 0.155 | 17.8 | 27.82 |
| `nvidia/Cosmos3-Nano` | VQA negation | 25/25 | 0.147 | 13.6 | 28.10 |
| `nvidia/Cosmos3-Nano` | GRAB-Lite graph analysis | 500/500 | 2.172 | 1.7 | 31.07 |
| `nvidia/Cosmos-Reason2-8B` | AI2D test | 100/100 | 0.165 | 12.1 | 18.76 |
| `nvidia/Cosmos-Reason2-8B` | TextVQA validation | 100/100 | 0.238 | 18.9 | 18.76 |
| `nvidia/Cosmos-Reason2-8B` | VQA truth | 25/25 | 0.148 | 13.5 | 18.20 |
| `nvidia/Cosmos-Reason2-8B` | VQA counting | 25/25 | 0.148 | 13.8 | 18.20 |
| `nvidia/Cosmos-Reason2-8B` | VQA description | 25/25 | 0.169 | 19.8 | 18.22 |
| `nvidia/Cosmos-Reason2-8B` | VQA affordance | 25/25 | 0.203 | 13.6 | 18.55 |
| `nvidia/Cosmos-Reason2-8B` | VQA negation | 25/25 | 0.202 | 9.9 | 19.09 |
| `nvidia/Cosmos-Reason2-8B` | GRAB-Lite graph analysis | 500/500 | 1.705 | 2.2 | 23.82 |
| `nvidia/Cosmos3-Edge` | AI2D test | 100/100 | 0.110 | 18.2 | 6.41 |
| `nvidia/Cosmos3-Edge` | TextVQA validation | 100/100 | 0.146 | 29.2 | 6.41 |
| `nvidia/Cosmos3-Edge` | VQA truth | 25/25 | 0.106 | 18.9 | 6.19 |
| `nvidia/Cosmos3-Edge` | VQA counting | 25/25 | 0.104 | 19.9 | 6.19 |
| `nvidia/Cosmos3-Edge` | VQA description | 25/25 | 0.117 | 26.6 | 6.21 |
| `nvidia/Cosmos3-Edge` | VQA affordance | 25/25 | 0.126 | 24.1 | 6.32 |
| `nvidia/Cosmos3-Edge` | VQA negation | 25/25 | 0.128 | 15.6 | 6.67 |
| `nvidia/Cosmos3-Edge` | GRAB-Lite graph analysis | 500/500 | 1.110 | 4.0 | 9.40 |
| `nvidia/Cosmos-Reason2-2B` | AI2D test | 100/100 | 0.113 | 17.7 | 7.35 |
| `nvidia/Cosmos-Reason2-2B` | TextVQA validation | 100/100 | 0.145 | 32.1 | 7.35 |
| `nvidia/Cosmos-Reason2-2B` | VQA truth | 25/25 | 0.109 | 18.6 | 7.19 |
| `nvidia/Cosmos-Reason2-2B` | VQA counting | 25/25 | 0.109 | 19.1 | 7.19 |
| `nvidia/Cosmos-Reason2-2B` | VQA description | 25/25 | 0.126 | 26.1 | 7.19 |
| `nvidia/Cosmos-Reason2-2B` | VQA affordance | 25/25 | 0.133 | 20.5 | 7.19 |
| `nvidia/Cosmos-Reason2-2B` | VQA negation | 25/25 | 0.130 | 15.3 | 7.64 |
| `nvidia/Cosmos-Reason2-2B` | GRAB-Lite graph analysis | 500/500 | 0.849 | 4.1 | 10.69 |

## Context and hardware fit

Advertised context comes from the cached checkpoint configuration. The run-context column shows explicit server caps where configured. Cached weight size sums local safetensors shards. Model-load peak is this process's CUDA allocation; task peak is whole-device GPU use.

| Model | Advertised context | Run context | Cached weights | Model-load peak allocated | Peak whole GPU |
|---|---:|---|---:|---:|---:|
| `nvidia/Cosmos-Reason1-7B` | 128,000 | Checkpoint default | 15.445 GiB | 15.48 GiB | 22.89 GiB |
| `nvidia/Cosmos3-Nano` | 262,144 | 16,384 tokens | 32.498 GiB | — GiB | 31.07 GiB |
| `nvidia/Cosmos-Reason2-8B` | 262,144 | Checkpoint default | 16.330 GiB | 16.37 GiB | 23.82 GiB |
| `nvidia/Cosmos3-Edge` | 131,072 | Checkpoint default | 8.501 GiB | 4.57 GiB | 9.40 GiB |
| `nvidia/Cosmos-Reason2-2B` | 262,144 | Checkpoint default | 4.542 GiB | 4.87 GiB | 10.69 GiB |

## GRAB-Lite task groups

The overall score is exact-match accuracy across all 500 questions. Group scores below use the five prepared task categories.

| Model | Properties | Function | Series | Transform | Real figures |
|---|---:|---:|---:|---:|---:|
| `nvidia/Cosmos-Reason1-7B` | 15.0% | 10.0% | 9.0% | 4.0% | 13.0% |
| `nvidia/Cosmos3-Nano` | 13.0% | 9.0% | 6.0% | 4.0% | 13.0% |
| `nvidia/Cosmos-Reason2-8B` | 21.0% | 11.0% | 8.0% | 5.0% | 19.0% |
| `nvidia/Cosmos3-Edge` | 8.0% | 7.0% | 6.0% | 3.0% | 8.0% |
| `nvidia/Cosmos-Reason2-2B` | 8.0% | 5.0% | 5.0% | 5.0% | 13.0% |

## Methodology

- **AI2D:** cached test split, 100 fixed examples, exact A/B/C/D answer accuracy. [Dataset card](https://huggingface.co/datasets/lmms-lab/ai2d).
- **TextVQA:** cached validation split, 100 fixed examples, normalized human-answer consensus score. [Dataset card](https://huggingface.co/datasets/facebook/textvqa).
- **VQA panel:** 25 fixed examples each for truth, counting, visual description, affordance/function, and negation; the same 100 TextVQA examples provide OCR. Panel macro averages these six categories equally. It is a local metric. Sources: [VQAv2](https://visualqa.org/download.html), [PercepTax](https://huggingface.co/datasets/RyanWW/PercepTaxBench), and [NegVQA](https://arxiv.org/abs/2505.22946).
- **GRAB-Lite:** complete 500-question fixed split, normalized exact match, with five category scores. [Dataset card](https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite).
- **Cosmos Reason1 image support:** the official card lists image and video inputs and documents the Qwen2.5-VL architecture. This run loads its Qwen2.5-VL conditional-generation class. [Model card](https://huggingface.co/nvidia/Cosmos-Reason1-7B).
- **Cosmos3 Nano support route:** the official card documents vLLM 0.21.0 for CUDA 13 or 0.19.1 for CUDA 12.8, plus the `vllm-cosmos3` plugin and a reasoner architecture override. This run used 16,384-token context, one sequence, and 88% GPU memory utilization to leave headroom for full-resolution vision activations on the RTX 5090. [Model card](https://huggingface.co/nvidia/Cosmos3-Nano).
- Requests were serial and single-image, using the shared task prompts, scorers, prepared panel data, and seed 3407. Dataset Arrow files and cached model snapshots were read offline.
- AI2D and open-answer tasks used a 256-token cap except Cosmos Reason1 and Cosmos3 Nano, which used a 4096-token cap as recommended by their model cards. No Reason1 output reached that cap. Reason2 used checkpoint generation defaults with deterministic seeds. Cosmos3 Edge used `enable_thinking=False` in its cached chat template so outputs follow the shared answer-only contract; the same deterministic seed and 256-token limit were used. No output reached the cap in the completed Cosmos3 Edge run.
- The Transformers cache used HQQ 8-bit KV storage where supported. The cached model revisions are recorded in `results.json`.

## Coverage and limitations

| Checkpoint | Cached revision | License label | Coverage status |
|---|---|---|---|
| `nvidia/Cosmos-Reason1-7B` | `375e24000b24baed78f4618d3dd779e47cd96323` | NVIDIA Open Model License | complete: all eight task groups completed |
| `nvidia/Cosmos3-Nano` | `e59a53c25979a090fa8706c9acc0c254a6e89b92` | OpenMDW1.1 | complete: all eight task groups completed |
| `nvidia/Cosmos-Reason2-8B` | `a9fae2cf89dc64db96b12860417f0eb403013bb9` | NVIDIA Open Model License | complete: all eight task groups completed |
| `nvidia/Cosmos3-Edge` | `344d602b128d1bbdacb43b08d0a3626f46343e29` | OpenMDW1.1 | complete: all eight task groups completed |
| `nvidia/Cosmos-Reason2-2B` | `9ce19a195e423419c349abfc86fd07178b230561` | NVIDIA Open Model License | complete: all eight task groups completed |

- **Cosmos3 Nano run note:** The managed vLLM endpoint stopped responding during the initial GRAB-Lite pass. Failed fixed-seed graph requests were retried in short controller-managed CUDA 13 sessions. All 825 selected examples ultimately completed. The profile used 16,384-token context, one sequence, and 88% GPU memory utilization.
- Cosmos3 Edge ran with reasoning disabled through a supported chat-template option. Its scores are therefore for answer-only decoding and are not directly equivalent to a default-thinking run.
- Results are local fixed-sample comparisons, not broad accuracy estimates or leaderboard scores. No model-load, licensing acceptance, or context-capacity claims are inferred from cache presence.

## Artifacts

- [Aggregate-only report page](index.html)
- Authoritative per-item results, errors, latencies, token counts, and memory observations: [`results/results.json`](results/results.json).
- Aggregate site payload: [`data.json`](data.json).
- Benchmark runner: [`run_benchmark.py`](run_benchmark.py).
