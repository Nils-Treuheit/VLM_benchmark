# Phi vision benchmark

This benchmark compares the cached Phi-3.5 Vision, Phi-3 Vision, Phi-4 Multimodal, and converted Phi-3.5 checkpoints against the shared image-task suite.

Run the fixed suite with `./run_benchmark.sh --resume`. The launcher releases managed engines, selects the compatible Phi runtime, records all task results, and refreshes the report-site data bundle.

The runner uses fixed prompts and scorers from `qwenVL_Bench`. It runs one image request at a time and records incomplete cache entries instead of silently omitting them. See [report.md](results/phi_family/report.md) for the full measurements and [the local report page](report-site/index.html) for interactive summaries.

Phi-3 remote model code needs the compatibility patch in `benchmark.py`. The tested run uses Transformers 4.57.1 for that legacy code. Phi-4 still-image evaluation uses its cached processor and requires `peft`, `scipy`, and `backoff`.

The runner expects the shared prepared VQA and GRAB-Lite data under the repository's `data/` directory. It resolves checkpoints from the local Hugging Face cache and does not download weights or datasets.

Serve the completed page locally with `python3 -m http.server 8174 --directory report-site` from this directory.
