# Image SGG benchmark

This benchmark evaluates direct image-to-scene-graph generation on three fixed
200-image panels. It reports VG-150 canonical test, Open Images V7 validation
relationships, and **PSG box-adapted SGG** separately. PSG's box score is not
the official mask-grounded metric.

Prepare panels offline from the local datasets:

```bash
uv run python qwenVL_Bench/image_SGG_Bench/prepare_data.py \
  --dataset all \
  --data-root /mnt/HDD1/Project_Data/public_datasets \
  --output-dir qwenVL_Bench/image_SGG_Bench/data \
  --seed 3407 --audit
```

The preparer writes manifests with IDs and source paths. It does not copy source
images. The VG audit writes temporary overlays for twelve deterministic panel
images for manual inspection.

VG-150 preparation keeps boxes strictly inside the source image bounds without
clipping. The current source artifact excludes 12,005 test rows with boxes
outside those bounds; the manifest records this subset limitation.

Run the cached Qwen VL entries directly on the host GPU, without the benchmark
queue or model-control service:

```bash
./.venv/bin/python qwenVL_Bench/image_SGG_Bench/runner.py \
  --direct-host --dataset all --wait-for-gpu
```

Pass `--list-models` to inspect the cached inventory or repeat `--model MODEL_ID`
to select exact model entries. Direct-host mode checks GPU 0 with `nvidia-smi`
and CUDA, enforces a 30 GiB whole-device limit, runs batch size one, and never
downloads weights. Each model runs in a fresh process. GGUF entries launch
llama.cpp as an owned child, which stops when the model finishes. Transformers
entries run first through the existing loader; GGUF entries follow.
Deterministic decoding allows up to
8192 output tokens to reduce truncation under the 50-object/100-relation caps.
With `--wait-for-gpu`, it checks readiness every 60 seconds by default; set
`--wait-interval-seconds N` to change the interval. The direct host mode waits
until the whole-device free-memory budget is available without calling
model-control endpoints.
If a host session ends unexpectedly, rerun with `--resume-incomplete`; it
validates model, manifest, prompt, evaluator, and runtime provenance before
continuing an incomplete GGUF record.

GGUF runs leave GPU-layer placement automatic so llama.cpp can fit the model
under the recorded cap. Speculative MTP is disabled because cached GGUF variants
do not all include draft layers. The direct runner writes each server log beside
that model's result record. llama.cpp uses a 32,768-token context so the 72B
quant's KV cache can fit within the same VRAM cap. It sets a 1,024-token
minimum image resolution for Qwen-VL grounding tasks.
Item outputs and raw responses remain under the ignored `results/` directory.
Use `--max-items N` for a partial diagnostic run; the aggregator rejects runs
that do not contain every image in each fixed panel.
The per-item VRAM field is a whole-device maximum sampled every 250 ms. Brief
allocation peaks may fall between samples. Item records distinguish source
image dimensions from the backend's processor output, visual grid, and configured
pixel limits when exposed. llama.cpp records its PNG input and notes that its
chat API does not expose mmproj crop dimensions or max-pixel settings.

Aggregate only runs that match the panel, vocabulary, prompt, evaluator, and
source provenance:

```bash
uv run python qwenVL_Bench/image_SGG_Bench/aggregate.py
```

The Markdown report contains aggregate metrics only. Per-item records retain
failures in the denominator and preserve available timing, token, memory, and
model revision metadata.
