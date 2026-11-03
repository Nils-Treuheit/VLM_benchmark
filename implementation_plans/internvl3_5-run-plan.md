# InternVL 3.5 30B-A3B single-GPU run plan

## Goal and current evidence

Run the exact cached `OpenGVLab/InternVL3_5-30B-A3B` checkpoint on the RTX 5090
within the benchmark's 30 GiB process cap. Keep its checkpoint identity and
image preprocessing fixed. Do not publish a score until the real-image
preflight and all eight shared tasks complete.

The cached revision is
`ba1762d830e2aeddb3272b66ba9e0abf7e69f626`. It has all 13 weight shards and
about 57.46 GiB of BF16 checkpoint files. Its config declares BF16 weights and
no quantization configuration. The benchmark must apply bitsandbytes NF4 at
load time, with BF16 compute and double quantization, to fit this checkpoint.
A prior runtime tried that configuration with `device_map="auto"`; 12 of 24
startup cells dispatched modules to CPU or disk and failed before inference.
Do not repeat that placement.

This cache is the repository's GitHub-format custom checkpoint. Keep it
separate from InternVL's distinct Hugging Face-format or `-Instruct` repos.
The official InternVL documentation describes both formats and provides
conversion scripts; conversion is a fallback only if it preserves this exact
revision and weights.

## Staged execution

### 1. Confirm the local checkpoint and software path without loading weights

- Resolve the pinned revision and verify the config, tokenizer, custom model
  code, safetensors index, and all 13 indexed shards locally.
- The General benchmark environment has Transformers 4.57.1, Accelerate
  1.15.0, bitsandbytes 0.50.2, FlashAttention 2.8.4, and CUDA 13.0 PyTorch,
  but lacks `timm`; dynamic module discovery also left a nested config file
  out of its custom-code cache. These are specific compatibility blockers.
- Use the existing `/mnt/HDD1/Project_Code/robotics_VLM/.venv` runtime for
  this model. Its Python 3.10.12 environment has Transformers 5.19.0,
  PyTorch 2.13.0, Accelerate 1.15.0, bitsandbytes 0.50.2, and `timm` 1.0.27.
  Keep model-code caches under
  `general_transformers_Bench/environments/internvl/` and do not modify that
  shared runtime.
- Offline class discovery passed with the candidate runtime: the pinned
  `InternVLChatModel` class imports and exposes `chat`, the local tokenizer
  loads as `Qwen2Tokenizer`, and all 13 indexed shards resolve. Transformers
  emitted its tokenizer regex warning; keep the documented `use_fast=False`
  path and preserve that warning in runtime provenance.
- The pinned `config.json` sets `torch_dtype` to BF16 and has
  `quantization_config: null`. NF4 is applied by this benchmark at load time,
  not claimed as a property of the checkpoint files.
- Inspect the checkpoint's `AutoModel` class and module names with
  `local_files_only=True`. Keep any model-code cache under the benchmark's
  `environments/internvl/` folder. Do not change package versions unless this
  static compatibility step identifies a specific blocker.
- Verify the exact checkpoint's Apache-2.0 metadata from its [official model
  page](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B) and preserve the
  revision and cache path in every resulting artifact.

**Pass:** every required file resolves to the pinned revision; imports and
class discovery succeed offline; the checkpoint still declares NF4/BF16.
Otherwise stop before any GPU work and fix only the identified dependency or
cache issue.

### Controlled attempt record

The first explicit all-GPU NF4 load was attempted on 2026-11-03 and stopped
before inference. Transformers 5.19's allocator warmup requested 30.15 GiB
with 29.93 GiB free. A CPU meta-skeleton audit found 30.85B parameters, but
only 1.22B parameters in 290 `Linear4bit` modules were eligible for BNB NF4.
The Qwen3-MoE fused expert tensors remain BF16, so the current quantization
route cannot fit the 30 GiB cap. Do not retry this same loader/configuration.
A revised route must preserve the pinned checkpoint, all-GPU placement, and
both memory limits, and must be reviewed before another GPU attempt.

### 2. Make one controlled placement attempt

- Run `python3 src/executor.py release-local` from
  `/home/nils/Project_Code/codex_opencode_pipeline` immediately before GPU
  inspection, then record device free memory and other device residency.
  Leave RustDesk open as requested.
- Use one model process and the existing 30 GiB hard cap. Target at most 26
  GiB of `torch.cuda.max_memory_allocated()` to leave device headroom for the
  desktop and CUDA. Enforce the hard cap against sampled NVML/nvidia-smi
  per-process memory and label that peak as sampled. Record
  `torch.cuda.max_memory_reserved()` separately; it is not the 26 GiB pass
  metric. If available memory is below the planned budget, stop before
  loading.
- Load the pinned custom model class with the checkpoint's NF4 configuration,
  BF16 compute, low CPU-memory loading, and an explicit all-GPU module map.
  Start with `device_map={"": 0}` after confirming the custom composite model
  accepts a root GPU mapping. Do not use unconstrained `device_map="auto"`,
  CPU offload, or disk offload.
- Before generation, inspect `hf_device_map` and verify every model module is
  on `cuda:0`. Record the map, model footprint, CUDA allocated and reserved
  peaks, and any non-quantized modules.

**Pass:** loading completes with peak allocated memory below 26 GiB and sampled
per-process memory below 30 GiB; no module is on CPU or disk; one real image
can be encoded and generated from with finite output.
An OOM, CPU/disk placement, or loader exception ends this attempt. Unload the
failed model through the normal model-control workflow and diagnose that
specific cause before another GPU attempt.

### 3. Validate the real benchmark adapter

- Preserve the shared prompt and answer contract. Use the checkpoint's
  `model.chat` API, tokenizer, and official dynamic image tiling. Keep one
  image per request, batch size one, greedy decoding, and the shared 256-token
  output cap. Record the actual number of image tiles for each request.
- Run the shared AI2D and TextVQA real-image preflights. Require parseable
  choice output and a nonempty TextVQA answer before scoring begins.
- Check completion and parse errors explicitly; do not turn loader or parser
  failures into skipped examples.

**Pass:** both preflights pass and the model remains within the memory cap.
If either fails, do not run a partial benchmark or update the leaderboard.

### 4. Run and close the full suite

- Run AI2D 100, TextVQA 100, five VQA panels at 25 each, and GRAB-Lite 500,
  serially on the same pinned model and runtime.
- Require exactly 825 completed examples, zero infrastructure failures, and
  explicit per-task parse-error counts. Keep all output rows in the run
  artifacts, including invalid answers scored by the shared scorer.
- Record exact Transformers, PyTorch/CUDA, bitsandbytes and FlashAttention
  versions, checkpoint revision, quantization, precision, tile policy,
  generation settings, and peak task/memory.
- Retain a per-task row of requested indices and one item record per selected
  index. Reconcile all 825 rows against the prepared dataset and shared scorer:
  recompute predictions, scores, and parse-error flags; verify each task mean,
  each GRAB-Lite group mean, and the eight-task macro. Parse failures that the
  shared scorer can score remain completed scored rows; infrastructure errors
  do not. Require 825 scored rows and zero infrastructure failures before
  marking the model complete or publishing a macro.

Only then add the model to the aggregate and report page. If coverage is
incomplete, leave it visible as partial with the specific failure reason.

## If the explicit GPU map cannot fit

Do not fall back to CPU/disk dispatch. First inspect which modules remain in
BF16 and whether the custom loader's map is compatible with Transformers and
Accelerate. Make one evidence-based change at a time, retaining the 26 GiB
target. FlashAttention may be evaluated only after the default attention path
loads and returns valid output; it can reduce attention memory but cannot
make oversized weights fit.

If the custom checkpoint format is the blocker, evaluate the official
`custom2hf` conversion on a copy of this exact pinned snapshot. Verify tensor
names, tensor counts, dtypes, and hashes or equivalent per-tensor identity
evidence before loading the converted form. Keep conversion outputs outside
the source cache. If neither route achieves all-GPU placement within the cap,
leave the model unrun. Switching to an `-Instruct`, AWQ, or other quantized
repository changes the checkpoint and requires an explicit benchmark-plan
revision.

## References

- [Official InternVL3.5 model and format documentation](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-HF/blob/main/README.md)
- [Official InternVL3.5 Hugging Face usage example](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct)
- [Transformers bitsandbytes quantization guide](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- Prior local loader: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/internvl.py`
- Prior local runtime record: `/mnt/HDD1/Project_Code/robotics_VLM/configs/runtimes/internvl3_5.yaml`
