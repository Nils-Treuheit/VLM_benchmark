# Cosmos family implementation plan

## Scope and locally cached checkpoints

- `nvidia/Cosmos-Reason1-7B` (Cosmos 1)
- `nvidia/Cosmos-Reason2-2B` and `nvidia/Cosmos-Reason2-8B` (Cosmos 2)
- `nvidia/Cosmos3-Edge` and `nvidia/Cosmos3-Nano` (Cosmos 3)

All listed cache entries were present. The robotics adapter pins revisions, expected files, generation defaults, and Cosmos 3 setup details. Verify exact cache completeness and any license requirements before loading.

## Runner and prompt approach

Use Transformers and the checkpoint's local `AutoProcessor` plus documented model class. Prefer its chat template. Preserve the shared task wording and answer constraints; set deterministic generation where the model card supports it. Keep thinking tokens or reasoning traces out of the answer parser while preserving raw output for audit.

The common tasks supply still images. Use the model's supported image-input path for them. Do not turn an image into synthetic video or change task data. If a Cosmos checkpoint only supports video or another input type, mark it incompatible with the shared suite and retain that reason in results.

## References

- Primary local adapter and pinned setup records: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/cosmos_reason.py`.
- Transformers loader and prior task prompts: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` and `scripts/benchmark_vqa.py`.
- Model cards: [Cosmos Reason1](https://huggingface.co/nvidia/Cosmos-Reason1-7B), [Cosmos Reason2 2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B), [Cosmos Reason2 8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B), [Cosmos3 Edge](https://huggingface.co/nvidia/Cosmos3-Edge), [Cosmos3 Nano](https://huggingface.co/nvidia/Cosmos3-Nano).

## Implementation order

Start with Reason2-2B and Reason1-7B. Add Reason2-8B, then Cosmos3 Edge and Nano after the loader and static-image path are confirmed from their local model cards and cached processor configuration. Keep one checkpoint active at a time. Treat missing or unaccepted model license requirements as a blocked model setup, not a task result.

## Report site delivery

After completing the Cosmos benchmark and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Keep per-checkpoint compatibility and task coverage visible.
