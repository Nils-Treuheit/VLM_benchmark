# G-Substrate cross-task benchmark

This suite measures the pinned G-Substrate model on official GAR, MGD, and ERE test subsets. It joins these results with the reviewed 50-image PCIs SGG panel.

The new panels contain 200 GAR examples, 200 MGD examples, and 200 ERE examples. Decoding follows the official inference script: temperature 0.7, top-p 0.8, top-k 20, and a 4,096-token cap. Sampling uses seed 42. The runner saves source rows, predictions, hashes, official scores, bootstrap intervals, and a report.

Run from the repository root after releasing managed local models:

```bash
cd .agentic_workflow
python3 src/executor.py release-local
cd ../
python3 qwenVL_Bench/gsubstrate_Bench/run_official_subsample.py \
  --model-id zmli/G-Substrate-Qwen3-VL-2B \
  --model-revision e3c7d3c8ffd278884522739ae7bb9e05b4c18a1a \
  --dataset-dir .agentic_workflow/workspace/g-substrate-cross-task/data/test \
  --image-root /mnt/HDD1/Project_Data/public_datasets/visual_genome \
  --output-dir .agentic_workflow/workspace/g-substrate-cross-task/runs/gsubstrate-v1 \
  --seed 42
```

The runner refuses to overwrite prior attempts and stops on inference errors or empty outputs. It scores nonempty token-capped responses as partial predictions, matching the official pipeline, and reports their count separately. The scorer checks each prediction against its exact pinned source row before computing metrics.

The report compares CT, CD, SP, BM, BLEU-4, ROUGE-L, PCIs R@50, MA-S, MA-T, MA-C, and HiE with advertised values. The SGG score remains a distinct Visual Genome panel and is not treated as an HF test-split sample.
