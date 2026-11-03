# Phi vision benchmark

This suite compares cached Phi-3 Vision, Phi-3.5 Vision, and Phi-4 Multimodal checkpoints on the shared image-task set.

The completed comparison uses fixed samples and scoring from the [Qwen-VL benchmark](../qwenVL_Bench/README.md). See the [results report](report.md) for task coverage, scores, and runtime measurements, or open the [interactive report](index.html).

Run the fixed suite with `./run_benchmark.sh --resume` after preparing shared data and local model environments. The runner evaluates one image request at a time and records incomplete or failed entries.
