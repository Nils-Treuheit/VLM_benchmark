# Local VLM benchmark

This repository compares vision-language models already available on the local system. The suites share fixed image tasks where their capabilities overlap, and report results separately by model family.

## Benchmark reports

| Suite | Status | Report |
|---|---|---|
| Qwen-VL | Complete | [Open report](qwenVL_Bench/) |
| LLaVA | Complete | [Open report](llava_Bench/) |
| Cosmos | Complete | [Open report](cosmos_Bench/) |
| General Transformers | Complete for the current supported candidate set | [Open report](general_transformers_Bench/) |
| Local foundation and GGUF | Complete for the current cached and runnable candidates | [Open report](local_foundation_Bench/) |
| Phi vision | Complete | [Open report](phi_Bench/) |
| Video VLMs | Complete | [Open report](video_Bench/) |
| Grounding specialists | In progress; local data preparation remains | [Open report](grounding_Bench/) |

The report pages use fixed local samples and support model selection on this system. They are not leaderboard results. Each report describes its tasks, sample sizes, scoring rules, model coverage, and runtime limits.

## Running benchmarks

Runs require prepared datasets and model checkpoints in the local Hugging Face cache. Start from the README in the relevant benchmark folder. Model inference runs serially on the host GPU; benchmark commands do not download model weights.

The image suites reuse task definitions and scoring implemented in [Qwen-VL](qwenVL_Bench/README.md). The [implementation plan index](implementation_plans/README.md) tracks active work; completed suite reports retain their model inventories and results.

## Report hosting

Each published report is the `index.html` at its benchmark folder root. The repository [landing page](index.html) links to available reports. A GitHub Pages deployment should preserve folder names so each suite is available under `/<benchmark-folder>/`.
