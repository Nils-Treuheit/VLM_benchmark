# Local foundation and GGUF VLMs implementation plan

## Scope and locally cached checkpoints

- Qwen3.8 local variants: `Qwen/Qwen3.8-27B`, `turboderp/Qwen3.8-27B-exl3`, `HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3`, and `unsloth/Qwen3.8-27B-NVFP4`.
- DiffusionGemma: `SVPG-Labs/diffusiongemma-26b-djev-v10-step160-NVFP4` and `unsloth/diffusiongemma-26B-A4B-it-GGUF`.
- `unsloth/MiMo-VL-7B-RL-GGUF`.
- `unsloth/LFM2.5-VL-1.6B-GGUF`.

All listed entries were present in the local cache. Exact quantizations and required multimodal projector files must be discovered from snapshots before implementation. The project-level Qwen3.8 instruction uses the configured executor route; follow it rather than adding an unrelated serving path.

## Compatibility and runner approach

First confirm that each exact artifact accepts images and can produce text answers. Some repositories may contain text-only or image-generation/diffusion models rather than general image-question-answering VLMs. Do not infer capability from a model name. If image-to-text VQA is not documented, mark the checkpoint incompatible with this shared benchmark.

For Qwen3.8, prefer its configured local route and use the shared prompts verbatim. For MiMo-VL and LFM2.5-VL GGUF, use llama.cpp only if the installed binary and cached model/projector files support that architecture. Otherwise use the documented local adapter. Do not convert weights or fetch missing files during a benchmark run. Record backend, quantization, projector identity, context, and generation settings.

DiffusionGemma must be evaluated as its cached artifact and documented inference pipeline. The prior VLMExperiments code wraps it through `run.py` and llama-diffusion-cli; assess whether that pipeline provides image-conditioned text output directly. Do not include external detectors or other added model components in the comparable score.

## References

- Robotics Qwen3.8 routes: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/qwen38_ollama.py` and `qwen38_nvfp4.py`.
- Robotics LFM2.5-VL adapter: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/lfm25_vl.py`.
- Prior Llama / diffusion runner setup: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_llama_vision`, `load_diffusion_gemma`) and its task scripts.
- GUI comparison settings: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py`, `config.py`, and `multi_vlm_infer.py`.
- Model cards: [Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B), [DiffusionGemma GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF), [MiMo-VL GGUF](https://huggingface.co/unsloth/MiMo-VL-7B-RL-GGUF), [LFM2.5-VL GGUF](https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF).

## Implementation order

Start with MiMo-VL and LFM2.5-VL once their GGUF and projector pair is resolved. Evaluate Qwen3.8 through its configured executor after the required `release-local` command. Gate DiffusionGemma on direct image-question-answering capability. Keep checkpoints that fail the capability or hardware gate in the inventory with a precise status, not in the scored ranking.

## Report site delivery

After completing this group and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Identify each backend and quantization, and distinguish measured results from estimates.
