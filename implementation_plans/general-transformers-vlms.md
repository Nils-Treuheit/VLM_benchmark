# General Transformers VLMs implementation plan

## Scope and locally cached checkpoints

- `OpenGVLab/InternVL3_5-30B-A3B`
- `allenai/Molmo2-8B`
- `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- `google/paligemma2-3b-mix-224`

These four checkpoints are present in `HF_HOME`. Keep each as a separate model configuration even though they share a Transformers backend.

`microsoft/Florence-2-large-ft` is also cached, but keep it as a conditional candidate rather than a scored member of this group for now. The local GUI project records an incompatibility between Florence-2's legacy cache/model code and the current Transformers version, and its current loader is disabled. Promote it only after a supported loading route is established; do not treat the old VLMExperiments monkey patch as verified compatibility.

## Runner and prompt approach

Use each model's processor and documented input format. Preserve the shared benchmark's common task wording and output constraints. Use a chat template where the model supports one; otherwise apply only the required task prefix or input wrapper. InternVL's model-card path may use tiled/high-resolution image preprocessing; use the documented processor defaults, record effective image size/crops, and do not alter the benchmark source image. Molmo2 requires its specific processor/model implementation; use its official local configuration and decode only the generated completion. Llama Vision BNB weights use the Transformers model-card path, not llama.cpp.

For PaliGemma 2, use the cached `mix-224` checkpoint and `PaliGemmaProcessor`. Prefix the unchanged shared question with `answer en `, following the model's documented VQA format. Do not use a chat template. Record its 224-pixel input resolution because it differs from higher-resolution candidates. If the shared suite includes captioning, use the documented `describe en` prefix for that task while keeping the benchmark's caption instruction intact.

If Florence-2 becomes loadable through a supported implementation, use its documented task-token interface and `processor.post_process_generation` for structured outputs. Its `<VQA>` task plus the shared question is suitable for a VQA comparison; `<CAPTION>` or `<DETAILED_CAPTION>` can serve caption tasks only if those tasks are already in the common suite. Keep its parsed structured results and raw generated text in artifacts. Do not add Florence-only tasks to the shared score; any detection or grounding capability should be a separately reported task group.

The shared Qwen-VL prompt remains the semantic comparison contract. Do not import extra task instructions from the older broad VLMExperiments benchmark. Add only the minimum architecture-required chat formatting, task prefix, or task token, and record that wrapper in result metadata.

## References

- Robotics adapters: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/internvl.py` and `molmo2.py`.
- Prior loaders and prompt examples: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_internvl3`, `load_llama_vision`) and `scripts/benchmark_vqa.py`.
- GUI config and inference examples: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py` and `inference.py`.
- PaliGemma 2 local examples: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_paligemma`) and `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/inference.py`.
- Florence-2 local compatibility notes: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py` (`_load_florence2`) and `parameter_tests/test_config.py`; legacy patched loader: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_florence2`).
- Model cards: [InternVL 3.5 30B-A3B](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B), [Molmo2 8B](https://huggingface.co/allenai/Molmo2-8B), [Llama 3.2 Vision 11B BNB 4-bit](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit), [PaliGemma 2 3B Mix 224](https://huggingface.co/google/paligemma2-3b-mix-224), [Florence-2 Large FT](https://huggingface.co/microsoft/Florence-2-large-ft).

## Implementation order

Start with Molmo2, PaliGemma 2, and Llama Vision. Add InternVL3.5 with a documented quantization or device-map configuration that fits the 30 GiB limit; do not silently quantize a different checkpoint. Resolve Florence-2's supported loading route before implementation or scoring. Preserve exact checkpoint identity, quantization, Transformers version, dtype, and image preprocessing in result metadata.

## Report site delivery

After completing this group and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Show Florence-2 as conditional or unsupported unless its loading route is resolved and measured.
