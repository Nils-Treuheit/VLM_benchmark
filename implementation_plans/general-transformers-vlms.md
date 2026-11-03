# General Transformers VLMs implementation plan

## Scope and locally cached checkpoints

- `OpenGVLab/InternVL3_5-30B-A3B`
- `allenai/Molmo2-8B`
- `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- `google/paligemma2-3b-mix-224`

The cache has entries for all four, but the Llama Vision snapshot was incomplete
when this plan was written. The user authorized downloading and benchmarking
`unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` for non-commercial academic
local functional comparison. The user also confirmed their authorization under
Meta's terms. Keep each model as a separate configuration, pin its revision,
verify all indexed weight shards and processor files, and run the same real-image
preflight plus all eight shared task groups.

`microsoft/Florence-2-large-ft` was initially a conditional candidate because its legacy remote code did not load through the current shared Transformers runtime. The benchmark now has an isolated Transformers 4.44.2 loader for the exact cached snapshot. It passed real-image AI2D and TextVQA preflights and completed the shared suite. Its `<VQA>` route remains experimental because the processor does not register that token; the report discloses the text fallback and choice-label normalization. Do not treat the old VLMExperiments monkey patch as verified compatibility.

## Runner and prompt approach

Use each model's processor and documented input format. Preserve the shared benchmark's common task wording and output constraints. Use a chat template where the model supports one; otherwise apply only the required task prefix or input wrapper. InternVL's model-card path may use tiled/high-resolution image preprocessing; use the documented processor defaults, record effective image size/crops, and do not alter the benchmark source image. Molmo2 requires its specific processor/model implementation; use its official local configuration and decode only the generated completion. Llama Vision BNB weights use the Transformers model-card path, not llama.cpp.

For PaliGemma 2, use the cached `mix-224` checkpoint and `PaliGemmaProcessor`. Prefix the unchanged shared question with `answer en `, following the model's documented VQA format. Do not use a chat template. Record its 224-pixel input resolution because it differs from higher-resolution candidates. If the shared suite includes captioning, use the documented `describe en` prefix for that task while keeping the benchmark's caption instruction intact.

Florence's isolated loader and experimental `<VQA>` route have now passed real-image preflight and completed the shared suite. Because its cached processor does not register `<VQA>`, the current adapter falls back to text and normalizes exact option-text matches for choice scoring. Keep that caveat and raw generated text in artifacts. Do not add Florence-only tasks to the shared score; any detection or grounding capability should be a separately reported task group.

The shared Qwen-VL prompt remains the semantic comparison contract. Do not import extra task instructions from the older broad VLMExperiments benchmark. Add only the minimum architecture-required chat formatting, task prefix, or task token, and record that wrapper in result metadata.

## References

- Robotics adapters: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/internvl.py` and `molmo2.py`.
- Prior loaders and prompt examples: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_internvl3`, `load_llama_vision`) and `scripts/benchmark_vqa.py`.
- GUI config and inference examples: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py` and `inference.py`.
- PaliGemma 2 local examples: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_paligemma`) and `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/inference.py`.
- Florence-2 local compatibility notes: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py` (`_load_florence2`) and `parameter_tests/test_config.py`; legacy patched loader: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_florence2`).
- Model cards: [InternVL 3.5 30B-A3B](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B), [Molmo2 8B](https://huggingface.co/allenai/Molmo2-8B), [Llama 3.2 Vision 11B BNB 4-bit](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit), [PaliGemma 2 3B Mix 224](https://huggingface.co/google/paligemma2-3b-mix-224), [Florence-2 Large FT](https://huggingface.co/microsoft/Florence-2-large-ft).

## Implementation order

Molmo2, PaliGemma 2, and Florence-2 have complete measured runs. The pinned InternVL3.5 NF4 load failed before inference because the Qwen3-MoE fused expert tensors remain BF16; the diagnosis and attempt record are in [`internvl3_5-run-plan.md`](internvl3_5-run-plan.md). Do not retry that route or silently switch checkpoints. The user has since authorized local Llama Vision comparison and confirmed license applicability; its complete snapshot and benchmark are being prepared.

The current cache and failed-placement evidence for InternVL3.5 are recorded in
[`internvl3_5-run-plan.md`](internvl3_5-run-plan.md). Any new route needs a
reviewed, checkpoint-preserving way to quantize fused expert weights and meet
the single-GPU memory limits.

## Report site delivery

Keep the static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Show completed runs and unresolved candidates with their actual status. Keep Florence-2's experimental route caveat visible.
