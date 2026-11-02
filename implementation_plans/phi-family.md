# Phi vision family implementation plan

## Scope and locally cached checkpoints

- `microsoft/Phi-3-vision-128k-instruct`
- `microsoft/Phi-3.5-vision-instruct`
- `microsoft/Phi-4-multimodal-instruct`
- `sovitrath/Phi-3.5-vision-instruct` (cached converted copy; compare contents/config before treating as a separate checkpoint)

The first three Microsoft cache entries and the Sovitrath entry were present. Avoid counting the converted Phi-3.5 copy as an independent model unless its weights or quantization differ and are identified in metadata.

## Runner and prompt approach

Use each model's local processor and documented chat format. Phi-3 and Phi-3.5 use their image-token/message format; Phi-4 multimodal uses its own audio/video/image conventions. The common suite supplies still images, so use only the documented still-image mode. Keep user task wording unchanged and use the same decoding and scoring rules as the shared harness.

Phi-3 model cards may require `trust_remote_code=True`; make this an explicit per-checkpoint setting and load only cached code. Keep the compatibility patch isolated to the Phi adapter and record the Transformers version and attention implementation. Do not add that setting globally.

## References

- Phi-3.5 example loader and compatibility patch reference: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` (`load_phi_vision`).
- Prior VQA prompts: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/benchmark_vqa.py`.
- GUI model config and inference paths: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/models.py` and `inference.py`.
- Model cards: [Phi-3 Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), [Phi-3.5 Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [Phi-4 Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).

## Implementation order

Implement Phi-3.5 first using the existing local example, then Phi-3 Vision, then Phi-4 Multimodal. Check the converted copy for weight identity and configuration drift before inclusion. Run only on GPU and within the common memory ceiling.

## Report site delivery

After completing the Phi benchmark and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Preserve checkpoint identity and compatibility status in comparisons.
