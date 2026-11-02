# LLaVA family implementation plan

## Scope and locally cached checkpoints

- `xtuner/llava-phi-3-mini-hf`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `llava-hf/LLaVA-NeXT-Video-7B-hf`
- `llava-hf/LLaVA-NeXT-Video-34B-DPO-hf`

All five cache entries were present under `HF_HOME/hub`. Resolve snapshots and weight completeness when implementing. Keep OneVision, NeXT Video, and classic LLaVA as separate model identities and prompt templates.

## Runner and prompt approach

Start with Transformers and each checkpoint's cached processor/tokenizer. Use the model's own `apply_chat_template` when available. Adapt only the message wrapper and image placement; keep the shared task text and answer constraints byte-for-byte identical. For single images, provide one image and one user question. Ask AI2D for only the answer letter. Preserve TextVQA's shortest-exact-visible-text instruction. Do not use video frame templates for the shared still-image tasks.

For each response, decode only newly generated tokens where the architecture supports it. Keep the common final-response cleanup and answer parsers. Record any model-specific output cleanup as runner metadata rather than changing the scorer.

## References

- Local implementation reference: `/mnt/HDD1/Project_Code/robotics_VLM/src/robotics_vlm/models/adapters/llava_video.py`. It documents LLaVA-NeXT-Video and keeps OneVision as a distinct setup.
- Prior prompt and loader examples: `/mnt/HDD1/Project_Code/VLMexperiments/VLMbenchmark/scripts/common.py` and `scripts/benchmark_vqa.py`.
- GUI inference/config examples: `/mnt/HDD1/Project_Code/visual-LLM-testground/GUI_app/inference.py`, `models.py`, and `config.py`.
- Model cards: [LLaVA-NeXT-Video](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf), [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf), [LLaVA 1.6 Mistral](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), [LLaVA-Phi-3](https://huggingface.co/xtuner/llava-phi-3-mini-hf).

## Compatibility and implementation order

Begin with LLaVA-NeXT-Video 7B and OneVision 7B, then LLaVA 1.6 and LLaVA-Phi-3. Attempt the 34B checkpoint only if it fits the same 30 GiB ceiling with the fixed one-image requests. Record a clear load failure if it does not. Include each model in the full benchmark only after all four shared tasks complete; retain partial task results without treating them as a full-panel ranking.

## Report site delivery

After completing the LLaVA benchmark and writing `report.md`, keep its static report page and assets at the benchmark folder root, following the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Keep incomplete models visible and base memory-tier recommendations on measured full-benchmark peaks unless the report clearly documents a separate estimate.
