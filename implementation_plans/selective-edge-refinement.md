# Selective Edge Limited benchmark refinement

**Status checked:** 2026-11-05. This plan remains active; the user has paused GPU work after the Cosmos3-Edge 16 GiB run. Four 8 GiB runs are independently verified: `LiquidAI/LFM3-VL-2B-GGUF:NVFP4` scored 58.54% at 128,000 context tokens with a 6.793 GiB peak; `unsloth/LFM2.5-VL-3B-GGUF` scored 58.44% at 32,768 tokens with a 4.573 GiB peak; `LiquidAI/LFM3-VL-2B-GGUF` scored 55.73% at 128,000 tokens with a 5.430 GiB peak; and `google/paligemma2-3b-mix-224` scored 54.53% at 1,800 tokens with a 7.955 GiB peak. Their artifacts appear in `../edge-refinement-data.json` and the root page. Four other 8 GiB Qwen candidates lack cached weights and baseline GRAB-Lite scores, so remain unavailable. At 10 GiB, five runs are verified: LFM3 NVFP4 (58.54%, 128,000 tokens, 5.693 GiB), LFM2.5-VL-3B (58.44%, 32,768 tokens, 4.554 GiB), PaliGemma2 (54.53%, 4,418 tokens, 9.961 GiB), Cosmos3-Edge (52.59%, 23,552 tokens, 9.82 GiB), and LLaVA Phi-3 Mini (43.65%, 1,258 tokens, 9.865 GiB). The LLaVA probe retained nine sampled, under-cap OOM bounds. The 12 GiB stage has six verified runs: Cosmos3-Edge (52.59%, 37,888 tokens, 11.891 GiB), LFM3 NVFP4 (58.54%, 128,000, 5.554 GiB), LFM2.5-VL-3B (58.44%, 32,768, 4.554 GiB), PaliGemma2 (54.53%, 6,065, 11.883 GiB), LLaVA Phi-3 Mini (43.65%, 3,880, 11.869 GiB), and Qwen3-VL-8B Q4 (60.60%, 32,768, 11.324 GiB). An earlier Qwen 12 GiB attempt exceeded the cap and remains preserved separately. No selected 12 or 14 GiB candidates remain pending. The 14 GiB stage has six verified candidates: Cosmos3-Edge (52.59%, 48,128 tokens, 13.553 GiB), LLaVA Phi-3 Mini (43.65%, 4,096, 12.264 GiB), LFM3 NVFP4 (58.54%, 128,000, 5.558 GiB), LFM2.5-VL-3B (58.44%, 32,768, 4.567 GiB), PaliGemma2 (54.53%, 6,065, 12.482 GiB), and Qwen3-VL-8B Q4 (60.60%, 65,536, 13.355 GiB). Three earlier Cosmos runs and PaliGemma2’s first high-context probe exceeded the cap; all are preserved and excluded. At 16 GiB, Cosmos3-Edge completed all 825 samples at 52.59% macro but exceeded the cap: its probe and whole-run peak reached 16.073 GiB. Independent review confirmed exclusion. The run is recorded as unverified; five other selected 16 GiB candidates remain pending. Two earlier failed Cosmos runs at 12 GiB and three at 14 GiB also remain preserved. `release-local` succeeded before each GPU run.

The 87 model-cap entries in `../edge-refinement-candidates.json` remain screening candidates, not a runnable queue. The inventory now labels each row as a category or overall leader, a nonleader, or a selected but unavailable checkpoint, and records verified artifacts. Exact cached weights and runner support were checked for every cap. Missing-weight leaders remain visible as unavailable; nonleaders are marked not selected.

The Qwen and suite runners now use controlled real-image context probes, whole-device cap checks, and cap-verification metadata. Local Foundation, General Transformers, and Cosmos Edge runs at 8 and 10 GiB were exercised on the GPU and independently reviewed. The LLaVA Edge runner passed code review. `../aggregate_edge_refinement.py` merges cap-specific records into root data, while only verified runs qualify for rankings. The 10 GiB stage has five verified candidates. The 12 GiB stage has six verified candidates: Cosmos3-Edge, LFM3 NVFP4, LFM2.5-VL-3B, PaliGemma2, LLaVA Phi-3 Mini, and Qwen3-VL-8B Q4. Six other selected Qwen leaders are unavailable because their exact 12 GiB checkpoints are not cached. The 14 GiB stage also has six verified candidates: Cosmos3-Edge, LLaVA Phi-3 Mini, LFM3 NVFP4, LFM2.5-VL-3B, PaliGemma2, and Qwen3-VL-8B Q4. The 16 GiB stage has no verified candidates yet. Its Cosmos3-Edge run completed, but the context probe and whole-device peak reached 16.073 GiB against a 16 GiB cap; the artifact remains preserved and excluded. Five selected 16 GiB runs remain pending while GPU work is paused at the user's request.

## Purpose

Replace preliminary VRAM-filtered leader lists with measured quality, peak memory, and context limits for the 8, 10, 12, 14, and 16 GiB stages. Rerun only the strongest candidates at each stage rather than every cached VLM.

## Current evidence and limits

The root landing page's current scores come from completed benchmark runs. The current memory filters do not represent the same experiment for every suite:

- Qwen-VL estimates 8K residency with a 1 GiB desktop reserve and interpolates a maximum context from its context profile. Its recorded benchmark peaks can exceed the selected edge tier because some runs used much longer contexts.
- LLaVA and Φ-Vision record process-local peak VRAM across their eight shared tasks at the benchmark settings. Their checkpoint context values are advertised metadata, not context sweeps.
- Cosmos records whole-device peak VRAM including desktop use. Cosmos3-Nano has an explicit 16,384-token run cap. Other Cosmos runs use the checkpoint default.
- Other Big VLMs record process-local peak VRAM across the benchmark tasks. Their checkpoint context values do not establish the maximum context that fits an edge tier.

Use these results only to select candidates. Do not treat the current Edge Limited ranking as a controlled accuracy comparison under a selected VRAM cap.

## Working hypothesis

At fixed model, inputs, context length, precision, and decoding, a VRAM ceiling alone should have little effect on answer quality. It mainly affects whether a run fits and its speed. More context increases KV-cache memory and can help or hurt depending on the task and how well the model retrieves relevant details. Context-driven score changes are expected to be smaller on the current mostly single-image tasks than on multi-image, video, or long-prompt tasks. This is a hypothesis to test, not an established result. See the [KV-cache documentation](https://huggingface.co/docs/transformers/kv_cache) and [long-context retrieval study](https://arxiv.org/abs/2307.03172).

## Selective candidate selection

Run five stages, one for each cap: 8, 10, 12, 14, and 16 GiB.

For each stage, select every complete model that currently leads the overall eight-task macro or any shared category among candidates at that preliminary tier. Include all ties. Deduplicate identical model configurations within the stage, but keep a separate candidate-stage record for every VRAM cap. Across stages, a model may be rerun at several caps because its usable context and memory behavior can change. Include the completed Local Foundation and GGUF suite in candidate screening; its measurements were added after the original landing-page filter was written. The current screening roster is `../edge-refinement-candidates.json`: 8, 16, 22, 22, and 19 model-cap records at 8, 10, 12, 14, and 16 GiB respectively. These counts are preliminary candidate counts, not measured successes.

Use these eight shared measures: AI2D, TextVQA/OCR, VQA truth, counting, description, affordance, negation, and GRAB-Lite. Overall leaders require complete results on all eight measures. Category leaders require complete results for that category. Preserve model identity, quantization, backend, and task prompts. Do not change a model's quantization between cap stages.

## Controlled run protocol

1. Reuse the shared prepared examples, seed 3407, prompts, output constraints, scorers, image handling, and task sample counts from the Qwen-VL reference benchmark.
2. Run one model at a time on the same GPU and host configuration. Before any local GPU work, follow the repository workflow's `release-local` procedure. Keep the GPU job in host context and record the GPU model, driver, software versions, desktop baseline, backend, and model configuration.
3. Define each selected cap as maximum whole-device GPU use reported by `nvidia-smi`, including desktop baseline. Record baseline use immediately before loading; if the idle baseline already reaches a stage cap, mark that stage unavailable on this host. Sample whole-device use throughout each run, and record process-local allocated/reserved memory when available. Apply supported backend limits where available and record their exact settings.
4. Measure context fit independently from checkpoint metadata with a controlled context probe through the model's real image path. When the processor exposes visual-token counts, choose the sampled image with the most visual tokens and search in 1,024-token increments. Otherwise, sample 100 fixed AI2D examples and choose the image with the largest measured total input-token count; record that selection as a proxy, not a visual-token count. Use deterministic filler and backend-reported token counts to find the largest accepted context. Generate a short fixed output and retain per-attempt peaks. Record the image, selection method, target/actual token counts, output length, peak, and whether the runner's configured limit was reached. If reliable token counts are unavailable, leave the candidate unverified. This probe measures memory fit; it does not contribute quality scores.
5. Run the full shared suite with the runner configured to the selected measured context. Save per-task completion counts, measured scores, peak VRAM, peak-producing task, runtime, outputs/errors, exact run context, and cap-control evidence. Do not infer an under-cap score from an earlier run at a different context.
6. If the context probe or any scored task exceeds the whole-device cap, lower the context and rerun the probe and full suite. If the backend cannot reliably verify whole-device peak use, record the candidate as unverified for that tier. Do not publish it as an edge-limited measured result.
7. For a candidate selected at multiple caps, add paired runs at a shared context that fits each compared cap. Hold model, input examples, prompts, precision, decoding, and backend settings fixed. Compare these runs to test the fixed-context hypothesis separately from each cap's maximum-context run.

## Result artifacts and aggregation

Keep cap-specific raw outputs under each suite's existing results directory. Produce a compact aggregate with one record per model and cap containing:

- exact model, quantization, backend, and run configuration;
- requested whole-device VRAM cap, desktop baseline, cap enforcement method, and process-local memory measurements where available;
- largest context passed by the controlled image-context probe, actual token count, and measured peak VRAM during that probe;
- context configured for the scored suite, whether the complete suite passed under the cap, and measured peak VRAM during scoring;
- complete task scores, per-task peaks, completion counts, and the task that produced the peak;
- run date, software versions, hardware, and links to the detailed report.

Recompute the overall macro and each category leader from these cap-specific runs only. Show ties. Require all eight completed scores for overall. Do not mix an estimate, checkpoint context value, or score from another cap into a measured tier result.

## Landing page completion criteria

After the refinement runs are complete, update the root Edge Limited section to read only the cap-specific measured aggregate. Display a model at a tier only when its complete benchmark run passed under that tier's verified whole-device cap and its controlled context probe passed. Show measured score, measured maximum tested context, probe peak VRAM, scoring-run peak VRAM, and any supplementary process-local memory figure.

Remove estimated context-fit values from the ranking logic and generated leader rows. Keep estimates only in the measurement disclaimer or historical report notes. If a stage has no verified candidates, show an explicit empty state instead of a provisional estimate. The root landing page must show no estimate as an Edge Limited result after this plan is implemented.

Write the refinement results to each suite's `report.md`, regenerate the existing static report site, and update the Edge Limited summary on the root page. Keep the exact task set and report-site structure defined in the [general VLM report site guideline](general-vlm-report-site-guideline.md).

## References

- Shared tasks, prompts, sampling, and scorers: [Qwen-VL benchmark README](../qwenVL_Bench/README.md)
- Qwen estimate formula and existing task runner: [Qwen-VL data generator](../qwenVL_Bench/generate_data.py), [benchmark runner](../qwenVL_Bench/benchmark.py)
- Existing report memory definitions: [Qwen-VL](../qwenVL_Bench/index.html), [LLaVA](../llava_Bench/index.html), [Cosmos](../cosmos_Bench/index.html), [Other Big VLMs](../general_transformers_Bench/index.html), and [Φ-Vision](../phi_Bench/index.html)
- Static report publishing rules: [general VLM report site guideline](general-vlm-report-site-guideline.md)
