# Selective Edge Limited benchmark refinement

**Status: Completed 2026-11-06.** All selected runnable 16 GiB leaders now have either a verified complete run or an evidence-based blocker status. Qwen3-VL-8B Q8 completed 825/825 at 59.9125%, with a measured 262,144-token context and 15.753 GiB whole-device peak; an independent review accepted it for aggregation. LLaVA Mistral 7B remains blocked at this cap: seven reserve settings were tried, and the final 0.62 GiB reserve plus expandable segments still OOMed on the minimum 3,012-token context attempt. Its 15.996 GiB peak leaves about 4 MiB under the cap; independent review found further reserve tuning is not cap-safe. Preserve all attempts and exclude them from rankings. Keep `PLAN.md` and all `grounding_Bench` files untouched. A selected Qwen3-VL-4B-Thinking checkpoint remains unavailable because its exact weights are not cached. Lower-cap stages remain complete: 8 GiB has four verified candidates, 10 GiB has five, and each of 12 and 14 GiB has six. The aggregate reports five verified 16 GiB candidates with Qwen Q8 as the overall leader. The root Edge Limited section was served and checked.

The 87 model-cap entries in `../edge-refinement-candidates.json` remain screening candidates, not a runnable queue. The inventory now labels each row as a category or overall leader, a nonleader, or a selected but unavailable checkpoint, and records verified artifacts. Exact cached weights and runner support were checked for every cap. Missing-weight leaders remain visible as unavailable; nonleaders are marked not selected.

The Qwen and suite runners now use controlled real-image context probes, whole-device cap checks, and cap-verification metadata. Local Foundation, General Transformers, and Cosmos Edge runs at 8 and 10 GiB were exercised on the GPU and independently reviewed. The LLaVA Edge runner passed code review. `../aggregate_edge_refinement.py` merges cap-specific records into root data, while only verified runs qualify for rankings. The 10 GiB stage has five verified candidates. The 12 GiB stage has six verified candidates: Cosmos3-Edge, LFM3 NVFP4, LFM2.5-VL-3B, PaliGemma2, LLaVA Phi-3 Mini, and Qwen3-VL-8B Q4. Six other selected Qwen leaders are unavailable because their exact 12 GiB checkpoints are not cached. The 14 GiB stage also has six verified candidates: Cosmos3-Edge, LLaVA Phi-3 Mini, LFM3 NVFP4, LFM2.5-VL-3B, PaliGemma2, and Qwen3-VL-8B Q4. The 16 GiB stage has five verified candidates: Cosmos3-Edge scored 52.59% at a 58,368-token context, with a 15.446 GiB probe peak and 15.907 GiB reused score-run peak; LFM3 NVFP4 scored 58.54% at 128,000 tokens, with a 5.305 GiB probe peak and 5.981 GiB scoring/whole-run peak, and passed independent review on the matching b12097 backend; LFM2.5-VL-3B scored 58.44% at 32,768 tokens, with a 4.644 GiB probe peak and reconciled 5.158 GiB scoring/whole-run peak, and its corrected artifact passed independent review; PaliGemma2 scored 54.525% at 8,189 tokens, with a 15.693 GiB probe and whole-device peak and a 7.547 GiB scoring peak; Qwen3-VL-8B Q8 scored 59.9125% at a measured 262,144-token context, with a 15.753 GiB whole-device peak, and passed independent review. The LLaVA Mistral leader remains excluded after seven failed cap-specific context probes; the last reserve 0.62 GiB attempt peaked at 15.996 GiB and was independently adjudicated as not cap-safe to tune further. Qwen3-VL-4B-Thinking remains unavailable because its exact weights are not cached.

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
