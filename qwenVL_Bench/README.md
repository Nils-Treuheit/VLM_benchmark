# Qwen VLM benchmark

This project compares runnable local Qwen2-VL, Qwen2.5-VL, and Qwen3-VL checkpoints on diagram reasoning and a broader six-category visual-question panel. A focused GRAB-Lite graph-analysis comparison evaluates G-Substrate against the Qwen3-VL baseline, best-overall, and long-context recommendations. Runs use one image and one request at a time, without batching.

The benchmark includes cached Unsloth GGUF quantizations, original Qwen Transformers checkpoints, G-Substrate, and the complete Unsloth Qwen3-VL-8B Thinking BNB exports. GGUF files run through llama.cpp with Q8_0 K/V cache and MTP requested when available. BNB safetensors cannot be loaded as GGUF, so those two cached exports follow their Transformers model-card path with the same HQQ 8-bit KV cache used by the Transformers baselines.

## Run

```bash
uv sync
./run_benchmark.sh --samples 100
```

Run in host context. The launcher runs `release-local` before starting one model at a time. The results are saved to `results/corrected/results.json` and `results/corrected/report.md`.

Use `--list-models` to check locally discovered checkpoints. Pass exact model IDs after `--models` to select a subset. Add `--resume` to append missing runs while keeping completed model results. llama.cpp attempts the largest supported context that completes a real-image warmup under the 30 GiB target. Thinking models can generate up to 2,048 tokens so the final answer can follow their reasoning text.

## Benchmarks

### AI2D diagram reasoning

The benchmark reads the cached **AI2D test split** (`lmms-lab/ai2d`), containing 3,088 examples. It selects 100 examples with seed 3407. Each item contains a diagram, a four-option question, and the answer index. The prompt asks for only A, B, C, or D. The score is exact answer-letter accuracy; missing or unparsable letters count as incorrect. See the [AI2D dataset card](https://huggingface.co/datasets/lmms-lab/ai2d).

### TextVQA image text reading

The benchmark reads the cached **TextVQA validation split** (`facebook/textvqa`), containing 5,000 examples. It selects 100 examples with seed 3407. Each item contains a natural image, an open question, and human answer references. The prompt asks for the shortest exact text visible in the image. The scorer normalizes case, punctuation, articles, and common number words, then averages leave-one-reference-out reference matches, capped at three matches per three annotators. See the [TextVQA dataset card](https://huggingface.co/datasets/facebook/textvqa).

The sample indices are fixed across models. Dataset loading reads cached Arrow files directly and does not run dataset scripts or download examples. Images are passed at their stored resolution; each model's processor applies its own preprocessing.

### VQA skill panel

TextVQA alone measures image-text reading. The additional panel reports six question-type scores: visible-content yes/no, counting, visual description, OCR, affordance/function, and negation. There is no single public dataset with reliable coverage of all six, so the panel combines VQAv2 validation, the existing TextVQA validation sample, PercepTax real/test, and NegVQA dev. It selects 25 fixed examples for each new category using seed 3407; the existing 100-example TextVQA result supplies the OCR category.

- **Truth questions:** 25 VQAv2 validation questions with `answer_type=yes/no`.
- **Counting:** 25 VQAv2 validation questions with `answer_type=number` and “How many” wording.
- **Visual description:** 25 VQAv2 validation `answer_type=other` questions filtered to direct questions about visible content.
- **OCR:** the same 100 TextVQA validation examples described above.
- **Affordance and function:** 25 real-image PercepTax test questions with affordance, function, or repurposing labels.
- **Negation:** 25 NegVQA development questions. Models choose A or B.

VQAv2 uses normalized human-answer consensus. PercepTax uses normalized open-answer matching, and NegVQA uses exact A/B accuracy. The panel total is the unweighted macro average of these six category scores; it is a local selection metric, not an official score from one dataset. See the [VQAv2 data page](https://visualqa.org/download.html), [PercepTax dataset](https://huggingface.co/datasets/RyanWW/PercepTaxBench), and [NegVQA paper and dataset](https://arxiv.org/abs/2505.22946).

Prepare the fixed public examples once before GPU runs:

```bash
uv run python prepare_vqa_panel.py
```

Run the five new categories across every discovered model:

```bash
./run_benchmark.sh --samples 25 --seed 3407 --tasks vqa_truth vqa_count vqa_description vqa_affordance vqa_negation --output-dir results/vqa_panel --resume
```

The report site joins these results with AI2D and TextVQA. The “Fastest average generation with acceptable quality” pick requires a mean above 75% on AI2D and TextVQA and at least 50% on the six-category VQA panel macro score. Other recommendations use AI2D plus the six VQA category scores and only include models complete on the full panel.

### Graph analysis

The graph specialist evaluation uses the complete 500-question GRAB-Lite split from `jonathan-roberts1/GRAB-lite`. Its five balanced task groups cover graph properties, functions, series, transformations, and real figures. The overall score is exact-match accuracy across all 500 examples; the report also gives per-group scores and deltas against `Qwen/Qwen3-VL-2B-Instruct`.

In the completed four-model comparison, `unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL` scored 14.0%, `Qwen/Qwen2.5-VL-7B-Instruct` scored 12.6%, the Qwen3-VL-2B baseline scored 6.2%, and G-Substrate scored 4.4%. On this split, G-Substrate did not outperform its Qwen3-VL-2B baseline; the UD Q4 8B model led overall and on properties and real-figure questions.

Prepare the local image set before running the graph comparison:

```bash
uv run python prepare_graph_panel.py
```

Run the graph specialist, Qwen3-VL-2B baseline, and the final best-overall and long-context picks with `--tasks graph_grab_lite --samples 500 --seed 3407 --output-dir results/graph_panel --resume`. The site combines graph results with the VQA results and shows the graph specialist’s mean score across all eight benchmark measures, GRAB-Lite score, configured context, and peak VRAM.

## Image scene graph generation

The separate [image SGG benchmark](image_SGG_Bench/README.md) evaluates direct image-to-graph generation on fixed VG-150, Open Images V7, and PSG panels. It discovers runnable entries from the requested local Qwen VL cache folders, includes supported GGUF quantizations, and never downloads models. Prepare the panels once, then run the suite serially on the GPU.

## Measures and limits

- Each task scores 100 requests. Calls run serially with one image per request; no benchmark batching is used.
- Mean seconds is the time for a generation call. Output tokens/second divides generated tokens by that call time. Inference includes model-side vision encoding and prefill performed inside generation.
- Peak VRAM is sampled across the whole GPU during the task, including desktop baseline use.
- The configured context is recorded per model. Context-tier VRAM estimates use the model's KV dimensions and an 8-bit cache. Edge tables estimate 8K residency plus a 1 GiB desktop reserve and interpolate the maximum context that fits under 12 or 16 GiB. Those are estimates, not new accuracy runs at edge limits.
- Parse failures score as incorrect. Thinking text before the final response is removed before scoring.
- Results are small-sample selection evidence, not leaderboard scores. Dataset tasks, sample counts, scoring details, recommendations, and sortable results are shown in the [benchmark report page](index.html). The page and its adjacent `app.js`, `styles.css`, and `data.json` files are kept in this benchmark folder so a Pages workflow can publish this folder at `/qwenVL_Bench/`.

## Run guidance

The GGUF route follows the [Unsloth Qwen3-VL llama.cpp guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune). Non-GGUF checkpoints follow their model-card Transformers examples. The local llama.cpp binary is `/usr/local/lib/ollama/llama-server`.
