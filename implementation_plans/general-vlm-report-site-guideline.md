# General VLM benchmark report site design guideline

Use this guideline after a benchmark group has completed its planned runs and written its results to `report.md`. The report site presents that report in a readable, navigable form; it does not replace the report. Keep the site in that benchmark group's `report-site/` directory and link the source report from the page or its documentation.

The visual and information architecture draws on the completed [Qwen-VL report site](../qwenVL_Bench/report-site/dist/index.html) and [LLaVA report site](../llava_Bench/report-site/dist/index.html). Adapt labels and metrics to the group. Do not force specialized video or grounding results into an image-QA layout when that would obscure their task structure.

## Information order

1. **Header and hero.** Identify the benchmark family, purpose, and intended comparison. Show report update time, tested model count, runtime/library versions, execution mode, and completion state. Use a compact family mark in the header and favicon; use a local asset where practical.
2. **Paired summary cards.** Summarize tested model variants and task coverage. Include exact sample counts or duration/example totals. Make unsupported and incomplete coverage visible.
3. **Selection guide.** When results support recommendations, show the best complete model overall and per task, plus practical memory-tier picks where measurements justify them. State the selection rule and evidence type. Exclude incomplete models from quality picks; do not present a partial macro as a full-suite win.
4. **Methodology near the top.** Describe each dataset and split, sample selection, prompt/output contract, scorer, aggregation, decoding settings, and task-specific evaluation. Link primary dataset and evaluation references. Label locally defined metrics as local comparison scores.
5. **Quality results.** Present full-suite or group-level aggregates and per-task scores. Keep every tested model visible with completion, skipped, unsupported, or failed status. Sort and rank only comparable, valid scores. Explain macro weighting and minimum coverage directly beside the aggregate.
6. **Runtime and resource results.** Show task completion, generation latency, generated-token throughput when available, peak VRAM, output/parse errors, and relevant backend settings. Define measurement scope, including whole-device versus process memory and whether desktop baseline is included. For video or grounding, add domain-specific measurements such as duration, frames, temporal IoU, localization accuracy, and parser validity.
7. **Context and hardware fit.** Distinguish advertised context, configured run context, cached weight size, measured load/peak VRAM, and modeled estimates. Name the task producing a peak. Mark estimates as estimates, state assumptions, and do not imply that metadata or a context profile is a constrained-device accuracy rerun. Show empty states when no model meets a stated cap.
8. **Limitations and provenance.** Explain partial runs, failures, unsupported tasks, known compatibility issues, license/access limits, measurement boundaries, and selection caveats near the relevant result. Link the full `report.md` and any allowed aggregate data artifact.

## Visual system and interaction

- Reuse the Qwen/LLaVA dark palette, typography, spacing, panel treatment, rounded cards, compact sticky header, and restrained accent colors. Keep family identity in its mark and labels rather than changing the whole design system.
- Use green for primary highlights, cyan for secondary emphasis, amber for caveats, and red for errors. Pair color with text, symbols, or labels so meaning remains clear without color.
- Make tables horizontally scrollable on small screens and bound their vertical height when useful. Keep model names readable, stick headers where practical, use tabular numerals, and add accessible table captions and column labels.
- Provide useful sorting on result tables when the number of rows warrants it. Support chained sorting if needed, identify the active sort, and provide a clear-sort control. Rank each metric in its correct direction; use neutral styling for coverage and error counts.
- Make layouts responsive: summary and recommendation cards should collapse cleanly, methodology cards should reflow, and tables should remain usable without clipping the page.
- Keep text comfortably inset from every card, panel, and table border. Use consistent padding and line height; wrap long names without letting labels touch or collide with outlines. Inspect rendered desktop and narrow layouts and correct any crowded edges before delivery.
- Use semantic headings, keyboard-operable controls, visible focus styles, descriptive link text, and concise page metadata. Avoid animation that distracts from comparison.

## Data and implementation rules

- Build the page from the completed `report.md` and the benchmark's authoritative result artifacts. Generate a compact site data file from those sources instead of copying measurements into display code by hand.
- Treat recorded results as the source of truth. Do not invent recommendations, fill missing values, relabel estimates as measurements, or silently drop failed models. Recompute aggregates only when the rule is explicit and matches the report.
- Keep item-level prompts, images, and predictions out of the public site unless the benchmark report explicitly calls for them and data-use terms permit publication. Prefer aggregate-only data bundles.
- Include a visible timestamp and enough runtime/model metadata to identify what was run. Preserve checkpoint names, quantization, backend, context, precision, and relevant package versions where applicable.
- Keep the site static and locally usable unless the group requires another delivery method. Do not start a server or publish the site as part of authoring unless requested.
- Check generated links, table labels, responsive behavior, and consistency with `report.md` before delivery. Do not add fake data or mock measurements to fill empty sections.

## Suggested shared page shell

```text
Header / family identity
Hero / summary metadata
Paired cards / model variants + task coverage
Selection guide (when supported by completed results)
Benchmark methodology
Quality results
Runtime and resource results
Memory or domain-specific analysis
Limitations and provenance
Footer / benchmark identity + source report + data disclosure
```

Skip a section when the benchmark has no defensible metric for it. Explain relevant benchmark-specific evidence instead of adding generic or speculative estimates.
