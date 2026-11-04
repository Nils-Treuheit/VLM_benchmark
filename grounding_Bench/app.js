(() => {
  "use strict";
  const data = window.GROUNDING_REPORT;
  if (!data) return;
  const $ = (selector, root = document) => root.querySelector(selector);
  const esc = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
  const num = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
  const status = (value) => `<span class="status ${esc(String(value).toLowerCase().replaceAll(" ", "-"))}">${esc(value)}</span>`;
  const date = new Date(data.generated_at);
  $("#hero-meta").innerHTML = [
    `<span class="pill">Updated ${esc(Number.isNaN(date.valueOf()) ? data.generated_at : date.toLocaleString())}</span>`,
    `<span class="pill">${data.tested_model_count} tested · ${data.candidate_model_count} candidates</span>`,
    `<span class="pill">100 grounding · 25 VQAv2 · 100 TextVQA · seed ${data.seed}</span>`,
    `<span class="pill">${esc(data.runtime_summary)}</span>`,
    status(data.status)
  ].join("");
  $("#summary").innerHTML = [
    `<article class="card"><div class="label">Model coverage</div><div class="value">${data.tested_model_count} / ${data.candidate_model_count}</div><p>Tested checkpoints / pinned candidates. Every task keeps its own coverage.</p></article>`,
    `<article class="card"><div class="label">Task plan</div><div class="value">${Object.keys(data.tasks).length + Object.keys(data.aux_tasks || {}).length} task panels</div><p>Localization, text recognition, VQA, counting, and COCO vision quality. Samples are shared across models.</p></article>`
  ].join("");
  const complete = data.best_complete;
  const overallCopy = complete
    ? `<p class="selection-copy"><strong>${esc(complete.model_id)}</strong> leads the complete suite with a local task macro of <strong>${pct(complete.local_macro)}</strong>. It equally weights one primary score from each complete task. This local score is separate from image-QA rankings.</p>`
    : `<p class="selection-copy">No complete model is eligible for an overall quality pick. Per-task picks require complete task coverage.</p>`;
  const taskPicks = Object.entries(data.task_winners).map(([task, model]) => `<li>${esc(data.tasks[task].label)}: ${model ? esc(model) : "No complete result"}</li>`).join("");
  $("#selection-content").innerHTML = `${overallCopy}<ul>${taskPicks}</ul>`;
  $("#method-grid").innerHTML = Object.entries(data.tasks).map(([id, task]) => {
    const extra = id === "coco" ? "Categories are joined with the model's native separator. AP uses COCOeval on the selected image IDs when local annotations are supplied." :
      id === "hiertext" ? "Predicted boxes are matched one-to-one with ground-truth boxes at IoU 0.5." :
      id === "screenspot_pro" ? "A predicted point is correct when it falls inside the annotated target box." :
      "One annotated phrase is sent with the model's native referring-expression prompt.";
    return `<article class="method-card"><h3>${esc(task.label)}</h3><p><strong>${esc(task.split)}</strong> · ${task.n} examples. ${esc(task.measure)}. ${esc(extra)}</p></article>`;
  }).join("");
  const answerCards = Object.entries(data.aux_tasks || {}).map(([taskId, task]) => {
    const rows = data.models.map((model) => {
      const result = model.tasks[taskId];
      const metrics = result.metrics || {};
      const score = metrics.vqa_score == null ? "—" : pct(metrics.vqa_score);
      const exact = taskId === "vqa_count" ? metrics.count_exact_match : metrics.exact_match;
      return `<tr><td>${esc(model.model_id)}</td><td>${status(result.status)}</td><td>${result.completed}/${task.n}</td><td class="metric">${score}<br><span class="caveat">${taskId === "vqa_count" ? "count exact" : "canonical exact"} ${pct(exact)}</span></td><td>${result.mean_latency_s == null ? "—" : `${num(result.mean_latency_s * 1000, 0)} ms`}</td><td>${num(result.mean_output_tokens_per_s, 2)} tok/s</td><td>${num(result.peak_whole_device_vram_gib)} GiB</td></tr>`;
    }).join("");
    return `<article class="task-card"><h3>${esc(task.label)}</h3><p class="subhead">${esc(task.split)} · target ${task.n} examples · ${esc(task.measure)}</p><div class="table-wrap" tabindex="0" role="region" aria-label="${esc(task.label)} results table"><table><caption>${esc(task.label)} scores. VQA consensus is the primary measure; exact match is supplementary.</caption><thead><tr><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Coverage</th><th scope="col">Consensus / exact</th><th scope="col">Mean latency</th><th scope="col">Output rate</th><th scope="col">Peak VRAM</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }).join("");
  $("#answer-task-results").innerHTML = `<p class="subhead">The grounding checkpoints do not document general VQA as a native prompt task. These answer-generation scores are exploratory. VQAv2 panels reuse the same question IDs as the local Qwen panel. TextVQA tests OCR-dependent answers; HierText reports direct OCR exact accuracy and normalized edit similarity.</p>${answerCards}`;
  const scoreCells = (taskId, t) => {
    const m = t.metrics || {};
    if (taskId === "refcocog") return `Mean IoU ${pct(m.iou)}<br>Acc@0.5 ${pct(m.acc_iou_05)}`;
    if (taskId === "coco") {
      const ap = t.coco_official_ap || {};
      return ap.status === "complete" ? `COCO AP ${pct(ap.AP)}<br>AP50 ${pct(ap.AP50)}` : `Local micro F1 ${pct(m.micro_f1)}<br><span class="caveat">COCO AP unavailable</span>`;
    }
    if (taskId === "screenspot_pro") return `Point hit ${pct(m.point_in_target)}<br>Acc@0.5 ${pct(m.acc_iou_05)}`;
    return `Box F1 ${pct(m.f1)} · P ${pct(m.precision)} · R ${pct(m.recall)}<br>OCR exact ${pct(m.ocr_exact_accuracy)} · NED ${pct(m.ocr_normalized_edit_similarity)}`;
  };
  $("#task-results").innerHTML = Object.entries(data.tasks).map(([taskId, task]) => {
    const rows = data.models.map((model) => {
      const result = model.tasks[taskId];
      return `<tr><td>${esc(model.model_id)}</td><td>${status(result.status)}</td><td>${result.completed}/${task.n}</td><td class="metric">${scoreCells(taskId, result)}</td><td>${result.mean_latency_s == null ? "—" : `${num(result.mean_latency_s * 1000, 0)} ms`}</td><td>${num(result.peak_whole_device_vram_gib)} GiB</td><td>${result.malformed} malformed<br>${result.invalid_outputs} invalid<br>${result.truncated} truncated<br>${result.empty_outputs} empty<br>${result.duplicate_predictions} duplicate</td></tr>`;
    }).join("");
    return `<article class="task-card"><h3>${esc(task.label)}</h3><p class="subhead">${esc(task.split)} · target ${task.n} examples · ${esc(task.measure)}</p><div class="table-wrap" tabindex="0" role="region" aria-label="${esc(task.label)} results table"><table><caption>${esc(task.label)} scorecard; unsorted model rows are shown because runs may be incomplete.</caption><thead><tr><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Coverage</th><th scope="col">Score</th><th scope="col">Mean latency</th><th scope="col">Peak VRAM</th><th scope="col">Output issues</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }).join("");
  const visionRows = (data.vision || []).map((model) => {
    const det = model.detection || {}, seg = model.segmentation || {};
    const cls = model.classification || {}, grounding = model.grounding || {}, runtime = model.runtime || {};
    const metric = (v) => v == null ? "—" : pct(v);
    return `<tr><td>${esc(model.label)}<br><code>${esc(model.model_id)}</code></td><td>${status(model.status)}</td><td>${model.sample_count}/100</td><td class="metric">${metric(det.AP)}<br>AP50 ${metric(det.AP50)}</td><td class="metric">${metric(seg.AP)}<br>AP50 ${metric(seg.AP50)}${seg.status === "unsupported" ? `<br><span class="caveat">${esc(seg.reason)}</span>` : ""}</td><td>${metric(cls.accuracy_on_matched_objects)} matched<br>${metric(cls.joint_detection_and_classification_accuracy)} joint<br>F1 ${metric(det.class_aware_F1_IoU50)}</td><td>${metric(grounding.acc_iou_05)}<br>${grounding.sample_count || 0} phrases · ${grounding.mean_latency_s == null ? "—" : `${num(grounding.mean_latency_s * 1000, 0)} ms`}</td><td>${runtime.latency_mean_s == null ? "—" : `${num(runtime.latency_mean_s * 1000, 0)} ms`}<br>${runtime.images_per_s == null ? "—" : `${num(runtime.images_per_s, 2)} img/s`}</td><td>${num(runtime.peak_whole_device_vram_gib)} GiB</td></tr>`;
  }).join("");
  $("#vision-task-results").innerHTML = `<article class="task-card"><p class="subhead">COCO val2017 · shared 100-image manifest · serial batch-1 inference. Box and mask AP use COCOeval. Label accuracy is calculated over predictions matched geometrically to any object at IoU ≥ 0.5; class-aware F1 counts misses and wrong labels. YOLOE and SAM 3.1 can also run on the shared RefCOCOg text-grounding manifest. Grounding specialists lack detection confidence scores, so their COCO AP uses uniform scores; use class-aware F1 for direct comparisons.</p><div class="table-wrap" tabindex="0" role="region" aria-label="Object detection, classification, segmentation, and grounding results"><table><caption>Vision baseline quality and speed. A dash means the run or metric is unavailable.</caption><thead><tr><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Images</th><th scope="col">Box AP / AP50</th><th scope="col">Mask AP / AP50</th><th scope="col">Label accuracy / class F1</th><th scope="col">RefCOCOg grounding</th><th scope="col">Latency / throughput</th><th scope="col">Peak VRAM</th></tr></thead><tbody>${visionRows}</tbody></table></div></article>`;
  const dota = data.dota_track || {};
  const dotaMetrics = dota.metrics || {}, dotaRuntime = dota.runtime || {};
  const datasetRows = (data.dataset_tracks || []).map((track) => `<tr><td>${esc(track.dataset)}</td><td>${status(track.status)}</td><td>${esc(track.task)}</td><td>${esc(track.note)}</td></tr>`).join("");
  const domainRows = (data.domain_benchmarks || []).map((result) => `<tr><td>${esc(result.dataset)}</td><td>${esc(result.model)}</td><td>${status(result.status)}</td><td>${result.sample_count}/${result.expected_count}</td><td class="metric">${pct(result.AP)} / ${pct(result.AP50)}</td><td>${pct(result.class_aware_F1_IoU50)}</td><td>${pct(result.mask_AP)}</td><td>${result.latency_s == null ? "—" : `${num(result.latency_s * 1000, 1)} ms`}</td><td>${num(result.images_per_s, 2)}</td></tr>`).join("");
  $("#dataset-track-results").innerHTML = `<article class="task-card"><h3>Shared domain detection results</h3><p class="subhead">The DOTA and HOPE v2 manifests use identical images and category prompts for YOLOE-26-L, SAM 3.1, LocateAnything-3B, and GroundAnything-VLM. Box AP uses COCOeval; class-aware F1 remains useful when VLM scores lack confidence values. HOPE mask AP is available for models that return masks. YOLO uses 1024 pixels on DOTA and 640 on HOPE; SAM and VLMs use their native image processors.</p><div class="table-wrap" tabindex="0" role="region" aria-label="Shared DOTA and HOPE domain results"><table><caption>Shared benchmark results for domain-shift datasets.</caption><thead><tr><th scope="col">Dataset</th><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Images</th><th scope="col">Input size</th><th scope="col">Box AP / AP50</th><th scope="col">Class F1</th><th scope="col">Mask AP</th><th scope="col">Mean latency</th><th scope="col">Images/s</th></tr></thead><tbody>${domainRows || `<tr><td colspan="10">No domain runs recorded yet.</td></tr>`}</tbody></table></div></article><article class="task-card"><h3>DOTA v1.0 oriented detection</h3><p class="subhead">YOLO26-L-OBB · official DOTA validation protocol · DOTA-pretrained in-domain checkpoint reference. This result complements the shared horizontal-box comparison. Validation speed excludes model loading; peak whole-device VRAM sampling includes it.</p><div class="table-wrap" tabindex="0" role="region" aria-label="DOTA oriented detection result"><table><caption>DOTA OBB quality and inference speed.</caption><thead><tr><th scope="col">Status</th><th scope="col">OBB AP / AP50 / AP75</th><th scope="col">Inference latency</th><th scope="col">Images/s</th><th scope="col">Peak VRAM</th></tr></thead><tbody><tr><td>${status(dota.status || "not run")}</td><td class="metric">${pct(dotaMetrics.AP)} / ${pct(dotaMetrics.AP50)} / ${pct(dotaMetrics.AP75)}</td><td>${dotaRuntime.inference_ms_per_image == null ? "—" : `${num(dotaRuntime.inference_ms_per_image, 2)} ms`}</td><td>${dotaRuntime.images_per_s_inference_only == null ? "—" : num(dotaRuntime.images_per_s_inference_only, 2)}</td><td>${num(dotaRuntime.peak_whole_device_vram_gib)} GiB</td></tr></tbody></table></div></article><article class="task-card"><h3>Dataset selection and download plan</h3><div class="table-wrap" tabindex="0" role="region" aria-label="Dataset track fit"><table><caption>Dataset selection with task fit and known limitations.</caption><thead><tr><th scope="col">Dataset</th><th scope="col">Status</th><th scope="col">Task</th><th scope="col">Fit and limitation</th></tr></thead><tbody>${datasetRows}</tbody></table></div></article>`;
  const taskCatalog = {...data.tasks, ...(data.aux_tasks || {})};
  const runtimeRows = data.models.map((model) => Object.entries(model.tasks).map(([taskId, task]) =>
    `<tr><td>${esc(model.model_id)}</td><td>${esc(taskCatalog[taskId]?.label || taskId)}</td><td>${status(task.status)}</td><td>${task.mean_latency_s == null ? "—" : `${num(task.mean_latency_s * 1000, 0)} ms`}</td><td>${num(task.mean_output_tokens_per_s, 2)} tok/s</td><td>${num(task.peak_whole_device_vram_gib)} GiB</td></tr>`).join("")).join("");
  const metadataRows = data.models.map((model) => `<tr><td>${esc(model.model_id)}</td><td><code>${esc(model.revision || "not run")}</code></td><td>${esc(model.backend || "not run")}</td><td>${esc(model.precision || "—")}</td><td>${esc(model.generation_mode || "—")}</td><td>${num(model.model_load_peak_whole_device_vram_gib)} GiB</td><td>${esc(Object.entries(model.package_versions || {}).filter(([,v]) => v).map(([k,v]) => `${k} ${v}`).join(" · ") || "not recorded")}</td></tr>`).join("");
  $("#runtime-results").innerHTML = `<div class="table-wrap" tabindex="0" role="region" aria-label="Model runtime metadata"><table><caption>Checkpoint and runtime metadata. Package versions describe the client runner; GroundAnything model load occurs before this client run.</caption><thead><tr><th scope="col">Model</th><th scope="col">Revision</th><th scope="col">Backend</th><th scope="col">Precision</th><th scope="col">Generation mode</th><th scope="col">Load peak VRAM</th><th scope="col">Runner package versions</th></tr></thead><tbody>${metadataRows}</tbody></table></div><div class="table-wrap" tabindex="0" role="region" aria-label="Runtime results table" style="margin-top:12px"><table><caption>Per-task runtime; blank fields mean no recorded measurement.</caption><thead><tr><th scope="col">Model</th><th scope="col">Task</th><th scope="col">Status</th><th scope="col">Mean latency</th><th scope="col">Output rate</th><th scope="col">Peak VRAM</th></tr></thead><tbody>${runtimeRows}</tbody></table></div>`;
  const dataCopy = data.data_status === "local manifests used"
    ? "Local manifests were used for completed runs. Partial runs and failures remain visible in the scorecards."
    : data.data_status === "local auxiliary manifests prepared"
      ? `Local manifests are prepared for ${(data.prepared_tasks || []).join(", ")}. Any remaining task data still needs local manifests.`
      : data.data_status === "some local manifests prepared"
        ? `Local manifests are prepared for ${(data.prepared_tasks || []).join(", ")}; remaining task data is still needed.`
      : "No local manifests have been prepared for this report yet.";
  $("#limitations-content").innerHTML = `<p>${dataCopy} Dataset assets and item-level results stay local. The scorecards use the fixed samples and task-specific metrics described above. The full report links to per-item run artifacts and runtime details.</p><p>Direct OCR uses HierText word boxes with a local IoU-based matching score; it is not the official polygon benchmark. TextVQA is OCR-dependent VQA. VQA/counting prompts are exploratory transfer measurements because these checkpoints document grounding and OCR tasks, not general VQA. The COCO vision track requires local YOLO/SAM runtimes, weights, images, and annotations.</p>`;
  $("#source-links").innerHTML = Object.entries(data.references).map(([label, url]) => `<li><a href="${esc(url)}" rel="noreferrer">${esc(label)}</a></li>`).join("");
})();
