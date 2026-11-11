(() => {
  "use strict";
  const data = window.GROUNDING_REPORT;
  if (!data) return;

  const $ = (selector, root = document) => root.querySelector(selector);
  const esc = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;", "'":"&#39;"}[c]));
  const pct = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
  const num = (value, digits = 1) => value == null ? "—" : Number(value).toFixed(digits);
  const status = (value) => `<span class="status ${esc(String(value).toLowerCase().replaceAll(" ", "-"))}">${esc(value)}</span>`;
  const table = (caption, headings, rows, className = "") => `
    <div class="table-wrap ${className}" tabindex="0" role="region" aria-label="${esc(caption)}">
      <table><caption>${esc(caption)}</caption><thead><tr>${headings.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>
    </div>`;
  const taskMetric = (id, task) => {
    const m = task.metrics || {};
    if (id === "refcocog") return `Acc@0.5 <strong>${pct(m.acc_iou_05)}</strong><br>Mean IoU ${pct(m.iou)}`;
    if (id === "coco") return `COCO AP <strong>${pct(task.coco_official_ap?.AP)}</strong><br>AP50 ${pct(task.coco_official_ap?.AP50)} · class F1 ${pct(m.micro_f1)}`;
    if (id === "screenspot_pro") return `Point hit <strong>${pct(m.point_in_target)}</strong><br>Acc@0.5 ${pct(m.acc_iou_05)}`;
    return `Box F1 <strong>${pct(m.f1)}</strong><br>P ${pct(m.precision)} · R ${pct(m.recall)} · OCR exact ${pct(m.ocr_exact_accuracy)} · NED ${pct(m.ocr_normalized_edit_similarity)}`;
  };

  const date = new Date(data.generated_at);
  const latest = data.latest_rf100_result_date
    ? new Date(`${data.latest_rf100_result_date}T12:00:00Z`).toLocaleDateString(undefined, {day:"numeric", month:"short", year:"numeric", timeZone:"UTC"})
    : "date not recorded";
  $("#hero-meta").innerHTML = [
    `<span class="pill">Latest RF100-VL results · ${esc(latest)}</span>`,
    `<span class="pill">${data.tested_model_count} pinned checkpoints · seed ${data.seed}</span>`,
    `<span class="pill">Report compiled ${esc(Number.isNaN(date.valueOf()) ? data.generated_at : date.toLocaleDateString())}</span>`
  ].join("");

  const completeTasks = data.models.filter((model) => model.complete_suite).length;
  const coreCount = Object.keys(data.tasks).length;
  const rf = data.rf100_summary || {};
  $("#summary").innerHTML = [
    `<article class="card"><div class="label">Grounding task coverage</div><div class="value">${completeTasks} / ${data.models.length} complete</div><p>${coreCount} tasks · ${coreCount * 100} fixed examples per checkpoint</p></article>`,
    `<article class="card"><div class="label">Current RF100-VL comparison</div><div class="value">${rf.panel_count || 0} / ${rf.panel_total || 0} panels</div><p>Up to 20 shared test images per subset; ${rf.models?.length || 0} checkpoints with complete subset coverage.</p></article>`,
    `<article class="card"><div class="label">Evidence handling</div><div class="value">By task</div><p>No combined score across localization, OCR, open-set detection, or runtime.</p></article>`
  ].join("");

  const highlightFor = (id, modelId) => {
    const t = data.models.find((m) => m.model_id === modelId)?.tasks[id];
    if (!t) return "No complete result";
    const metric = id === "coco" ? `COCO AP ${pct(t.coco_official_ap?.AP)}`
      : id === "screenspot_pro" ? `Point hit ${pct(t.metrics?.point_in_target)}`
        : id === "hiertext" ? `Box F1 ${pct(t.metrics?.f1)}` : `Acc@0.5 ${pct(t.metrics?.acc_iou_05)}`;
    return modelId ? `${modelId.replace("GroundingPI/", "").replace("nvidia/", "")} · ${metric}` : "No complete result";
  };
  const winners = [
    ["refcocog", "Referring expression", "100 phrases"],
    ["coco", "COCO open-vocabulary", "100 images · uniform VLM box scores"],
    ["screenspot_pro", "GUI grounding", "100 screenshots"],
    ["hiertext", "Text regions and OCR", "100 images · local word-box metric"]
  ];
  $("#selection").innerHTML = winners.map(([id, title, sample]) => {
    const modelId = data.task_winners[id];
    const result = modelId ? data.models.find((model) => model.model_id === modelId)?.tasks[id] : null;
    const metric = id === "coco" ? pct(result?.coco_official_ap?.AP)
      : id === "screenspot_pro" ? pct(result?.metrics?.point_in_target)
        : id === "hiertext" ? pct(result?.metrics?.f1) : pct(result?.metrics?.acc_iou_05);
    return `<article class="selection-card"><p class="eyebrow">${esc(sample)}</p><h3>${esc(title)}</h3><strong>${metric}</strong><p class="winner">${esc(highlightFor(id, modelId))}</p></article>`;
  }).join("");

  $("#task-results").innerHTML = Object.entries(data.tasks).map(([id, task]) => {
    const rows = data.models.map((model) => {
      const result = model.tasks[id];
      const issues = [[result.malformed, "malformed"], [result.invalid_outputs, "invalid"], [result.truncated, "truncated"], [result.empty_outputs, "empty"], [result.duplicate_predictions, "duplicate"]]
        .filter(([count]) => count > 0).map(([count, label]) => `${count} ${label}`).join(" · ") || "None recorded";
      return `<tr><td>${esc(model.model_id)}</td><td>${status(result.status)}</td><td>${result.completed}/${task.n}</td><td class="metric">${taskMetric(id, result)}</td><td>${esc(issues)}</td></tr>`;
    }).join("");
    return `<article class="task-card"><h3>${esc(task.label)}</h3><p class="subhead">${esc(task.split)} · ${task.n} examples · ${esc(task.measure)}</p>${table(`${task.label} current scorecard`, ["Model", "Status", "Coverage", "Scores", "Output issues"], rows)}</article>`;
  }).join("");

  const answerCards = Object.entries(data.aux_tasks || {}).map(([taskId, task]) => {
    const rows = data.models.map((model) => {
      const result = model.tasks[taskId];
      const metrics = result.metrics || {};
      const exact = taskId === "vqa_count" ? metrics.count_exact_match : metrics.exact_match;
      return `<tr><td>${esc(model.model_id)}</td><td>${result.completed}/${task.n}</td><td class="metric">${pct(metrics.vqa_score)}<br><span class="caveat">Exact ${pct(exact)}</span></td></tr>`;
    }).join("");
    return `<article class="task-card"><h3>${esc(task.label)}</h3><p class="subhead">${esc(task.split)} · ${task.n} examples</p>${table(`${task.label} exploratory results`, ["Model", "Coverage", "Consensus / exact"], rows)}</article>`;
  }).join("");
  $("#answer-task-results").innerHTML = answerCards;

  const visionRows = (data.vision || []).map((model) => {
    const det = model.detection || {}, seg = model.segmentation || {};
    const cls = model.classification || {}, runtime = model.runtime || {};
    return `<tr><td>${esc(model.label)}</td><td>${status(model.status)}</td><td>${model.sample_count}/100</td><td class="metric">${pct(det.AP)}<br>AP50 ${pct(det.AP50)}</td><td>${pct(cls.joint_detection_and_classification_accuracy)} joint<br>F1 ${pct(det.class_aware_F1_IoU50)}</td><td>${pct(seg.AP)}${seg.status === "unsupported" ? `<br><span class="caveat">No masks</span>` : ""}</td><td>${runtime.latency_mean_s == null ? "—" : `${num(runtime.latency_mean_s * 1000, 0)} ms`}</td></tr>`;
  }).join("");
  $("#vision-task-results").innerHTML = `<article class="task-card"><h3>COCO val2017 · shared 100-image panel</h3><p class="subhead">Box and mask AP use COCOeval. Grounding models emit no per-box confidence, so their AP uses uniform scores; class-aware F1 is shown alongside it.</p>${table("COCO vision baseline results", ["Model", "Status", "Coverage", "Box AP / AP50", "Classification", "Mask AP", "Mean latency"], visionRows)}</article>`;

  const currentDomain = (data.domain_benchmarks || []).filter((row) => !row.dataset.startsWith("RF100-VL:") && row.dataset !== "COCO val2017");
  const domainRows = currentDomain.map((result) => `<tr><td>${esc(result.dataset)}</td><td>${esc(result.model)}</td><td>${result.sample_count}/${result.expected_count}</td><td class="metric">${pct(result.AP)} / ${pct(result.AP50)}</td><td>${pct(result.class_aware_F1_IoU50)}</td><td>${pct(result.mask_AP)}</td><td>${result.latency_s == null ? "—" : `${num(result.latency_s * 1000)} ms`}</td></tr>`).join("");
  const dota = data.dota_track || {};
  const dotaMetrics = dota.metrics || {}, dotaRuntime = dota.runtime || {};
  const obbRow = `<tr><td>${status(dota.status || "not run")}</td><td class="metric">${pct(dotaMetrics.AP)} / ${pct(dotaMetrics.AP50)} / ${pct(dotaMetrics.AP75)}</td><td>458</td><td>${dotaRuntime.inference_ms_per_image == null ? "—" : `${num(dotaRuntime.inference_ms_per_image, 2)} ms`}</td><td>${dotaRuntime.images_per_s_inference_only == null ? "—" : num(dotaRuntime.images_per_s_inference_only, 2)}</td></tr>`;
  $("#domain-summary").innerHTML = `<article class="task-card"><h3>Current shared domain panels</h3><p class="subhead">These comparisons use the latest selected run for each model and dataset. DOTA and HOPE results are local sampled panels; LVIS is a 30-category panel, not the full leaderboard.</p>${table("Current domain-specific benchmark results", ["Dataset", "Model", "Images", "Box AP / AP50", "Class F1", "Mask AP", "Latency"], domainRows || `<tr><td colspan="7">No domain results recorded.</td></tr>`)}</article><article class="task-card"><h3>DOTA v1.0 · oriented-box reference</h3><p class="subhead">Full official validation split with a DOTA-pretrained YOLO26-L-OBB checkpoint. This in-domain reference complements, but is not directly comparable to, the sampled horizontal-box panels.</p>${table("DOTA oriented-box benchmark result", ["Status", "OBB AP / AP50 / AP75", "Images", "Inference latency", "Images / second"], obbRow)}</article>`;

  const rfModels = rf.models || [];
  const rfRows = (data.domain_benchmarks || []).filter((row) => row.dataset.startsWith("RF100-VL:")).map((row) => `<tr><td>${esc(row.dataset)}</td><td>${esc(row.model)}</td><td>${row.sample_count}/${row.expected_count}</td><td class="metric">${pct(row.AP)} / ${pct(row.AP50)}</td><td>${pct(row.class_aware_F1_IoU50)}</td><td>${row.latency_s == null ? "—" : `${num(row.latency_s * 1000)} ms`}</td></tr>`).join("");
  const rfSummaryRows = rfModels.map((model) => `<tr><td>${esc(model.model)}</td><td>${model.complete_panels}/${rf.panel_total}</td><td>${model.ap_panels}/${rf.panel_total}</td><td class="metric">${pct(model.macro_AP)}</td><td>${model.f1_panels}/${rf.panel_total}</td><td class="metric">${pct(model.macro_F1)}</td></tr>`).join("");
  $("#rf-summary").innerHTML = `<h3>RF100-VL · equally weighted subset summary</h3>${table("Current RF100-VL subset macro results", ["Model", "Complete panels", "AP panels", "Macro box AP", "F1 panels", "Macro class F1"], rfSummaryRows)}`;
  $("#rf100-table").innerHTML = table("Current per-subset RF100-VL results", ["Subset", "Model", "Images", "Box AP / AP50", "Class F1", "Latency"], rfRows);

  const methodCards = Object.entries(data.tasks).map(([id, task]) => {
    const specifics = id === "coco" ? "AP uses COCOeval and uniform box scores for VLM outputs without confidence values."
      : id === "screenspot_pro" ? "A point is correct when it falls inside the annotated target box."
        : id === "hiertext" ? "Word boxes are matched one-to-one at IoU 0.5; this is not the official polygon benchmark."
          : "The lmms-lab test derivative is sampled deterministically and is not the canonical direct RefCOCOg export.";
    return `<article class="method-card"><h3>${esc(task.label)} · ${esc(task.split)}</h3><p>${task.n} examples. ${esc(task.measure)}. ${esc(specifics)}</p></article>`;
  }).join("");
  $("#method-grid").innerHTML = methodCards;
  $("#method-notes").innerHTML = `<p><strong>Sampling:</strong> fixed manifests use seed ${data.seed}; panels are shared where stated.</p><p><strong>Timing:</strong> latency timing scopes differ by backend. LocateAnything's historical core measurements cover model generation only; GroundAnything measures the HTTP request through its complete response. Whole-device VRAM includes desktop baseline use.</p><p><strong>Selection:</strong> domain panels use the highest-coverage saved run, with the newest file breaking coverage ties. Core task scores use the current aggregate artifact after saved attempts are compared for higher coverage. Incomplete historical attempts remain in local run artifacts and are excluded from the current RF100-VL macro.</p>`;

  const taskCatalog = {...data.tasks, ...(data.aux_tasks || {})};
  const runtimeRows = data.models.flatMap((model) => Object.entries(model.tasks).map(([id, task]) => `<tr><td>${esc(model.model_id)}</td><td>${esc(taskCatalog[id]?.label || id)}</td><td>${status(task.status)}</td><td>${task.completed}/${task.n_examples}</td><td>${task.mean_latency_s == null ? "—" : `${num(task.mean_latency_s * 1000)} ms`}</td><td>${num(task.mean_output_tokens_per_s, 2)} tok/s</td><td>${num(task.peak_whole_device_vram_gib, 2)} GiB</td></tr>`)).join("");
  const metadataRows = data.models.map((model) => `<tr><td>${esc(model.model_id)}</td><td><code>${esc(model.revision || "not run")}</code></td><td>${esc(model.backend || "not run")}</td><td>${esc(model.precision || "—")}</td><td>${esc(model.generation_mode || "—")}</td><td>${esc(Object.entries(model.package_versions || {}).filter(([,v]) => v).map(([k,v]) => `${k} ${v}`).join(" · ") || "not recorded")}</td></tr>`).join("");
  $("#runtime-results").innerHTML = table("Pinned checkpoint and runner metadata", ["Model", "Revision", "Backend", "Precision", "Generation", "Runner packages"], metadataRows) + `<div style="height:12px"></div>` + table("Current per-task runtime measurements", ["Model", "Task", "Status", "Coverage", "Mean latency", "Output rate", "Peak whole-device VRAM"], runtimeRows);

  const limits = (data.dataset_tracks || []).map((track) => `<li><strong>${esc(track.dataset)}:</strong> ${esc(track.status)}. ${esc(track.note)}</li>`).join("");
  $("#limitations-content").innerHTML = `<p>Dataset assets and item-level results stay local. Scorecards show aggregate metrics from fixed samples. See the full report for detailed per-task runtime values, selected run configuration, and result notes.</p><ul class="limitations-list">${limits}</ul>`;
  $("#source-links").innerHTML = Object.entries(data.references).map(([label, url]) => `<li><a href="${esc(url)}" rel="noreferrer">${esc(label)}</a></li>`).join("");
})();
