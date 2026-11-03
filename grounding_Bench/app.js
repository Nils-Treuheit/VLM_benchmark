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
    `<span class="pill">${data.sample_count_per_task} planned examples per task · seed ${data.seed}</span>`,
    `<span class="pill">${esc(data.runtime_summary)}</span>`,
    status(data.status)
  ].join("");
  $("#summary").innerHTML = [
    `<article class="card"><div class="label">Model coverage</div><div class="value">${data.tested_model_count} / ${data.candidate_model_count}</div><p>Tested checkpoints / pinned candidates. Every task keeps its own coverage.</p></article>`,
    `<article class="card"><div class="label">Task plan</div><div class="value">${Object.keys(data.tasks).length} × ${data.sample_count_per_task} planned</div><p>RefCOCOg, COCO, ScreenSpot-Pro, and HierText; samples are shared across models.</p></article>`
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
  const scoreCells = (taskId, t) => {
    const m = t.metrics || {};
    if (taskId === "refcocog") return `Mean IoU ${pct(m.iou)}<br>Acc@0.5 ${pct(m.acc_iou_05)}`;
    if (taskId === "coco") {
      const ap = t.coco_official_ap || {};
      return ap.status === "complete" ? `COCO AP ${pct(ap.AP)}<br>AP50 ${pct(ap.AP50)}` : `Local micro F1 ${pct(m.micro_f1)}<br><span class="caveat">COCO AP unavailable</span>`;
    }
    if (taskId === "screenspot_pro") return `Point hit ${pct(m.point_in_target)}<br>Acc@0.5 ${pct(m.acc_iou_05)}`;
    return `F1 ${pct(m.f1)} · P ${pct(m.precision)} · R ${pct(m.recall)}`;
  };
  $("#task-results").innerHTML = Object.entries(data.tasks).map(([taskId, task]) => {
    const rows = data.models.map((model) => {
      const result = model.tasks[taskId];
      return `<tr><td>${esc(model.model_id)}</td><td>${status(result.status)}</td><td>${result.completed}/${task.n}</td><td class="metric">${scoreCells(taskId, result)}</td><td>${result.mean_latency_s == null ? "—" : `${num(result.mean_latency_s * 1000, 0)} ms`}</td><td>${num(result.peak_whole_device_vram_gib)} GiB</td><td>${result.malformed} malformed<br>${result.invalid_outputs} invalid<br>${result.truncated} truncated<br>${result.empty_outputs} empty<br>${result.duplicate_predictions} duplicate</td></tr>`;
    }).join("");
    return `<article class="task-card"><h3>${esc(task.label)}</h3><p class="subhead">${esc(task.split)} · target ${task.n} examples · ${esc(task.measure)}</p><div class="table-wrap" tabindex="0" role="region" aria-label="${esc(task.label)} results table"><table><caption>${esc(task.label)} scorecard; unsorted model rows are shown because runs may be incomplete.</caption><thead><tr><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Coverage</th><th scope="col">Score</th><th scope="col">Mean latency</th><th scope="col">Peak VRAM</th><th scope="col">Output issues</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }).join("");
  const runtimeRows = data.models.map((model) => Object.entries(model.tasks).map(([taskId, task]) =>
    `<tr><td>${esc(model.model_id)}</td><td>${esc(data.tasks[taskId].label)}</td><td>${status(task.status)}</td><td>${task.mean_latency_s == null ? "—" : `${num(task.mean_latency_s * 1000, 0)} ms`}</td><td>${num(task.mean_output_tokens_per_s, 2)} tok/s</td><td>${num(task.peak_whole_device_vram_gib)} GiB</td></tr>`).join("")).join("");
  const metadataRows = data.models.map((model) => `<tr><td>${esc(model.model_id)}</td><td><code>${esc(model.revision || "not run")}</code></td><td>${esc(model.backend || "not run")}</td><td>${esc(model.precision || "—")}</td><td>${esc(model.generation_mode || "—")}</td><td>${num(model.model_load_peak_whole_device_vram_gib)} GiB</td><td>${esc(Object.entries(model.package_versions || {}).filter(([,v]) => v).map(([k,v]) => `${k} ${v}`).join(" · ") || "not recorded")}</td></tr>`).join("");
  $("#runtime-results").innerHTML = `<div class="table-wrap" tabindex="0" role="region" aria-label="Model runtime metadata"><table><caption>Checkpoint and runtime metadata. Package versions describe the client runner; GroundAnything model load occurs before this client run.</caption><thead><tr><th scope="col">Model</th><th scope="col">Revision</th><th scope="col">Backend</th><th scope="col">Precision</th><th scope="col">Generation mode</th><th scope="col">Load peak VRAM</th><th scope="col">Runner package versions</th></tr></thead><tbody>${metadataRows}</tbody></table></div><div class="table-wrap" tabindex="0" role="region" aria-label="Runtime results table" style="margin-top:12px"><table><caption>Per-task runtime; blank fields mean no recorded measurement.</caption><thead><tr><th scope="col">Model</th><th scope="col">Task</th><th scope="col">Status</th><th scope="col">Mean latency</th><th scope="col">Output rate</th><th scope="col">Peak VRAM</th></tr></thead><tbody>${runtimeRows}</tbody></table></div>`;
  $("#limitations-content").innerHTML = `<p>${data.data_status === "local manifests used" ? "Local manifests were used for these runs. Partial runs and failures remain visible in the scorecards." : "No annotations or image assets were found in the inspected local Hugging Face dataset cache when this report was generated. No assets were downloaded or copied."} Dataset assets and item-level results stay local. The scorecards use the fixed sample and task-specific metrics described above. The full report links to per-item run artifacts and runtime details.</p>`;
  $("#source-links").innerHTML = Object.entries(data.references).map(([label, url]) => `<li><a href="${esc(url)}" rel="noreferrer">${esc(label)}</a></li>`).join("");
})();
