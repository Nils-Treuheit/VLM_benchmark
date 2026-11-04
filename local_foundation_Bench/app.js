const TASKS = [
  ["ai2d", "AI2D"], ["textvqa", "TextVQA / OCR"],
  ["vqa_truth", "Truth"], ["vqa_count", "Counting"],
  ["vqa_description", "Description"], ["vqa_affordance", "Affordance"],
  ["vqa_negation", "Negation"], ["graph_grab_lite", "GRAB-Lite"],
];
const fmtPct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const fmt = value => value == null ? "—" : Number(value).toFixed(2);
const escapeText = value => value == null ? "—" : String(value);
const sortState = new Map();
const taskDone = item => item && item.completed_n === item.requested_n;
const taskScoreText = item => item ? `${fmtPct(item.score)}${taskDone(item) ? "" : ` · ${item.completed_n ?? 0}/${item.requested_n ?? "—"}`}` : "—";

function rankRows(rows, rules) {
  rows.forEach(row => { row.ranks = {}; });
  for (const [column, rule] of Object.entries(rules)) {
    const groups = new Map();
    rows.forEach(row => {
      const value = row.sort[column];
      if (!rule.eligible(row) || typeof value !== "number" || !Number.isFinite(value)) return;
      const group = rule.groupBy ? rule.groupBy(row) : "all";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push({row, value});
    });
    for (const entries of groups.values()) {
      const values = [...new Set(entries.map(entry => entry.value))]
        .sort((a, b) => rule.direction === "min" ? a - b : b - a);
      const positions = new Map(values.map((value, index) => [value, index + 1]));
      entries.forEach(({row, value}) => {
        const rank = positions.get(value);
        if (rank <= 5) row.ranks[column] = rank === 1 ? "best" : "top-five";
      });
    }
  }
}

function tableShell(container, headers, rows, sortable = false, rankRules = {}) {
  rankRows(rows, rankRules);
  container.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No recorded run is available.";
    container.append(empty);
    return;
  }
  const table = document.createElement("table");
  const originalRows = [...rows];
  if (sortable) {
    const controls = document.createElement("div");
    controls.className = "table-controls";
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "clear-sort";
    clear.textContent = "Clear sorting";
    clear.addEventListener("click", () => {
      sortState.delete(container.id);
      tableShell(container, headers, originalRows, sortable, rankRules);
    });
    controls.append(clear);
    if (Object.keys(rankRules).length) {
      const legend = document.createElement("p");
      legend.className = "comparison-legend";
      legend.innerHTML = '<span class="rank-badge rank-best">Best</span> highest or lowest valid result <span class="rank-badge rank-top-five">Top 5</span> next four ranks';
      controls.append(legend);
    }
    container.append(controls);
  }
  const caption = document.createElement("caption");
  caption.className = "sr-only";
  caption.textContent = headers.join(", ");
  table.append(caption);
  const head = table.createTHead().insertRow();
  const activeSort = sortState.get(container.id) || {index:-1, direction:"desc"};
  headers.forEach((header, index) => {
    const th = document.createElement("th");
    th.scope = "col";
    if (sortable) {
      th.setAttribute("aria-sort", activeSort.index === index ? (activeSort.direction === "desc" ? "descending" : "ascending") : "none");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = header + (activeSort.index === index ? (activeSort.direction === "desc" ? " ↓" : " ↑") : "");
      button.addEventListener("click", () => {
        const direction = activeSort.index === index && activeSort.direction === "desc" ? "asc" : "desc";
        sortState.set(container.id, {index, direction});
        rows.sort((a, b) => {
          const left = a.sort[index];
          const right = b.sort[index];
          if (left == null && right != null) return 1;
          if (right == null && left != null) return -1;
          const order = typeof left === "number" && typeof right === "number"
            ? right - left : String(left ?? "").localeCompare(String(right ?? ""));
          return direction === "desc" ? order : -order;
        });
        tableShell(container, headers, rows, sortable, rankRules);
      });
      th.append(button);
    } else th.textContent = header;
    head.append(th);
  });
  const body = table.createTBody();
  rows.forEach(rowData => {
    const tr = body.insertRow();
    rowData.cells.forEach((value, index) => {
      const isObject = value && typeof value === "object";
      const td = tr.insertCell();
      if (index && typeof value === "number") td.classList.add("numeric");
      if (isObject && value.className) td.classList.add(value.className);
      if (rowData.ranks?.[index]) {
        const rank = rowData.ranks[index];
        td.classList.add(rank === "best" ? "best-rank-1" : "best-rank-2to5");
      }
      if (isObject && value.href) {
        const link = document.createElement("a");
        link.href = value.href; link.target = "_blank"; link.rel = "noreferrer";
        link.textContent = value.text;
        td.append(link);
      } else {
        const text = isObject ? value.text : value;
        td.append(document.createTextNode(escapeText(text)));
      }
      const rank = rowData.ranks?.[index];
      if (rank) {
        const badge = document.createElement("span");
        badge.className = `rank-badge ${rank === "best" ? "rank-best" : "rank-top-five"}`;
        badge.textContent = rank === "best" ? "Best" : "Top 5";
        td.append(document.createTextNode(" "), badge);
      }
    });
  });
  container.append(table);
}

function renderPicks(models) {
  const target = document.getElementById("picks");
  target.replaceChildren();
  const complete = models.filter(model => model.full_benchmark_complete && model.full_benchmark_macro != null);
  if (!complete.length) {
    target.textContent = "No complete model qualifies for a selection pick.";
    return;
  }
  const addPick = (label, model, reason, metricLabel, metric) => {
    const article = document.createElement("article");
    article.className = "pick-card";
    const eyebrow = document.createElement("p"); eyebrow.className = "eyebrow"; eyebrow.textContent = label;
    const title = document.createElement("h3"); title.textContent = model.model_id;
    const why = document.createElement("p"); why.textContent = reason;
    const details = document.createElement("div"); details.className = "pick-metrics";
    const context = model.advertised_context
      ? `${model.configured_context ?? "—"} configured / ${model.advertised_context} advertised tokens`
      : `${model.configured_context ?? "—"} configured tokens`;
    details.textContent = `${metricLabel}: ${metric} · Peak: ${fmt(model.whole_device_peak_gib)} GiB · Context: ${context}`;
    article.append(eyebrow, title, why, details); target.append(article);
  };
  const best = [...complete].sort((a,b) => b.full_benchmark_macro - a.full_benchmark_macro)[0];
  addPick("BEST COMPLETE FULL-SUITE SCORE", best, "Highest unweighted mean across all eight task scores among complete runs.", "Macro", fmtPct(best.full_benchmark_macro));
  for (const [key, label] of TASKS) {
    const candidates = complete.filter(m => m.tasks?.[key]?.score != null && taskDone(m.tasks[key])).sort((a,b) => b.tasks[key].score - a.tasks[key].score);
    if (candidates.length) addPick(`BEST COMPLETE · ${label.toUpperCase()}`, candidates[0], `Highest completed ${label} score among the full-suite-complete models.`, "Score", fmtPct(candidates[0].tasks[key].score));
  }
}

function render(data) {
  const models = data.models || [];
  const complete = models.filter(model => model.full_benchmark_complete);
  document.getElementById("updated").textContent = `Updated ${data.updated_at || data.generated_at || "—"}`;
  document.getElementById("count").textContent = `${models.length} measured routes · ${complete.length} complete`;
  const versions = [...new Set(models.flatMap(model => Object.entries(model.runtime_versions || {}).map(([name, version]) => `${name} ${version}`)))];
  document.getElementById("runtime-versions").textContent = `Runtime: ${versions.join(" · ") || "version unavailable"}`;
  document.getElementById("model-count").textContent = String(models.length);
  document.getElementById("models-note").textContent = `${complete.length} routes completed all 825 requests; other runs remain visible with their status.`;
  renderPicks(models);

  const scoreRows = models.map(model => {
    const cells = [model.model_card ? {text:model.model_id,href:model.model_card} : model.model_id, model.status, model.full_benchmark_complete ? fmtPct(model.full_benchmark_macro) : "—", model.vqa_panel_macro == null ? "—" : fmtPct(model.vqa_panel_macro), ...TASKS.map(([key]) => taskScoreText(model.tasks?.[key]))];
    const sort = [model.model_id, model.status, model.full_benchmark_complete ? model.full_benchmark_macro : null, model.vqa_panel_macro, ...TASKS.map(([key]) => taskDone(model.tasks?.[key]) ? model.tasks[key].score : null)];
    cells[1] = {text:model.status, className:statusClass(model.status)};
    return {cells, sort, model};
  });
  const scoreRanks = {
    2: {direction:"max", eligible:row => row.model.full_benchmark_complete},
    3: {direction:"max", eligible:row => row.model.vqa_panel_macro != null},
  };
  TASKS.forEach(([key], index) => {
    scoreRanks[index + 4] = {direction:"max", eligible:row => taskDone(row.model.tasks?.[key])};
  });
  tableShell(document.getElementById("score-table"), ["Model", "Status", "Full-suite macro", "VQA panel macro", ...TASKS.map(([,label]) => label)], scoreRows, true, scoreRanks);

  const runtimeRows = [];
  for (const model of models) for (const [key, label] of TASKS) {
    const item = model.tasks?.[key];
    if (!item) continue;
    runtimeRows.push({cells:[model.model_id,label,`${item.completed_n ?? 0}/${item.requested_n ?? "—"}`,fmt(item.mean_generation_s),fmt(item.generated_tokens_per_s)],sort:[model.model_id,label,item.completed_n,item.mean_generation_s,item.generated_tokens_per_s],item});
  }
  tableShell(document.getElementById("runtime-table"), ["Model","Task","Completed","Mean seconds","Output tokens/s"], runtimeRows, true, {
    3: {direction:"min", groupBy:row => row.sort[1], eligible:row => taskDone(row.item)},
    4: {direction:"max", groupBy:row => row.sort[1], eligible:row => taskDone(row.item)},
  });

  const inventoryRows = (data.candidates || []).map(candidate => ({
    cells:[candidate.name,{text:candidate.checkpoint,href:candidate.source},candidate.license || "See model card",{text:candidate.status,className:statusClass(candidate.status)},candidate.detail],
    sort:[candidate.name,candidate.checkpoint,candidate.license,candidate.status,candidate.detail],
  }));
  tableShell(document.getElementById("inventory-table"), ["Candidate","Checkpoint","License","Status","Disposition"], inventoryRows, true);
  const notes = document.getElementById("notes");
  notes.replaceChildren();
  (data.notes || []).forEach(text => {const li=document.createElement("li");li.textContent=text;notes.append(li);});
}

function statusClass(status) {
  const normalized = String(status || "").toLowerCase();
  if (["failed", "error", "preflight_failed"].includes(normalized)) return "status-label status-error";
  if (["incomplete", "unsupported", "not_run", "partial", "skipped"].includes(normalized)) return "status-label status-warning";
  return "status-label status-neutral";
}

fetch("data.json").then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(render).catch(error => {
  const box = document.createElement("p"); box.className = "load-error";
  box.textContent = `Could not load report data: ${error.message}. Open report.md for the source results.`;
  document.querySelector("main").prepend(box);
});
