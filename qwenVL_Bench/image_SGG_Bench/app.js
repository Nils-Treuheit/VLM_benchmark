const fmt = value => Number.isFinite(value) ? (100 * value).toFixed(2) + "%" : "—";
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

function tableFor(block) {
  const rows = (block.models || []).map(model => {
    const m = model.metrics || {};
    const r50 = m.r_at_k?.["50"];
    const mr50 = m.mr_at_k?.["50"];
    return `<tr><th scope="row">${esc(model.id)}</th><td>${esc(model.backend || "—")}</td><td>${fmt(mr50)}</td><td>${fmt(r50)}</td><td>${fmt(m.triplet_f1_at_50)}</td><td>${fmt(m.object_recall_at_50)}</td><td>${model.valid_json ?? 0}/${model.requested ?? 0}</td><td>${model.completed ?? 0}/${model.requested ?? 0}</td></tr>`;
  }).join("");
  if (!rows) return `<p class="empty">No complete model run is available for this panel yet.</p>`;
  return `<div class="table-wrap"><table><caption>${esc(block.name)} · split ${esc(block.split)} · ${block.panel_n} fixed images · ${block.class_count} object labels · ${block.predicate_count} predicates</caption><thead><tr><th scope="col">Model</th><th scope="col">Backend</th><th scope="col">mR@50</th><th scope="col">R@50</th><th scope="col">Triplet F1@50</th><th scope="col">Object R@50</th><th scope="col">Valid JSON</th><th scope="col">Completed</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

fetch("data.json").then(response => {
  if (!response.ok) throw new Error(`data.json returned ${response.status}`);
  return response.json();
}).then(data => {
  document.querySelector("#updated").textContent = `Updated ${new Date(data.generated_at).toLocaleString()}`;
  document.querySelector("#coverage").textContent = `${data.completed_model_count} of ${data.candidate_count} cached candidates have complete panels`;
  document.querySelector("#model-count").textContent = `${data.completed_model_count} / ${data.candidate_count}`;
  const macros = Object.values(data.suite_macro_mr_at_50 || {});
  document.querySelector("#macro").textContent = macros.length ? fmt(Math.max(...macros)) : "Pending";
  const pending = data.pending_models || [];
  const state = document.querySelector("#run-state");
  if (data.run_complete) {
    state.className = "run-state complete";
    state.textContent = `All ${data.candidate_count} cached candidates have full three-panel records.`;
  } else {
    const incomplete = data.incomplete_models || [];
    state.className = "run-state pending";
    const detail = incomplete.length
      ? `${incomplete.length} model${incomplete.length === 1 ? " has" : "s have"} partial records (${incomplete.map(esc).join(", ")}); ${pending.length} candidate${pending.length === 1 ? "" : "s"} lack a complete three-panel record.`
      : `${pending.length} model${pending.length === 1 ? " has" : "s have"} no complete record: ${pending.map(esc).join(", ") || "complete aggregate data"}.`;
    state.innerHTML = `<b>Partial comparison.</b> ${detail} The tables show available full-panel scores.`;
  }
  const labels = {vg150:"VG-150 canonical test", openimages_v7:"Open Images V7 relationships", psg:"PSG box-adapted SGG"};
  document.querySelector("#datasets").innerHTML = Object.entries(labels).map(([key, label]) => {
    const block = data.datasets?.[key];
    return `<section class="dataset"><div class="section-head"><div><p class="eyebrow">${key === "psg" ? "BOX-ADAPTED TRACK" : "FIXED SUBSET"}</p><h2>${esc(label)}</h2></div><a href="report.md">Metrics and provenance ↗</a></div>${block ? tableFor(block) : `<p class="empty">${esc(label)} results are not aggregated yet.</p>`}</section>`;
  }).join("");
}).catch(error => {
  document.querySelector("#datasets").innerHTML = `<p class="error">Could not load aggregate report data: ${esc(error.message)}</p>`;
  document.querySelector("#run-state").className = "run-state pending";
  document.querySelector("#run-state").textContent = "Report data is not ready.";
});
