const $ = (id) => document.getElementById(id);
const pct = (v) => v == null ? '—' : `${(Number(v) * 100).toFixed(1)}%`;
const nfmt = (v, digits = 2) => v == null ? '—' : Number(v).toFixed(digits);
const esc = (v) => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const taskOrder = ['ai2d','textvqa','vqa_truth','vqa_count','vqa_description','vqa_affordance','vqa_negation','graph_grab_lite'];

function table(targetId, caption, headers, rows) {
  const target = $(targetId);
  if (!rows.length) { target.innerHTML = '<div class="panel empty">No measurements recorded.</div>'; return; }
  target.innerHTML = `<div class="table-shell"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
}

function card(eyebrow, title, copy) {
  return `<article class="selection-card"><p class="eyebrow">${esc(eyebrow)}</p><h3>${esc(title)}</h3><p>${esc(copy)}</p></article>`;
}

function render(data) {
  const models = data.models || [];
  const full = models.filter(m => m.full_benchmark_complete && m.full_benchmark_macro != null);
  $('generated').textContent = `Updated ${data.generated_at || '—'}`;
  $('runtime-version').textContent = [...new Set(models.map(m => `Transformers ${m.runtime_versions?.transformers || '—'}`))].join(' · ');
  $('run-state').textContent = `${models.filter(m=>m.status==='complete').length}/${models.length} complete`;
  $('variant-count').textContent = `${models.length} configured`;
  $('variant-names').textContent = models.map(m=>m.model_id).join(' · ') || 'No model runs recorded yet.';
  if (data.status && data.status !== 'complete') {
    const banner = $('report-status'); banner.hidden = false;
    banner.className = data.status === 'blocked' ? 'status-banner error' : 'status-banner';
    banner.textContent = data.status_message || `Run state: ${data.status}. Read the full report for details.`;
  }

  const selection = $('selection'); selection.replaceChildren();
  if (!full.length) selection.innerHTML = '<div class="panel empty">No complete full-suite runs support quality recommendations.</div>';
  else {
    const overall = [...full].sort((a,b)=>b.full_benchmark_macro-a.full_benchmark_macro)[0];
    selection.insertAdjacentHTML('beforeend', card('BEST COMPLETE OVERALL', overall.model_id,
      `${pct(overall.full_benchmark_macro)} unweighted macro across eight fully completed tasks.`));
    for (const task of taskOrder) {
      const candidates = full.filter(m=>m.tasks?.[task]?.score != null);
      if (!candidates.length) continue;
      const best = [...candidates].sort((a,b)=>b.tasks[task].score-a.tasks[task].score)[0];
      selection.insertAdjacentHTML('beforeend', card(`BEST · ${(data.task_labels?.[task] || task).toUpperCase()}`,
        best.model_id, `${pct(best.tasks[task].score)} on ${best.tasks[task].completed_n}/${best.tasks[task].requested_n} examples.`));
    }
  }

  const qualityRows = models.map(m=>`<tr><td class="model">${esc(m.model_id)}</td><td class="status-${esc(m.status)}">${esc(m.status)}</td><td>${esc(m.preflight?.passed ? 'passed' : m.preflight ? 'failed' : 'not run')}</td>${taskOrder.map(t=>`<td class="numeric">${pct(m.tasks?.[t]?.score)}</td>`).join('')}<td class="numeric">${m.complete_tasks ?? 0}/8</td><td class="numeric">${pct(m.full_benchmark_macro)}</td></tr>`);
  table('quality-table','Model task scores and completion',['Model','Status','Preflight',...(data.task_order||taskOrder).map(t=>data.task_labels?.[t]||t),'Tasks complete','Full-suite macro'],qualityRows);

  const groupNames = ['function','properties','real','series','transform'];
  const groupRows = models.map(m=>{
    const groups=m.tasks?.graph_grab_lite?.group_scores||{};
    return `<tr><td class="model">${esc(m.model_id)}</td>${groupNames.map(g=>`<td class="numeric">${pct(groups[g])}</td>`).join('')}</tr>`;
  });
  table('graph-group-table','GRAB-Lite graph-analysis groups',['Model','Function','Properties','Real figures','Series','Transformations'],groupRows);

  const runtimeRows = [];
  for (const m of models) for (const task of taskOrder) {
    const d=m.tasks?.[task]||{};
    runtimeRows.push(`<tr><td class="model">${esc(m.model_id)}</td><td>${esc(data.task_labels?.[task]||task)}</td><td class="numeric">${d.completed_n??0}/${d.requested_n??data.task_sizes?.[task]??'—'}</td><td class="numeric">${nfmt(d.mean_generation_s)}</td><td class="numeric">${nfmt(d.generated_tokens_per_s)}</td><td class="numeric">${nfmt(d.peak_allocated_gib)}</td><td class="numeric">${d.parse_errors??0}</td></tr>`);
  }
  table('runtime-table','Per-task runtime and memory',['Model','Task','Completed / requested','Mean seconds','Tokens / second','Peak allocated GiB','Parse errors'],runtimeRows);

  const memoryRows = models.map(m=>`<tr><td class="model">${esc(m.model_id)}</td><td>${esc(m.checkpoint_context ?? '—')}</td><td>${esc(m.image_resolution_px ? `${m.image_resolution_px}px` : 'processor default')}</td><td class="numeric">${nfmt(m.weight_gib)}</td><td class="numeric">${nfmt(m.peak_allocated_gib)}</td><td>${esc(m.runtime_versions?.python||'—')}</td><td>${esc(m.runtime_versions?.transformers||'—')}</td><td>${esc(m.backend||'—')}</td></tr>`);
  table('memory-table','Checkpoint and runtime metadata',['Model','Advertised context','Input resolution','Cached weights GiB','Peak allocated GiB','Python','Transformers','Backend'],memoryRows);

  const notes = [];
  for (const m of models) {
    notes.push(`<li><b>${esc(m.model_id)}:</b> ${esc(m.status)}. ${esc(m.error || m.scoring_route?.evidence || m.preflight?.error || 'No additional run note.')}</li>`);
  }
  notes.push('<li>Peak allocation is process-local; whole-device desktop baseline is not included.</li>');
  notes.push('<li>Florence uses a separate Transformers 4.44.2 environment. Its unregistered &lt;VQA&gt; path is only scored after real-image AI2D and TextVQA preflights pass.</li>');
  $('limitations').innerHTML = `<ul>${notes.join('')}</ul>`;
}

fetch('data.json').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then(render)
  .catch(error => { $('report-status').hidden = false; $('report-status').className = 'status-banner'; $('report-status').textContent = `Report data has not been generated yet (${error.message}).`; });
