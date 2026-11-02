const TASKS = { ai2d: 'AI2D', textvqa: 'TextVQA' };
const states = new Map();

const esc = value => String(value ?? '—');
const pct = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const pp = value => value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(1)} pp`;
const n = (value, digits = 2, suffix = '') => value == null ? '—' : `${Number(value).toFixed(digits)}${suffix}`;
const context = value => {
  if (value == null) return '—';
  const v = Number(value);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(v % 1_000_000 ? 2 : 0)}M`;
  if (v >= 1024) return `${(v / 1024).toFixed(v % 1024 ? 1 : 0)}K`;
  return String(v);
};
const profileCell = (model, name) => {
  const tier = model.context_profile?.tiers?.find(item => item.tier === name);
  return tier || {};
};

function modelSortValue(value) {
  if (value == null || value === '—') return null;
  if (typeof value === 'number') return value;
  return value;
}

function compareRows(a, b, columns, chain) {
  for (const entry of chain) {
    const column = columns.find(item => item.key === entry.key);
    if (!column) continue;
    const av = modelSortValue(column.sortValue ? column.sortValue(a) : a[column.key]);
    const bv = modelSortValue(column.sortValue ? column.sortValue(b) : b[column.key]);
    if (av == null && bv == null) continue;
    if (av == null) return 1;
    if (bv == null) return -1;
    let cmp;
    if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv;
    else cmp = String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: 'base' });
    if (cmp) return entry.direction === 'asc' ? cmp : -cmp;
  }
  return a.__index - b.__index;
}

function renderTable(targetId, title, sourceRows, columns, tableKey) {
  const target = document.getElementById(targetId);
  const shell = document.createElement('div');
  shell.className = 'table-shell';
  const toolbar = document.createElement('div');
  toolbar.className = 'sort-toolbar';
  const chainLabel = document.createElement('div');
  chainLabel.className = 'sort-chain';
  chainLabel.setAttribute('aria-live', 'polite');
  const legend = document.createElement('span');
  legend.className = 'comparison-legend';
  const referenceName = window.BENCHMARK_REFERENCE_MODEL_ID || 'best overall quality model';
  legend.textContent = `Green triangle = better · red triangle = worse · no marker = tied (vs ${referenceName})`;
  const clear = document.createElement('button');
  clear.className = 'sort-clear';
  clear.type = 'button';
  clear.textContent = 'Clear sort';
  const table = document.createElement('table');
  const caption = document.createElement('caption');
  caption.className = 'sr-only';
  caption.textContent = title;
  table.append(caption);
  const thead = document.createElement('thead');
  const labels = document.createElement('tr');
  const controls = document.createElement('tr');
  controls.className = 'sort-row';
  const body = document.createElement('tbody');

  columns.forEach((column, index) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = column.label;
    labels.append(th);

    const controlTh = document.createElement('th');
    const select = document.createElement('select');
    select.setAttribute('aria-label', `Sort ${title} by ${column.label}`);
    select.innerHTML = '<option value="">None</option><option value="asc">Ascending</option><option value="desc">Descending</option>';
    select.dataset.key = column.key;
    select.addEventListener('change', () => {
      const chain = states.get(tableKey) || [];
      const position = chain.findIndex(item => item.key === column.key);
      if (!select.value) {
        if (position !== -1) chain.splice(position, 1);
      } else if (position === -1) {
        chain.push({ key: column.key, direction: select.value });
      } else {
        chain[position].direction = select.value;
      }
      states.set(tableKey, chain);
      draw();
    });
    controlTh.append(select);
    controls.append(controlTh);
  });
  thead.append(labels, controls);
  table.append(thead, body);

  function draw() {
    const chain = states.get(tableKey) || [];
    controls.querySelectorAll('select').forEach(select => {
      select.value = chain.find(item => item.key === select.dataset.key)?.direction || '';
    });
    chainLabel.replaceChildren();
    if (!chain.length) {
      const hint = document.createElement('span');
      hint.textContent = 'No sort applied · choose a column to start';
      chainLabel.append(hint);
    } else {
      chain.forEach((item, index) => {
        const column = columns.find(col => col.key === item.key);
        const chip = document.createElement('span');
        chip.className = 'sort-chip';
        chip.textContent = `${index + 1}. ${column?.label || item.key} ${item.direction === 'asc' ? '↑' : '↓'}`;
        chainLabel.append(chip);
      });
    }
    const indexed = sourceRows.map((row, index) => ({ ...row, __index: index }));
    indexed.sort((a, b) => compareRows(a, b, columns, chain));
    const metricRanks = new Map();
    const keepModels = new Set(window.BENCHMARK_KEEP_MODELS || []);
    const referenceModelId = window.BENCHMARK_REFERENCE_MODEL_ID;
    const metricReferences = new Map();
    columns.filter(column => column.best).forEach(column => {
      const groups = new Map();
      sourceRows.forEach((row, index) => {
        const group = column.rankGroup ? String(row[column.rankGroup] ?? '') : '';
        if (!groups.has(group)) groups.set(group, []);
        const value = modelSortValue(column.sortValue ? column.sortValue(row) : row[column.key]);
        if (value != null) groups.get(group).push({ index, value });
      });
      const ranks = new Map();
      groups.forEach(items => {
        items.sort((a, b) => {
          let cmp = typeof a.value === 'number' && typeof b.value === 'number'
            ? a.value - b.value
            : String(a.value).localeCompare(String(b.value), undefined, { numeric: true, sensitivity: 'base' });
          if (column.best === 'max') cmp = -cmp;
          return cmp || a.index - b.index;
        });
        items.forEach((item, index) => ranks.set(item.index, index + 1));
      });
      metricRanks.set(column.key, ranks);

      const referenceValues = new Map();
      sourceRows.forEach(row => {
        if (row.model_id !== referenceModelId) return;
        const value = modelSortValue(column.sortValue ? column.sortValue(row) : row[column.key]);
        if (typeof value !== 'number' || !Number.isFinite(value)) return;
        const group = column.rankGroup ? String(row[column.rankGroup] ?? '') : '';
        referenceValues.set(group, value);
      });
      metricReferences.set(column.key, referenceValues);
    });
    body.replaceChildren();
    indexed.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(column => {
        const td = document.createElement('td');
        const value = column.display ? column.display(row) : esc(row[column.key]);
        const metricValue = column.numeric && column.best
          ? modelSortValue(column.sortValue ? column.sortValue(row) : row[column.key])
          : null;
        const group = column.rankGroup ? String(row[column.rankGroup] ?? '') : '';
        const reference = metricReferences.get(column.key)?.get(group);
        if (typeof metricValue === 'number' && Number.isFinite(reference) && Math.abs(metricValue - reference) > 1e-9) {
          td.append(document.createTextNode(value));
          const better = column.best === 'max' ? metricValue > reference : metricValue < reference;
          const mark = document.createElement('span');
          mark.className = `comparison-mark ${better ? 'better' : 'worse'}`;
          mark.setAttribute('aria-hidden', 'true');
          mark.textContent = better === (column.best === 'max') ? '▲' : '▼';
          td.append(mark);
          td.setAttribute('aria-label', `${value}, ${better ? 'better' : 'worse'} than ${referenceName}`);
        } else {
          td.textContent = value;
        }
        if (column.className) td.className = column.className(row);
        if (column.numeric) td.classList.add('numeric');
        if (column.key === 'model_id' || column.key === 'model') td.classList.add('model-cell');
        if (keepModels.has(row.model_id) && column.numeric) td.classList.add('keep-model-cell');
        const rank = metricRanks.get(column.key)?.get(row.__index);
        if (rank === 1) td.classList.add('best-rank-1');
        else if (rank >= 2 && rank <= 5) td.classList.add('best-rank-2to5');
        tr.append(td);
      });
      body.append(tr);
    });
  }
  clear.addEventListener('click', () => { states.set(tableKey, []); draw(); });
  toolbar.append(chainLabel, legend, clear);
  shell.append(toolbar, table);
  target.replaceChildren(shell);
  draw();
}

function buildScoreRows(models) {
  const byId = new Map(models.map(model => [model.model_id, model]));
  return models.map(model => {
    const reference = byId.get(model.comparison_reference);
    const ai2d = model.tasks?.ai2d?.score;
    const textvqa = model.tasks?.textvqa?.score;
    return {
      model_id: model.model_id,
      backend: model.backend,
      reference: model.comparison_reference,
      ai2d,
      ai2d_delta: ai2d == null || reference?.tasks?.ai2d?.score == null ? null : (ai2d - reference.tasks.ai2d.score) * 100,
      textvqa,
      textvqa_delta: textvqa == null || reference?.tasks?.textvqa?.score == null ? null : (textvqa - reference.tasks.textvqa.score) * 100,
      status: model.status || 'unknown',
    };
  });
}

function buildRuntimeRows(models) {
  const rows = [];
  models.forEach(model => Object.entries(model.tasks || {}).forEach(([task, metrics]) => {
    rows.push({
      model_id: model.model_id, task, requested_n: metrics.requested_n, completed_n: metrics.completed_n,
      parse_errors: metrics.parse_errors, peak_gpu_used_gib: metrics.peak_gpu_used_gib,
      mean_wall_s: metrics.mean_wall_s, generated_tokens_per_s: metrics.generated_tokens_per_s,
      context: model.configured_context || model.checkpoint_context, kv_cache: model.kv_cache,
    });
  }));
  return rows;
}

function buildPanelRows(models) {
  return models.map(model => {
    const keys = ['vqa_truth','vqa_count','vqa_description','textvqa','vqa_affordance','vqa_negation'];
    const categories = keys.map(key => model.tasks?.[key]).filter(metrics => metrics?.score != null);
    const complete = categories.filter(metrics => metrics.completed_n === metrics.requested_n);
    const partial = categories.length - complete.length;
    return ({
    model_id: model.model_id,
    overall: model.vqa_panel_score,
    available_mean: complete.length ? complete.reduce((sum, metrics) => sum + metrics.score, 0) / complete.length : null,
    category_coverage: `${complete.length}/6 complete${partial ? ` · ${partial} partial` : ''}`,
    truth: model.tasks?.vqa_truth?.score,
    counting: model.tasks?.vqa_count?.score,
    description: model.tasks?.vqa_description?.score,
    ocr: model.tasks?.textvqa?.score,
    affordance: model.tasks?.vqa_affordance?.score,
    negation: model.tasks?.vqa_negation?.score,
  });
  });
}

function buildGraphRows(graph) {
  return (graph?.models || []).map(model => ({
    model_id: model.model_id,
    score: model.score,
    delta_vs_qwen3_2b: model.delta_vs_qwen3_2b,
    properties: model.task_scores?.properties,
    function: model.task_scores?.function,
    series: model.task_scores?.series,
    transform: model.task_scores?.transform,
    real: model.task_scores?.real,
    completed_n: model.completed_n,
    peak_gpu_used_gib: model.peak_gpu_used_gib,
    mean_wall_s: model.mean_wall_s,
    generated_tokens_per_s: model.generated_tokens_per_s,
    context: model.context,
  }));
}

function buildContextRows(models) {
  return models.map(model => {
    const min = profileCell(model, 'min');
    const dft = profileCell(model, 'default');
    const max = profileCell(model, 'max');
    return {
      model_id: model.model_id,
      min_vram: min.estimated_vram_gib, min_context: min.context_tokens,
      default_vram: dft.estimated_vram_gib, default_context: dft.context_tokens,
      max_vram: max.estimated_vram_gib, max_context: max.context_tokens,
      peak_gpu_used_gib: model.peak_gpu_used_gib,
      mtp_supported: model.mtp_supported || 'no', backend: model.backend,
    };
  });
}

function renderRecommendations(data) {
  const target = document.getElementById('recommendation-grid');
  target.replaceChildren();
  (data.recommendations || []).forEach(item => {
    const card = document.createElement('article');
    card.className = 'recommendation-card';
    const tag = document.createElement('p');
    tag.className = 'eyebrow';
    tag.textContent = item.label;
    const title = document.createElement('h3');
    title.textContent = item.model_id;
    const rationale = document.createElement('p');
    rationale.className = 'recommendation-reason';
    rationale.textContent = item.rationale;
    const metrics = document.createElement('dl');
    metrics.className = 'recommendation-metrics';
    const addMetric = (label, value) => {
      const wrapper = document.createElement('div');
      const dt = document.createElement('dt'); dt.textContent = label;
      const dd = document.createElement('dd'); dd.textContent = value;
      wrapper.append(dt, dd); metrics.append(wrapper);
    };
    if (item.label === 'Graph specialist') {
      addMetric('Mean Score · all benchmarks', pct(item.all_benchmark_mean_score));
      addMetric('Graph Specific Benchmark Score', pct(item.graph_score));
      addMetric('Configured context', context(item.context));
      addMetric('Peak VRAM', `${n(item.peak_gpu_gib, 2)} GiB`);
    } else {
      addMetric('AI2D', pct(item.ai2d_score));
      addMetric('TextVQA', pct(item.textvqa_score));
      if (item.vqa_panel_score != null) addMetric('VQA panel', pct(item.vqa_panel_score));
    }
    if (item.edge_limit_gib) {
      addMetric(`Edge ${item.edge_limit_gib} GiB`, `${n(item.edge_estimated_vram_8k_gib, 1)} GiB @8K`);
      addMetric('Max context fit', context(item.edge_max_context));
    } else if (item.label === 'Fastest average generation with acceptable quality') {
      addMetric('Mean output speed', `${n(item.mean_tokens_s, 1)} tok/s`);
    } else if (item.label !== 'Graph specialist') {
      addMetric('Mean score', pct(item.combined_score));
      if (item.context) addMetric('Configured context', context(item.context));
    }
    card.append(tag, title, rationale, metrics);
    target.append(card);
  });
}

const percentColumns = (key, label, best = 'max', rankGroup = null) => ({ key, label, numeric: true, best, rankGroup, sortValue: row => row[key], display: row => pct(row[key]), className: row => row[key] == null ? 'dim' : '' });
const numberColumn = (key, label, digits = 2, suffix = '', best = null, rankGroup = null) => ({ key, label, numeric: true, best, rankGroup, sortValue: row => row[key], display: row => n(row[key], digits, suffix) });
const contextColumn = (key, label) => ({ key, label, numeric: true, best: 'max', sortValue: row => row[key], display: row => context(row[key]) });
const plainColumn = (key, label, className) => ({ key, label, sortValue: row => row[key], display: row => esc(row[key]), className: className ? () => className : null });

async function init() {
  const response = await fetch('data.json?v=summary-pair-1');
  if (!response.ok) throw new Error(`Could not load benchmark data (${response.status})`);
  const data = await response.json();
  const models = data.models;
  window.BENCHMARK_KEEP_MODELS = (data.recommendations || []).map(item => item.model_id).filter(Boolean);
  const overallPick = (data.recommendations || []).find(item => item.title === 'Best overall quality');
  window.BENCHMARK_REFERENCE_MODEL_ID = overallPick?.model_id || null;
  document.getElementById('generated').textContent = `Updated ${data.generated_at}`;
  if (data.preliminary) {
    const banner = document.getElementById('preliminary-banner');
    banner.textContent = `PRELIMINARY RESULTS · ${data.preliminary_note}`;
    banner.hidden = false;
  }
  const modelCount = document.getElementById('model-count');
  if (modelCount) modelCount.textContent = models.length;
  const modelTotal = document.getElementById('model-total');
  if (modelTotal) modelTotal.textContent = `${models.length} benchmarked variants`;
  renderRecommendations(data);

  renderTable('scores-table', 'Task scores and deltas', buildScoreRows(models), [
    plainColumn('model_id','Model'), plainColumn('backend','Runner'), plainColumn('reference','Reference'),
    percentColumns('ai2d','AI2D'), {key:'ai2d_delta',label:'Δ AI2D',numeric:true,best:'max',sortValue:r=>r.ai2d_delta,display:r=>pp(r.ai2d_delta)},
    percentColumns('textvqa','TextVQA'), {key:'textvqa_delta',label:'Δ TextVQA',numeric:true,best:'max',sortValue:r=>r.textvqa_delta,display:r=>pp(r.textvqa_delta)},
    plainColumn('status','Run status'),
  ], 'scores');

  renderTable('vqa-panel-table', 'VQA skill panel scores', buildPanelRows(models), [
    plainColumn('model_id','Model'), percentColumns('overall','Complete 6-category avg'),
    percentColumns('available_mean','Available-category avg'), { ...plainColumn('category_coverage','Coverage'), best:'max', sortValue: row => Number((row.category_coverage.match(/^(\d+)/) || [0,0])[1]) },
    percentColumns('truth','Truth yes/no'), percentColumns('counting','Counting'),
    percentColumns('description','Visual description'), percentColumns('ocr','OCR / TextVQA'),
    percentColumns('affordance','Affordance / function'), percentColumns('negation','Negation'),
  ], 'vqa-panel');

  renderTable('graph-table', 'GRAB-Lite graph scores', buildGraphRows(data.graph_benchmark), [
    plainColumn('model_id','Model'), percentColumns('score','Overall'),
    {key:'delta_vs_qwen3_2b',label:'Δ vs Qwen3-2B',numeric:true,best:'max',sortValue:r=>r.delta_vs_qwen3_2b,display:r=>pp(r.delta_vs_qwen3_2b)},
    percentColumns('properties','Properties'), percentColumns('function','Functions'),
    percentColumns('series','Series'), percentColumns('transform','Transforms'), percentColumns('real','Real'),
    {key:'completed_n',label:'Completed',numeric:true,best:'max',sortValue:r=>r.completed_n,display:r=>`${r.completed_n ?? 0}/500`},
    numberColumn('peak_gpu_used_gib','Peak VRAM GiB',2,'','min'), numberColumn('mean_wall_s','Mean sec/example',2,'','min'),
    numberColumn('generated_tokens_per_s','Output tokens/s',1,'','max'), contextColumn('context','Context'),
  ], 'graph');

  renderTable('runtime-table', 'Per-task VRAM and speed', buildRuntimeRows(models), [
    plainColumn('model_id','Model'), plainColumn('task','Task'),
    {key:'completed_n',label:'Completed',numeric:true,best:'max',rankGroup:'task',sortValue:r=>r.completed_n,display:r=>`${r.completed_n ?? 0}/${r.requested_n ?? 0}`},
    numberColumn('parse_errors','Parse errors',0,'','min','task'), numberColumn('peak_gpu_used_gib','Peak GPU GiB',2,'','min','task'),
    numberColumn('mean_wall_s','Mean sec/example',2,'','min','task'), numberColumn('generated_tokens_per_s','Output tokens/s',1,'','max','task'),
    contextColumn('context','Context'), plainColumn('kv_cache','KV cache'),
  ], 'runtime');

  renderTable('context-table', 'VRAM and context profiles', buildContextRows(models), [
    plainColumn('model_id','Model'), numberColumn('min_vram','Min VRAM GiB',2,'','min'), contextColumn('min_context','Min context'),
    numberColumn('default_vram','Default VRAM GiB',2,'','min'), contextColumn('default_context','Default context'),
    numberColumn('max_vram','Max VRAM GiB',2,'','min'), contextColumn('max_context','Max context'),
    numberColumn('peak_gpu_used_gib','Observed peak GiB',2,'','min'), plainColumn('mtp_supported','MTP'), plainColumn('backend','Runner'),
  ], 'context');

  for (const ceiling of [12,16]) {
    const rows = data.edge_tables[String(ceiling)] || [];
    document.getElementById(`edge${ceiling}-count`).textContent = `${rows.length} candidates`;
    renderTable(`edge${ceiling}-table`, `Estimated ${ceiling} GiB edge profiles`, rows, [
      plainColumn('model_id','Model'), numberColumn('benchmark_peak_gib','Benchmark peak GiB',2,'','min'),
      numberColumn('estimated_vram_8k_gib','Est. total @8K GiB',2,'','min'), contextColumn('max_context_fit','Max context fit'),
      percentColumns('ai2d_score','AI2D'), percentColumns('textvqa_score','TextVQA'),
      percentColumns('vqa_panel_score','VQA panel'), contextColumn('benchmark_context','Test context'),
      numberColumn('ai2d_mean_s','AI2D sec/example',2,'','min'), numberColumn('textvqa_mean_s','TextVQA sec/example',2,'','min'),
      numberColumn('ai2d_tokens_s','AI2D tok/s',1,'','max'), numberColumn('textvqa_tokens_s','TextVQA tok/s',1,'','max'),
    ], `edge${ceiling}`);
  }

}

init().catch(error => {
  const message = document.createElement('p');
  message.className = 'load-error';
  message.textContent = `Benchmark data could not be loaded: ${error.message}`;
  document.querySelector('main').prepend(message);
});
