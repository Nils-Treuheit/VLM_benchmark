const states = new Map();
const pct = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const num = (value, digits = 2, suffix = '') => value == null ? '—' : `${Number(value).toFixed(digits)}${suffix}`;
const formatModel = model => model?.model_id || '—';

function renderTable(targetId, title, sourceRows, columns, tableKey, defaults = []) {
  const target = document.getElementById(targetId);
  const shell = document.createElement('div'); shell.className = 'table-shell';
  const toolbar = document.createElement('div'); toolbar.className = 'sort-toolbar';
  const sortChain = document.createElement('div'); sortChain.className = 'sort-chain'; sortChain.setAttribute('aria-live', 'polite');
  const legend = document.createElement('span'); legend.className = 'comparison-legend';
  legend.textContent = 'Gold = top result · cyan = ranks 2–5 · choose columns to add sort keys';
  const clear = document.createElement('button'); clear.className = 'sort-clear'; clear.type = 'button'; clear.textContent = 'Clear sort';
  const table = document.createElement('table');
  const caption = document.createElement('caption'); caption.className = 'sr-only'; caption.textContent = title; table.append(caption);
  const thead = document.createElement('thead'); const labels = document.createElement('tr'); const controls = document.createElement('tr');
  controls.className = 'sort-row'; const body = document.createElement('tbody');
  columns.forEach(column => {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = column.label; labels.append(th);
    const controlTh = document.createElement('th'); const select = document.createElement('select');
    select.setAttribute('aria-label', `Sort ${title} by ${column.label}`);
    select.innerHTML = '<option value="">None</option><option value="asc">Ascending</option><option value="desc">Descending</option>';
    select.dataset.key = column.key;
    select.addEventListener('change', () => {
      const chain = states.get(tableKey) || [];
      const position = chain.findIndex(item => item.key === column.key);
      if (!select.value) { if (position !== -1) chain.splice(position, 1); }
      else if (position === -1) chain.push({ key: column.key, direction: select.value });
      else chain[position].direction = select.value;
      states.set(tableKey, chain); draw();
    });
    controlTh.append(select); controls.append(controlTh);
  });
  thead.append(labels, controls); table.append(thead, body);

  function draw() {
    const chain = states.get(tableKey) || [];
    controls.querySelectorAll('select').forEach(select => {
      select.value = chain.find(item => item.key === select.dataset.key)?.direction || '';
    });
    sortChain.replaceChildren();
    if (!chain.length) {
      const hint = document.createElement('span'); hint.textContent = 'No sort applied · choose a column to start'; sortChain.append(hint);
    } else chain.forEach((item, index) => {
      const column = columns.find(col => col.key === item.key); const chip = document.createElement('span');
      chip.className = 'sort-chip'; chip.textContent = `${index + 1}. ${column?.label || item.key} ${item.direction === 'asc' ? '↑' : '↓'}`;
      sortChain.append(chip);
    });

    const indexed = sourceRows.map((row, index) => ({ ...row, __index: index }));
    indexed.sort((a, b) => {
      for (const item of chain) {
        const column = columns.find(col => col.key === item.key); if (!column) continue;
        const av = column.value(a); const bv = column.value(b);
        if (av == null && bv == null) continue;
        if (av == null) return 1; if (bv == null) return -1;
        const cmp = typeof av === 'number' && typeof bv === 'number'
          ? av - bv : String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: 'base' });
        if (cmp) return item.direction === 'asc' ? cmp : -cmp;
      }
      return a.__index - b.__index;
    });

    const ranks = new Map();
    columns.filter(column => column.best).forEach(column => {
      const ordered = sourceRows.map((row, index) => ({ index, value: column.value(row) })).filter(item => item.value != null);
      ordered.sort((a, b) => {
        const cmp = a.value - b.value;
        return (column.best === 'max' ? -cmp : cmp) || a.index - b.index;
      });
      const byIndex = new Map(); ordered.forEach((item, index) => byIndex.set(item.index, index + 1));
      ranks.set(column.key, byIndex);
    });

    body.replaceChildren();
    indexed.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(column => {
        const td = document.createElement('td'); td.textContent = column.display(row);
        if (column.numeric) td.classList.add('numeric');
        if (column.model) td.classList.add('model-cell');
        if (column.className) { const className = column.className(row); if (className) td.classList.add(className); }
        const rank = ranks.get(column.key)?.get(row.__index);
        if (rank === 1) td.classList.add('best-rank-1');
        else if (rank >= 2 && rank <= 5) td.classList.add('best-rank-2to5');
        tr.append(td);
      });
      body.append(tr);
    });
  }
  clear.addEventListener('click', () => { states.set(tableKey, []); draw(); });
  toolbar.append(sortChain, legend, clear); shell.append(toolbar, table); target.replaceChildren(shell);
  states.set(tableKey, defaults); draw();
}

const percentColumn = (key, label, getValue) => ({
  key, label, numeric: true, best: 'max', value: getValue,
  display: row => pct(getValue(row)),
});

function card(parent, eyebrow, heading, description, metrics = []) {
  const element = document.createElement('article'); element.className = 'recommendation-card';
  const tag = document.createElement('p'); tag.className = 'eyebrow'; tag.textContent = eyebrow;
  const title = document.createElement('h3'); title.textContent = heading;
  const copy = document.createElement('p'); copy.className = 'recommendation-reason'; copy.textContent = description;
  element.append(tag, title, copy);
  if (metrics.length) {
    const list = document.createElement('dl'); list.className = 'recommendation-metrics';
    metrics.forEach(([label, value]) => {
      const wrapper = document.createElement('div'); const term = document.createElement('dt'); const detail = document.createElement('dd');
      term.textContent = label; detail.textContent = value; wrapper.append(term, detail); list.append(wrapper);
    });
    element.append(list);
  }
  parent.append(element); return element;
}

function renderSelection(data, complete) {
  const byId = new Map(complete.map(model => [model.model_id, model]));
  const best = byId.get(data.leaders.overall);
  document.getElementById('model-count').textContent = String(complete.length);

  const overall = document.getElementById('overall-pick');
  card(overall, 'BEST OVERALL QUALITY', formatModel(best), 'Highest complete eight-task macro score.', [
    ['Full-suite macro', pct(best?.full_benchmark_macro)],
    ['AI2D', pct(best?.tasks.ai2d.score)],
    ['TextVQA / OCR', pct(best?.tasks.textvqa.score)],
    ['Peak task VRAM', num(Math.max(...Object.values(best?.tasks || {}).map(task => task.peak_allocated_gib || 0)), 2, ' GiB')],
  ]);

  const taskGrid = document.getElementById('task-picks');
  data.task_order.forEach(taskId => {
    const metrics = complete.map(model => ({ model, score: model.tasks[taskId]?.score })).filter(item => item.score != null);
    const max = Math.max(...metrics.map(item => item.score));
    const winners = metrics.filter(item => Math.abs(item.score - max) < 1e-12);
    const names = winners.map(item => item.model.model_id).join(' · ');
    card(taskGrid, `BEST · ${data.task_labels[taskId].toUpperCase()}`, names, `${pct(max)} on ${data.task_samples[taskId]} fixed examples.`);
  });

  const budgetGrid = document.getElementById('budget-picks');
  [12, 16].forEach(limit => {
    const fitting = complete.map(model => ({
      model,
      peak: Math.max(...Object.values(model.tasks).map(task => task.peak_allocated_gib || 0)),
    })).filter(item => item.peak <= limit);
    const pick = fitting.sort((a, b) => b.model.full_benchmark_macro - a.model.full_benchmark_macro)[0];
    card(budgetGrid, `BEST MEASURED FIT · ${limit} GIB`, formatModel(pick?.model),
      pick ? `Highest full-suite score among models whose measured peak stayed within ${limit} GiB.` : `No complete model stayed within ${limit} GiB.`,
      pick ? [['Full-suite macro', pct(pick.model.full_benchmark_macro)], ['Measured peak', num(pick.peak, 2, ' GiB')]] : []);
  });
}

function buildScoreRows(models) {
  return models.map(model => ({
    model_id: model.model_id,
    macro: model.full_benchmark_macro,
    ai2d: model.tasks.ai2d.score,
    textvqa: model.tasks.textvqa.score,
    status: model.status,
    coverage: model.complete_tasks,
    model,
  }));
}

function buildTaskRows(models) {
  return models.filter(model => model.status === 'complete').map(model => ({ ...model, ...model.tasks }));
}

function buildRuntimeRows(models, taskOrder, taskLabels) {
  return models.filter(model => model.status === 'complete').flatMap(model => taskOrder.map(taskId => {
    const task = model.tasks[taskId];
    return { model_id: model.model_id, task: taskLabels[taskId], samples: `${task.completed_n}/${task.requested_n}`,
      completed_n: task.completed_n, mean_generation_s: task.mean_generation_s,
      generated_tokens_per_s: task.generated_tokens_per_s, peak_allocated_gib: task.peak_allocated_gib,
      parse_errors: task.parse_errors };
  }));
}

function buildContextRows(models) {
  return models.filter(model => model.status === 'complete').map(model => {
    const tasks = Object.values(model.tasks);
    const peak = Math.max(...tasks.map(task => task.peak_allocated_gib).filter(Number.isFinite));
    const peakTask = tasks.find(task => task.peak_allocated_gib === peak)?.label;
    return { model_id: model.model_id, checkpoint_context: model.checkpoint_context,
      weight_gib: model.weight_gib, peak_allocated_gib: peak, peak_task: peakTask };
  }).filter(row => Number.isFinite(row.checkpoint_context) && Number.isFinite(row.peak_allocated_gib));
}

function renderEdgeTables(models) {
  const complete = models.filter(model => model.status === 'complete');
  const rows = complete.map(model => {
    const tasks = Object.values(model.tasks);
    const peak = Math.max(...tasks.map(task => task.peak_allocated_gib).filter(Number.isFinite));
    const peakTask = tasks.find(task => task.peak_allocated_gib === peak)?.label;
    return { model_id: model.model_id, full_benchmark_macro: model.full_benchmark_macro,
      checkpoint_context: model.checkpoint_context, peak_allocated_gib: peak, peak_task: peakTask };
  }).filter(row => Number.isFinite(row.peak_allocated_gib));

  [12, 16].forEach(limit => {
    const candidates = rows.filter(row => row.peak_allocated_gib <= limit)
      .sort((a, b) => b.full_benchmark_macro - a.full_benchmark_macro);
    document.getElementById(`edge${limit}-count`).textContent = `${candidates.length} candidates`;
    if (!candidates.length) {
      const empty = document.createElement('p'); empty.className = 'edge-caveat';
      empty.textContent = `No completed model stayed within ${limit} GiB at its measured peak across all eight tasks.`;
      document.getElementById(`edge${limit}-table`).replaceChildren(empty);
      return;
    }
    renderTable(`edge${limit}-table`, `Measured ${limit} GiB edge candidates`, candidates, [
      { key: 'model_id', label: 'Model', model: true, value: row => row.model_id, display: row => row.model_id },
      { key: 'full_benchmark_macro', label: '8-task macro', numeric: true, best: 'max', value: row => row.full_benchmark_macro, display: row => pct(row.full_benchmark_macro) },
      { key: 'peak_allocated_gib', label: 'Measured peak VRAM', numeric: true, best: 'min', value: row => row.peak_allocated_gib, display: row => num(row.peak_allocated_gib, 2, ' GiB') },
      { key: 'peak_task', label: 'Task at peak', value: row => row.peak_task, display: row => row.peak_task || '—' },
      { key: 'checkpoint_context', label: 'Checkpoint context', numeric: true, value: row => row.checkpoint_context, display: row => Number.isFinite(row.checkpoint_context) ? `${Math.round(row.checkpoint_context / 1024)}K tokens` : '—' },
    ], `edge${limit}`, [{ key: 'full_benchmark_macro', direction: 'desc' }]);
  });
}

function averageRanks(values) {
  const sorted = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const ranks = Array(values.length); let start = 0;
  while (start < sorted.length) {
    let end = start;
    while (end + 1 < sorted.length && sorted[end + 1].value === sorted[start].value) end++;
    const rank = (start + end) / 2 + 1;
    for (let index = start; index <= end; index++) ranks[sorted[index].index] = rank;
    start = end + 1;
  }
  return ranks;
}

function correlation(a, b) {
  if (a.length < 2 || a.length !== b.length) return null;
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  const numerator = a.reduce((sum, value, index) => sum + (value - meanA) * (b[index] - meanB), 0);
  const denominator = Math.sqrt(a.reduce((sum, value) => sum + (value - meanA) ** 2, 0)
    * b.reduce((sum, value) => sum + (value - meanB) ** 2, 0));
  return denominator ? numerator / denominator : null;
}

function renderContextCorrelation(rows) {
  const host = document.getElementById('context-correlation');
  const contexts = rows.map(row => Math.log2(row.checkpoint_context));
  const peaks = rows.map(row => row.peak_allocated_gib);
  const pearson = correlation(contexts, peaks);
  const spearman = correlation(averageRanks(contexts), averageRanks(peaks));
  host.textContent = pearson == null
    ? 'Context and VRAM correlation is unavailable for the current completed models.'
    : `Completed models (n=${rows.length}) · Pearson r=${pearson.toFixed(2)} for log₂ checkpoint context vs. measured peak VRAM · Spearman ρ=${spearman.toFixed(2)}. Descriptive only; context is model metadata, not a context-length VRAM sweep.`;
}

async function init() {
  const response = await fetch('data.json?v=5');
  if (!response.ok) throw new Error(`Could not load report data (${response.status})`);
  const data = await response.json();
  const complete = data.models.filter(model => model.status === 'complete');
  document.getElementById('generated').textContent = `Updated ${data.generated_at}`;
  document.getElementById('runtime-version').textContent = `Transformers ${data.runtime_versions?.transformers || '—'}`;
  document.getElementById('model-total').textContent = `${complete.length} complete models · fixed samples across all models`;
  renderSelection(data, complete);

  const scores = buildScoreRows(data.models);
  renderTable('scores-table', 'Task scores and coverage', scores, [
    { key: 'model_id', label: 'Model', model: true, value: row => row.model_id, display: row => row.model_id },
    percentColumn('macro', '8-task macro', row => row.macro),
    percentColumn('ai2d', 'AI2D', row => row.ai2d),
    percentColumn('textvqa', 'TextVQA / OCR', row => row.textvqa),
    { key: 'coverage', label: 'Complete tasks', numeric: true, value: row => row.coverage, display: row => `${row.coverage}/8` },
    { key: 'status', label: 'Status', value: row => row.status, display: row => row.status === 'skipped' ? 'Skipped · over memory budget' : row.status,
      className: row => row.status === 'skipped' ? 'dim' : '' },
  ], 'scores', [{ key: 'macro', direction: 'desc' }]);

  const taskRows = buildTaskRows(data.models);
  renderTable('task-table', 'Accuracy by task', taskRows, [
    { key: 'model_id', label: 'Model', model: true, value: row => row.model_id, display: row => row.model_id },
    ...data.task_order.map(taskId => percentColumn(taskId, data.task_labels[taskId], row => row.tasks[taskId]?.score)),
    percentColumn('macro', 'Macro', row => row.full_benchmark_macro),
  ], 'tasks', [{ key: 'macro', direction: 'desc' }]);

  const runtimeRows = buildRuntimeRows(data.models, data.task_order, data.task_labels);
  renderTable('runtime-table', 'Per-task VRAM and speed', runtimeRows, [
    { key: 'model_id', label: 'Model', model: true, value: row => row.model_id, display: row => row.model_id },
    { key: 'task', label: 'Task', value: row => row.task, display: row => row.task },
    { key: 'completed_n', label: 'Samples', numeric: true, value: row => row.completed_n,
      display: row => `${row.completed_n}/${row.samples.split('/')[1]}` },
    { key: 'mean_generation_s', label: 'Mean generation', numeric: true, best: 'min', value: row => row.mean_generation_s,
      display: row => num(row.mean_generation_s, 2, ' s') },
    { key: 'generated_tokens_per_s', label: 'Output speed', numeric: true, best: 'max', value: row => row.generated_tokens_per_s,
      display: row => num(row.generated_tokens_per_s, 2, ' tok/s') },
    { key: 'peak_allocated_gib', label: 'Peak allocated VRAM', numeric: true, best: 'min', value: row => row.peak_allocated_gib,
      display: row => num(row.peak_allocated_gib, 2, ' GiB') },
    { key: 'parse_errors', label: 'Parse errors', numeric: true, value: row => row.parse_errors,
      display: row => String(row.parse_errors) },
  ], 'runtime', [{ key: 'model_id', direction: 'asc' }, { key: 'task', direction: 'asc' }]);

  const contextRows = buildContextRows(data.models);
  renderContextCorrelation(contextRows);
  renderTable('context-vram-table', 'Checkpoint context and measured VRAM', contextRows, [
    { key: 'model_id', label: 'Model', model: true, value: row => row.model_id, display: row => row.model_id },
    { key: 'checkpoint_context', label: 'Checkpoint context', numeric: true, value: row => row.checkpoint_context,
      display: row => `${Math.round(row.checkpoint_context / 1024)}K tokens` },
    { key: 'weight_gib', label: 'Cached weights · file size', numeric: true, value: row => row.weight_gib,
      display: row => num(row.weight_gib, 2, ' GiB') },
    { key: 'peak_allocated_gib', label: 'Peak measured VRAM', numeric: true, best: 'min', value: row => row.peak_allocated_gib,
      display: row => num(row.peak_allocated_gib, 2, ' GiB') },
    { key: 'peak_task', label: 'Task at peak', value: row => row.peak_task, display: row => row.peak_task },
  ], 'context-vram', [{ key: 'checkpoint_context', direction: 'asc' }]);
  renderEdgeTables(data.models);
}

init().catch(error => {
  const notice = document.createElement('p'); notice.className = 'load-error'; notice.textContent = error.message;
  document.querySelector('main').prepend(notice);
});
