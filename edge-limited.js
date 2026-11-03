(() => {
  'use strict';

  const suites = [
    { key: 'qwen', name: 'Qwen-VL', href: 'qwenVL_Bench/', path: 'qwenVL_Bench/data.json' },
    { key: 'llava', name: 'LLaVA', href: 'llava_Bench/', path: 'llava_Bench/data.json' },
    { key: 'cosmos', name: 'Cosmos', href: 'cosmos_Bench/', path: 'cosmos_Bench/data.json' },
    { key: 'other', name: 'Other Big VLMs', href: 'general_transformers_Bench/', path: 'general_transformers_Bench/data.json' },
    { key: 'phi', name: 'Φ-Vision', href: 'phi_Bench/', path: 'phi_Bench/data.json' },
  ];
  const tasks = [
    ['ai2d', 'AI2D'],
    ['textvqa', 'TextVQA / OCR'],
    ['vqa_truth', 'VQA truth'],
    ['vqa_count', 'VQA counting'],
    ['vqa_description', 'VQA description'],
    ['vqa_affordance', 'VQA affordance'],
    ['vqa_negation', 'VQA negation'],
    ['graph_grab_lite', 'GRAB-Lite graph analysis'],
  ];
  const status = document.getElementById('edge-status');
  const limitSelect = document.getElementById('edge-vram-limit');
  const overall = document.getElementById('edge-overall');
  const categoryHost = document.getElementById('edge-categories');

  function numeric(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function qwenEdgeRows(data, cap) {
    const savedRows = data.edge_tables?.[String(cap)];
    if (savedRows) return new Map(savedRows.map(row => [row.model_id, row]));

    const rows = [];
    for (const model of data.models || []) {
      if (model.status !== 'complete' || !model.tasks?.ai2d || !model.tasks?.textvqa) continue;
      const tiers = model.context_profile?.tiers || [];
      const low = tiers.find(tier => tier.tier === 'min');
      if (!low) continue;
      const high = tiers.reduce((best, tier) =>
        numeric(tier.context_tokens) > numeric(best.context_tokens) ? tier : best, low);
      const lowContext = numeric(low.context_tokens) ?? 8192;
      const highContext = numeric(high.context_tokens) ?? lowContext;
      const lowVram = numeric(low.estimated_vram_gib) ?? 0;
      const highVram = numeric(high.estimated_vram_gib) ?? lowVram;
      const estimatedVram8k = lowVram + 1;
      if (estimatedVram8k >= cap) continue;

      const contextCap = Math.min(
        numeric(model.context_profile?.estimated_max_context) ?? highContext,
        numeric(model.checkpoint_context) ?? highContext,
      );
      const modelBudget = Math.max(lowVram, cap - 1);
      let maxContextFit;
      if (contextCap <= lowContext || highContext <= lowContext || highVram <= lowVram) {
        maxContextFit = Math.min(lowContext, contextCap);
      } else if (modelBudget >= highVram) {
        maxContextFit = contextCap;
      } else {
        const estimate = lowContext + (modelBudget - lowVram)
          * (highContext - lowContext) / (highVram - lowVram);
        maxContextFit = Math.min(contextCap, Math.max(lowContext, Math.floor(estimate / 1024) * 1024));
      }
      const imagePeaks = ['ai2d', 'textvqa'].map(taskId => numeric(model.tasks[taskId].peak_gpu_used_gib))
        .filter(value => value !== null);
      rows.push({
        model_id: model.model_id,
        benchmark_peak_gib: imagePeaks.length ? Math.max(...imagePeaks) : null,
        estimated_vram_8k_gib: estimatedVram8k,
        max_context_fit: maxContextFit,
        benchmark_context: model.configured_context ?? model.checkpoint_context,
      });
    }
    return new Map(rows.map(row => [row.model_id, row]));
  }

  function taskScore(model, taskId) {
    if (taskId === 'graph_grab_lite' && model.suite === 'qwen') {
      return model.raw.graph_complete_n === 500 ? numeric(model.raw.graph_score) : null;
    }
    const task = model.raw.tasks?.[taskId];
    if (!task || (numeric(task.completed_n) !== null && numeric(task.requested_n) !== null
      && task.completed_n < task.requested_n)) return null;
    return numeric(task.score);
  }

  function qwenEligible(model, cap, edgeRows, taskId = null) {
    if (!edgeRows.has(model.id)) return false;
    if (taskId === 'graph_grab_lite' || taskId === 'overall') {
      const graphPeak = numeric(model.raw.graph_peak_gpu_used_gib);
      if (graphPeak === null || graphPeak > cap) return false;
    }
    return true;
  }

  function measuredPeak(model) {
    const peaks = Object.values(model.raw.tasks || {}).map(task =>
      numeric(task.peak_gpu_used_gib ?? task.peak_allocated_gib)).filter(value => value !== null);
    if (model.suite === 'cosmos') {
      return numeric(model.raw.peak_gpu_used_gib);
    }
    if (model.suite === 'other') {
      return numeric(model.raw.peak_allocated_gib)
        ?? (peaks.length ? Math.max(...peaks) : null);
    }
    return peaks.length ? Math.max(...peaks) : null;
  }

  function measuredPeakTask(model) {
    if (model.suite === 'qwen') {
      const entries = ['ai2d', 'textvqa'].map(taskId => [taskId, model.raw.tasks?.[taskId]])
        .map(([taskId, task]) => [taskId, numeric(task?.peak_gpu_used_gib)])
        .filter(([, peak]) => peak !== null);
      const peak = entries.length ? Math.max(...entries.map(([, value]) => value)) : null;
      return entries.find(([, value]) => value === peak)?.[0] || 'not recorded';
    }
    if (model.raw.peak_memory_task) return model.raw.peak_memory_task;
    const entries = Object.entries(model.raw.tasks || {})
      .map(([taskId, task]) => [taskId, numeric(task.peak_gpu_used_gib ?? task.peak_allocated_gib)])
      .filter(([, peak]) => peak !== null);
    const peak = entries.length ? Math.max(...entries.map(([, value]) => value)) : null;
    return entries.find(([, value]) => value === peak)?.[0] || 'not recorded';
  }

  function isEligible(model, cap, edgeRows, taskId = null) {
    if (model.raw.status !== 'complete') return false;
    if (model.suite === 'qwen') return qwenEligible(model, cap, edgeRows, taskId);
    const peak = measuredPeak(model);
    return peak !== null && peak <= cap;
  }

  function suiteModels(suite, data) {
    return (data.models || []).map(raw => ({ id: raw.model_id, suite: suite.key, suiteName: suite.name,
      suiteHref: suite.href, raw }));
  }

  function createLink(model) {
    const link = document.createElement('a');
    link.href = model.suiteHref;
    link.textContent = model.id;
    return link;
  }

  function formatTokens(value) {
    return numeric(value) === null ? 'not recorded' : `${Math.round(value).toLocaleString()} tokens`;
  }

  function taskLabel(taskId) {
    return tasks.find(([id]) => id === taskId)?.[1] || taskId;
  }

  function profileLines(model, cap, edgeRows, taskId = null) {
    if (model.suite === 'qwen') {
      const row = edgeRows.get(model.id);
      const graphPeak = numeric(model.raw.graph_peak_gpu_used_gib);
      const imagePeak = numeric(row?.benchmark_peak_gib);
      const lines = [
        `Estimated max context under ${cap} GiB: ${formatTokens(row?.max_context_fit)}.`,
        `Estimated residency at that context: ≤${cap} GiB including the desktop reserve; estimated 8K total: ${numeric(row?.estimated_vram_8k_gib)?.toFixed(2) ?? 'not recorded'} GiB.`,
      ];
      if (taskId !== 'graph_grab_lite') {
        lines.push(`Measured image-task peak: ${imagePeak?.toFixed(2) ?? 'not recorded'} GiB during ${taskLabel(measuredPeakTask(model))} at ${formatTokens(row?.benchmark_context)}.`);
      }
      if ((taskId === 'graph_grab_lite' || taskId === 'overall') && graphPeak !== null) {
        lines.push(`Measured GRAB-Lite peak: ${graphPeak.toFixed(2)} GiB; graph context was not recorded.`);
      }
      return lines;
    }

    const checkpointContext = numeric(model.raw.checkpoint_context);
    const runContext = numeric(model.raw.run_context_tokens);
    const contextLine = runContext !== null
      ? `Maximum context under this cap: not measured. Recorded run cap: ${formatTokens(runContext)}; checkpoint advertises ${formatTokens(checkpointContext)}.`
      : `Maximum context under this cap: not measured. Checkpoint advertises ${formatTokens(checkpointContext)}; per-request run context was not recorded.`;
    const peak = measuredPeak(model);
    const scope = model.suite === 'cosmos' ? 'whole-device peak' : 'process-local peak';
    return [contextLine, `Observed ${scope}: ${peak?.toFixed(2) ?? 'not recorded'} GiB (≤${cap} GiB) during ${taskLabel(measuredPeakTask(model))}.`];
  }

  function appendProfile(host, model, cap, edgeRows, taskId = null) {
    const profile = document.createElement('div');
    profile.className = 'edge-profile';
    for (const line of profileLines(model, cap, edgeRows, taskId)) {
      const paragraph = document.createElement('p');
      paragraph.textContent = line;
      profile.append(paragraph);
    }
    host.append(profile);
  }

  function renderOverall(candidates, cap, edgeRows) {
    const ranked = candidates.filter(model => model.suite !== 'qwen'
      || isEligible(model, cap, edgeRows, 'overall')).map(model => {
      const scores = tasks.map(([taskId]) => taskScore(model, taskId));
      return scores.every(score => score !== null)
        ? { model, score: scores.reduce((sum, value) => sum + value, 0) / scores.length }
        : null;
    }).filter(Boolean).sort((a, b) => b.score - a.score);
    overall.replaceChildren();
    const title = document.createElement('h3');
    title.textContent = `Best overall · complete on all eight shared tasks · ${cap} GiB cap`;
    overall.append(title);
    if (!ranked.length) {
      const note = document.createElement('p');
      note.textContent = `No model has complete results across all eight shared tasks and meets the ${cap} GiB eligibility rule.`;
      overall.append(note);
      return;
    }
    const winner = ranked[0];
    const score = document.createElement('p');
    score.className = 'edge-score';
    score.textContent = `${(winner.score * 100).toFixed(1)}%`;
    const model = document.createElement('p');
    model.className = 'edge-model';
    model.append(createLink(winner.model));
    const suite = document.createElement('p');
    suite.className = 'subtle';
    suite.textContent = winner.model.suiteName;
    overall.append(score, model, suite);
    appendProfile(overall, winner.model, cap, edgeRows, 'overall');
  }

  function renderCategories(candidates, cap, edgeRows) {
    const leaders = tasks.map(([taskId, label]) => {
      const scored = candidates.filter(model => isEligible(model, cap, edgeRows, taskId))
        .map(model => ({ model, score: taskScore(model, taskId) }))
        .filter(row => row.score !== null).sort((a, b) => b.score - a.score);
      return { label, winners: scored.length ? scored.filter(row => row.score === scored[0].score) : [] };
    });
    const table = document.createElement('table');
    table.className = 'edge-table';
    const caption = document.createElement('caption');
    caption.textContent = 'Highest completed category score among models meeting each report’s published edge-fit rule; ties are shown.';
    table.append(caption);
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const label of ['Category', 'Score', 'Model', 'Suite', 'Context and peak VRAM']) {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = label;
      headRow.append(cell);
    }
    thead.append(headRow);
    const tbody = document.createElement('tbody');
    for (const leader of leaders) {
      if (!leader.winners.length) {
        const row = document.createElement('tr');
        const category = document.createElement('td');
        category.textContent = leader.label;
        const empty = document.createElement('td');
        empty.colSpan = 4;
        empty.className = 'subtle';
        empty.textContent = 'No eligible completed result';
        row.append(category, empty);
        tbody.append(row);
        continue;
      }
      leader.winners.forEach((winner, index) => {
        const row = document.createElement('tr');
        const category = document.createElement('td');
        if (index === 0) category.textContent = leader.label;
        const score = document.createElement('td');
        score.textContent = `${(winner.score * 100).toFixed(1)}%`;
        const model = document.createElement('td');
        model.append(createLink(winner.model));
        const suite = document.createElement('td');
        suite.textContent = winner.model.suiteName;
        const profileCell = document.createElement('td');
        appendProfile(profileCell, winner.model, cap, edgeRows, taskId);
        row.append(category, score, model, suite, profileCell);
        tbody.append(row);
      });
    }
    table.append(thead, tbody);
    categoryHost.replaceChildren(table);
  }

  let allModels = [];
  let reportData = new Map();

  function render() {
    const cap = Number(limitSelect.value);
    const qwenData = reportData.get('qwen');
    const edgeRows = qwenEdgeRows(qwenData, cap);
    const candidates = allModels.filter(model => isEligible(model, cap, edgeRows));
    renderOverall(candidates, cap, edgeRows);
    renderCategories(candidates, cap, edgeRows);
    status.textContent = `${candidates.length} completed models meet the published ${cap} GiB filter for at least one shared category.`;
  }

  async function init() {
    try {
      const loaded = await Promise.all(suites.map(async suite => {
        const response = await fetch(suite.path);
        if (!response.ok) throw new Error(`${suite.name} report returned ${response.status}`);
        return [suite, await response.json()];
      }));
      reportData = new Map(loaded.map(([suite, data]) => [suite.key, data]));
      allModels = loaded.flatMap(([suite, data]) => suiteModels(suite, data));
      limitSelect.addEventListener('change', render);
      render();
    } catch (error) {
      status.textContent = `Could not load the benchmark data for Edge Limited: ${error.message}`;
      overall.replaceChildren();
      categoryHost.replaceChildren();
    }
  }

  init();
})();
