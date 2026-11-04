(() => {
  'use strict';

  const videoHost = document.getElementById('video-specialist-leaders');
  const groundingHost = document.getElementById('grounding-specialist-leaders');
  const videoHref = 'video_Bench/';
  const groundingHref = 'grounding_Bench/';

  function isScore(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  function pct(value) {
    return isScore(value) ? `${(value * 100).toFixed(1)}%` : '—';
  }

  function renderTable(host, captionText, rows, emptyText) {
    const table = document.createElement('table');
    table.className = 'leader-table specialist-table';
    const caption = document.createElement('caption');
    caption.textContent = captionText;
    table.append(caption);

    const thead = document.createElement('thead');
    const header = document.createElement('tr');
    for (const label of ['Category', 'Measure', 'Best score', 'Winner', 'Report']) {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = label;
      header.append(cell);
    }
    thead.append(header);

    const tbody = document.createElement('tbody');
    if (!rows.length) {
      const tr = document.createElement('tr');
      const empty = document.createElement('td');
      empty.colSpan = 5;
      empty.textContent = emptyText;
      tr.append(empty);
      tbody.append(tr);
    }
    for (const row of rows) {
      const tr = document.createElement('tr');
      const category = document.createElement('td');
      category.textContent = row.category;
      const measure = document.createElement('td');
      measure.textContent = row.measure;
      const score = document.createElement('td');
      score.textContent = pct(row.score);
      const winner = document.createElement('td');
      if (row.modelName) {
        const modelLink = document.createElement('a');
        modelLink.href = row.href;
        modelLink.textContent = row.modelName;
        winner.append(modelLink);
      } else {
        winner.textContent = row.emptyText || emptyText;
      }
      const report = document.createElement('td');
      const reportLink = document.createElement('a');
      reportLink.href = row.href;
      reportLink.textContent = row.reportName;
      report.append(reportLink);
      tr.append(category, measure, score, winner, report);
      tbody.append(tr);
    }
    table.append(thead, tbody);
    host.replaceChildren(table);
  }

  function rankedTrackRows(data) {
    const models = Object.values(data.models || {});
    const tracks = [
      { id: 'mvbench', metric: 'task_macro_accuracy', category: 'MVBench video QA', measure: '19-task macro accuracy' },
      { id: 'vue_tr', metric: 'success_auc', category: 'VUE-TR temporal retrieval', measure: 'Success AUC' },
      { id: 'vue_tr', metric: 'precision_auc', category: 'VUE-TR temporal retrieval', measure: 'Precision AUC' },
      { id: 'vue_tr', metric: 'recall_auc', category: 'VUE-TR temporal retrieval', measure: 'Recall AUC' },
    ];
    return tracks.flatMap(track => {
      const candidates = models.map(model => ({ model, result: model.tracks?.[track.id] }))
        .filter(({ result }) => result?.status === 'complete'
          && result.completed === result.expected && isScore(result[track.metric]))
        .sort((a, b) => b.result[track.metric] - a.result[track.metric]);
      if (!candidates.length) {
        return [{ category: track.category, measure: track.measure, score: null, modelName: null,
          href: videoHref, reportName: 'Video VLM report' }];
      }
      const best = candidates[0].result[track.metric];
      return candidates.filter(candidate => candidate.result[track.metric] === best).map(({ model, result }) => ({
        category: track.category,
        measure: track.measure,
        score: result[track.metric],
        modelName: model.name,
        href: videoHref,
        reportName: 'Video VLM report',
      }));
    });
  }

  function groundingRows(data) {
    const definitions = [
      { id: 'refcocog', category: 'Referring expression', measure: 'Acc@IoU 0.5' },
      { id: 'coco', category: 'Open-vocabulary detection', measure: null, emptyMeasure: 'COCO AP / local F1@IoU 0.5' },
      { id: 'screenspot_pro', category: 'GUI grounding', measure: 'Point-in-target accuracy' },
      { id: 'hiertext', category: 'Text-region detection', measure: 'F1@IoU 0.5' },
    ];
    return definitions.flatMap(definition => {
      const taskInfo = data.tasks?.[definition.id] || {};
      const candidates = (data.models || []).map(model => ({ model, result: model.tasks?.[definition.id] }))
        .filter(({ result }) => result?.status === 'complete'
          && result.completed === taskInfo.n_examples && isScore(result.primary_score));
      const metricGroups = new Map();
      for (const candidate of candidates) {
        const metric = candidate.result.primary_metric || definition.measure || 'Primary score';
        if (!metricGroups.has(metric)) metricGroups.set(metric, []);
        metricGroups.get(metric).push(candidate);
      }
      if (!metricGroups.size) {
        return [{ category: definition.category,
          measure: definition.measure || definition.emptyMeasure || taskInfo.primary || 'Primary score', score: null,
          modelName: null, href: groundingHref, reportName: 'Grounding report',
          emptyText: `No complete result yet (${data.tested_model_count ?? 0} models tested)` }];
      }
      return [...metricGroups.entries()].flatMap(([metric, group]) => {
        const ranked = group.sort((a, b) => b.result.primary_score - a.result.primary_score);
        const best = ranked[0].result.primary_score;
        return ranked.filter(candidate => candidate.result.primary_score === best).map(({ model, result }) => ({
          category: definition.category,
          measure: definition.measure || metric,
          score: result.primary_score,
          modelName: model.model_id,
          href: groundingHref,
          reportName: 'Grounding report',
        }));
      });
    });
  }

  async function init() {
    try {
      const response = await fetch('video_Bench/data.json');
      if (!response.ok) throw new Error(`Video report returned ${response.status}`);
      const videoData = await response.json();
      renderTable(videoHost, 'MVBench: 95 questions across 19 tasks. VUE-TR: 100 vision-only queries. Best complete result per measure; ties are shown.',
        rankedTrackRows(videoData), 'No complete result yet');
    } catch (error) {
      renderTable(videoHost, 'Video benchmark result data is unavailable.', [], error.message);
    }

    const groundingData = window.GROUNDING_REPORT;
    if (groundingData) {
      const tested = groundingData.tested_model_count ?? 0;
      renderTable(groundingHost,
        'Grounding tasks use 100 examples each. Best complete result per task and comparable primary metric; ties are shown.',
        groundingRows(groundingData), `No complete result yet (${tested} models tested)`);
    } else {
      renderTable(groundingHost, 'Grounding benchmark result data is unavailable.', [], 'No grounding result data');
    }
  }

  init();
})();
