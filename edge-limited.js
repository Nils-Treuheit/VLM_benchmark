(() => {
  'use strict';

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
  const stageHost = document.getElementById('edge-stage-summary');

  function textCell(row, value) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.append(cell);
    return cell;
  }

  function headingCell(row, value) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = value;
    row.append(cell);
    return cell;
  }

  function capRecord(data, cap) {
    return (data.caps || []).find(stage => stage.cap_gib === cap) || null;
  }

  function validRecord(record, cap, requireFullSuite = true) {
    return record && record.status === 'verified' && (!requireFullSuite || record.complete === true)
      && (requireFullSuite || record.task_complete === true)
      && record.context_probe_passed === true && record.whole_device_cap_gib === cap
      && Number.isFinite(record.score) && Number.isFinite(record.context_tokens)
      && Number.isFinite(record.context_probe_peak_gib)
      && record.context_probe_peak_gib <= cap
      && Number.isFinite(record.load_peak_gib) && record.load_peak_gib > 0
      && record.load_peak_gib <= cap
      && Number.isFinite(record.scoring_peak_gib) && record.scoring_peak_gib > 0
      && record.scoring_peak_gib <= cap
      && Number.isFinite(record.whole_device_peak_gib) && record.whole_device_peak_gib > 0
      && record.whole_device_peak_gib <= cap;
  }

  function renderStageSummary(data) {
    const table = document.createElement('table');
    table.className = 'edge-table';
    const caption = document.createElement('caption');
    caption.textContent = 'Only complete runs with a passing real-image context probe qualify.';
    table.append(caption);
    const thead = document.createElement('thead');
    const heading = document.createElement('tr');
    ['Whole-device cap', 'Verified complete candidates', 'Stage status'].forEach(label => headingCell(heading, label));
    thead.append(heading);
    const tbody = document.createElement('tbody');
    for (const cap of [8, 10, 12, 14, 16]) {
      const stage = capRecord(data, cap);
      const row = document.createElement('tr');
      textCell(row, `${cap} GiB`);
      const verified = (stage?.records || []).filter(record => validRecord(record, cap));
      textCell(row, String(verified.length));
      const stageState = verified.length ? 'Measured' : 'No verified cap-specific run';
      textCell(row, stage?.status === 'unavailable_baseline' ? 'Unavailable: baseline exceeds cap' : stageState);
      tbody.append(row);
    }
    table.append(thead, tbody);
    stageHost.replaceChildren(table);
  }

  function appendRunEvidence(host, record) {
    const detail = document.createElement('p');
    const processPeak = Number.isFinite(record.process_peak_gib)
      ? `; process peak ${record.process_peak_gib.toFixed(2)} GiB` : '';
    detail.textContent = `Measured context ${Math.round(record.context_tokens).toLocaleString()} tokens; context-probe peak ${record.context_probe_peak_gib.toFixed(2)} GiB; scoring-run peak ${record.scoring_peak_gib.toFixed(2)} GiB${processPeak}.`;
    host.append(detail);
  }

  function renderOverall(stage, cap) {
    const header = document.createElement('h3');
    header.textContent = `Best overall among currently verified runs · complete on all eight tasks · ${cap} GiB cap`;
    overall.replaceChildren(header);
    const records = (Array.isArray(stage?.overall) ? stage.overall : stage?.overall ? [stage.overall] : [])
      .filter(record => validRecord(record, cap));
    if (!records.length) {
      const empty = document.createElement('p');
      empty.textContent = stage?.status === 'unavailable_baseline'
        ? 'This stage is unavailable because the measured idle whole-device baseline reached the cap.'
        : 'No verified complete run is recorded for this cap.';
      overall.append(empty);
      return;
    }
    for (const record of records) {
      const score = document.createElement('p');
      score.className = 'edge-score';
      score.textContent = `${(record.score * 100).toFixed(1)}%`;
      const model = document.createElement('p');
      model.className = 'edge-model';
      const link = document.createElement('a');
      link.href = record.suite_href;
      link.textContent = record.model_id;
      model.append(link);
      const suite = document.createElement('p');
      suite.className = 'subtle';
      suite.textContent = record.suite_name;
      overall.append(score, model, suite);
      appendRunEvidence(overall, record);
    }
  }

  function renderCategories(stage, cap) {
    const table = document.createElement('table');
    table.className = 'edge-table';
    const caption = document.createElement('caption');
    caption.textContent = 'Best scores among currently verified cap-specific runs. Selected candidates may remain unmeasured; ties are shown.';
    table.append(caption);
    const thead = document.createElement('thead');
    const heading = document.createElement('tr');
    ['Category', 'Score', 'Model', 'Suite', 'Measured context and peaks'].forEach(label => headingCell(heading, label));
    thead.append(heading);
    const tbody = document.createElement('tbody');
    for (const [task, label] of tasks) {
      const row = document.createElement('tr');
      textCell(row, label);
      const records = (stage?.category_leaders?.[task] || []).filter(record => validRecord(record, cap, false));
      if (!records.length) {
        const empty = document.createElement('td');
        empty.colSpan = 4;
        empty.className = 'subtle';
        empty.textContent = stage?.status === 'unavailable_baseline'
          ? 'Stage unavailable: idle baseline reached the cap'
          : 'No verified cap-specific run';
        row.append(empty);
      } else {
        const record = records[0];
        textCell(row, `${(record.score * 100).toFixed(1)}%`);
        const model = document.createElement('td');
        const link = document.createElement('a');
        link.href = record.suite_href;
        link.textContent = record.model_id;
        model.append(link);
        row.append(model);
        textCell(row, record.suite_name);
        const evidence = document.createElement('td');
        appendRunEvidence(evidence, record);
        row.append(evidence);
        tbody.append(row);
        for (const tied of records.slice(1)) {
          const tie = document.createElement('tr');
          textCell(tie, '');
          textCell(tie, `${(tied.score * 100).toFixed(1)}%`);
          const tiedModel = document.createElement('td');
          const tiedLink = document.createElement('a');
          tiedLink.href = tied.suite_href;
          tiedLink.textContent = tied.model_id;
          tiedModel.append(tiedLink);
          tie.append(tiedModel);
          textCell(tie, tied.suite_name);
          const tiedEvidence = document.createElement('td');
          appendRunEvidence(tiedEvidence, tied);
          tie.append(tiedEvidence);
          tbody.append(tie);
        }
        continue;
      }
      tbody.append(row);
    }
    table.append(thead, tbody);
    categoryHost.replaceChildren(table);
  }

  let data = null;
  function render() {
    const cap = Number(limitSelect.value);
    const stage = capRecord(data, cap);
    const records = (stage?.records || []).filter(record => validRecord(record, cap));
    const categoryCount = tasks.reduce((count, [task]) => count +
      (stage?.category_leaders?.[task] || []).filter(record => validRecord(record, cap, false)).length, 0);
    status.textContent = records.length
      ? `${records.length} complete model-cap runs passed both the controlled context probe and full suite at ${cap} GiB. Leaders below rank the currently verified runs; refinement may add selected candidates.`
      : categoryCount
        ? `No complete eight-task run is verified at ${cap} GiB; ${categoryCount} category result(s) have a passing cap probe and complete task coverage.`
        : `No verified cap-specific run is available at ${cap} GiB. Provisional estimates are not shown as results.`;
    renderOverall(stage, cap);
    renderCategories(stage, cap);
  }

  async function init() {
    try {
      const response = await fetch('edge-refinement-data.json');
      if (!response.ok) throw new Error(`Edge refinement data returned ${response.status}`);
      data = await response.json();
      renderStageSummary(data);
      limitSelect.addEventListener('change', render);
      render();
    } catch (error) {
      status.textContent = `Could not load cap-specific measurements: ${error.message}`;
      overall.replaceChildren();
      categoryHost.replaceChildren();
      stageHost.replaceChildren();
    }
  }

  init();
})();
