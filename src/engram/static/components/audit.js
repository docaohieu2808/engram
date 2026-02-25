/**
 * Audit tab — feedback history, audit log, benchmark runner.
 */
const Audit = {
  _loaded: false,

  async load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-audit').innerHTML = this._shell();
    }
    await Promise.all([this._loadFeedback(), this._loadAudit()]);
  },

  _shell() {
    return `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
        <div class="card"><h3>Feedback History</h3><div id="audit-feedback"></div></div>
        <div class="card"><h3>Audit Log</h3>
          <div style="margin-bottom:8px"><select id="audit-filter" onchange="Audit._loadAudit()"><option value="">All</option><option value="modification">Modifications</option><option value="semantic">Semantic</option></select></div>
          <div id="audit-log"></div>
        </div>
      </div>
      <div class="card">
        <h3>Benchmark</h3>
        <div class="form-group"><label>Questions JSON</label>
          <textarea id="bench-input" rows="5" placeholder='[{"question": "What is X?", "expected": "X is ..."}]'></textarea>
        </div>
        <button class="btn btn-primary" id="bench-btn" onclick="Audit.runBenchmark()">Run Benchmark</button>
        <div id="bench-results" style="margin-top:12px"></div>
      </div>`;
  },

  async _loadFeedback() {
    const el = document.getElementById('audit-feedback');
    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';
    try {
      const res = await API.feedbackHistory(30);
      const entries = res.entries || [];
      if (!entries.length) { el.innerHTML = '<span style="color:var(--text-muted)">No feedback yet</span>'; return; }
      el.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Time</th><th>Resource</th><th>Change</th></tr></thead><tbody>` +
        entries.map(e => `<tr>
          <td style="white-space:nowrap;font-size:11px">${App.formatDate(e.timestamp)}</td>
          <td><span class="truncate" style="max-width:100px" title="${e.resource_id || ''}">${(e.resource_id || '').slice(0, 8)}</span></td>
          <td style="font-size:11px">${this._formatChange(e)}</td>
        </tr>`).join('') + '</tbody></table></div>';
    } catch (e) { el.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`; }
  },

  _formatChange(e) {
    const before = e.before_value;
    const after = e.after_value;
    if (!before && !after) return e.description || '—';
    return `<span style="color:var(--text-muted)">${JSON.stringify(before)}</span> → <span>${JSON.stringify(after)}</span>`;
  },

  async _loadAudit() {
    const el = document.getElementById('audit-log');
    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';
    try {
      const res = await API.auditLog(50);
      let entries = res.entries || [];
      const filter = document.getElementById('audit-filter')?.value;
      if (filter) entries = entries.filter(e => (e.operation || '').includes(filter));
      if (!entries.length) { el.innerHTML = '<span style="color:var(--text-muted)">No audit entries</span>'; return; }
      el.innerHTML = `<div class="table-wrap" style="max-height:400px;overflow-y:auto"><table><thead><tr><th>Time</th><th>Operation</th><th>Resource</th><th>Details</th></tr></thead><tbody>` +
        entries.slice(0, 50).map(e => `<tr>
          <td style="white-space:nowrap;font-size:11px">${App.formatDate(e.timestamp)}</td>
          <td><span class="badge" style="background:var(--bg-tertiary)">${e.operation || e.mod_type || '?'}</span></td>
          <td><span class="truncate" style="max-width:100px" title="${e.resource_id || ''}">${(e.resource_id || '').slice(0, 12)}</span></td>
          <td style="font-size:11px;cursor:pointer" onclick="this.querySelector('.json-view')?.classList.toggle('hidden')" title="Click to expand">
            ${App.truncate(JSON.stringify(e.details || e.description || ''), 60)}
            <div class="json-view hidden" style="display:none;margin-top:4px">${JSON.stringify(e, null, 2)}</div>
          </td>
        </tr>`).join('') + '</tbody></table></div>';
      // Make JSON expandable
      el.querySelectorAll('td[title="Click to expand"]').forEach(td => {
        td.addEventListener('click', () => {
          const jv = td.querySelector('.json-view');
          if (jv) jv.style.display = jv.style.display === 'none' ? 'block' : 'none';
        });
      });
    } catch (e) { el.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`; }
  },

  async runBenchmark() {
    const btn = document.getElementById('bench-btn');
    const results = document.getElementById('bench-results');
    let questions;
    try { questions = JSON.parse(document.getElementById('bench-input').value); }
    catch { App.toast('Invalid JSON', 'error'); return; }
    if (!Array.isArray(questions) || !questions.length) { App.toast('Provide array of questions', 'error'); return; }

    btn.disabled = true; btn.innerHTML = '<div class="spinner"></div>';
    results.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Running...</div>';
    try {
      const res = await API.runBenchmark(questions);
      results.innerHTML = `
        <div style="display:flex;gap:16px;margin-bottom:12px">
          <div class="card stat-card" style="padding:12px"><div class="value" style="font-size:22px">${res.accuracy}%</div><div class="label">Accuracy</div></div>
          <div class="card stat-card" style="padding:12px"><div class="value" style="font-size:22px">${res.avg_latency_ms}ms</div><div class="label">Avg Latency</div></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>Question</th><th>Expected</th><th>Actual</th><th>Correct</th><th>Latency</th></tr></thead><tbody>` +
        (res.results || []).map(r => `<tr>
          <td>${App.truncate(r.question, 80)}</td>
          <td style="font-size:11px">${App.truncate(r.expected, 60)}</td>
          <td style="font-size:11px">${App.truncate(r.actual, 60)}</td>
          <td>${r.correct ? '<span style="color:var(--success)">✓</span>' : '<span style="color:var(--danger)">✗</span>'}</td>
          <td>${r.latency_ms}ms</td>
        </tr>`).join('') + '</tbody></table></div>';
    } catch (e) { results.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`; }
    finally { btn.disabled = false; btn.textContent = 'Run Benchmark'; }
  },
};
