/**
 * Think/Ask tab — query interface with mode selector, options, history.
 */
const Think = {
  _loaded: false,
  _history: [],

  load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-think').innerHTML = this._shell();
    }
  },

  _shell() {
    return `<div class="think-layout">
      <div>
        <div class="card">
          <h3>Query</h3>
          <div class="form-group"><textarea id="think-input" rows="4" placeholder="Ask a question or search your memories..."></textarea>
            <div style="text-align:right;font-size:11px;color:var(--text-muted)" id="think-charcount">0</div>
          </div>
          <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end">
            <div class="form-group" style="margin:0"><label>Mode</label>
              <select id="think-mode"><option value="think">Think (LLM)</option><option value="recall">Recall (search)</option></select>
            </div>
            <div class="form-group" style="margin:0"><label>Limit</label>
              <input id="think-limit" type="number" min="1" max="20" value="5" style="width:60px">
            </div>
            <div class="form-group" style="margin:0"><label>Type filter</label>
              <select id="think-type"><option value="">All</option><option value="fact">Fact</option><option value="preference">Preference</option><option value="decision">Decision</option><option value="todo">Todo</option></select>
            </div>
            <button class="btn btn-primary" id="think-go" onclick="Think.execute()">Go</button>
          </div>
        </div>
        <div id="think-results" class="think-results"></div>
      </div>
      <div>
        <div class="card"><h3>History</h3><div class="query-history" id="think-history"><span style="color:var(--text-muted)">No queries yet</span></div></div>
      </div>
    </div>`;
  },

  async execute() {
    const input = document.getElementById('think-input');
    const q = input.value.trim();
    if (!q) return;
    const mode = document.getElementById('think-mode').value;
    const btn = document.getElementById('think-go');
    const results = document.getElementById('think-results');
    btn.disabled = true; btn.innerHTML = '<div class="spinner"></div>';
    results.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Processing...</div>';

    try {
      if (mode === 'think') {
        const res = await API.think(q);
        results.innerHTML = `<div class="card"><h3>Answer</h3>
          <div class="think-answer markdown-body">${this._md(res.answer || 'No answer')}</div>
          <div style="margin-top:8px"><button class="btn btn-sm" onclick="navigator.clipboard.writeText(document.querySelector('.think-answer').textContent);App.toast('Copied','info')">Copy</button></div>
        </div>`;
      } else {
        const limit = parseInt(document.getElementById('think-limit').value) || 5;
        const type = document.getElementById('think-type').value;
        const params = { query: q, limit };
        if (type) params.memory_type = type;
        const res = await API.recall(q, params);
        const items = res.results || [];
        if (!items.length) {
          results.innerHTML = '<div class="card"><h3>Results</h3><span style="color:var(--text-muted)">No results found</span></div>';
        } else {
          results.innerHTML = `<div class="card"><h3>Results (${items.length})</h3><table><thead><tr><th>Content</th><th>Score</th><th>Type</th></tr></thead><tbody>` +
            items.map(r => {
              const mt = r.memory_type || r.metadata?.memory_type || 'fact';
              const score = r.score !== undefined ? (r.score * 100).toFixed(0) + '%' : '—';
              return `<tr><td class="markdown-body" style="max-width:500px">${this._md(r.content || r.document || '')}</td><td>${score}</td><td>${App.typeBadge(mt)}</td></tr>`;
            }).join('') + '</tbody></table></div>';
        }
        // Show graph results if present
        if (res.graph_results?.length) {
          results.innerHTML += `<div class="card" style="margin-top:8px"><h3>Related Entities</h3>` +
            res.graph_results.map(g => `<div style="margin-bottom:6px"><strong>${g.node?.name || '?'}</strong> <span class="badge" style="background:var(--bg-secondary)">${g.node?.type || ''}</span></div>`).join('') + '</div>';
        }
      }
      // Add to history
      this._history.unshift({ query: q, mode, time: new Date() });
      if (this._history.length > 10) this._history.pop();
      this._renderHistory();
    } catch (e) {
      results.innerHTML = `<div class="card" style="color:var(--error)">${e.message}</div>`;
    } finally {
      btn.disabled = false; btn.textContent = 'Go';
    }
  },

  _renderHistory() {
    const el = document.getElementById('think-history');
    if (!this._history.length) { el.innerHTML = '<span style="color:var(--text-muted)">No queries yet</span>'; return; }
    el.innerHTML = this._history.map((h, i) => `<div class="query-item" onclick="Think.rerun(${i})">
      <span class="badge" style="font-size:9px;margin-right:4px">${h.mode}</span>
      ${App.truncate(h.query, 60)}
      <div style="font-size:10px;color:var(--text-muted)">${App.formatDate(h.time.toISOString())}</div>
    </div>`).join('');
  },

  rerun(idx) {
    const h = this._history[idx];
    if (!h) return;
    document.getElementById('think-input').value = h.query;
    document.getElementById('think-mode').value = h.mode;
    this.execute();
  },

  _esc(s) { return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
  _md(s) {
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
      // Collapse 3+ newlines to 2 before parsing to avoid empty <p> tags
      const clean = (s || '').replace(/\n{3,}/g, '\n\n');
      const html = marked.parse(clean, { gfm: true, breaks: false });
      return DOMPurify.sanitize(
        html.replace(/<li><p>([\s\S]*?)<\/p>\s*<\/li>/g, '<li>$1</li>')
            .replace(/<p>\s*<\/p>/g, '')
      );
    }
    return this._esc(s);
  },
};

// Character count
document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('think-input');
  if (el) el.addEventListener('input', () => {
    const cc = document.getElementById('think-charcount');
    if (cc) cc.textContent = el.value.length;
  });
});
