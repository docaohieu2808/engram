/**
 * Dashboard tab — stats cards with icons/gradients, recent memories with pagination, quick actions.
 */
const Dashboard = {
  _loaded: false,
  _offset: 0,
  _limit: 10,
  _total: 0,

  async load() {
    if (this._loaded) return;
    this._loaded = true;
    const el = document.getElementById('tab-dashboard');
    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading dashboard...</div>';
    try {
      const [statusRes, healthRes] = await Promise.all([API.status(), API.health()]);
      this._render(el, statusRes, healthRes);
    } catch (e) {
      el.innerHTML = `<div class="card" style="color:var(--error)">Failed to load dashboard: ${e.message}</div>`;
    }
  },

  reload() { this._loaded = false; this._offset = 0; this.load(); },

  _render(el, status, health) {
    const ep = status.episodic || {};
    const sem = status.semantic || {};
    const healthStatus = (health.status || '?').toUpperCase();
    const healthColor = healthStatus === 'OK' || healthStatus === 'HEALTHY' ? 'var(--success)' : 'var(--warning)';
    el.innerHTML = `
      <div class="stats-grid">
        <div class="card stat-card stat-card--memories">
          <div class="stat-icon">${Icons.statBrain}</div>
          <div class="value">${ep.count || 0}</div>
          <div class="label">Episodic Memories</div>
        </div>
        <div class="card stat-card stat-card--nodes">
          <div class="stat-icon">${Icons.statNodes}</div>
          <div class="value">${sem.node_count || 0}</div>
          <div class="label">Semantic Nodes</div>
        </div>
        <div class="card stat-card stat-card--edges">
          <div class="stat-icon">${Icons.statEdges}</div>
          <div class="value">${sem.edge_count || 0}</div>
          <div class="label">Semantic Edges</div>
        </div>
        <div class="card stat-card stat-card--health">
          <div class="stat-icon">${Icons.statHealth}</div>
          <div class="value" style="font-size:32px;color:${healthColor}">${healthStatus}</div>
          <div class="label">Health</div>
        </div>
      </div>
      ${sem.node_types ? this._nodeTypesHtml(sem.node_types) : ''}
      <div class="dash-actions">
        <button class="btn btn-primary" onclick="Dashboard.showRememberModal()">${Icons.plus} New Memory</button>
        <button class="btn" onclick="Dashboard.showThinkModal()">Think</button>
      </div>
      <div class="card">
        <h3>Recent Memories</h3>
        <div id="dash-recent"><div class="loading-overlay"><div class="spinner"></div></div></div>
      </div>`;
    this._offset = 0;
    this._loadRecent();
  },

  _nodeTypesHtml(types) {
    const items = Object.entries(types).map(([t, c]) => `<span class="pill">${t}: ${c}</span>`).join(' ');
    return `<div class="card" style="margin-bottom:20px"><h3>Node Types</h3>${items}</div>`;
  },

  async _loadRecent() {
    const el = document.getElementById('dash-recent');
    try {
      const res = await API.listMemories({ limit: this._limit, offset: this._offset });
      this._total = res.total || 0;

      if (this._offset === 0) {
        el.innerHTML = `
          <div class="table-info">Showing ${Math.min(this._limit, res.memories.length)} of ${this._total.toLocaleString()} memories</div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Content</th><th>Type</th><th>Priority</th><th>Confidence</th><th>Created</th></tr></thead>
              <tbody id="dash-recent-tbody"></tbody>
            </table>
          </div>
          <div id="dash-load-more"></div>
        `;
      }

      const tbody = document.getElementById('dash-recent-tbody');
      res.memories.forEach(m => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.onclick = () => Memories.showDetail(m.id);
        tr.innerHTML = `
          <td><span class="truncate truncate-wide" title="${this._esc(m.content)}">${App.truncate(m.content)}</span></td>
          <td>${App.typeBadge(m.memory_type)}</td>
          <td>${App.priorityBar(m.priority)}</td>
          <td>${App.confidenceBar(m.confidence)}</td>
          <td style="white-space:nowrap">${App.formatDate(m.timestamp)}</td>
        `;
        tbody.appendChild(tr);
      });

      const shown = this._offset + res.memories.length;
      const infoEl = el.querySelector('.table-info');
      if (infoEl) infoEl.textContent = `Showing ${shown} of ${this._total.toLocaleString()} memories`;

      const loadMoreEl = document.getElementById('dash-load-more');
      if (shown < this._total) {
        loadMoreEl.innerHTML = `
          <button class="btn" onclick="Dashboard.loadMore()">
            Load More (${(this._total - shown).toLocaleString()} remaining)
          </button>
        `;
      } else {
        loadMoreEl.innerHTML = '<span class="text-muted">All memories loaded</span>';
      }
    } catch (e) {
      el.innerHTML = `<span class="text-error">${e.message}</span>`;
    }
  },

  async loadMore() {
    this._offset += this._limit;
    await this._loadRecent();
  },

  _esc(s) { return (s || '').replace(/"/g, '&quot;').replace(/</g, '&lt;'); },
  _md(s) {
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
      const clean = (s || '').replace(/\n{3,}/g, '\n\n');
      const html = marked.parse(clean, { gfm: true, breaks: false });
      return DOMPurify.sanitize(
        html.replace(/<li><p>([\s\S]*?)<\/p>\s*<\/li>/g, '<li>$1</li>')
            .replace(/<p>\s*<\/p>/g, '')
      );
    }
    return this._esc(s);
  },

  showRememberModal() {
    App.showModal(`
      <h2>New Memory</h2>
      <div class="form-group"><label>Content</label><textarea id="rm-content" rows="4" placeholder="What to remember..."></textarea></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="form-group"><label>Type</label>
          <select id="rm-type"><option value="fact">Fact</option><option value="preference">Preference</option><option value="decision">Decision</option><option value="todo">Todo</option><option value="error">Error</option><option value="context">Context</option><option value="workflow">Workflow</option></select>
        </div>
        <div class="form-group"><label>Priority (1-10)</label><input id="rm-priority" type="number" min="1" max="10" value="5"></div>
      </div>
      <div class="form-group"><label>Tags (comma-separated)</label><input id="rm-tags" placeholder="tag1, tag2"></div>
      <div class="form-group"><label>Entities (comma-separated)</label><input id="rm-entities" placeholder="entity1, entity2"></div>
      <div class="modal-actions">
        <button class="btn" onclick="App.closeModal()">Cancel</button>
        <button class="btn btn-primary" onclick="Dashboard.doRemember()">Save</button>
      </div>`);
  },

  async doRemember() {
    const content = document.getElementById('rm-content').value.trim();
    if (!content) { App.toast('Content required', 'error'); return; }
    try {
      const tags = document.getElementById('rm-tags').value.split(',').map(t => t.trim()).filter(Boolean);
      const entities = document.getElementById('rm-entities').value.split(',').map(e => e.trim()).filter(Boolean);
      await API.remember({
        content,
        memory_type: document.getElementById('rm-type').value,
        priority: parseInt(document.getElementById('rm-priority').value) || 5,
        tags, entities,
      });
      App.closeModal();
      App.toast('Memory saved', 'success');
      this.reload();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  showThinkModal() {
    App.showModal(`
      <h2>Think</h2>
      <div class="form-group"><label>Question</label><textarea id="think-q" rows="3" placeholder="Ask a question..."></textarea></div>
      <div class="modal-actions">
        <button class="btn" onclick="App.closeModal()">Cancel</button>
        <button class="btn btn-primary" id="think-btn" onclick="Dashboard.doThink()">Think</button>
      </div>
      <div id="think-result" style="margin-top:12px"></div>`);
  },

  async doThink() {
    const q = document.getElementById('think-q').value.trim();
    if (!q) return;
    const btn = document.getElementById('think-btn');
    btn.disabled = true; btn.innerHTML = '<div class="spinner"></div>';
    try {
      const res = await API.think(q);
      document.getElementById('think-result').innerHTML = `<div class="think-answer markdown-body">${this._md(res.answer || 'No answer')}</div>`;
    } catch (e) {
      document.getElementById('think-result').innerHTML = `<span style="color:var(--error)">${e.message}</span>`;
    } finally { btn.disabled = false; btn.textContent = 'Think'; }
  },
};
