/**
 * Dashboard tab — stats cards, recent memories, quick actions.
 */
const Dashboard = {
  _loaded: false,

  async load() {
    if (this._loaded) return;
    this._loaded = true;
    const el = document.getElementById('tab-dashboard');
    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading dashboard...</div>';
    try {
      const [statusRes, healthRes] = await Promise.all([API.status(), API.health()]);
      this._render(el, statusRes, healthRes);
    } catch (e) {
      el.innerHTML = `<div class="card" style="color:var(--danger)">Failed to load dashboard: ${e.message}</div>`;
    }
  },

  reload() { this._loaded = false; this.load(); },

  _render(el, status, health) {
    const ep = status.episodic || {};
    const sem = status.semantic || {};
    el.innerHTML = `
      <div class="stats-grid">
        <div class="card stat-card"><div class="value">${ep.count || 0}</div><div class="label">Episodic Memories</div></div>
        <div class="card stat-card"><div class="value">${sem.node_count || 0}</div><div class="label">Semantic Nodes</div></div>
        <div class="card stat-card"><div class="value">${sem.edge_count || 0}</div><div class="label">Semantic Edges</div></div>
        <div class="card stat-card"><div class="value" style="font-size:18px;color:var(--success)">${health.status || '?'}</div><div class="label">Health</div></div>
      </div>
      ${sem.node_types ? this._nodeTypesHtml(sem.node_types) : ''}
      <div style="display:flex;gap:12px;margin-bottom:16px">
        <button class="btn btn-primary" onclick="Dashboard.showRememberModal()">+ New Memory</button>
        <button class="btn" onclick="Dashboard.showThinkModal()">Think</button>
      </div>
      <div class="card">
        <h3>Recent Memories</h3>
        <div id="dash-recent"><div class="loading-overlay"><div class="spinner"></div></div></div>
      </div>`;
    this._loadRecent();
  },

  _nodeTypesHtml(types) {
    const items = Object.entries(types).map(([t, c]) => `<span class="pill">${t}: ${c}</span>`).join(' ');
    return `<div class="card" style="margin-bottom:16px"><h3>Node Types</h3>${items}</div>`;
  },

  async _loadRecent() {
    try {
      const res = await API.listMemories({ limit: 10 });
      const el = document.getElementById('dash-recent');
      if (!res.memories || !res.memories.length) { el.innerHTML = '<span style="color:var(--text-muted)">No memories yet</span>'; return; }
      el.innerHTML = '<table><thead><tr><th>Content</th><th>Type</th><th>Priority</th><th>Confidence</th><th>Created</th></tr></thead><tbody>' +
        res.memories.map(m => `<tr style="cursor:pointer" onclick="Memories.showDetail('${m.id}')">
          <td><span class="truncate truncate-wide" title="${this._esc(m.content)}">${App.truncate(m.content)}</span></td>
          <td>${App.typeBadge(m.memory_type)}</td>
          <td>${App.priorityBar(m.priority)}</td>
          <td>${App.confidenceBar(m.confidence)}</td>
          <td style="white-space:nowrap">${App.formatDate(m.timestamp)}</td>
        </tr>`).join('') + '</tbody></table>';
    } catch (e) {
      document.getElementById('dash-recent').innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    }
  },

  _esc(s) { return (s || '').replace(/"/g, '&quot;').replace(/</g, '&lt;'); },

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
      document.getElementById('think-result').innerHTML = `<div class="think-answer">${(res.answer || '').replace(/</g, '&lt;')}</div>`;
    } catch (e) {
      document.getElementById('think-result').innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    } finally { btn.disabled = false; btn.textContent = 'Think'; }
  },
};
