/**
 * Settings tab — config display, scheduler, danger zone.
 */
const Settings = {
  _loaded: false,

  async load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-settings').innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading settings...</div>';
    }
    await this._render();
  },

  async _render() {
    const el = document.getElementById('tab-settings');
    let status = {}, scheduler = [];
    try { status = await API.status(); } catch {}
    try { const s = await API.schedulerTasks(); scheduler = s.tasks || []; } catch {}

    el.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
        <div class="card">
          <h3>Episodic Store</h3>
          ${this._kvTable(status.episodic || {})}
        </div>
        <div class="card">
          <h3>Semantic Graph</h3>
          ${this._kvTable(status.semantic || {})}
        </div>
      </div>
      <div class="card" style="margin-bottom:12px">
        <h3>Scheduler Tasks</h3>
        ${scheduler.length ? this._schedulerTable(scheduler) : '<span style="color:var(--text-muted)">No tasks registered (scheduler may not be running)</span>'}
      </div>
      <div class="card" style="border-color:var(--danger)">
        <h3 style="color:var(--danger)">Danger Zone</h3>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn btn-danger" onclick="Settings.cleanup()">Cleanup Expired</button>
          <button class="btn btn-danger" onclick="Settings.clearAll()">Clear All Memories</button>
        </div>
      </div>`;
  },

  _kvTable(obj) {
    const entries = Object.entries(obj);
    if (!entries.length) return '<span style="color:var(--text-muted)">—</span>';
    return '<table>' + entries.map(([k, v]) => {
      const display = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return `<tr><td style="color:var(--text-secondary);padding:3px 10px 3px 0">${k}</td><td style="padding:3px 0">${display}</td></tr>`;
    }).join('') + '</table>';
  },

  _schedulerTable(tasks) {
    return `<div class="table-wrap"><table><thead><tr><th>Task</th><th>Interval</th><th>Runs</th><th>Next In</th><th>LLM</th><th>Error</th><th>Action</th></tr></thead><tbody>` +
      tasks.map(t => `<tr>
        <td>${t.name}</td>
        <td>${this._formatInterval(t.interval_seconds)}</td>
        <td>${t.run_count}</td>
        <td>${this._formatInterval(t.next_run_in)}</td>
        <td>${t.requires_llm ? '<span style="color:var(--warning)">Yes</span>' : 'No'}</td>
        <td style="color:var(--danger);font-size:11px">${t.last_error || '—'}</td>
        <td><button class="btn btn-sm" onclick="Settings.forceRun('${t.name}')">Run</button></td>
      </tr>`).join('') + '</tbody></table></div>';
  },

  _formatInterval(s) {
    if (!s && s !== 0) return '—';
    if (s >= 3600) return `${Math.round(s / 3600)}h`;
    if (s >= 60) return `${Math.round(s / 60)}m`;
    return `${Math.round(s)}s`;
  },

  async forceRun(name) {
    try {
      await API.forceRunTask(name);
      App.toast(`Task ${name} triggered`, 'success');
    } catch (e) { App.toast(e.message, 'error'); }
  },

  async cleanup() {
    if (!confirm('Delete all expired memories?')) return;
    try {
      const res = await API.cleanup();
      App.toast(`Cleaned up ${res.deleted || 0} expired memories`, 'success');
    } catch (e) { App.toast(e.message, 'error'); }
  },

  clearAll() {
    App.showModal(`<h2 style="color:var(--danger)">Clear All Memories</h2>
      <p>This will permanently delete ALL episodic memories. This cannot be undone.</p>
      <p>Type <strong>DELETE</strong> to confirm:</p>
      <div class="form-group"><input id="confirm-delete" placeholder="Type DELETE"></div>
      <div class="modal-actions">
        <button class="btn" onclick="App.closeModal()">Cancel</button>
        <button class="btn btn-danger" onclick="Settings.doClearAll()">Delete Everything</button>
      </div>`);
  },

  async doClearAll() {
    if (document.getElementById('confirm-delete')?.value !== 'DELETE') {
      App.toast('Type DELETE to confirm', 'error'); return;
    }
    try {
      // Fetch all memory IDs and bulk-delete
      const all = await API.exportMemories({ limit: 1000 });
      const ids = (all.memories || []).map(m => m.id);
      if (ids.length) await API.bulkDelete(ids);
      App.closeModal();
      App.toast(`Deleted ${ids.length} memories`, 'success');
    } catch (e) { App.toast(e.message, 'error'); }
  },
};
