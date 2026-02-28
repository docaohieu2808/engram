/**
 * Settings tab — live config editor, scheduler, danger zone.
 */
const Settings = {
  _loaded: false,
  _config: null,
  _restartSections: new Set(),
  _dirty: {},  // key_path → new value

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
    try {
      const res = await API.getConfig();
      this._config = res.config || {};
      this._restartSections = new Set(res.restart_required_sections || []);
    } catch { this._config = null; }

    el.innerHTML = `
      ${this._config ? this._modelSelector() : ''}
      ${this._config ? this._configEditor() : ''}
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
        ${scheduler.length ? this._schedulerTable(scheduler) : '<span style="color:var(--text-muted)">No tasks registered</span>'}
      </div>
      <div class="card" style="border-color:var(--error)">
        <h3 style="color:var(--error)">Danger Zone</h3>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn btn-danger" onclick="Settings.cleanup()">Cleanup Expired</button>
          <button class="btn btn-danger" onclick="Settings.clearAll()">Clear All Memories</button>
        </div>
      </div>`;
    this._dirty = {};
  },

  // --- Config editor ---

  // Config sections to show, grouped for readability
  _configGroups: [
    { label: 'LLM', sections: ['llm'] },
    { label: 'Embedding', sections: ['embedding'] },
    { label: 'Extraction', sections: ['extraction'] },
    { label: 'Recall', sections: ['recall', 'recall_pipeline'] },
    { label: 'Scoring', sections: ['scoring'] },
    { label: 'Scheduler', sections: ['scheduler'] },
    { label: 'Consolidation', sections: ['consolidation'] },
    { label: 'Resolution', sections: ['resolution'] },
    { label: 'Feedback', sections: ['feedback'] },
    { label: 'Episodic', sections: ['episodic'] },
    { label: 'Semantic', sections: ['semantic'] },
    { label: 'Capture', sections: ['capture'] },
    { label: 'Hooks', sections: ['hooks'] },
    { label: 'Logging', sections: ['logging'] },
    { label: 'Security', sections: ['security', 'auth'] },
    { label: 'Cache', sections: ['cache'] },
    { label: 'Rate Limit', sections: ['rate_limit'] },
    { label: 'Retrieval Audit', sections: ['retrieval_audit'] },
    { label: 'Server', sections: ['serve'] },
  ],

  // Known model families and their thinking model status
  _modelOptions: [
    { value: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6', thinking: false },
    { value: 'anthropic/claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5', thinking: false },
    { value: 'gemini/gemini-2.5-flash', label: 'Gemini 2.5 Flash', thinking: true },
    { value: 'gemini/gemini-2.5-pro', label: 'Gemini 2.5 Pro', thinking: true },
    { value: 'openai/gpt-4o', label: 'GPT-4o', thinking: false },
    { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini', thinking: false },
  ],

  _modelSelector() {
    const currentModel = this._config?.llm?.model || '';
    const disableThinking = this._config?.llm?.disable_thinking || false;
    const opts = this._modelOptions.map(m =>
      `<option value="${m.value}"${m.value === currentModel ? ' selected' : ''}>${m.label}</option>`
    ).join('');

    return `<div class="card" style="margin-bottom:12px">
      <h3 style="margin:0 0 8px">Model</h3>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <div>
          <label style="font-size:11px;color:var(--text-muted)">LLM Model</label>
          <select id="model-select" onchange="Settings._onModelChange(this.value)" style="padding:4px 8px">
            ${opts}
            <option value="_custom"${!this._modelOptions.find(m => m.value === currentModel) ? ' selected' : ''}>Custom...</option>
          </select>
        </div>
        <div id="model-custom-wrap" style="display:${this._modelOptions.find(m => m.value === currentModel) ? 'none' : 'block'}">
          <label style="font-size:11px;color:var(--text-muted)">Custom model ID</label>
          <input id="model-custom" type="text" value="${currentModel}" style="width:220px" onchange="Settings._onChange('llm.model',this.value)">
        </div>
        <div>
          <label style="font-size:11px;color:var(--text-muted)">Disable thinking</label>
          <select id="model-thinking" onchange="Settings._onChange('llm.disable_thinking',this.value==='true')" style="padding:4px 8px">
            <option value="false"${!disableThinking ? ' selected' : ''}>No</option>
            <option value="true"${disableThinking ? ' selected' : ''}>Yes</option>
          </select>
        </div>
        <button class="btn btn-sm" onclick="Settings.testModel()" id="model-test-btn">Test</button>
        <span id="model-test-result" style="font-size:12px"></span>
      </div>
    </div>`;
  },

  _onModelChange(value) {
    const customWrap = document.getElementById('model-custom-wrap');
    if (value === '_custom') {
      customWrap.style.display = 'block';
      return;
    }
    customWrap.style.display = 'none';
    this._onChange('llm.model', value);
    // Auto-set disable_thinking based on model family
    const known = this._modelOptions.find(m => m.value === value);
    if (known) {
      const thinkEl = document.getElementById('model-thinking');
      if (thinkEl) { thinkEl.value = String(known.thinking); }
      this._onChange('llm.disable_thinking', known.thinking);
    }
  },

  async testModel() {
    const btn = document.getElementById('model-test-btn');
    const result = document.getElementById('model-test-result');
    btn.disabled = true; btn.textContent = 'Testing...';
    result.textContent = '';
    try {
      const res = await API.think('ping');
      result.style.color = 'var(--success)';
      result.textContent = 'OK — model responded';
    } catch (e) {
      result.style.color = 'var(--error)';
      result.textContent = `Error: ${e.message}`;
    }
    btn.disabled = false; btn.textContent = 'Test';
  },

  _configEditor() {
    const groups = this._configGroups.map(g => {
      const sectionHtml = g.sections.map(s => {
        const data = this._config[s];
        if (!data || typeof data !== 'object') return '';
        const needsRestart = this._restartSections.has(s);
        return this._sectionFields(s, data, needsRestart);
      }).join('');
      if (!sectionHtml) return '';
      return `<div class="config-group"><h4 style="margin:0 0 6px;color:var(--primary)">${g.label}</h4>${sectionHtml}</div>`;
    }).filter(Boolean).join('');

    return `<div class="card" style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h3 style="margin:0">Configuration</h3>
        <div style="display:flex;gap:6px">
          <button class="btn btn-sm" onclick="Settings._render()" title="Reload">↻ Reload</button>
          <button class="btn btn-sm btn-primary" id="cfg-save-btn" onclick="Settings.saveConfig()" disabled>Save Changes</button>
        </div>
      </div>
      <div id="cfg-status" style="display:none;margin-bottom:8px;padding:6px 10px;border-radius:4px;font-size:12px"></div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px">${groups}</div>
    </div>`;
  },

  // Fields with known dropdown options: "section.field" → [{value, label}]
  _dropdownOptions: {
    'embedding.provider': [
      { value: 'gemini', label: 'Gemini' },
      { value: 'openai', label: 'OpenAI' },
      { value: 'cohere', label: 'Cohere' },
    ],
    'embedding.model': [
      { value: 'gemini-embedding-001', label: 'Gemini Embedding 001' },
      { value: 'text-embedding-3-small', label: 'OpenAI text-embedding-3-small' },
      { value: 'text-embedding-3-large', label: 'OpenAI text-embedding-3-large' },
    ],
    'embedding.key_strategy': [
      { value: 'failover', label: 'Failover (primary first)' },
      { value: 'round-robin', label: 'Round Robin (spread quota)' },
    ],
    'llm.provider': [
      { value: 'anthropic', label: 'Anthropic' },
      { value: 'gemini', label: 'Google Gemini' },
      { value: 'openai', label: 'OpenAI' },
    ],
    'llm.model': [
      { value: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
      { value: 'anthropic/claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
      { value: 'gemini/gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
      { value: 'gemini/gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
      { value: 'openai/gpt-4o', label: 'GPT-4o' },
      { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
    ],
    'extraction.llm_model': [
      { value: '', label: '(inherit from LLM model)' },
      { value: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
      { value: 'anthropic/claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
      { value: 'gemini/gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
      { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
    ],
    'consolidation.llm_model': [
      { value: '', label: '(inherit from LLM model)' },
      { value: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
      { value: 'gemini/gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
      { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
    ],
    'resolution.llm_model': [
      { value: '', label: '(inherit from LLM model)' },
      { value: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
      { value: 'gemini/gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
      { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
    ],
    'episodic.provider': [
      { value: 'chromadb', label: 'ChromaDB' },
    ],
    'episodic.mode': [
      { value: 'embedded', label: 'Embedded (local)' },
      { value: 'http', label: 'HTTP (remote)' },
    ],
    'semantic.provider': [
      { value: 'sqlite', label: 'SQLite' },
      { value: 'postgresql', label: 'PostgreSQL' },
    ],
    'logging.format': [
      { value: 'text', label: 'Text' },
      { value: 'json', label: 'JSON' },
    ],
    'logging.level': [
      { value: 'DEBUG', label: 'DEBUG' },
      { value: 'INFO', label: 'INFO' },
      { value: 'WARNING', label: 'WARNING' },
      { value: 'ERROR', label: 'ERROR' },
    ],
    'audit.backend': [
      { value: 'file', label: 'File' },
    ],
    'event_bus.backend': [
      { value: 'memory', label: 'In-Memory' },
      { value: 'redis', label: 'Redis' },
    ],
  },

  _sectionFields(section, data, needsRestart) {
    const badge = needsRestart ? ' <span style="background:var(--warning);color:#000;padding:1px 5px;border-radius:3px;font-size:10px">restart required</span>' : '';
    const rows = Object.entries(data).map(([field, value]) => {
      const keyPath = `${section}.${field}`;
      const inputId = `cfg-${keyPath.replace(/\./g, '-')}`;
      const type = typeof value;
      let input;
      // Check if this field has known dropdown options
      const opts = this._dropdownOptions[keyPath];
      if (opts) {
        const currentVal = String(value ?? '');
        const hasCustom = currentVal && !opts.find(o => String(o.value) === currentVal);
        input = `<select id="${inputId}" onchange="Settings._onChange('${keyPath}',this.value)" style="min-width:160px">` +
          opts.map(o => `<option value="${o.value}"${String(o.value) === currentVal ? ' selected' : ''}>${o.label}</option>`).join('') +
          (hasCustom ? `<option value="${currentVal}" selected>${currentVal}</option>` : '') +
          `</select>`;
      } else if (type === 'boolean') {
        input = `<select id="${inputId}" onchange="Settings._onChange('${keyPath}',this.value==='true')">
          <option value="true"${value ? ' selected' : ''}>true</option>
          <option value="false"${!value ? ' selected' : ''}>false</option>
        </select>`;
      } else if (type === 'number') {
        const step = Number.isInteger(value) ? '1' : '0.01';
        input = `<input id="${inputId}" type="number" step="${step}" value="${value}" onchange="Settings._onChange('${keyPath}',Number(this.value))" style="width:100px">`;
      } else if (type === 'object' && value !== null) {
        return ''; // skip nested objects/arrays in flat editor
      } else {
        const displayVal = String(value || '').replace(/"/g, '&quot;');
        input = `<input id="${inputId}" type="text" value="${displayVal}" onchange="Settings._onChange('${keyPath}',this.value)" style="width:180px">`;
      }
      return `<tr><td style="color:var(--text-secondary);padding:2px 8px 2px 0;font-size:12px;white-space:nowrap">${field}</td><td style="padding:2px 0">${input}</td></tr>`;
    }).filter(Boolean).join('');
    if (!rows) return '';
    return `<div style="margin-bottom:6px"><div style="font-size:11px;color:var(--text-muted);font-weight:600">${section}${badge}</div><table style="font-size:12px">${rows}</table></div>`;
  },

  _onChange(keyPath, value) {
    this._dirty[keyPath] = value;
    const btn = document.getElementById('cfg-save-btn');
    if (btn) { btn.disabled = false; btn.textContent = `Save (${Object.keys(this._dirty).length})`; }
  },

  async saveConfig() {
    if (!Object.keys(this._dirty).length) return;
    const statusEl = document.getElementById('cfg-status');
    try {
      const res = await API.updateConfig(this._dirty);
      if (statusEl) {
        const msg = res.restart_required
          ? `Saved ${res.changed.length} changes. <strong>Server restart required</strong> for some settings to take effect.`
          : `Saved ${res.changed.length} changes.`;
        statusEl.style.display = 'block';
        statusEl.style.background = res.restart_required ? 'var(--warning-bg, #fff3cd)' : 'var(--success-bg, #d4edda)';
        statusEl.style.color = res.restart_required ? 'var(--warning, #856404)' : 'var(--success, #155724)';
        statusEl.innerHTML = msg;
      }
      this._dirty = {};
      const btn = document.getElementById('cfg-save-btn');
      if (btn) { btn.disabled = true; btn.textContent = 'Save Changes'; }
      App.toast('Config saved', 'success');
    } catch (e) {
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.style.background = 'var(--error-bg, #f8d7da)';
        statusEl.style.color = 'var(--error, #721c24)';
        statusEl.textContent = e.message;
      }
      App.toast(e.message, 'error');
    }
  },

  // --- Helpers ---

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
        <td style="color:var(--error);font-size:11px">${t.last_error || '—'}</td>
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
    App.showModal(`<h2 style="color:var(--error)">Clear All Memories</h2>
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
      const all = await API.exportMemories({ limit: 1000 });
      const ids = (all.memories || []).map(m => m.id);
      if (ids.length) await API.bulkDelete(ids);
      App.closeModal();
      App.toast(`Deleted ${ids.length} memories`, 'success');
    } catch (e) { App.toast(e.message, 'error'); }
  },
};
