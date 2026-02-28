/**
 * Settings tab — live config editor, scheduler, danger zone.
 */
const Settings = {
  _loaded: false,
  _config: null,
  _restartSections: new Set(),
  _dirty: {},  // key_path -> new value
  _fetchedModels: null,  // { anthropic: [...], gemini: [...], openai: [...] }

  async load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-settings').innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading settings...</div>';
      // Auto-fetch live models from provider APIs
      try {
        const res = await API.listModels('all');
        const fetched = res.models || {};
        for (const [provider, models] of Object.entries(fetched)) {
          if (models.length) {
            this._llmModels[provider] = models.map(m => ({
              value: m, label: m.replace(/^(anthropic|gemini|openai)\//, ''), thinking: false,
            }));
          }
        }
      } catch {}
    }
    await this._render();
  },

  async _render() {
    const el = document.getElementById('tab-settings');
    let scheduler = [];
    try {
      const res = await API.getConfig();
      this._config = res.config || {};
      this._restartSections = new Set(res.restart_required_sections || []);
    } catch { this._config = null; }
    try { const s = await API.schedulerTasks(); scheduler = s.tasks || []; } catch {}

    el.innerHTML = `
      ${this._config ? this._modelSelector() : ''}
      ${this._config ? this._configEditor() : ''}
      <div class="card settings-card">
        <h3>Scheduler Tasks</h3>
        ${scheduler.length ? this._schedulerTable(scheduler) : '<span class="empty-state">No tasks registered</span>'}
      </div>
      <div class="card settings-card danger-card">
        <h3>Danger Zone</h3>
        <div class="danger-actions">
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

  _modelSelector() {
    const currentModel = this._config?.llm?.model || '';
    const currentProvider = this._config?.llm?.provider || '';
    const disableThinking = this._config?.llm?.disable_thinking || false;
    const models = this._getLlmModels(currentProvider);
    const hasModels = models.length > 0;
    const currentInList = hasModels && models.find(m => m.value === currentModel);
    const opts = models.map(m =>
      `<option value="${m.value}"${m.value === currentModel ? ' selected' : ''}>${m.label}</option>`
    ).join('');

    return `<div class="card settings-card">
      <h3>Model</h3>
      <div class="model-selector">
        <div class="form-group">
          <label>LLM Model</label>
          ${hasModels
            ? `<select id="model-select" onchange="Settings._onModelChange(this.value)">
                ${opts}
                <option value="_custom"${!currentInList ? ' selected' : ''}>Custom...</option>
              </select>`
            : `<select id="model-select" disabled><option>No models available</option></select>`}
          ${!hasModels ? `<div class="model-warning">No models loaded — set API key for ${currentProvider} or click Refresh</div>` : ''}
        </div>
        <div class="form-group" id="model-custom-wrap" style="display:${currentInList ? 'none' : 'block'}">
          <label>Custom model ID</label>
          <input id="model-custom" type="text" value="${currentModel}" onchange="Settings._onChange('llm.model',this.value)">
        </div>
        <div class="form-group">
          <label>Disable thinking</label>
          <select id="model-thinking" onchange="Settings._onChange('llm.disable_thinking',this.value==='true')">
            <option value="false"${!disableThinking ? ' selected' : ''}>No</option>
            <option value="true"${disableThinking ? ' selected' : ''}>Yes</option>
          </select>
        </div>
      </div>
      <div class="model-actions">
        <button class="btn btn-sm" onclick="Settings.testModel()" id="model-test-btn">Test</button>
        <button class="btn btn-sm" onclick="Settings.refreshModels()" id="model-refresh-btn" title="Fetch latest models from providers">Refresh Models</button>
        <span id="model-test-result" class="model-test-result"></span>
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
    const known = Object.values(this._llmModels).flat().find(m => m.value === value);
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
      const res = await API.testModel();
      result.className = 'model-test-result';
      result.style.color = 'var(--success)';
      result.textContent = `OK — ${res.response}`;
    } catch (e) {
      result.className = 'model-test-result';
      result.style.color = 'var(--error)';
      result.textContent = `Error: ${e.message}`;
    }
    btn.disabled = false; btn.textContent = 'Test';
  },

  async refreshModels() {
    const btn = document.getElementById('model-refresh-btn');
    const result = document.getElementById('model-test-result');
    btn.disabled = true; btn.textContent = 'Loading...';
    try {
      const res = await API.listModels('all');
      this._fetchedModels = res.models || {};
      // Update _llmModels with fetched data (only if non-empty, keep hardcoded fallback)
      for (const [provider, models] of Object.entries(this._fetchedModels)) {
        if (models.length) {
          this._llmModels[provider] = models.map(m => ({
            value: m, label: m.replace(/^(anthropic|gemini|openai)\//, ''), thinking: false,
          }));
        }
      }
      // Re-render to pick up new models
      await this._render();
      const total = Object.values(this._fetchedModels).reduce((s, a) => s + a.length, 0);
      App.toast(`Loaded ${total} models from providers`, 'success');
    } catch (e) {
      if (result) { result.style.color = 'var(--error)'; result.textContent = e.message; }
    }
    btn.disabled = false; btn.textContent = 'Refresh Models';
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
      return `<div class="config-group">
        <div class="config-group-title">${g.label}</div>
        ${sectionHtml}
      </div>`;
    }).filter(Boolean).join('');

    return `<div class="card settings-card">
      <div class="card-header">
        <h3>Configuration</h3>
        <div class="card-header-actions">
          <button class="btn btn-sm" onclick="Settings._render()" title="Reload">Reload</button>
          <button class="btn btn-sm btn-primary" id="cfg-save-btn" onclick="Settings.saveConfig()" disabled>Save Changes</button>
        </div>
      </div>
      <div id="cfg-status" class="cfg-status"></div>
      <div class="config-grid">${groups}</div>
    </div>`;
  },

  // Model catalogs per provider (used for cascading dropdowns)
  _llmModels: { anthropic: [], gemini: [], openai: [] },
  _embeddingModels: {
    gemini: [
      { value: 'gemini-embedding-001', label: 'Gemini Embedding 001' },
    ],
    openai: [
      { value: 'text-embedding-3-small', label: 'text-embedding-3-small' },
      { value: 'text-embedding-3-large', label: 'text-embedding-3-large' },
    ],
    cohere: [
      { value: 'embed-english-v3.0', label: 'Embed English v3.0' },
      { value: 'embed-multilingual-v3.0', label: 'Embed Multilingual v3.0' },
    ],
  },

  // Get LLM models filtered by current provider (or all if no match)
  _getLlmModels(provider) {
    return this._llmModels[provider] || Object.values(this._llmModels).flat();
  },
  _getEmbeddingModels(provider) {
    return this._embeddingModels[provider] || Object.values(this._embeddingModels).flat();
  },

  // Build dropdown options dynamically — provider-aware for model fields
  _getDropdownOptions(keyPath) {
    const staticOpts = {
      'embedding.provider': [
        { value: 'gemini', label: 'Gemini' },
        { value: 'openai', label: 'OpenAI' },
        { value: 'cohere', label: 'Cohere' },
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
      'episodic.provider': [{ value: 'chromadb', label: 'ChromaDB' }],
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
      'audit.backend': [{ value: 'file', label: 'File' }],
      'event_bus.backend': [
        { value: 'memory', label: 'In-Memory' },
        { value: 'redis', label: 'Redis' },
      ],
    };
    if (staticOpts[keyPath]) return staticOpts[keyPath];

    // Dynamic: model fields depend on their provider
    const llmProvider = this._dirty['llm.provider'] || this._config?.llm?.provider || '';
    const embProvider = this._dirty['embedding.provider'] || this._config?.embedding?.provider || '';

    if (keyPath === 'embedding.model') return this._getEmbeddingModels(embProvider);
    if (keyPath === 'llm.model') return this._getLlmModels(llmProvider);
    // Sub-model fields: inherit list from current llm.provider + add "(inherit)" option
    if (['extraction.llm_model', 'consolidation.llm_model', 'resolution.llm_model'].includes(keyPath)) {
      return [{ value: '', label: '(inherit from LLM model)' }, ...Object.values(this._llmModels).flat()];
    }
    return null;
  },

  // Cascade: when a provider changes, re-render dependent model dropdown
  _cascadeProvider(keyPath) {
    const cascadeMap = {
      'llm.provider': 'llm.model',
      'embedding.provider': 'embedding.model',
    };
    const targetKey = cascadeMap[keyPath];
    if (!targetKey) return;
    const targetId = `cfg-${targetKey.replace(/\./g, '-')}`;
    const el = document.getElementById(targetId);
    if (!el) return;
    const opts = this._getDropdownOptions(targetKey) || [];
    if (!opts.length) return;  // Don't cascade if no models available
    const firstVal = opts[0].value;
    el.innerHTML = opts.map(o =>
      `<option value="${o.value}"${o.value === firstVal ? ' selected' : ''}>${o.label}</option>`
    ).join('');
    this._onChange(targetKey, firstVal);

    // For llm.provider, also update the top Model selector card
    if (keyPath === 'llm.provider') {
      const modelSelect = document.getElementById('model-select');
      if (modelSelect) {
        const allModels = this._getLlmModels(this._dirty['llm.provider'] || this._config?.llm?.provider);
        modelSelect.innerHTML = allModels.map(m =>
          `<option value="${m.value}"${m.value === firstVal ? ' selected' : ''}>${m.label}</option>`
        ).join('') + '<option value="_custom">Custom...</option>';
      }
      // Auto-set disable_thinking
      const known = Object.values(this._llmModels).flat().find(m => m.value === firstVal);
      if (known) {
        const thinkEl = document.getElementById('model-thinking');
        if (thinkEl) thinkEl.value = String(known.thinking);
        this._onChange('llm.disable_thinking', known.thinking);
      }
    }
  },

  _sectionFields(section, data, needsRestart) {
    const badge = needsRestart ? '<span class="badge-restart">restart required</span>' : '';
    const rows = Object.entries(data).map(([field, value]) => {
      const keyPath = `${section}.${field}`;
      const inputId = `cfg-${keyPath.replace(/\./g, '-')}`;
      const type = typeof value;
      let input;
      // Check if this field has known dropdown options
      const opts = this._getDropdownOptions(keyPath);
      if (opts) {
        const currentVal = String(value ?? '');
        const isModel = keyPath.endsWith('.model') || keyPath.endsWith('.llm_model');
        const hasCustom = currentVal && !opts.find(o => String(o.value) === currentVal);
        const isProvider = keyPath.endsWith('.provider');
        const changeHandler = isProvider
          ? `Settings._onDropdownChange('${keyPath}','${inputId}');Settings._cascadeProvider('${keyPath}')`
          : `Settings._onDropdownChange('${keyPath}','${inputId}')`;
        input = `<select id="${inputId}" onchange="${changeHandler}">` +
          opts.map(o => `<option value="${o.value}"${String(o.value) === currentVal ? ' selected' : ''}>${o.label}</option>`).join('') +
          (isModel ? `<option value="__custom"${hasCustom ? ' selected' : ''}>Custom...</option>` : '') +
          (hasCustom && !isModel ? `<option value="${currentVal}" selected>${currentVal}</option>` : '') +
          `</select>` +
          (isModel ? `<input id="${inputId}-custom" type="text" value="${hasCustom ? currentVal : ''}" ` +
            `placeholder="e.g. gemini/gemini-2.5-pro-preview" ` +
            `style="display:${hasCustom ? 'inline' : 'none'}" ` +
            `onchange="Settings._onChange('${keyPath}',this.value)">` : '');
      } else if (type === 'boolean') {
        input = `<select id="${inputId}" onchange="Settings._onChange('${keyPath}',this.value==='true')">
          <option value="true"${value ? ' selected' : ''}>true</option>
          <option value="false"${!value ? ' selected' : ''}>false</option>
        </select>`;
      } else if (type === 'number') {
        const step = Number.isInteger(value) ? '1' : '0.01';
        input = `<input id="${inputId}" type="number" step="${step}" value="${value}" onchange="Settings._onChange('${keyPath}',Number(this.value))">`;
      } else if (type === 'object' && value !== null) {
        return ''; // skip nested objects/arrays in flat editor
      } else {
        const displayVal = String(value || '').replace(/"/g, '&quot;');
        input = `<input id="${inputId}" type="text" value="${displayVal}" onchange="Settings._onChange('${keyPath}',this.value)">`;
      }
      return `<div class="config-field">
        <span class="config-field-label">${field}</span>
        <div class="config-field-input">${input}</div>
      </div>`;
    }).filter(Boolean).join('');
    if (!rows) return '';
    return `<div class="config-fields">
      ${badge ? `<div class="config-field-label">${section} ${badge}</div>` : ''}
      ${rows}
    </div>`;
  },

  _onDropdownChange(keyPath, inputId) {
    const select = document.getElementById(inputId);
    const customInput = document.getElementById(inputId + '-custom');
    if (select.value === '__custom') {
      if (customInput) { customInput.style.display = 'inline'; customInput.focus(); }
    } else {
      if (customInput) { customInput.style.display = 'none'; }
      this._onChange(keyPath, select.value);
    }
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
        const restartBtn = '<button class="btn-restart-inline" onclick="Settings.restartServer()">Restart Now</button>';
        const msg = res.restart_required
          ? `Saved ${res.changed.length} changes. <strong>Server restart required.</strong>${restartBtn}`
          : `Saved ${res.changed.length} changes.`;
        statusEl.style.display = 'block';
        statusEl.className = res.restart_required ? 'cfg-status cfg-status--warning' : 'cfg-status cfg-status--success';
        statusEl.innerHTML = msg;
      }
      this._dirty = {};
      const btn = document.getElementById('cfg-save-btn');
      if (btn) { btn.disabled = true; btn.textContent = 'Save Changes'; }
      App.toast('Config saved', 'success');
    } catch (e) {
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.className = 'cfg-status cfg-status--error';
        statusEl.textContent = e.message;
      }
      App.toast(e.message, 'error');
    }
  },

  async restartServer() {
    if (!confirm('Restart the server? The page will reload in a few seconds.')) return;
    const statusEl = document.getElementById('cfg-status');
    try {
      await API.restartServer();
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.className = 'cfg-status cfg-status--success';
        statusEl.innerHTML = 'Server restarting... page will reload automatically.';
      }
      // Poll until server is back, then reload
      Settings._waitForServer();
    } catch (e) {
      // Server may already be shutting down — still wait for it
      Settings._waitForServer();
    }
  },

  _waitForServer(attempt = 0) {
    if (attempt > 20) { // 20s max wait
      App.toast('Server did not come back. Please refresh manually.', 'error');
      return;
    }
    // First 3 attempts wait longer to give server time to restart
    const delay = attempt < 3 ? 2000 : 1000;
    setTimeout(async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) { window.location.reload(); return; }
      } catch {}
      Settings._waitForServer(attempt + 1);
    }, delay);
  },

  // --- Helpers ---

  _kvTable(obj) {
    const entries = Object.entries(obj);
    if (!entries.length) return '<span class="empty-state">—</span>';
    return '<table>' + entries.map(([k, v]) => {
      const display = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return `<tr><td class="config-field-label" style="padding:3px 10px 3px 0">${k}</td><td style="padding:3px 0">${display}</td></tr>`;
    }).join('') + '</table>';
  },

  _schedulerTable(tasks) {
    return `<div class="table-wrap"><table><thead><tr><th>Task</th><th>Interval</th><th>Runs</th><th>Next In</th><th>LLM</th><th>Error</th><th>Action</th></tr></thead><tbody>` +
      tasks.map(t => `<tr>
        <td>${t.name}</td>
        <td>${this._formatInterval(t.interval_seconds)}</td>
        <td>${t.run_count}</td>
        <td>${this._formatInterval(t.next_run_in)}</td>
        <td>${t.requires_llm ? '<span class="badge" style="--src-color:var(--warning);background:rgba(245,158,11,.12);color:var(--warning)">Yes</span>' : 'No'}</td>
        <td class="text-error" style="font-size:11px">${t.last_error || '—'}</td>
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
