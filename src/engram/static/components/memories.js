/**
 * Memories tab — table with filters, pagination, CRUD, bulk actions, feedback.
 */
const Memories = {
  _loaded: false,
  _page: 0,
  _limit: 20,
  _total: 0,
  _selected: new Set(),
  _memories: [],

  async load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-memories').innerHTML = this._shell();
    }
    await this.search();
  },

  _shell() {
    return `
      <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:end">
        <div class="form-group" style="flex:1;min-width:200px;margin:0"><label>Search</label><input id="mem-search" placeholder="Search content..." onkeydown="if(event.key==='Enter')Memories.search()"></div>
        <div class="form-group" style="margin:0"><label>Type</label>
          <select id="mem-type" onchange="Memories.search()"><option value="">All</option><option value="fact">Fact</option><option value="preference">Preference</option><option value="decision">Decision</option><option value="todo">Todo</option><option value="error">Error</option><option value="context">Context</option><option value="workflow">Workflow</option></select>
        </div>
        <div class="form-group" style="margin:0"><label>Priority</label>
          <select id="mem-priority" onchange="Memories.search()"><option value="">All</option><option value="7-10">High (7-10)</option><option value="4-6">Medium (4-6)</option><option value="1-3">Low (1-3)</option></select>
        </div>
        <button class="btn btn-primary" onclick="Memories.search()">Search</button>
        <button class="btn" onclick="Memories.exportAll()">Export JSON</button>
      </div>
      <div id="mem-bulk" style="display:none" class="bulk-bar">
        <span id="mem-sel-count">0</span> selected
        <button class="btn btn-sm btn-danger" onclick="Memories.bulkDelete()">Delete Selected</button>
        <button class="btn btn-sm" onclick="Memories.clearSelection()">Clear</button>
      </div>
      <div class="card"><div class="table-wrap" id="mem-table"></div></div>
      <div class="pagination" id="mem-pagination"></div>`;
  },

  async search() {
    this._page = 0;
    this._selected.clear();
    this._updateBulkBar();
    await this._fetch();
  },

  async _fetch() {
    const el = document.getElementById('mem-table');
    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';
    try {
      const params = { offset: this._page * this._limit, limit: this._limit };
      const q = document.getElementById('mem-search')?.value?.trim();
      if (q) params.search = q;
      const type = document.getElementById('mem-type')?.value;
      if (type) params.memory_type = type;
      const pri = document.getElementById('mem-priority')?.value;
      if (pri) { const [min, max] = pri.split('-'); params.priority_min = min; params.priority_max = max; }

      const res = await API.listMemories(params);
      this._memories = res.memories || [];
      this._total = res.total || 0;
      this._renderTable(el);
      this._renderPagination();
    } catch (e) {
      el.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    }
  },

  _renderTable(el) {
    if (!this._memories.length) { el.innerHTML = '<div style="padding:20px;color:var(--text-muted);text-align:center">No memories found</div>'; return; }
    el.innerHTML = `<table><thead><tr>
      <th><input type="checkbox" onchange="Memories.toggleAll(this.checked)"></th>
      <th>ID</th><th>Content</th><th>Type</th><th>Priority</th><th>Confidence</th><th>Tags</th><th>Entities</th><th>Access</th><th>Created</th><th>Actions</th>
    </tr></thead><tbody>${this._memories.map(m => this._row(m)).join('')}</tbody></table>`;
  },

  _row(m) {
    const checked = this._selected.has(m.id) ? 'checked' : '';
    const shortId = m.id.slice(0, 8);
    return `<tr>
      <td><input type="checkbox" ${checked} onchange="Memories.toggleSelect('${m.id}',this.checked)"></td>
      <td><span class="truncate" style="max-width:70px;cursor:pointer" title="${m.id}" onclick="navigator.clipboard.writeText('${m.id}');App.toast('ID copied','info')">${shortId}</span></td>
      <td><span class="truncate truncate-wide" title="${this._esc(m.content)}">${App.truncate(m.content, 150)}</span></td>
      <td>${App.typeBadge(m.memory_type)}</td>
      <td>${App.priorityBar(m.priority)}</td>
      <td>${App.confidenceBar(m.confidence)}</td>
      <td>${App.pills(m.tags)}</td>
      <td>${App.pills(m.entities)}</td>
      <td style="text-align:center">${m.access_count}</td>
      <td style="white-space:nowrap">${App.formatDate(m.timestamp)}</td>
      <td style="white-space:nowrap">
        <button class="btn-icon" title="View" onclick="Memories.showDetail('${m.id}')">${Icons.eye}</button>
        <button class="btn-icon" title="Edit" onclick="Memories.showEdit('${m.id}')">${Icons.edit}</button>
        <button class="btn-icon" title="Positive" onclick="Memories.feedback('${m.id}','positive')">${Icons.thumbsUp}</button>
        <button class="btn-icon" title="Negative" onclick="Memories.feedback('${m.id}','negative')">${Icons.thumbsDown}</button>
        <button class="btn-icon" title="Delete" onclick="Memories.confirmDelete('${m.id}')">${Icons.trash}</button>
      </td>
    </tr>`;
  },

  _esc(s) { return (s || '').replace(/"/g, '&quot;').replace(/</g, '&lt;'); },

  _renderPagination() {
    const pages = Math.ceil(this._total / this._limit);
    const el = document.getElementById('mem-pagination');
    if (pages <= 1) { el.innerHTML = `<span class="page-info">${this._total} memories</span>`; return; }
    el.innerHTML = `
      <button class="btn btn-sm" ${this._page === 0 ? 'disabled' : ''} onclick="Memories.goPage(${this._page - 1})">Prev</button>
      <span class="page-info">Page ${this._page + 1} of ${pages} (${this._total} total)</span>
      <button class="btn btn-sm" ${this._page >= pages - 1 ? 'disabled' : ''} onclick="Memories.goPage(${this._page + 1})">Next</button>`;
  },

  async goPage(p) { this._page = Math.max(0, p); await this._fetch(); },

  // Selection
  toggleSelect(id, checked) {
    if (checked) this._selected.add(id); else this._selected.delete(id);
    this._updateBulkBar();
  },
  toggleAll(checked) {
    this._memories.forEach(m => { if (checked) this._selected.add(m.id); else this._selected.delete(m.id); });
    this._updateBulkBar();
    this._renderTable(document.getElementById('mem-table'));
  },
  clearSelection() { this._selected.clear(); this._updateBulkBar(); this._renderTable(document.getElementById('mem-table')); },
  _updateBulkBar() {
    const bar = document.getElementById('mem-bulk');
    if (!bar) return;
    bar.style.display = this._selected.size > 0 ? 'flex' : 'none';
    const cnt = document.getElementById('mem-sel-count');
    if (cnt) cnt.textContent = this._selected.size;
  },

  // Detail modal
  async showDetail(id) {
    try {
      const res = await API.getMemory(id);
      const m = res.memory;
      const entList = m.entities || [];
      const entLimit = 10;
      const entHtml = entList.length <= entLimit
        ? App.pills(entList)
        : `<div class="entities-list collapsed" id="ent-list">${entList.map(e => `<span class="pill">${e}</span>`).join('')}</div>
           <span class="entities-toggle" onclick="document.getElementById('ent-list').classList.toggle('collapsed');this.textContent=this.textContent.includes('Show')?'Show less':'Show ${entList.length - entLimit} more'">Show ${entList.length - entLimit} more</span>`;
      App.showModal(`<h2>Memory Detail</h2><div class="memory-detail">
        <div class="field"><div class="field-label">ID</div><div class="field-value" style="font-family:var(--mono);font-size:12px">${m.id}</div></div>
        <div class="field"><div class="field-label">Content</div><div class="field-value markdown-body" style="line-height:1.6">${typeof DOMPurify !== 'undefined' && typeof marked !== 'undefined' ? DOMPurify.sanitize(marked.parse(m.content || '')) : this._esc(m.content)}</div></div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
          <div class="field"><div class="field-label">Type</div><div class="field-value">${App.typeBadge(m.memory_type)}</div></div>
          <div class="field"><div class="field-label">Priority</div><div class="field-value">${App.priorityBar(m.priority)}</div></div>
          <div class="field"><div class="field-label">Confidence</div><div class="field-value">${App.confidenceBar(m.confidence)}</div></div>
        </div>
        <div class="field"><div class="field-label">Tags</div><div class="field-value">${App.pills(m.tags)}</div></div>
        <div class="field"><div class="field-label">Entities</div><div class="field-value">${entHtml}</div></div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
          <div class="field"><div class="field-label">Access Count</div><div class="field-value">${m.access_count}</div></div>
          <div class="field"><div class="field-label">Decay Rate</div><div class="field-value">${m.decay_rate}</div></div>
          <div class="field"><div class="field-label">Neg. Count</div><div class="field-value">${m.negative_count}</div></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          <div class="field"><div class="field-label">Created</div><div class="field-value">${App.formatDate(m.timestamp)}</div></div>
          <div class="field"><div class="field-label">Expires</div><div class="field-value">${m.expires_at ? App.formatDate(m.expires_at) : '—'}</div></div>
        </div>
        ${m.topic_key ? `<div class="field"><div class="field-label">Topic Key</div><div class="field-value">${m.topic_key} (rev ${m.revision_count})</div></div>` : ''}
        ${m.metadata && Object.keys(m.metadata).length ? `<div class="field"><div class="field-label">Metadata</div><div class="json-view">${JSON.stringify(m.metadata, null, 2)}</div></div>` : ''}
      </div>
      <div class="modal-actions"><button class="btn" onclick="App.closeModal()">Close</button></div>`);
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Edit modal
  async showEdit(id) {
    const m = this._memories.find(x => x.id === id);
    if (!m) return;
    App.showModal(`<h2>Edit Memory</h2>
      <div class="form-group"><label>Type</label>
        <select id="edit-type">${['fact','preference','decision','todo','error','context','workflow'].map(t => `<option value="${t}" ${m.memory_type === t ? 'selected' : ''}>${t}</option>`).join('')}</select>
      </div>
      <div class="form-group"><label>Priority (1-10)</label><input id="edit-priority" type="number" min="1" max="10" value="${m.priority}"></div>
      <div class="form-group"><label>Tags (comma-separated)</label><input id="edit-tags" value="${(m.tags || []).join(', ')}"></div>
      <div class="form-group"><label>Entities (comma-separated)</label><input id="edit-entities" value="${(m.entities || []).join(', ')}"></div>
      <div class="modal-actions">
        <button class="btn" onclick="App.closeModal()">Cancel</button>
        <button class="btn btn-primary" onclick="Memories.doEdit('${id}')">Save</button>
      </div>`);
  },

  async doEdit(id) {
    try {
      await API.updateMemory(id, {
        memory_type: document.getElementById('edit-type').value,
        priority: parseInt(document.getElementById('edit-priority').value),
        tags: document.getElementById('edit-tags').value.split(',').map(t => t.trim()).filter(Boolean),
        entities: document.getElementById('edit-entities').value.split(',').map(e => e.trim()).filter(Boolean),
      });
      App.closeModal();
      App.toast('Memory updated', 'success');
      await this._fetch();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Feedback
  async feedback(id, type) {
    try {
      const res = await API.feedback(id, type);
      App.toast(`Feedback: ${type} (confidence: ${Math.round((res.confidence || 0) * 100)}%)`, 'success');
      if (type === 'negative') {
        const m = this._memories.find(x => x.id === id);
        if (m && (m.negative_count || 0) >= 2) App.toast('Warning: memory close to auto-delete threshold', 'error');
      }
      await this._fetch();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Delete
  confirmDelete(id) {
    App.showModal(`<h2>Delete Memory</h2><p>Are you sure you want to delete this memory?</p><p style="font-family:var(--mono);color:var(--text-secondary)">${id}</p>
      <div class="modal-actions"><button class="btn" onclick="App.closeModal()">Cancel</button><button class="btn btn-danger" onclick="Memories.doDelete('${id}')">Delete</button></div>`);
  },

  async doDelete(id) {
    try {
      await API.deleteMemory(id);
      App.closeModal();
      App.toast('Memory deleted', 'success');
      await this._fetch();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Bulk delete
  async bulkDelete() {
    if (!this._selected.size) return;
    App.showModal(`<h2>Bulk Delete</h2><p>Delete ${this._selected.size} memories?</p>
      <div class="modal-actions"><button class="btn" onclick="App.closeModal()">Cancel</button><button class="btn btn-danger" onclick="Memories.doBulkDelete()">Delete All</button></div>`);
  },

  async doBulkDelete() {
    try {
      await API.bulkDelete([...this._selected]);
      App.closeModal();
      App.toast(`${this._selected.size} memories deleted`, 'success');
      this._selected.clear();
      this._updateBulkBar();
      await this._fetch();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Export
  async exportAll() {
    try {
      const res = await API.exportMemories({ limit: 1000 });
      const blob = new Blob([JSON.stringify(res.memories, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'engram-export.json'; a.click();
      URL.revokeObjectURL(url);
      App.toast(`Exported ${res.count} memories`, 'success');
    } catch (e) { App.toast(e.message, 'error'); }
  },
};
