/**
 * Engram WebUI — Main app shell: routing, toast, modal, state.
 */
const App = {
  currentTab: 'dashboard',

  init() {
    this._bindTabs();
    this._bindSearch();
    this._bindKeyboard();
    this.switchTab('dashboard');
  },

  _bindTabs() {
    document.querySelectorAll('#tabs button').forEach(btn => {
      btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
    });
  },

  _bindSearch() {
    const input = document.getElementById('search-global');
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        this.switchTab('memories');
        setTimeout(() => {
          const memSearch = document.getElementById('mem-search');
          if (memSearch) { memSearch.value = input.value; Memories.search(); }
        }, 100);
      }
    });
  },

  _bindKeyboard() {
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.closeModal();
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('search-global').focus();
      }
    });
  },

  switchTab(tab) {
    this.currentTab = tab;
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById('tab-' + tab);
    const btn = document.querySelector(`#tabs button[data-tab="${tab}"]`);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
    // Lazy-load tab content
    const loaders = { dashboard: Dashboard, memories: Memories, graph: Graph, think: Think, audit: Audit, settings: Settings };
    if (loaders[tab] && loaders[tab].load) loaders[tab].load();
  },

  // Toast notifications
  toast(msg, type = 'info') {
    const c = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  },

  // Modal system
  showModal(html) {
    const root = document.getElementById('modal-root');
    root.innerHTML = `<div class="modal-overlay" onclick="if(event.target===this)App.closeModal()"><div class="modal">${html}</div></div>`;
  },

  closeModal() {
    document.getElementById('modal-root').innerHTML = '';
  },

  // Utility: loading indicator
  loading(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading...</div>';
  },

  // Utility: format date
  formatDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  },

  // Utility: type badge HTML
  typeBadge(type) {
    return `<span class="badge type-${type}">${type}</span>`;
  },

  // Utility: confidence bar
  confidenceBar(val) {
    const pct = Math.round((val || 0) * 100);
    return `<div class="bar-wrap bar-confidence"><div class="bar-fill" style="width:${pct}%"></div></div> <span style="font-size:11px">${pct}%</span>`;
  },

  // Utility: priority bar
  priorityBar(val) {
    const pct = Math.round(((val || 5) / 10) * 100);
    return `<div class="bar-wrap bar-priority"><div class="bar-fill" style="width:${pct}%"></div></div> <span style="font-size:11px">${val}/10</span>`;
  },

  // Utility: pills
  pills(items) {
    if (!items || !items.length) return '<span style="color:var(--text-muted)">—</span>';
    return items.map(t => `<span class="pill">${t}</span>`).join('');
  },

  // Utility: truncate
  truncate(str, len = 100) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
  },
};
