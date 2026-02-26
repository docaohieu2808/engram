/**
 * Engram WebUI — SVG Icons, auth, routing, toast, modal, state.
 */
const Icons = {
  logout: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>`,
  brain: `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3 2.5 2.5 0 0 0 .47 4.89 2.5 2.5 0 0 0 3 3.45A2.5 2.5 0 0 0 12 19.5"/><path d="M12 4.5a2.5 2.5 0 0 1 4.96-.46 2.5 2.5 0 0 1 1.98 3 2.5 2.5 0 0 1-.47 4.89 2.5 2.5 0 0 1-3 3.45A2.5 2.5 0 0 1 12 19.5"/><path d="M12 4.5v15"/></svg>`,
  plus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  search: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  check: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
  x: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  edit: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`,
  trash: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
  thumbsUp: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>`,
  thumbsDown: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>`,
  eye: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`,
  loader: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinner-icon"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>`,
  circle: (color) => `<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5" fill="${color}"/></svg>`,
  sun: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
  moon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`,
};

const App = {
  currentTab: 'dashboard',

  async init() {
    this._applyTheme();
    if (!this.isAuthenticated()) {
      this.showLoginPage();
      return;
    }
    this._bindTabs();
    this._bindSearch();
    this._bindKeyboard();
    this._injectLogoutIcon();
    this._injectThemeIcon();
    this.switchTab('dashboard');
  },

  isAuthenticated() {
    return localStorage.getItem('engram_token') || localStorage.getItem('engram_api_key');
  },

  showLoginPage() {
    document.getElementById('app').innerHTML = `
      <div class="login-wrapper">
        <div class="login-card">
          <div class="login-logo">${Icons.brain}</div>
          <h2>Engram</h2>
          <p class="login-subtitle">Enter your API key to access memories</p>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="login-key" placeholder="engram_xxxxx" autofocus>
          </div>
          <button class="btn btn-primary btn-block" onclick="App.doLogin()">Sign In</button>
          <p class="login-hint">Get your API key by running: engram auth create-key</p>
        </div>
      </div>
    `;
    document.getElementById('login-key').addEventListener('keypress', e => {
      if (e.key === 'Enter') App.doLogin();
    });
  },

  async doLogin() {
    const key = document.getElementById('login-key').value.trim();
    if (!key) { this.toast('API key required', 'error'); return; }
    try {
      const res = await fetch('/api/v1/status', {
        headers: { 'X-API-Key': key }
      });
      if (res.ok) {
        localStorage.setItem('engram_api_key', key);
        location.reload();
      } else {
        this.toast('Invalid API key', 'error');
      }
    } catch (e) {
      this.toast('Connection error', 'error');
    }
  },

  logout() {
    localStorage.removeItem('engram_token');
    localStorage.removeItem('engram_api_key');
    location.reload();
  },

  _applyTheme() {
    const theme = localStorage.getItem('engram_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
  },

  _injectLogoutIcon() {
    const btn = document.querySelector('.btn-logout');
    if (btn) btn.innerHTML = Icons.logout;
  },

  _injectThemeIcon() {
    const btn = document.querySelector('.btn-theme');
    if (btn) {
      const theme = document.documentElement.getAttribute('data-theme') || 'dark';
      btn.innerHTML = theme === 'dark' ? Icons.sun : Icons.moon;
      btn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
  },

  toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('engram_theme', next);
    this._injectThemeIcon();
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
    const loaders = { dashboard: Dashboard, memories: Memories, graph: Graph, think: Think, audit: Audit, settings: Settings };
    if (loaders[tab] && loaders[tab].load) loaders[tab].load();
  },

  toast(msg, type = 'info') {
    const c = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  },

  showModal(html) {
    const root = document.getElementById('modal-root');
    root.innerHTML = `<div class="modal-overlay" onclick="if(event.target===this)App.closeModal()"><div class="modal">${html}</div></div>`;
  },

  closeModal() {
    document.getElementById('modal-root').innerHTML = '';
  },

  loading(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading...</div>';
  },

  formatDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  },

  typeBadge(type) {
    return `<span class="badge type-${type}">${type}</span>`;
  },

  confidenceBar(val) {
    const pct = Math.round((val || 0) * 100);
    return `<div class="bar-wrap bar-confidence"><div class="bar-fill" style="width:${pct}%"></div></div> <span style="font-size:12px;color:var(--text-secondary)">${pct}%</span>`;
  },

  priorityBar(val) {
    const pct = Math.round(((val || 5) / 10) * 100);
    return `<div class="bar-wrap bar-priority"><div class="bar-fill" style="width:${pct}%"></div></div> <span style="font-size:12px;color:var(--text-secondary)">${val}/10</span>`;
  },

  pills(items) {
    if (!items || !items.length) return '<span style="color:var(--text-muted)">—</span>';
    return items.map(t => `<span class="pill">${t}</span>`).join('');
  },

  truncate(str, len = 100) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
  },
};
