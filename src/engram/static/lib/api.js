/**
 * Engram API client — wraps all /api/v1/ endpoints with auth + error handling.
 */
const API = {
  base: '/api/v1',

  _headers() {
    const h = { 'Content-Type': 'application/json' };
    const token = localStorage.getItem('engram_token');
    if (token) h['Authorization'] = `Bearer ${token}`;
    const apiKey = localStorage.getItem('engram_api_key');
    if (apiKey) h['X-API-Key'] = apiKey;
    return h;
  },

  async _fetch(path, opts = {}) {
    opts.headers = { ...this._headers(), ...opts.headers };
    const res = await fetch(this.base + path, opts);
    if (!res.ok) {
      if (res.status === 401) {
        // Session expired or invalid credentials — show login
        localStorage.removeItem('engram_token');
        localStorage.removeItem('engram_api_key');
        if (typeof App !== 'undefined') App.showLoginPage();
        throw new Error('Session expired — please log in again');
      }
      const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
      throw new Error(err.error?.message || err.detail || res.statusText);
    }
    return res.json();
  },

  // Status & health
  status() { return this._fetch('/status'); },
  health() { return fetch('/health').then(r => r.json()); },

  // Memories
  listMemories(params = {}) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') q.set(k, v);
    }
    return this._fetch('/memories?' + q.toString());
  },
  getMemory(id) { return this._fetch(`/memories/${id}`); },
  updateMemory(id, body) { return this._fetch(`/memories/${id}`, { method: 'PUT', body: JSON.stringify(body) }); },
  deleteMemory(id) { return this._fetch(`/memories/${id}`, { method: 'DELETE' }); },
  bulkDelete(ids) { return this._fetch('/memories/bulk-delete', { method: 'POST', body: JSON.stringify({ ids }) }); },
  exportMemories(params = {}) {
    const q = new URLSearchParams(params);
    return this._fetch('/memories/export?' + q.toString());
  },

  // Remember
  remember(body) { return this._fetch('/remember', { method: 'POST', body: JSON.stringify(body) }); },

  // Recall
  recall(query, opts = {}) {
    const q = new URLSearchParams({ query, ...opts });
    return this._fetch('/recall?' + q.toString());
  },

  // Think
  think(question) { return this._fetch('/think', { method: 'POST', body: JSON.stringify({ question }) }); },

  // Feedback
  feedback(memoryId, type) {
    return this._fetch('/feedback', { method: 'POST', body: JSON.stringify({ memory_id: memoryId, feedback: type }) });
  },
  feedbackHistory(last = 50) { return this._fetch(`/feedback/history?last=${last}`); },

  // Graph
  graphData() { return this._fetch('/graph/data'); },
  createNode(body) { return this._fetch('/graph/nodes', { method: 'POST', body: JSON.stringify(body) }); },
  updateNode(key, body) { return this._fetch(`/graph/nodes/${key}`, { method: 'PUT', body: JSON.stringify(body) }); },
  deleteNode(key) { return this._fetch(`/graph/nodes/${key}`, { method: 'DELETE' }); },
  createEdge(body) { return this._fetch('/graph/edges', { method: 'POST', body: JSON.stringify(body) }); },
  deleteEdge(key) { return this._fetch('/graph/edges', { method: 'DELETE', body: JSON.stringify({ key }) }); },

  // Audit
  auditLog(last = 50) { return this._fetch(`/audit/log?last=${last}`); },

  // Scheduler
  schedulerTasks() { return this._fetch('/scheduler/tasks'); },
  forceRunTask(name) { return this._fetch(`/scheduler/tasks/${name}/run`, { method: 'POST' }); },

  // Benchmark
  runBenchmark(questions) { return this._fetch('/benchmark/run', { method: 'POST', body: JSON.stringify({ questions }) }); },

  // Config
  getConfig() { return this._fetch('/config'); },
  updateConfig(body) { return this._fetch('/config', { method: 'PUT', body: JSON.stringify(body) }); },
  restartServer() { return this._fetch('/restart', { method: 'POST' }); },
  testModel() { return this._fetch('/test-model', { method: 'POST' }); },
  listModels(provider) { return this._fetch(`/models?provider=${provider || 'all'}`); },

  // Cleanup & summarize
  cleanup() { return this._fetch('/cleanup', { method: 'POST' }); },
  summarize(count = 20) { return this._fetch('/summarize', { method: 'POST', body: JSON.stringify({ count }) }); },
};
