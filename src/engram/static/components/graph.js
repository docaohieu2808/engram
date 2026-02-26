/**
 * Graph tab — vis-network visualization with node/edge CRUD and colored nodes.
 */
const NODE_TYPE_COLORS = {
  'Person':       { background: '#73bf69', border: '#5a9952', font: '#fff' },
  'Technology':   { background: '#5794f2', border: '#4477cc', font: '#fff' },
  'Project':      { background: '#ff9830', border: '#cc7a26', font: '#000' },
  'Service':      { background: '#b877d9', border: '#9360ae', font: '#fff' },
  'Organization': { background: '#f2495c', border: '#c23a4a', font: '#fff' },
  'Location':     { background: '#ffb357', border: '#cc8f46', font: '#000' },
  'default':      { background: '#6e6e6e', border: '#555555', font: '#fff' },
};

function getNodeColor(type) {
  return NODE_TYPE_COLORS[type] || NODE_TYPE_COLORS.default;
}

function renderLegend(nodeTypes) {
  const types = Object.keys(nodeTypes);
  if (!types.length) return '';
  return `
    <div class="graph-legend">
      <h4>Node Types</h4>
      ${types.map(t => {
        const c = getNodeColor(t);
        return `<div class="legend-item">
          <svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5" fill="${c.background}"/></svg>
          <span>${t} (${nodeTypes[t]})</span>
        </div>`;
      }).join('')}
    </div>
  `;
}

const Graph = {
  _loaded: false,
  _network: null,
  _nodes: [],
  _edges: [],
  _nodesDS: null,
  _edgesDS: null,

  async load() {
    if (!this._loaded) {
      this._loaded = true;
      document.getElementById('tab-graph').innerHTML = this._shell();
    }
    await this._fetchAndRender();
  },

  _shell() {
    return `
      <div class="graph-toolbar">
        <button class="btn btn-sm" onclick="Graph.zoomFit()">Fit</button>
        <button class="btn btn-sm" id="graph-physics-btn" onclick="Graph.togglePhysics()">Physics: ON</button>
        <input id="graph-search" placeholder="Search nodes..." style="padding:4px 10px;background:var(--bg-primary);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:12px;width:200px" oninput="Graph.filterNodes(this.value)">
        <div style="margin-left:auto;display:flex;gap:6px">
          <button class="btn btn-sm btn-primary" onclick="Graph.showAddNode()">+ Node</button>
          <button class="btn btn-sm btn-primary" onclick="Graph.showAddEdge()">+ Edge</button>
        </div>
      </div>
      <div style="display:flex;gap:12px">
        <div style="flex:1"><div id="graph-vis" class="card" style="padding:0"></div></div>
        <div style="width:260px;flex-shrink:0">
          <div class="card" style="margin-bottom:12px"><h3>Details</h3><div id="graph-detail"><span style="color:var(--text-muted)">Click a node or edge</span></div></div>
          <div class="card"><h3>Stats</h3><div id="graph-stats"></div></div>
        </div>
      </div>`;
  },

  async _fetchAndRender() {
    try {
      const data = await API.graphData();
      this._nodes = data.nodes || [];
      this._edges = data.edges || [];
      this._renderGraph();
      this._renderStats();
    } catch (e) {
      document.getElementById('graph-vis').innerHTML = `<div style="padding:20px;color:var(--danger)">${e.message}</div>`;
    }
  },

  _renderGraph() {
    const container = document.getElementById('graph-vis');
    const cs = getComputedStyle(document.documentElement);
    const edgeColor = cs.getPropertyValue('--border').trim();
    const bgColor = cs.getPropertyValue('--bg-canvas').trim();
    const accentColor = cs.getPropertyValue('--accent').trim();
    const mutedColor = cs.getPropertyValue('--text-muted').trim();
    const textColor = cs.getPropertyValue('--text-primary').trim();

    this._nodesDS = new vis.DataSet(this._nodes.map(n => {
      const colors = getNodeColor(n.group);
      return {
        ...n,
        title: `${n.group}: ${n.label}`,
        color: {
          background: colors.background,
          border: colors.border,
          highlight: { background: colors.background, border: '#fff' },
          hover: { background: colors.background, border: '#fff' },
        },
        font: { color: textColor, size: 13, strokeWidth: 3, strokeColor: bgColor },
        borderWidth: 2,
      };
    }));

    this._edgesDS = new vis.DataSet(this._edges.map(e => ({
      ...e, font: { color: mutedColor, size: 12, align: 'middle', strokeWidth: 0 },
      color: { color: edgeColor, hover: accentColor, highlight: accentColor },
      smooth: { type: 'dynamic', forceDirection: 'none', roundness: 0.5 },
    })));

    this._network = new vis.Network(container, { nodes: this._nodesDS, edges: this._edgesDS }, {
      physics: { enabled: true, forceAtlas2Based: { gravitationalConstant: -50, springLength: 120 }, solver: 'forceAtlas2Based', stabilization: { iterations: 150 } },
      interaction: { hover: true, tooltipDelay: 200 },
      nodes: { shape: 'dot', size: 18 },
      edges: { arrows: { to: { enabled: true, scaleFactor: 0.6 } }, width: 1.5 },
      background: { color: bgColor },
    });

    this._network.on('click', params => {
      if (params.nodes.length > 0) this._showNodeDetail(params.nodes[0]);
      else if (params.edges.length > 0) this._showEdgeDetail(params.edges[0]);
      else document.getElementById('graph-detail').innerHTML = '<span style="color:var(--text-muted)">Click a node or edge</span>';
    });
  },

  _showNodeDetail(nodeId) {
    const node = this._nodes.find(n => n.id === nodeId);
    if (!node) return;
    const connected = this._edges.filter(e => e.from === nodeId || e.to === nodeId);
    let html = `<div style="font-weight:600;font-size:14px;margin-bottom:6px;color:#d8d9da">${node.label}</div>`;
    html += `<span class="badge type-preference">${node.group}</span>`;
    if (node.attributes && Object.keys(node.attributes).length) {
      html += '<div style="margin-top:8px;font-size:11px">';
      for (const [k, v] of Object.entries(node.attributes)) html += `<div><span style="color:var(--text-secondary)">${k}:</span> ${v}</div>`;
      html += '</div>';
    }
    if (connected.length) {
      html += `<div style="margin-top:8px;font-size:11px;font-weight:600;color:var(--accent)">Connections (${connected.length})</div>`;
      connected.slice(0, 10).forEach(e => {
        const dir = e.from === nodeId ? '→' : '←';
        const otherId = e.from === nodeId ? e.to : e.from;
        const other = this._nodes.find(n => n.id === otherId);
        html += `<div style="font-size:11px;padding:2px 0;border-bottom:1px solid var(--border-light)">${dir} ${e.label} ${other ? other.label : otherId}</div>`;
      });
    }
    html += `<div style="margin-top:10px;display:flex;gap:6px">
      <button class="btn btn-sm btn-danger" onclick="Graph.deleteNode('${nodeId}')">Delete</button>
    </div>`;
    document.getElementById('graph-detail').innerHTML = html;
  },

  _showEdgeDetail(edgeId) {
    const edge = this._edgesDS.get(edgeId);
    if (!edge) return;
    const fromNode = this._nodes.find(n => n.id === edge.from);
    const toNode = this._nodes.find(n => n.id === edge.to);
    let html = `<div style="font-size:12px">
      <div><strong>${fromNode ? fromNode.label : edge.from}</strong></div>
      <div style="color:var(--accent);margin:4px 0">→ ${edge.label} →</div>
      <div><strong>${toNode ? toNode.label : edge.to}</strong></div>
      ${edge.weight !== undefined ? `<div style="margin-top:6px;color:var(--text-secondary)">Weight: ${edge.weight}</div>` : ''}
    </div>`;
    const edgeKey = `${edge.from}--${edge.label}-->${edge.to}`;
    html += `<div style="margin-top:10px"><button class="btn btn-sm btn-danger" onclick="Graph.deleteEdge('${edgeKey}')">Delete Edge</button></div>`;
    document.getElementById('graph-detail').innerHTML = html;
  },

  _renderStats() {
    const types = {};
    this._nodes.forEach(n => { types[n.group] = (types[n.group] || 0) + 1; });
    const rels = {};
    this._edges.forEach(e => { rels[e.label] = (rels[e.label] || 0) + 1; });
    let html = `<div style="font-size:12px"><strong>${this._nodes.length}</strong> nodes, <strong>${this._edges.length}</strong> edges</div>`;
    html += renderLegend(types);
    if (Object.keys(rels).length) {
      html += '<div style="margin-top:6px;font-size:11px;font-weight:600;color:var(--text-secondary)">Relations</div>';
      for (const [r, c] of Object.entries(rels)) html += `<div style="font-size:11px"><span class="pill">${r}: ${c}</span></div>`;
    }
    document.getElementById('graph-stats').innerHTML = html;
  },

  // Toolbar
  zoomFit() { if (this._network) this._network.fit({ animation: true }); },
  togglePhysics() {
    if (!this._network) return;
    const btn = document.getElementById('graph-physics-btn');
    const on = btn.textContent.includes('ON');
    this._network.setOptions({ physics: { enabled: !on } });
    btn.textContent = on ? 'Physics: OFF' : 'Physics: ON';
  },
  filterNodes(q) {
    if (!this._nodesDS) return;
    q = q.trim().toLowerCase();
    if (!q) { this._nodesDS.update(this._nodes.map(n => ({ id: n.id, opacity: 1 }))); return; }
    const matched = new Set(this._nodes.filter(n => n.label.toLowerCase().includes(q)).map(n => n.id));
    this._nodesDS.update(this._nodes.map(n => ({ id: n.id, opacity: matched.has(n.id) ? 1 : 0.15, borderWidth: matched.has(n.id) ? 3 : 1 })));
    if (matched.size > 0) this._network.fit({ nodes: [...matched], animation: true });
  },

  // Add node modal
  showAddNode() {
    const types = [...new Set(this._nodes.map(n => n.group))].sort();
    App.showModal(`<h2>Add Node</h2>
      <div class="form-group"><label>Type</label>
        <input id="gn-type" list="gn-types" placeholder="e.g. Person, Technology"><datalist id="gn-types">${types.map(t => `<option value="${t}">`).join('')}</datalist>
      </div>
      <div class="form-group"><label>Name</label><input id="gn-name" placeholder="Node name"></div>
      <div class="form-group"><label>Attributes (JSON)</label><textarea id="gn-attrs" rows="3" placeholder='{"key": "value"}'>{}</textarea></div>
      <div class="modal-actions"><button class="btn" onclick="App.closeModal()">Cancel</button><button class="btn btn-primary" onclick="Graph.doAddNode()">Create</button></div>`);
  },
  async doAddNode() {
    try {
      const attrs = JSON.parse(document.getElementById('gn-attrs').value || '{}');
      await API.createNode({ type: document.getElementById('gn-type').value, name: document.getElementById('gn-name').value, attributes: attrs });
      App.closeModal(); App.toast('Node created', 'success');
      this._loaded = false; this.load();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Add edge modal
  showAddEdge() {
    const nodeOpts = this._nodes.map(n => `<option value="${n.id}">${n.label} (${n.group})</option>`).join('');
    App.showModal(`<h2>Add Edge</h2>
      <div class="form-group"><label>From Node</label><select id="ge-from">${nodeOpts}</select></div>
      <div class="form-group"><label>To Node</label><select id="ge-to">${nodeOpts}</select></div>
      <div class="form-group"><label>Relation</label><input id="ge-rel" placeholder="e.g. uses, depends_on"></div>
      <div class="form-group"><label>Weight</label><input id="ge-weight" type="number" step="0.1" value="1.0"></div>
      <div class="modal-actions"><button class="btn" onclick="App.closeModal()">Cancel</button><button class="btn btn-primary" onclick="Graph.doAddEdge()">Create</button></div>`);
  },
  async doAddEdge() {
    try {
      await API.createEdge({
        from_node: document.getElementById('ge-from').value,
        to_node: document.getElementById('ge-to').value,
        relation: document.getElementById('ge-rel').value,
        weight: parseFloat(document.getElementById('ge-weight').value) || 1.0,
      });
      App.closeModal(); App.toast('Edge created', 'success');
      this._loaded = false; this.load();
    } catch (e) { App.toast(e.message, 'error'); }
  },

  // Delete
  async deleteNode(key) {
    if (!confirm(`Delete node ${key}?`)) return;
    try { await API.deleteNode(key); App.toast('Node deleted', 'success'); this._loaded = false; this.load(); }
    catch (e) { App.toast(e.message, 'error'); }
  },
  async deleteEdge(key) {
    if (!confirm(`Delete edge?`)) return;
    try { await API.deleteEdge(key); App.toast('Edge deleted', 'success'); this._loaded = false; this.load(); }
    catch (e) { App.toast(e.message, 'error'); }
  },
};
