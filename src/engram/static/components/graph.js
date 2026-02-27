/**
 * Graph tab — vis-network visualization with node/edge CRUD and colored nodes.
 */
const NODE_TYPE_COLORS = {
  'Person':       { background: '#73bf69', border: '#5a9952', font: '#fff' },
  'Technology':   { background: '#5794f2', border: '#4477cc', font: '#fff' },
  'Project':      { background: '#ff9830', border: '#cc7a26', font: '#000' },
  'Service':      { background: '#b877d9', border: '#9360ae', font: '#fff' },
  'Server':       { background: '#f2495c', border: '#c23a4a', font: '#fff' },
  'Environment':  { background: '#36a2eb', border: '#2b82bc', font: '#fff' },
  'Script':       { background: '#ffcd56', border: '#ccaa45', font: '#000' },
  'Organization': { background: '#ff6384', border: '#cc4f6a', font: '#fff' },
  'Location':     { background: '#ffb357', border: '#cc8f46', font: '#000' },
  'default':      { background: '#6e6e6e', border: '#555555', font: '#fff' },
};

function getNodeColor(type) {
  if (NODE_TYPE_COLORS[type]) return NODE_TYPE_COLORS[type];
  // Auto-generate a stable color for unknown types based on hash
  let h = 0;
  for (let i = 0; i < (type || '').length; i++) h = ((h << 5) - h + type.charCodeAt(i)) | 0;
  const hue = ((h >>> 0) % 360);
  return { background: `hsl(${hue}, 55%, 50%)`, border: `hsl(${hue}, 55%, 40%)`, font: '#fff' };
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

function stableSeed(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) || 1;
}

function createRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function toRgbaHexOrRgb(color, alpha) {
  const a = Math.max(0, Math.min(1, alpha));
  const c = (color || '').trim();

  if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(c)) {
    const raw = c.slice(1);
    const hex = raw.length === 3 ? raw.split('').map(ch => ch + ch).join('') : raw;
    const int = Number.parseInt(hex, 16);
    const r = (int >> 16) & 255;
    const g = (int >> 8) & 255;
    const b = int & 255;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }

  const rgb = c.match(/^rgb\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)$/i);
  if (rgb) return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${a})`;

  return c;
}

const Graph = {
  _loaded: false,
  _network: null,
  _nodes: [],
  _edges: [],
  _nodesDS: null,
  _edgesDS: null,
  _threadRenderer: null,

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
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#e0e0e0' : '#111111';

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
        font: { color: textColor, size: 13 },
        borderWidth: 2,
      };
    }));

    this._edgesDS = new vis.DataSet(this._edges.map(e => ({
      ...e,
      // Giữ edge để tương tác + label, nhưng ẩn hoàn toàn nét mặc định của vis
      font: { color: mutedColor, size: 12, align: 'middle', strokeWidth: 0 },
      color: {
        color: 'rgba(0,0,0,0)',
        hover: 'rgba(0,0,0,0)',
        highlight: 'rgba(0,0,0,0)',
        inherit: false,
      },
      smooth: { type: 'dynamic', roundness: 0.15 },
      width: 1,
      hoverWidth: 0,
      selectionWidth: 0,
      chosen: false,
    })));

    // Build vis-network groups config from our color map so group coloring matches legend
    const allGroups = [...new Set(this._nodes.map(n => n.group))];
    const visGroups = {};
    for (const g of allGroups) {
      const c = getNodeColor(g);
      visGroups[g] = {
        color: { background: c.background, border: c.border, highlight: { background: c.background, border: '#fff' }, hover: { background: c.background, border: '#fff' } },
        font: { color: textColor },
      };
    }

    this._network = new vis.Network(container, { nodes: this._nodesDS, edges: this._edgesDS }, {
      groups: visGroups,
      physics: { enabled: true, forceAtlas2Based: { gravitationalConstant: -50, springLength: 120 }, solver: 'forceAtlas2Based', stabilization: { iterations: 150 } },
      interaction: { hover: true, tooltipDelay: 200 },
      nodes: { shape: 'dot', size: 18 },
      edges: {
        arrows: { to: { enabled: false } },
        width: 1,
        color: {
          color: 'rgba(0,0,0,0)',
          hover: 'rgba(0,0,0,0)',
          highlight: 'rgba(0,0,0,0)',
          inherit: false,
        },
        hoverWidth: 0,
        selectionWidth: 0,
      },
      background: { color: bgColor },
    });

    // Custom thread-like renderer (vẽ trước node để ra cảm giác dây chỉ mềm)
    if (this._threadRenderer) this._network.off('beforeDrawing', this._threadRenderer);
    this._threadRenderer = (ctx) => this._drawThreadEdges(ctx, accentColor);
    this._network.on('beforeDrawing', this._threadRenderer);

    this._network.on('click', params => {
      if (params.nodes.length > 0) this._showNodeDetail(params.nodes[0]);
      else if (params.edges.length > 0) this._showEdgeDetail(params.edges[0]);
      else document.getElementById('graph-detail').innerHTML = '<span style="color:var(--text-muted)">Click a node or edge</span>';
    });
  },

  _drawThreadEdges(ctx, accentColor) {
    if (!this._network || !this._edges || !this._edges.length) return;

    const fanCounter = new Map();
    const baseColor = toRgbaHexOrRgb(accentColor, 0.62);
    const glowColor = toRgbaHexOrRgb(accentColor, 0.32);

    for (const e of this._edges) {
      const p1 = this._network.getPosition(e.from);
      const p2 = this._network.getPosition(e.to);
      if (!p1 || !p2) continue;

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 2) continue;

      const nx = -dy / dist;
      const ny = dx / dist;

      const fan = fanCounter.get(e.from) || 0;
      fanCounter.set(e.from, fan + 1);

      const key = `${e.from}|${e.label}|${e.to}`;
      const rng = createRng(stableSeed(key));
      const alt = fan % 2 === 0 ? 1 : -1;
      const baseOffset = alt * (10 + fan * 1.8 + rng() * 12);
      const tangent = (rng() - 0.5) * 30;

      // Tạo "anchor cloud" ở đầu/cuối để không bị quạt cứng từ 1 tâm
      const startJitter = 1.2 + Math.min(6.5, fan * 0.35);
      const endJitter = 0.8 + rng() * 2.2;
      const sx = p1.x + nx * ((rng() - 0.5) * startJitter * 2) + (dx / dist) * ((rng() - 0.5) * 2.8);
      const sy = p1.y + ny * ((rng() - 0.5) * startJitter * 2) + (dy / dist) * ((rng() - 0.5) * 2.8);
      const ex = p2.x + nx * ((rng() - 0.5) * endJitter * 2);
      const ey = p2.y + ny * ((rng() - 0.5) * endJitter * 2);

      // Cubic bezier với 2 control points lệch mạnh hơn để ra sợi chỉ lỏng
      const t1 = 0.18 + rng() * 0.16;
      const t2 = 0.58 + rng() * 0.22;
      const c1x = sx + (ex - sx) * t1 + nx * (baseOffset * (0.95 + rng() * 0.65));
      const c1y = sy + (ey - sy) * t1 + ny * (baseOffset * (0.95 + rng() * 0.65));
      const c2x = sx + (ex - sx) * t2 + nx * (baseOffset * (0.15 + rng() * 0.85)) + ((ex - sx) / dist) * tangent;
      const c2y = sy + (ey - sy) * t2 + ny * (baseOffset * (0.15 + rng() * 0.85)) + ((ey - sy) / dist) * tangent;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, ex, ey);
      ctx.lineCap = 'round';
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1.05 + Math.min(1.3, (e.weight || 1) * 0.26);
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 3.2;
      ctx.stroke();
      ctx.restore();
    }
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
