// Interactive weather for the Alta → Tromsø map (see webmap.py).
//
// Per pixel: wave height from the offshore wave height, the wind and the
// pixel's fetch (waves.sheltered_hs), discretised with the network's
// breakpoints, then the expected usv_risk looked up in geobn's precomputed
// table.  The risk-aware route is Dijkstra on the same 8-connected graph as
// routing.plan_route, with cost 1 + K_RISK · risk.

window.addEventListener("load", () => { altaMain().catch((err) => {
  document.getElementById("alta-stats").textContent = "Could not start: " + err;
  throw err;
}); });

async function altaMain() {
  const D = JSON.parse(document.getElementById("alta-data").textContent);
  const map = window[D.mapName];
  const G = 9.81;
  const [H, W] = D.shape;
  const N = H * W;

  // ── Decoding ────────────────────────────────────────────────────────────
  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const raw = (b64, Type) => new Type(bytes(b64).buffer);
  async function packed(b64, Type) {
    const stream = new Blob([bytes(b64)]).stream()
      .pipeThrough(new DecompressionStream("deflate"));
    return new Type(await new Response(stream).arrayBuffer());
  }

  const tableRisk = raw(D.tableRisk, Float32Array);
  const tableHigh = raw(D.tableHigh, Float32Array);
  const depthState = await packed(D.depthState, Uint8Array);
  const trafficState = await packed(D.trafficState, Uint8Array);
  const currentState = await packed(D.currentState, Uint8Array);
  const fetch = [];
  for (const f of D.fetch) fetch.push(await packed(f, Uint8Array));
  const shortest = raw(D.shortest, Int32Array);

  // Table strides, in the order of D.nodes (last node fastest).
  const size = Object.fromEntries(D.nodes.map((n, i) => [n, D.sizes[i]]));
  const stride = {};
  let s = 1;
  for (let i = D.nodes.length - 1; i >= 0; i--) { stride[D.nodes[i]] = s; s *= D.sizes[i]; }

  // np.digitize with the interior breakpoints: the number of edges <= value.
  function state(node, value) {
    const bp = D.breakpoints[node];
    let k = 0;
    for (let i = 1; i < bp.length - 1; i++) if (value >= bp[i]) k++;
    return k;
  }
  const waveEdges = D.breakpoints.wave_height.slice(1, -1);

  // ── Lattice interpolation (Web Mercator ↔ grid) ─────────────────────────
  function lattice(rows, cols, values) {
    const nc = cols.length;
    const seg = (pos, v) => {
      let k = Math.min(Math.floor(v / (pos[1] - pos[0])), pos.length - 2);
      k = Math.max(k, 0);
      return [k, (v - pos[k]) / (pos[k + 1] - pos[k])];
    };
    return (r, c) => {
      const [i, ti] = seg(rows, r);
      const [j, tj] = seg(cols, c);
      const a = values[i * nc + j], b = values[i * nc + j + 1];
      const d = values[(i + 1) * nc + j], e = values[(i + 1) * nc + j + 1];
      return (a * (1 - tj) + b * tj) * (1 - ti) + (d * (1 - tj) + e * tj) * ti;
    };
  }
  const ll = D.lonlat;
  const latAt = lattice(ll.rows, ll.cols, raw(ll.lat, Float64Array));
  const lonAt = lattice(ll.rows, ll.cols, raw(ll.lon, Float64Array));
  const toLatLng = (idx) => { const r = Math.floor(idx / W), c = idx % W; return [latAt(r, c), lonAt(r, c)]; };

  // Web Mercator pixel → grid pixel, once.
  const ov = D.overlay;
  const [OH, OW] = ov.shape;
  const srcRow = lattice(ov.rows, ov.cols, raw(ov.src_row, Float32Array));
  const srcCol = lattice(ov.rows, ov.cols, raw(ov.src_col, Float32Array));
  const pixelOf = new Int32Array(OH * OW);
  for (let r = 0; r < OH; r++) {
    for (let c = 0; c < OW; c++) {
      const sr = Math.floor(srcRow(r, c)), sc = Math.floor(srcCol(r, c));
      pixelOf[r * OW + c] = (sr >= 0 && sr < H && sc >= 0 && sc < W) ? sr * W + sc : -1;
    }
  }
  const canvas = document.createElement("canvas");
  canvas.width = OW; canvas.height = OH;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(OW, OH);
  const overlay = L.imageOverlay(canvas.toDataURL(), ov.bounds, { interactive: false }).addTo(map);
  // Show the whole area, to the right of the control panel.
  const panelWidth = document.getElementById("alta-panel").offsetWidth;
  map.options.zoomSnap = 0.25;
  map.fitBounds(ov.bounds, { paddingTopLeft: [panelWidth + 70, 10], paddingBottomRight: [10, 10] });

  // ── The model ───────────────────────────────────────────────────────────
  const risk = new Float32Array(N);
  const high = new Float32Array(N);
  const wave = new Float32Array(N);

  function computeRisk(p) {
    const f = fetch[Math.round(p.dir / (360 / fetch.length)) % fetch.length];
    const u = Math.max(p.u, 0.5);                      // waves.fetch_limited_hs
    const jA = 0.0016 * u * u / G, jCap = 0.243 * u * u / G, gu = G / (u * u);
    const base = stride.air_temperature * state("air_temperature", p.t)
               + stride.wind_speed * state("wind_speed", p.u);
    for (let i = 0; i < N; i++) {
      const d = depthState[i];
      if (d === 255) { risk[i] = NaN; high[i] = NaN; wave[i] = NaN; continue; }
      const q = f[i] / 255, F = q * q * D.fetchMax;          // webmap.dequantise_fetch
      const hs = Math.max(p.hs * Math.sqrt(Math.min(F / 50000, 1)),
                          Math.min(jA * Math.sqrt(gu * F), jCap));
      let sw = 0;
      for (let e = 0; e < waveEdges.length; e++) if (hs >= waveEdges[e]) sw++;
      const k = base + stride.water_depth * d + stride.wave_height * sw
              + stride.current_speed * currentState[i] + stride.vessel_traffic * trafficState[i];
      risk[i] = tableRisk[k]; high[i] = tableHigh[k]; wave[i] = hs;
    }
  }

  // ── Dijkstra (routing.plan_route) ───────────────────────────────────────
  const STEPS = [[0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1],
                 [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
  const dist = new Float64Array(N);
  const prev = new Int32Array(N);
  let heapKey = new Float64Array(1 << 16), heapVal = new Int32Array(1 << 16), heapLen = 0;

  function push(k, v) {
    if (heapLen === heapKey.length) {
      const k2 = new Float64Array(heapLen * 2); k2.set(heapKey); heapKey = k2;
      const v2 = new Int32Array(heapLen * 2); v2.set(heapVal); heapVal = v2;
    }
    let i = heapLen++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heapKey[p] <= k) break;
      heapKey[i] = heapKey[p]; heapVal[i] = heapVal[p]; i = p;
    }
    heapKey[i] = k; heapVal[i] = v;
  }
  function pop() {
    const v = heapVal[0], k = heapKey[0];
    const lk = heapKey[--heapLen], lv = heapVal[heapLen];
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= heapLen) break;
      if (c + 1 < heapLen && heapKey[c + 1] < heapKey[c]) c++;
      if (heapKey[c] >= lk) break;
      heapKey[i] = heapKey[c]; heapVal[i] = heapVal[c]; i = c;
    }
    heapKey[i] = lk; heapVal[i] = lv;
    return [k, v];
  }

  function planRoute(cost) {
    dist.fill(Infinity); prev.fill(-1); heapLen = 0;
    dist[D.start] = 0; push(0, D.start);
    while (heapLen > 0) {
      const [d, u] = pop();
      if (d > dist[u]) continue;
      if (u === D.goal) break;
      const r = Math.floor(u / W), c = u % W;
      for (const [dr, dc, step] of STEPS) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
        const v = rr * W + cc;
        if (!Number.isFinite(cost[v])) continue;
        if (dr && dc && !Number.isFinite(cost[r * W + cc]) && !Number.isFinite(cost[rr * W + c])) continue;
        const nd = d + step * D.pixelSize * 0.5 * (cost[u] + cost[v]);
        if (nd < dist[v]) { dist[v] = nd; prev[v] = u; push(nd, v); }
      }
    }
    const path = [];
    for (let v = D.goal; v !== -1; v = prev[v]) path.push(v);
    return { path: path.reverse(), cost: dist[D.goal] };
  }

  // Length, and length-weighted mean risk as in run_example.route_stats.
  function stats(path) {
    let length = 0, wsum = 0, rsum = 0, maxHigh = 0;
    const seg = [];
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1], b = path[i];
      const l = Math.hypot(Math.floor(a / W) - Math.floor(b / W), (a % W) - (b % W));
      seg.push(l); length += l;
    }
    for (let i = 0; i < path.length; i++) {
      const w = ((i > 0 ? seg[i - 1] : 0) + (i < seg.length ? seg[i] : 0)) / 2;
      wsum += w; rsum += w * risk[path[i]];
      maxHigh = Math.max(maxHigh, high[path[i]]);
    }
    return { km: length * D.pixelSize / 1000, meanRisk: rsum / wsum, maxHigh };
  }
  const riskWord = (r) => (r < 0.2 ? "low" : r <= 0.5 ? "medium" : "high");

  // ── Drawing ─────────────────────────────────────────────────────────────
  const halo = L.polyline([], { color: "white", weight: 8, opacity: 0.9, interactive: false }).addTo(map);
  const safeLine = L.polyline([], { color: "#0b0b0b", weight: 4 }).addTo(map);
  const shortLine = L.polyline(Array.from(shortest, toLatLng),
    { color: "#52514e", weight: 2.5, dashArray: "8 6" }).addTo(map);
  safeLine.bringToFront();
  for (const [idx, name] of [[D.start, "Alta"], [D.goal, "Tromsø"]]) {
    L.circleMarker(toLatLng(idx), { radius: 7, color: "#0b0b0b", weight: 2.5, fillColor: "white",
      fillOpacity: 1 }).bindTooltip(name).addTo(map);
  }

  let colourBy = "risk";
  function drawOverlay() {
    const lut = colourBy === "risk" ? D.riskLut : D.waveLut;
    const field = colourBy === "risk" ? risk : wave;
    const scale = colourBy === "risk" ? 255 : 255 / D.waveMax;
    const px = image.data;
    for (let j = 0; j < pixelOf.length; j++) {
      const i = pixelOf[j];
      const v = i < 0 ? NaN : field[i];
      if (!(v === v)) { px[4 * j + 3] = 0; continue; }
      const k = 4 * Math.min(255, Math.max(0, Math.round(v * scale)));
      px[4 * j] = lut[k]; px[4 * j + 1] = lut[k + 1]; px[4 * j + 2] = lut[k + 2]; px[4 * j + 3] = lut[k + 3];
    }
    ctx.putImageData(image, 0, 0);
    overlay.setUrl(canvas.toDataURL());
    const ramp = [0, 0.25, 0.5, 0.75, 1].map((t) => {
      const k = 4 * Math.round(t * 255);
      return `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
    });
    document.getElementById("legend-ramp").style.background = `linear-gradient(to right,${ramp})`;
    document.getElementById("legend-ticks").innerHTML = colourBy === "risk"
      ? "<span>low</span><span>passage risk</span><span>high</span>"
      : `<span>0 m</span><span>wave height</span><span>${D.waveMax} m</span>`;
  }

  function planAndShow() {
    const cost = new Float64Array(N);
    for (let i = 0; i < N; i++) cost[i] = 1 + D.kRisk * risk[i];   // NaN on land
    const route = planRoute(cost);
    const latlngs = route.path.map(toLatLng);
    halo.setLatLngs(latlngs); safeLine.setLatLngs(latlngs);
    const a = stats(route.path), b = stats(Array.from(shortest));
    const detour = a.km - b.km;
    safeLine.bindTooltip(`Risk-aware route: ${a.km.toFixed(0)} km, ${riskWord(a.meanRisk)} risk`, { sticky: true });
    shortLine.bindTooltip(`Shortest route: ${b.km.toFixed(0)} km, ${riskWord(b.meanRisk)} risk`, { sticky: true });
    document.getElementById("alta-stats").innerHTML =
      `<b>Risk-aware route:</b> ${a.km.toFixed(0)} km, ${riskWord(a.meanRisk)} risk<br>` +
      `<b>Shortest route:</b> ${b.km.toFixed(0)} km, ${riskWord(b.meanRisk)} risk<br>` +
      (detour >= 1 ? `Detour for safety: +${detour.toFixed(0)} km` : "Same way as the shortest route");
    return { route, a, b };
  }

  // ── Controls ────────────────────────────────────────────────────────────
  const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const inputs = { hs: "in-hs", u: "in-u", dir: "in-dir", t: "in-t" };
  const el = (id) => document.getElementById(id);
  const params = () => ({ hs: +el("in-hs").value, u: +el("in-u").value,
                          dir: +el("in-dir").value, t: +el("in-t").value });
  function showValues(p) {
    el("out-hs").textContent = `${p.hs.toFixed(1)} m`;
    el("out-u").textContent = `${p.u.toFixed(1)} m/s`;
    el("out-dir").textContent = `${COMPASS[Math.round(p.dir / 22.5) % 16]} (${p.dir}°)`;
    el("out-t").textContent = `${p.t.toFixed(1)} °C`;
  }

  let routeTimer = null;
  function update(planNow) {
    const p = params();
    showValues(p);
    computeRisk(p);
    drawOverlay();
    clearTimeout(routeTimer);
    if (planNow) return planAndShow();
    routeTimer = setTimeout(planAndShow, 150);
    return null;
  }
  for (const id of Object.values(inputs)) el(id).addEventListener("input", () => update(false));
  document.querySelectorAll('input[name="colour"]').forEach((r) =>
    r.addEventListener("change", () => { colourBy = r.value; drawOverlay(); }));

  function setWeather(p) {
    el("in-hs").value = p.hs; el("in-u").value = p.u;
    el("in-dir").value = Math.round(p.dir / 22.5) * 22.5 % 360; el("in-t").value = p.t;
    return update(true);
  }

  // ── Start: the initial weather, or a self-test from the URL ─────────────
  const test = location.hash.match(/^#selftest=([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)$/);
  if (test) {
    const [hs, u, dir, t] = test.slice(1).map(Number);
    const r = setWeather({ hs, u, dir, t });
    let riskSum = 0, water = 0;
    for (let i = 0; i < N; i++) if (risk[i] === risk[i]) { riskSum += risk[i]; water++; }
    const out = el("selftest");
    out.textContent = JSON.stringify({ water, riskSum, routeKm: r.a.km, routeCost: r.route.cost,
      meanRisk: r.a.meanRisk, shortestKm: r.b.km });
    out.hidden = false;
  } else {
    setWeather(D.initial);
  }
}
