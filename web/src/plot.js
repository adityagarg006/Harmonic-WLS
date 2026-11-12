/* Small helpers shared by the page: building DOM elements, form controls,
   and a canvas plotting function (lines, dots, bars, bands, histograms) with
   a hover read-out. No libraries, so the page works offline. */
const H = window.HWLS, D = window.SITE_DATA, TAU = 2 * Math.PI;
const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith("--")) e.style.setProperty(sk, sv); else e.style[sk] = sv; }
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}
const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d);
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const logspace = (a, b, n) => linspace(Math.log10(a), Math.log10(b), n).map(v => 10 ** v);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const variance = a => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); };
const absMax = a => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };
const db = v => 20 * Math.log10(Math.max(v, 1e-12));
const normPdf = x => Math.exp(-x * x / 2) / Math.sqrt(TAU);
const sleep0 = () => new Promise(r => setTimeout(r, 0));

/* ---------- controls ---------- */
function slider(id, label, { min, max, step, value, fmt: f = v => v, log = false }, onInput) {
  const out = el("output", { for: id });
  const toV = p => log ? 10 ** p : p, toP = v => log ? Math.log10(v) : v;
  const inp = el("input", { type: "range", id, min: toP(min), max: toP(max), step: log ? (toP(max) - toP(min)) / 200 : step, value: toP(value) });
  const ctl = el("div", { class: "ctl" }, el("label", { for: id }, label), el("div", { class: "row" }, inp, out));
  const get = () => { let v = toV(+inp.value); if (log && step) v = Math.round(v / step) * step; return v; };
  const sync = () => { out.textContent = f(get()); };
  inp.addEventListener("input", () => { sync(); onInput && onInput(get()); });
  sync();
  ctl.get = get; ctl.set = v => { inp.value = toP(v); sync(); }; ctl.input = inp;
  return ctl;
}
function select(id, label, options, value, onChange) {
  const s = el("select", { id }, options.map(([v, t]) => el("option", { value: v, selected: v === value }, t)));
  s.addEventListener("change", () => onChange && onChange(s.value));
  const ctl = el("div", { class: "ctl" }, el("label", { for: id }, label), s);
  ctl.get = () => s.value; ctl.set = v => { s.value = v; }; ctl.sel = s;
  return ctl;
}
let RADIO_GROUP = 0;
function segmented(options, value, onChange) { // a row of radio buttons
  const name = "rg" + (++RADIO_GROUP), box = el("div", { class: "radios", role: "radiogroup" });
  const inps = options.map(([v], i) => el("input", { type: "radio", name, value: i, checked: v === value, onchange: () => { value = v; onChange && onChange(v); } }));
  box.append(...options.map(([, t], i) => el("label", {}, inps[i], t)));
  box.get = () => value; box.set = v => { value = v; inps.forEach((b, i) => { b.checked = options[i][0] === v; }); };
  return box;
}
const controls = (title = "Parameters") => el("fieldset", { class: "controls" }, el("legend", {}, title));
// rows: [label, value, note]
const kvTable = rows => el("table", { class: "kv" }, el("tbody", {}, rows.map(([k, v, s]) => el("tr", {}, el("th", { scope: "row" }, k), el("td", {}, v), el("td", { class: "muted" }, s || "")))));
const windowOptions = () => Object.keys(H.WINDOWS).map(w => [w, H.DISPLAY[w]]);

/* ---------- colours from CSS tokens ---------- */
let COL = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["ink", "ink-2", "muted", "line", "grid", "panel", "panel-2", "accent", "band", "band-2", "good", "bad", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"])
    COL[k] = cs.getPropertyValue("--" + k).trim();
}
const col = c => (c && COL[c]) || c || COL.s1;
function withAlpha(c, a) {
  c = col(c);
  if (c.startsWith("#") && c.length === 7) return c + Math.round(a * 255).toString(16).padStart(2, "0");
  return c;
}

/* ---------- plotting ---------- */
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo; if (!(span > 0)) return [lo];
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function tickLabel(v, log) {
  if (log) { const r = +v.toPrecision(3); return r >= 0.01 && r < 1e4 ? String(r) : v.toExponential(0); }
  const a = Math.abs(v); if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(0);
  return String(+v.toPrecision(4));
}
const PLOT_FONT = "'Helvetica Neue', Arial, sans-serif";
const PLOTS = new Set();
function plot(canvas, cfg) {
  canvas._cfg = cfg; PLOTS.add(canvas);
  if (!canvas._wired) {
    canvas._wired = true;
    canvas.addEventListener("mousemove", ev => { const r = canvas.getBoundingClientRect(); canvas._hover = [ev.clientX - r.left, ev.clientY - r.top]; draw(canvas); });
    canvas.addEventListener("mouseleave", () => { canvas._hover = null; draw(canvas); });
    if (window.ResizeObserver) new ResizeObserver(() => draw(canvas)).observe(canvas);
  }
  draw(canvas);
}
function extent(cfg, axis) {
  let lo = Infinity, hi = -Infinity;
  const push = v => { if (v != null && isFinite(v) && !(cfg[axis].log && v <= 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  for (const L of cfg.layers) {
    if (L.noFit) continue;
    if (axis === "x") { if (Array.isArray(L.x)) L.x.forEach(push); else if (L.type === "vline") push(L.x); }
    else {
      for (const a of [L.y, L.lo, L.hi]) if (Array.isArray(a)) a.forEach(push);
      if (L.type === "hline") push(L.y);
      if (L.type === "bars") push(L.base ?? 0);
    }
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}
function draw(canvas) {
  const cfg = canvas._cfg; if (!cfg) return;
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, Hh = canvas.clientHeight;
  if (!W || !Hh) return;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hh * dpr);
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, Hh);
  const X = cfg.x = cfg.x || {}, Y = cfg.y = cfg.y || {};
  let [x0, x1] = extent(cfg, "x"), [y0, y1] = extent(cfg, "y");
  if (!Y.log && Y.pad !== 0) { const p = (y1 - y0) * 0.06; y0 -= p; y1 += p; }
  if (X.min != null) x0 = X.min; if (X.max != null) x1 = X.max; if (Y.min != null) y0 = Y.min; if (Y.max != null) y1 = Y.max;
  const pad = { l: cfg.padL ?? 52, r: 12, t: cfg.title ? 22 : 10, b: X.label ? 38 : 24 };
  const pw = W - pad.l - pad.r, ph = Hh - pad.t - pad.b;
  const tx = X.log ? v => pad.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * pw : v => pad.l + (v - x0) / (x1 - x0) * pw;
  const ty = Y.log ? v => pad.t + ph - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) * ph : v => pad.t + ph - (v - y0) / (y1 - y0) * ph;
  const ok = (v, A) => v != null && isFinite(v) && (!A.log || v > 0);
  g.font = "11px " + PLOT_FONT;
  // grid + ticks
  const ticksOf = (A, lo, hi) => A.ticks || (A.log ? (() => { const t = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); } return t; })() : niceTicks(lo, hi, A.n || (A === X ? Math.max(3, Math.floor(pw / 80)) : Math.max(3, Math.floor(ph / 45)))));
  g.strokeStyle = col("grid"); g.lineWidth = 1; g.fillStyle = col("ink-2");
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of ticksOf(Y, y0, y1)) { const yy = ty(v); if (yy < pad.t - 1 || yy > pad.t + ph + 1) continue; g.globalAlpha = 0.7; g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(pad.l + pw, yy); g.stroke(); g.globalAlpha = 1; const lab = (Y.fmt || (v => tickLabel(v, Y.log)))(v); if (lab) g.fillText(lab, pad.l - 6, yy); }
  g.textAlign = "center"; g.textBaseline = "top";
  for (const v of ticksOf(X, x0, x1)) { const xx = tx(v); if (xx < pad.l - 1 || xx > pad.l + pw + 1) continue; g.globalAlpha = 0.45; g.beginPath(); g.moveTo(xx, pad.t); g.lineTo(xx, pad.t + ph); g.stroke(); g.globalAlpha = 1; const lab = (X.fmt || (v => tickLabel(v, X.log)))(v); if (lab) g.fillText(lab, xx, pad.t + ph + 5); }
  g.strokeStyle = col("muted"); g.beginPath(); g.moveTo(pad.l, pad.t); g.lineTo(pad.l, pad.t + ph); g.lineTo(pad.l + pw, pad.t + ph); g.stroke();
  if (X.label) { g.fillStyle = col("ink-2"); g.textAlign = "center"; g.textBaseline = "bottom"; g.fillText(X.label, pad.l + pw / 2, Hh - 2); }
  if (Y.label) { g.save(); g.translate(11, pad.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = col("ink-2"); g.fillText(Y.label, 0, 0); g.restore(); }
  if (cfg.title) { g.fillStyle = col("ink"); g.textAlign = "left"; g.textBaseline = "top"; g.font = "600 12px " + PLOT_FONT; g.fillText(cfg.title, pad.l, 3); g.font = "11px " + PLOT_FONT; }
  g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();
  for (const L of cfg.layers) {
    g.globalAlpha = L.alpha ?? 1; g.setLineDash(L.dash || []);
    const c = col(L.color);
    if (L.type === "hspan") { g.fillStyle = c; g.fillRect(pad.l, ty(Math.max(L.y0, L.y1)), pw, Math.abs(ty(L.y0) - ty(L.y1))); }
    else if (L.type === "band") {
      g.fillStyle = c; g.beginPath(); let started = false;
      for (let i = 0; i < L.x.length; i++) if (ok(L.x[i], X) && ok(L.hi[i], Y)) { const p = [tx(L.x[i]), ty(L.hi[i])]; started ? g.lineTo(...p) : (g.moveTo(...p), started = true); }
      for (let i = L.x.length - 1; i >= 0; i--) if (ok(L.x[i], X) && ok(L.lo[i], Y)) g.lineTo(tx(L.x[i]), ty(L.lo[i]));
      g.closePath(); g.fill();
    } else if (L.type === "line") {
      g.strokeStyle = c; g.lineWidth = L.width ?? 2; g.lineJoin = "round"; g.beginPath(); let pen = false;
      for (let i = 0; i < L.x.length; i++) { if (!ok(L.x[i], X) || !ok(L.y[i], Y)) { pen = false; continue; } const p = [tx(L.x[i]), ty(L.y[i])]; pen ? g.lineTo(...p) : (g.moveTo(...p), pen = true); }
      g.stroke();
    } else if (L.type === "dots") {
      g.fillStyle = c; const r = L.r ?? 3;
      for (let i = 0; i < L.x.length; i++) if (ok(L.x[i], X) && ok(L.y[i], Y)) { g.beginPath(); g.arc(tx(L.x[i]), ty(L.y[i]), r, 0, TAU); g.fill(); }
    } else if (L.type === "bars") {
      g.fillStyle = c; const bw = L.width ?? 0.7, base = L.base ?? (Y.log ? y0 : 0);
      for (let i = 0; i < L.x.length; i++) {
        if (!ok(L.y[i], Y)) continue;
        const xa = tx(L.x[i] - bw / 2), xb = tx(L.x[i] + bw / 2), ya = ty(L.y[i]), yb = ty(Math.max(base, Y.log ? y0 : -Infinity));
        g.fillRect(xa + 1, Math.min(ya, yb), Math.max(1, xb - xa - 2), Math.abs(yb - ya));
      }
    } else if (L.type === "hist") { // x = bin edges, y = heights
      g.fillStyle = c;
      for (let i = 0; i < L.y.length; i++) { const xa = tx(L.x[i]), xb = tx(L.x[i + 1]); g.fillRect(xa + 0.5, ty(L.y[i]), Math.max(0.5, xb - xa - 1), ty(0) - ty(L.y[i])); }
    } else if (L.type === "errbar") {
      g.strokeStyle = c; g.lineWidth = L.width ?? 1.4;
      for (let i = 0; i < L.x.length; i++) { if (!ok(L.lo[i], Y) || !ok(L.hi[i], Y)) continue; const xx = tx(L.x[i]); g.beginPath(); g.moveTo(xx, ty(L.lo[i])); g.lineTo(xx, ty(L.hi[i])); g.stroke(); }
    } else if (L.type === "hline") { g.strokeStyle = c; g.lineWidth = L.width ?? 1; g.beginPath(); g.moveTo(pad.l, ty(L.y)); g.lineTo(pad.l + pw, ty(L.y)); g.stroke(); }
    else if (L.type === "vline") { g.strokeStyle = c; g.lineWidth = L.width ?? 1; g.beginPath(); g.moveTo(tx(L.x), pad.t); g.lineTo(tx(L.x), pad.t + ph); g.stroke(); }
    else if (L.type === "text") { g.fillStyle = c; g.textAlign = L.align || "left"; g.textBaseline = L.baseline || "middle"; g.font = (L.font || "11px ") + PLOT_FONT; g.fillText(L.text, tx(L.x), ty(L.y)); }
  }
  g.restore(); g.setLineDash([]); g.globalAlpha = 1;
  // hover readout: nearest point of the first hoverable layer
  if (canvas._hover && cfg.hover !== false) {
    const [mx, my] = canvas._hover;
    if (mx >= pad.l && mx <= pad.l + pw && my >= pad.t && my <= pad.t + ph) {
      let best = null;
      for (const L of cfg.layers) {
        if (!["line", "dots", "bars"].includes(L.type) || L.hover === false || !L.x) continue;
        for (let i = 0; i < L.x.length; i++) {
          if (!ok(L.x[i], X) || !ok(L.y[i], Y)) continue;
          const d = Math.abs(tx(L.x[i]) - mx) + 0.15 * Math.abs(ty(L.y[i]) - my);
          if (!best || d < best.d) best = { d, x: L.x[i], y: L.y[i], L };
        }
      }
      if (best) {
        const px = tx(best.x), py = ty(best.y);
        g.strokeStyle = col("muted"); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(px, pad.t); g.lineTo(px, pad.t + ph); g.stroke(); g.setLineDash([]);
        g.fillStyle = col(best.L.color); g.beginPath(); g.arc(px, py, 4.5, 0, TAU); g.fill(); g.strokeStyle = col("panel"); g.lineWidth = 2; g.stroke();
        const txt = (best.L.name ? best.L.name + ": " : "") + (X.hfmt || (v => fmt(v, 3)))(best.x) + ", " + (Y.hfmt || (v => fmt(v, 4)))(best.y);
        g.font = "11.5px " + PLOT_FONT;
        const tw = g.measureText(txt).width + 12, bx = Math.min(Math.max(px + 8, pad.l), pad.l + pw - tw), by = Math.max(pad.t, py - 26);
        g.fillStyle = col("panel"); g.strokeStyle = col("line"); g.lineWidth = 1; g.fillRect(bx, by, tw, 20); g.strokeRect(bx, by, tw, 20);
        g.fillStyle = col("ink"); g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(txt, bx + 6, by + 10);
      }
    }
  }
}
function legend(items) { return el("div", { class: "legend" }, items.map(([c, t, box]) => el("span", { class: box ? "box" : null, style: { "--c": col(c) } }, t))); }
function card(title, canvasClass = "", cap = null, extra = null) {
  const cv = el("canvas", { class: "plot " + canvasClass });
  const c = el("div", { class: "card" }, title ? el("h3", {}, title) : null, extra, cv, cap ? el("p", { class: "cap" }, cap) : null);
  c.canvas = cv; return c;
}
function histogram(vals, lo, hi, nb) {
  const e = linspace(lo, hi, nb + 1), h = new Array(nb).fill(0), w = (hi - lo) / nb;
  for (const v of vals) { const i = Math.floor((v - lo) / w); if (i >= 0 && i < nb) h[i]++; }
  return { edges: e, dens: h.map(c => c / (vals.length * w)) };
}


/* Audio: play a clip through the Web Audio API, and decode the 16-bit WAV
   clips that build.py embeds as base64. */
let AC = null, SRC_NODE = null;
function playSamples(x, fs) {
  try {
    AC = AC || new (window.AudioContext || window.webkitAudioContext)();
    stopAudio();
    let peak = 0; for (const v of x) peak = Math.max(peak, Math.abs(v));
    const g = peak > 0 ? 0.85 / peak : 1, buf = AC.createBuffer(1, x.length, fs), ch = buf.getChannelData(0);
    const fade = Math.min(400, x.length >> 3);
    for (let i = 0; i < x.length; i++) { let a = 1; if (i < fade) a = i / fade; if (i > x.length - fade) a = (x.length - i) / fade; ch[i] = x[i] * g * a; }
    SRC_NODE = AC.createBufferSource(); SRC_NODE.buffer = buf; SRC_NODE.connect(AC.destination); SRC_NODE.start();
  } catch (e) { console.warn("audio unavailable", e); }
}
function stopAudio() { if (SRC_NODE) try { SRC_NODE.stop(); } catch (e) { /* already stopped */ } }
function decodeWav(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const dv = new DataView(u8.buffer); let p = 12, fs = 44100, data = null;
  while (p < u8.length - 8) {
    const id = String.fromCharCode(u8[p], u8[p + 1], u8[p + 2], u8[p + 3]), sz = dv.getUint32(p + 4, true);
    if (id === "fmt ") fs = dv.getUint32(p + 12, true);
    if (id === "data") { data = [p + 8, sz]; break; }
    p += 8 + sz + (sz & 1);
  }
  const n = data[1] / 2, x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = dv.getInt16(data[0] + 2 * i, true) / 32768;
  return { x, fs };
}
