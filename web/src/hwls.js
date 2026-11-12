/* hwls.js: JavaScript port of the Python hwls package (the estimator of
   Irizarry 2002, eqs. 7-9, with Theorem-2 standard errors). Same algorithm,
   same defaults, so the numbers match the Python code.
   Frequencies are in radians per sample; t = 1..T as in the paper. */
(function (root) {
  "use strict";
  const PI = Math.PI, TAU = 2 * Math.PI;

  // ---------------------------------------------------------------- windows
  const WINDOWS = {
    rectangular: s => 1,
    hann: s => Math.sin(PI * s) ** 2,
    hamming: s => 0.54 - 0.46 * Math.cos(TAU * s),
    blackman: s => 0.42 - 0.5 * Math.cos(TAU * s) + 0.08 * Math.cos(2 * TAU * s),
    ramp: s => s,
  };
  const DISPLAY = {
    rectangular: "Rectangular (w = 1)", hann: "Hann", hamming: "Hamming", blackman: "Blackman",
    ramp: "Ramp w(s) = s",
  };
  function weights(win, T) {
    const f = WINDOWS[win], w = new Float64Array(T);
    for (let t = 1; t <= T; t++) w[t - 1] = f(t / T);
    return w;
  }
  // Simpson's rule on [0,1]
  function integrate(g, n = 4000) {
    const h = 1 / n; let s = g(0) + g(1);
    for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * g(i * h);
    return s * h / 3;
  }
  const _constCache = {};
  function constants(win, center = false) {
    const key = win + (center ? "c" : "");
    if (_constCache[key]) return _constCache[key];
    const f = WINDOWS[win], off = center ? 0.5 : 0;
    const m = {};
    for (let n = 0; n < 3; n++) {
      m["W" + n] = integrate(s => (s - off) ** n * f(s));
      m["U" + n] = integrate(s => (s - off) ** n * f(s) ** 2);
    }
    const { W0, W1, W2, U0, U1, U2 } = m, Wl = [W0, W1, W2];
    const a0 = (W0 * W2 - W1 * W1) ** -2;
    const b = n => Wl[n] ** 2 * U2 + Wl[n + 1] * (Wl[n + 1] * U0 - 2 * Wl[n] * U1);
    const c = Object.assign(m, {
      c0: a0 * b(0), c1: U0 / (W0 * W0), c2: a0 * b(1),
      c3: a0 * W1 / (W0 * W0) * (W0 * W0 * W1 * U2 - W1 ** 3 * U0 - 2 * W0 * W0 * W2 * U1 + 2 * W0 * W1 * W2 * U0),
      c4: a0 * (W0 * W1 * U2 - W1 * W1 * U1 - W0 * W2 * U1 + W1 * W2 * U0),
    });
    _constCache[key] = c;
    return c;
  }
  // |∫ w(s) e^{ibs} ds| / W0  (leakage kernel)
  function kernel(win, b, n = 800) {
    const f = WINDOWS[win]; let re = 0, im = 0, W0 = 0; const h = 1 / n;
    for (let i = 0; i <= n; i++) {
      const s = i * h, c = (i === 0 || i === n) ? 0.5 : 1, w = f(s) * c;
      re += w * Math.cos(b * s); im += w * Math.sin(b * s); W0 += w;
    }
    return Math.hypot(re, im) / W0;
  }

  // ---------------------------------------------------------------- FFT
  function fft(re, im) { // in-place radix-2, length power of 2
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -TAU / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  function paddedMagnitude(z, pad = 8) {
    const N = 1 << Math.ceil(Math.log2(pad * z.length));
    const re = new Float64Array(N), im = new Float64Array(N);
    re.set(z); fft(re, im);
    const half = N / 2 + 1, mag = new Float64Array(half);
    for (let i = 0; i < half; i++) mag[i] = Math.hypot(re[i], im[i]);
    return { mag, N };
  }

  // Z(ω) = Σ z_t e^{-iωt} and derivatives (t = 1..T)
  function dft3(z, w) {
    let r0 = 0, i0 = 0, r1 = 0, i1 = 0, r2 = 0, i2 = 0;
    // recurrence for e^{-iωt}
    const cr = Math.cos(w), ci = -Math.sin(w);
    let er = cr, ei = ci;
    for (let k = 0; k < z.length; k++) {
      const t = k + 1, zr = z[k] * er, zi = z[k] * ei;
      r0 += zr; i0 += zi;
      r1 += t * zi; i1 += -t * zr;           // (-i t) * z e
      r2 += -t * t * zr; i2 += -t * t * zi;
      const nr = er * cr - ei * ci; ei = er * ci + ei * cr; er = nr;
      if ((k & 255) === 255) { const a = -w * (t + 1); er = Math.cos(a); ei = Math.sin(a); }
    }
    return [r0, i0, r1, i1, r2, i2];
  }
  function absDft2(z, w) { const d = dft3(z, w); return d[0] * d[0] + d[1] * d[1]; }

  function refinePeak(z, w0, hw, trace) {
    const lo = w0 - hw, hi = w0 + hw; let w = w0;
    if (trace) trace.push(w);
    for (let it = 0; it < 30; it++) {
      const [r0, i0, r1, i1, r2, i2] = dft3(z, w);
      const g = 2 * (r0 * r1 + i0 * i1);
      const h = 2 * (r1 * r1 + i1 * i1 + r0 * r2 + i0 * i2);
      if (h >= 0) break;
      const step = -g / h, wn = w + step;
      if (wn < lo || wn > hi) break;
      w = wn; if (trace) trace.push(w);
      if (Math.abs(step) < 1e-13) return w;
    }
    // golden-section fallback on -|Z|^2
    let a = lo, b = hi; const gr = (Math.sqrt(5) - 1) / 2;
    let c = b - gr * (b - a), d = a + gr * (b - a), fc = absDft2(z, c), fd = absDft2(z, d);
    for (let i = 0; i < 80; i++) {
      if (fc > fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = absDft2(z, c); }
      else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = absDft2(z, d); }
    }
    w = (a + b) / 2; if (trace) trace.push(w);
    return w;
  }

  // residual periodogram at Fourier frequencies near ω, Daniell average (matches Python)
  function noiseSpectrum(resid, omegas, halfWidth = 12, notch = 2) {
    const T = resid.length, df = TAU / T, out = [];
    const cache = new Map();
    const I = j => {
      if (cache.has(j)) return cache.get(j);
      const lam = df * j; let re = 0, im = 0;
      for (let k = 0; k < T; k++) { re += resid[k] * Math.cos(lam * k); im -= resid[k] * Math.sin(lam * k); }
      const v = (re * re + im * im) / (TAU * T); cache.set(j, v); return v;
    };
    const jmax = Math.floor(T / 2);
    for (const w of omegas) {
      let hwid = halfWidth, vals;
      do {
        vals = [];
        const jc = Math.round(w / df);
        for (let j = Math.max(1, jc - hwid - 1); j <= Math.min(jmax, jc + hwid + 1); j++) {
          const lam = j * df;
          if (Math.abs(lam - w) > hwid * df) continue;
          if (omegas.some(o => Math.abs(lam - o) <= notch * df)) continue;
          vals.push(I(j));
        }
        hwid *= 2;
      } while (vals.length < 4 && hwid < T);
      out.push(vals.reduce((a, b) => a + b, 0) / vals.length);
    }
    return out;
  }

  /* fitHarmonics: eqs (7)-(9).  opts: {window, f0hint, searchFrac, pad, withSE, trace, detectWindow} */
  function fitHarmonics(y, K, opts = {}) {
    const win = opts.window || "hann", T = y.length, pad = opts.pad || 8;
    const w = weights(win, T), z = new Float64Array(T);
    for (let i = 0; i < T; i++) z[i] = w[i] * y[i];
    const { mag, N } = paddedMagnitude(z, pad);
    const cell = TAU / N, freqs = i => i * cell;
    let idx = [];
    if (opts.f0hint) {
      const sf = opts.searchFrac || 0.45, f0 = opts.f0hint;
      for (let k = 1; k <= K; k++) {
        const lo = Math.max((k - sf) * f0, 1e-3), hi = Math.min((k + sf) * f0, PI - 1e-3);
        let best = -1, bv = -1;
        for (let i = Math.ceil(lo / cell); i <= Math.floor(hi / cell); i++) if (mag[i] > bv) { bv = mag[i]; best = i; }
        idx.push(best);
      }
    } else {
      const minSep = opts.minSep || 4 * PI / T, loc = [];
      for (let i = 1; i < mag.length - 1; i++) if (mag[i] > mag[i - 1] && mag[i] >= mag[i + 1] && freqs(i) > 1e-3) loc.push(i);
      loc.sort((a, b) => mag[b] - mag[a]);
      for (const j of loc) {
        if (idx.every(c => Math.abs(freqs(j) - freqs(c)) >= minSep)) idx.push(j);
        if (idx.length === K) break;
      }
      idx.sort((a, b) => a - b);
    }
    const traces = [];
    let omega = idx.map(j => { const tr = opts.trace ? [] : null; const r = refinePeak(z, freqs(j), 1.5 * cell, tr); traces.push(tr); return r; });
    const order = omega.map((v, i) => i).sort((a, b) => omega[a] - omega[b]);
    omega = order.map(i => omega[i]);
    let sw = 0; for (let i = 0; i < T; i++) sw += w[i];
    const A = [], B = [];
    for (const om of omega) {
      let a = 0, b = 0;
      for (let k = 0; k < T; k++) { const t = k + 1; a += z[k] * Math.cos(om * t); b += z[k] * Math.sin(om * t); }
      A.push(2 * a / sw); B.push(2 * b / sw);
    }
    const fitted = new Float64Array(T), resid = new Float64Array(T);
    for (let k = 0; k < T; k++) {
      const t = k + 1; let s = 0;
      for (let j = 0; j < omega.length; j++) s += A[j] * Math.cos(omega[j] * t) + B[j] * Math.sin(omega[j] * t);
      fitted[k] = s; resid[k] = y[k] - s;
    }
    const rho = A.map((a, i) => Math.hypot(a, B[i]));
    const res = { omega, A, B, rho, T, window: win, y, z, w, fitted, resid, mag, N, peakIdx: idx, traces };
    if (opts.withSE !== false) {
      const c = constants(win), f = noiseSpectrum(resid, omega);
      res.f = f;
      res.se = {
        omega: omega.map((_, i) => Math.sqrt(4 * PI * f[i] * c.c0 / (T ** 3 * rho[i] ** 2))),
        A: omega.map((_, i) => Math.sqrt(4 * PI * f[i] * (c.c1 * A[i] ** 2 + c.c2 * B[i] ** 2) / (T * rho[i] ** 2))),
        B: omega.map((_, i) => Math.sqrt(4 * PI * f[i] * (c.c2 * A[i] ** 2 + c.c1 * B[i] ** 2) / (T * rho[i] ** 2))),
        rho: omega.map((_, i) => Math.sqrt(4 * PI * f[i] * c.c1 / T)),
      };
    }
    return res;
  }

  function analyzeFrames(x, fs, T, K, opts = {}) {
    const start = Math.round((opts.startS || 0) * fs), hop = opts.hop || T;
    const maxF = Math.floor((x.length - start - T) / hop) + 1;
    const n = Math.min(opts.nFrames || maxF, maxF);
    let f0 = opts.f0hz ? TAU * opts.f0hz / fs : null;
    const frames = [];
    for (let j = 0; j < n; j++) {
      const seg = x.subarray(start + j * hop, start + j * hop + T);
      const fit = fitHarmonics(seg, K, { window: opts.window || "hann", f0hint: f0 });
      if (opts.f0hz) f0 = fit.omega[0];
      fit.time = (start + j * hop) / fs;
      fit.freqHz = fit.omega.map(o => o * fs / TAU);
      fit.seHz = fit.se.omega.map(o => o * fs / TAU);
      frames.push(fit);
    }
    return frames;
  }

  // ---------------------------------------------------------------- inference helpers
  function pooledF0(f, se) {
    let num = 0, den = 0;
    f.forEach((v, i) => { const k = i + 1, w = 1 / se[i] ** 2; num += w * k * v; den += w * k * k; });
    const lam = num / den; let chi2 = 0;
    f.forEach((v, i) => { chi2 += (v - (i + 1) * lam) ** 2 / se[i] ** 2; });
    return { f0: lam, se: 1 / Math.sqrt(den), chi2, df: f.length - 1 };
  }
  const jndHz = (f, st = 0.03) => f * (2 ** (st / 12) - 1);
  function stiffFit(f, se, ks) { // Gauss–Newton for f_k = k f0 sqrt(1 + B k^2)
    ks = ks || f.map((_, i) => i + 1);
    let f0 = f[0], B = 0;
    for (let it = 0; it < 50; it++) {
      let JtJ = [0, 0, 0], Jtr = [0, 0];
      ks.forEach((k, i) => {
        const s = Math.sqrt(1 + B * k * k), m = k * f0 * s, r = (f[i] - m) / se[i];
        const j0 = k * s / se[i], j1 = k * f0 * k * k / (2 * s) / se[i];
        JtJ[0] += j0 * j0; JtJ[1] += j0 * j1; JtJ[2] += j1 * j1; Jtr[0] += j0 * r; Jtr[1] += j1 * r;
      });
      const det = JtJ[0] * JtJ[2] - JtJ[1] ** 2;
      const d0 = (JtJ[2] * Jtr[0] - JtJ[1] * Jtr[1]) / det, d1 = (JtJ[0] * Jtr[1] - JtJ[1] * Jtr[0]) / det;
      f0 += d0; B += d1;
      if (Math.abs(d0) < 1e-10 && Math.abs(d1) < 1e-14) {
        return { f0, B, seB: Math.sqrt(JtJ[0] / det), seF0: Math.sqrt(JtJ[2] / det) };
      }
    }
    return { f0, B, seB: NaN, seF0: NaN };
  }

  // ---------------------------------------------------------------- random numbers
  function rng(seed) {
    let a = seed >>> 0, spare = null;
    const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const n = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const m = Math.sqrt(-2 * Math.log(r) / r); spare = y * m; return x * m;
    };
    return { u, n };
  }
  function noise(R, T, kind, sigma, phi, g) {
    const e = new Float64Array(T);
    if (kind === "ar1") { let p = 0; for (let i = -200; i < T; i++) { p = phi * p + sigma * g.n(); if (i >= 0) e[i] = p; } }
    else for (let i = 0; i < T; i++) e[i] = sigma * g.n();
    return e;
  }
  const spectrum = (kind, sigma, phi) => w => kind === "ar1"
    ? sigma * sigma / (TAU * ((1 - phi * Math.cos(w)) ** 2 + (phi * Math.sin(w)) ** 2))
    : sigma * sigma / TAU;

  const api = { WINDOWS, DISPLAY, weights, constants, kernel, fft, paddedMagnitude, dft3, refinePeak,
    noiseSpectrum, fitHarmonics, analyzeFrames, pooledF0, jndHz, stiffFit, rng, noise, spectrum, TAU };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.HWLS = api;
})(typeof self !== "undefined" ? self : this);
