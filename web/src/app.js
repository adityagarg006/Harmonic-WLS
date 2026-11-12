/* The five steps of the demo page. Steps 1-3 compute everything live in the
   browser with hwls.js; steps 3-5 also show numbers for the real recordings,
   which come from the Python run (results/*.json, copied in by build.py). */

const STEPS = [];
const step = def => STEPS.push(def);
let CURRENT = 0;

/* A synthetic clarinet-like note: odd partials strong, 441 Hz, white noise.
   Synthetic so that every estimate can be checked against the truth. */
const NOTE = { f0: 441, K: 10, sigma: 0.02, T: 1025, fs: 44100, seed: 7, taper: "hann" };
const noteAmp = k => (k % 2 ? 1 : 0.06) / k ** 1.1;
function makeNote(seed = NOTE.seed, T = NOTE.T, K = NOTE.K) {
  const g = H.rng(seed), om = [], A = [], B = [];
  for (let k = 1; k <= K; k++) {
    const ph = g.u() * TAU, a = noteAmp(k);
    om.push(TAU * k * NOTE.f0 / NOTE.fs); A.push(a * Math.cos(ph)); B.push(-a * Math.sin(ph));
  }
  const e = H.noise(1, T, "white", NOTE.sigma, 0, g), y = new Float64Array(T);
  for (let i = 0; i < T; i++) {
    const t = i + 1; let v = 0;
    for (let j = 0; j < K; j++) v += A[j] * Math.cos(om[j] * t) + B[j] * Math.sin(om[j] * t);
    y[i] = v + e[i];
  }
  return { y, om, rho: A.map((a, i) => Math.hypot(a, B[i])) };
}
const hz = o => o * NOTE.fs / TAU;

function table(head, rows, cls = "") {
  return el("table", { class: cls }, el("thead", {}, el("tr", {}, head.map(h => el("th", {}, h)))),
    el("tbody", {}, rows.map(r => el("tr", {}, r.map(c => (c && c.nodeType) ? el("td", {}, c) : el("td", {}, c))))));
}
const figure = (canvasClass = "", caption = null) => {
  const cv = el("canvas", { class: "plot " + canvasClass });
  const f = el("figure", {}, cv, caption ? el("figcaption", {}, caption) : null);
  f.canvas = cv; return f;
};
const para = html => el("p", { html });

/* ---------- Step 1: the seven-step pipeline ---------- */
step({
  title: "The estimator, step by step",
  render(stage) {
    let sub = 0, kSel = 1, R, F;
    stage.append(para(`One 23&nbsp;ms frame of a synthetic clarinet-like note (441&nbsp;Hz, ${NOTE.K} partials, white noise with
      &sigma; = ${NOTE.sigma}, T = ${NOTE.T}). The estimator does not know the frequencies, amplitudes or phases.
      Go through the seven steps; the last one checks each estimate against the truth.`));
    const ctl = el("div", { class: "controls" });
    const tp = select("p-taper", "Taper", windowOptions(), NOTE.taper, v => { NOTE.taper = v; run(); });
    const again = el("button", { type: "button", onclick: () => { NOTE.seed = (NOTE.seed * 7919 + 13) % 100003; run(); } }, "New noise draw");
    ctl.append(tp, again);
    stage.append(ctl);
    const NAMES = ["Take the frame", "Taper it", "Tapered periodogram", "Refine each peak", "Amplitudes", "Noise level", "Standard errors"];
    const tabs = el("ol", { class: "substeps" }, NAMES.map((n, i) => el("li", {}, el("button", { type: "button", onclick: () => { sub = i; show(); } }, `${i + 1}. ${n}`))));
    const text = el("p", { class: "steptext" });
    const body = el("div");
    stage.append(tabs, text, body);
    const TEXT = [
      "The samples y<sub>1</sub>, &hellip;, y<sub>T</sub> of one frame.",
      "Multiply by the taper w(t/T). The frame now fades in and out, so its edges stop acting like sudden cuts, which is what causes leakage between partials.",
      "The periodogram of the tapered frame, |&Sigma; w(t/T) y<sub>t</sub> e<sup>&minus;i&omega;t</sup>|&sup2;, on an FFT grid 8 times finer than usual. Partial k is the highest point inside its own band around k&middot;f<sub>0</sub> (shaded).",
      "The grid only gives a starting point. Newton steps on the exact derivative climb to the top of the peak (eq. 9 of the paper). Pick a partial to see the steps.",
      "With the frequencies fixed, the amplitudes are weighted projections: &Acirc;<sub>k</sub> = 2&Sigma;w y cos(&omega;&#770;<sub>k</sub>t) / &Sigma;w, and B&#770;<sub>k</sub> the same with sin (eqs. 7&ndash;8). The fitted model is drawn over the data; what is left over should look like noise.",
      "Theorem 2 needs the noise spectrum at each partial. We average the periodogram of the left-over over &plusmn;12 frequency bins around each partial, skipping the bins right next to any fitted partial.",
      "Theorem 2: var(&omega;&#770;<sub>k</sub>) &asymp; 4&pi; c<sub>0</sub> f(&omega;<sub>k</sub>) / (T<sup>3</sup> &rho;<sub>k</sub><sup>2</sup>). A tick means the true value lies within &plusmn;2 s.e.; about 1 row in 20 should miss.",
    ];
    const nav = el("div", { class: "subnav" },
      el("button", { type: "button", onclick: () => { sub = Math.max(0, sub - 1); show(); } }, "Previous"),
      el("button", { type: "button", onclick: () => { sub = Math.min(6, sub + 1); show(); } }, "Next"));
    stage.append(nav);
    function run() {
      R = makeNote();
      F = H.fitHarmonics(R.y, NOTE.K, { window: NOTE.taper, f0hint: TAU * NOTE.f0 / NOTE.fs, trace: true });
      show();
    }
    function show() {
      tabs.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-current", String(i === sub)));
      text.innerHTML = TEXT[sub];
      body.innerHTML = "";
      const T = NOTE.T, t = Array.from({ length: T }, (_, i) => i + 1), fmax = (NOTE.K + 0.6) * NOTE.f0;
      if (sub === 0 || sub === 1) {
        const f = figure();
        body.append(f);
        const m = absMax(R.y);
        const L = sub === 0
          ? [{ type: "line", x: t, y: Array.from(R.y), color: "s1", width: 1 }]
          : [{ type: "line", x: t, y: Array.from(R.y), color: "muted", width: 0.8, alpha: 0.35, hover: false },
             { type: "line", x: t, y: Array.from(F.z), color: "s1", width: 1, name: "tapered" },
             { type: "line", x: t, y: Array.from(F.w).map(v => v * m), color: "s2", width: 1.4, dash: [5, 4], hover: false },
             { type: "line", x: t, y: Array.from(F.w).map(v => -v * m), color: "s2", width: 1.4, dash: [5, 4], hover: false }];
        plot(f.canvas, { x: { label: "t (samples)" }, layers: L });
      } else if (sub === 2) {
        const f = figure("tall");
        body.append(f);
        const mx = Math.max(...F.mag), idx = [];
        for (let i = 0; i < F.mag.length; i++) if (hz(i * TAU / F.N) <= fmax) idx.push(i);
        const bands = [];
        for (let k = 1; k <= NOTE.K; k++) bands.push({ type: "band", x: [(k - 0.45) * NOTE.f0, (k + 0.45) * NOTE.f0], lo: [-200, -200], hi: [10, 10], color: k % 2 ? "band" : "band-2", alpha: 0.5, noFit: true });
        plot(f.canvas, { x: { min: 0, max: fmax, label: "frequency (Hz)" }, y: { min: -110, max: 5, label: "dB" }, layers: [
          ...bands,
          { type: "line", x: idx.map(i => hz(i * TAU / F.N)), y: idx.map(i => db(F.mag[i] / mx)), color: "s1", width: 1, name: "periodogram" },
          { type: "dots", x: F.peakIdx.map(i => hz(i * TAU / F.N)), y: F.peakIdx.map(i => db(F.mag[i] / mx)), color: "s2", r: 4, name: "grid maximum" }] });
      } else if (sub === 3) {
        const pick = segmented(F.omega.map((_, i) => [i + 1, "k = " + (i + 1)]), kSel, v => { kSel = v; show(); });
        body.append(el("div", { class: "controls" }, el("span", { class: "lbl" }, "Partial"), pick));
        const tr = F.traces[kSel - 1] || [], fin = F.omega[kSel - 1];
        const span = Math.max(3 * Math.max(...tr.map(v => Math.abs(v - fin)), Math.abs(R.om[kSel - 1] - fin)), 0.15 * TAU / T);
        const ws = linspace(fin - span, fin + span, 240), val = ws.map(w => { const d = H.dft3(F.z, w); return d[0] * d[0] + d[1] * d[1]; });
        const mx = Math.max(...val), at = w => { const d = H.dft3(F.z, w); return (d[0] * d[0] + d[1] * d[1]) / mx; };
        const f = figure("tall", `Start ${hz(tr[0]).toFixed(4)} Hz (FFT grid), end ${hz(fin).toFixed(5)} Hz after ${Math.max(0, tr.length - 1)} steps. True value ${hz(R.om[kSel - 1]).toFixed(5)} Hz (green line).`);
        body.append(f);
        plot(f.canvas, { x: { label: "frequency (Hz)", hfmt: v => v.toFixed(4) }, y: { fmt: v => v.toFixed(4) }, padL: 60, layers: [
          { type: "line", x: ws.map(hz), y: val.map(v => v / mx), color: "s1", width: 1.6, name: "|Z|²" },
          { type: "vline", x: hz(R.om[kSel - 1]), color: "s3", dash: [4, 3], width: 1.4 },
          { type: "line", x: tr.map(hz), y: tr.map(at), color: "s2", width: 1, dash: [2, 2], hover: false },
          { type: "dots", x: tr.map(hz), y: tr.map(at), color: "s2", r: 4.5, name: "Newton step" }] });
      } else if (sub === 4) {
        const n = Math.round(NOTE.fs * 0.012);
        const f1 = figure("", "Data (solid) and fitted model (dashed), first 12 ms.");
        const f2 = figure("", null);
        body.append(el("div", { class: "pair" }, f1, f2));
        plot(f1.canvas, { x: { label: "t" }, layers: [
          { type: "line", x: t.slice(0, n), y: Array.from(R.y.slice(0, n)), color: "s1", width: 1.8, name: "y" },
          { type: "line", x: t.slice(0, n), y: Array.from(F.fitted.slice(0, n)), color: "s2", width: 1.2, dash: [5, 3], name: "fit" }] });
        plot(f2.canvas, { x: { label: "t" }, layers: [{ type: "line", x: t, y: Array.from(F.resid), color: "s4", width: 0.8, name: "left-over" }] });
        f2.append(el("figcaption", {}, `Left-over over the whole frame. Its variance is ${(variance(Array.from(F.resid)) / variance(Array.from(R.y))).toExponential(2)} of the signal's.`));
      } else if (sub === 5) {
        const df = TAU / T, jmax = Math.ceil(fmax / NOTE.fs * T), js = Array.from({ length: jmax }, (_, i) => i + 1);
        const I = js.map(j => { let re = 0, im = 0; const lam = j * df; for (let k = 0; k < T; k++) { re += F.resid[k] * Math.cos(lam * k); im -= F.resid[k] * Math.sin(lam * k); } return (re * re + im * im) / (TAU * T); });
        const f = figure("tall", "Grey: periodogram of the left-over. Orange: the estimate used at each partial. Green: the true noise level.");
        body.append(f);
        plot(f.canvas, { x: { label: "frequency (Hz)", min: 0, max: fmax }, y: { log: true, label: "spectrum" }, layers: [
          { type: "dots", x: js.map(j => hz(j * df)), y: I, color: "muted", r: 1.5, alpha: 0.7, hover: false },
          { type: "hline", y: NOTE.sigma ** 2 / TAU, color: "s3", width: 1.6 },
          { type: "dots", x: F.omega.map(hz), y: F.f, color: "s2", r: 4.5, name: "estimate" }] });
      } else {
        let hits = 0;
        const rows = F.omega.map((o, i) => {
          const z = (o - R.om[i]) / F.se.omega[i], ok = Math.abs(z) <= 2; hits += ok;
          return [String(i + 1), hz(R.om[i]).toFixed(3), hz(o).toFixed(4), (1000 * hz(F.se.omega[i])).toFixed(1), z.toFixed(2), el("span", { class: ok ? "ok" : "no" }, ok ? "✓" : "✗")];
        });
        body.append(table(["k", "true (Hz)", "estimate (Hz)", "s.e. (mHz)", "error / s.e.", "within ±2 s.e."], rows, "num"),
          el("p", { class: "small" }, `${hits} of ${F.omega.length} true frequencies inside ±2 s.e. c₀ = ${H.constants(NOTE.taper).c0.toFixed(2)} for this taper. Press "New noise draw" a few times.`));
      }
    }
    run();
  },
});

/* ---------- Step 2: Theorem 2, the formula and a simulation check ---------- */
step({
  title: "Theorem 2: is the error bar right?",
  render(stage) {
    stage.append(para(`For each partial, Theorem 2 gives
      <span class="eq">var(&omega;&#770;<sub>k</sub>) &asymp; 4&pi; c<sub>0</sub> f(&omega;<sub>k</sub>) / (&rho;<sub>k</sub><sup>2</sup> T<sup>3</sup>)</span>,
      where f is the noise spectrum and c<sub>0</sub> depends only on the taper (12 with no taper, 28.1 for Hann).
      Below, one sine wave (A = 1, B = 0.5) is simulated many times with a known frequency, estimated each time, and the
      spread of the errors is compared with the formula.`));
    const st = { taper: "hann", noise: "white", T: 512, R: 1000 };
    const ctl = el("div", { class: "controls" });
    ctl.append(select("t-taper", "Taper", windowOptions(), st.taper, v => st.taper = v),
      select("t-noise", "Noise", [["white", "white"], ["ar1", "AR(1), φ = 0.6"]], st.noise, v => st.noise = v),
      select("t-T", "T", [64, 128, 256, 512, 1024].map(v => [String(v), String(v)]), String(st.T), v => st.T = +v),
      select("t-R", "Runs", [300, 1000, 3000].map(v => [String(v), String(v)]), String(st.R), v => st.R = +v));
    const go1 = el("button", { type: "button" }, "Run simulation"), go2 = el("button", { type: "button" }, "Sweep T");
    const status = el("span", { class: "small" });
    ctl.append(go1, go2, status);
    stage.append(ctl);
    const f1 = figure("", "Standardised error T^(3/2)·(ω̂ − ω) / sd from Theorem 2, against the N(0, 1) curve.");
    const out = el("div");
    stage.append(el("div", { class: "pair" }, f1, out));
    const f2 = figure("", "Simulated / Theorem-2 variance as the frame grows (300 runs per point, white noise). Below T ≈ 128 some estimates land on a noise peak and the formula no longer applies.");
    f2.style.display = "none";
    stage.append(f2);
    stage.append(el("p", { class: "small" }, `Python study: 5 tapers × 3 noise types (white, AR(1), Student t₅) × ${D.mc.R} runs at T = 1024.
      Every simulated variance is within ${(100 * D.mc.worst).toFixed(1)}% of Theorem 2, and the 95% intervals cover the truth
      ${(100 * D.mc.coverage[0]).toFixed(0)}–${(100 * D.mc.coverage[1]).toFixed(0)}% of the time once T ≥ 256.`));
    const w0 = TAU * 0.1234, A0 = 1, B0 = 0.5, rho0 = Math.hypot(A0, B0);
    function once(g, T, taper, noise) {
      const y = new Float64Array(T), e = H.noise(1, T, noise, 1, 0.6, g);
      for (let i = 0; i < T; i++) y[i] = A0 * Math.cos(w0 * (i + 1)) + B0 * Math.sin(w0 * (i + 1)) + e[i];
      return H.fitHarmonics(y, 1, { window: taper });
    }
    async function run() {
      go1.disabled = true;
      const { taper, noise, T, R } = st, g = H.rng(99 + T), f = H.spectrum(noise, 1, 0.6)(w0);
      const sd = Math.sqrt(4 * Math.PI * f * H.constants(taper).c0 / rho0 ** 2);
      const z = []; let cover = 0;
      for (let r = 0; r < R; r++) {
        const fit = once(g, T, taper, noise);
        z.push(T ** 1.5 * (fit.omega[0] - w0) / sd);
        cover += Math.abs(fit.omega[0] - w0) <= 1.96 * fit.se.omega[0];
        if (r % 50 === 49) { status.textContent = `${r + 1} / ${R}`; await sleep0(); }
      }
      status.textContent = "";
      const h = histogram(z, -4, 4, 40), xs = linspace(-4, 4, 200);
      plot(f1.canvas, { x: { min: -4, max: 4, label: "standardised error" }, y: { min: 0 }, layers: [
        { type: "hist", x: h.edges, y: h.dens, color: "s1", alpha: 0.85 },
        { type: "line", x: xs, y: xs.map(normPdf), color: "ink", width: 1.6, name: "N(0,1)" }] });
      const kept = z.filter(v => Math.abs(v) < 50);
      out.innerHTML = "";
      out.append(table(["", "value"], [
        ["simulated / Theorem-2 variance", (variance(kept)).toFixed(3)],
        ["runs far off (noise peak chosen)", String(z.length - kept.length)],
        ["coverage of ω̂ ± 1.96 s.e.", (100 * cover / R).toFixed(1) + "%"],
        ["c₀ for this taper", H.constants(taper).c0.toFixed(2)]]),
        el("p", { class: "small" }, "The s.e. in the coverage line is fully plug-in: the noise level is estimated from each simulated frame, as on a real recording."));
      go1.disabled = false;
    }
    async function sweep() {
      go2.disabled = true;
      f2.style.display = "";
      const Ts = [32, 64, 128, 256, 512, 1024, 2048], tapers = ["rectangular", "hann", "blackman"], layers = [];
      const colors = { rectangular: "s1", hann: "s2", blackman: "s3" };
      for (const taper of tapers) {
        const ratio = [];
        for (const T of Ts) {
          const g = H.rng(T + 7), f = H.spectrum("white", 1, 0)(w0), th = 4 * Math.PI * f * H.constants(taper).c0 / rho0 ** 2;
          const v = [];
          for (let r = 0; r < 300; r++) v.push(T ** 1.5 * (once(g, T, taper, "white").omega[0] - w0));
          ratio.push(v.reduce((s, x) => s + x * x, 0) / v.length / th);
          status.textContent = `${taper}, T = ${T}`; await sleep0();
        }
        layers.push({ type: "line", x: Ts, y: ratio, color: colors[taper], width: 1.6, name: H.DISPLAY[taper] },
          { type: "dots", x: Ts, y: ratio, color: colors[taper], r: 3, hover: false });
      }
      status.textContent = "";
      plot(f2.canvas, { x: { log: true, label: "frame length T" }, y: { log: true, label: "ratio", min: 0.5, max: 300 }, layers: [{ type: "hline", y: 1, color: "ink", width: 1 }, ...layers] });
      go2.disabled = false;
    }
    go1.onclick = run; go2.onclick = sweep;
    f2.append(legend([["s1", "rectangular"], ["s2", "Hann"], ["s3", "Blackman"]]));
    run();
  },
});

/* ---------- Step 3: Extension 1, every partial ---------- */
step({
  title: "Extension 1: use every partial",
  render(stage) {
    const E = D.ext1;
    stage.append(para(`For a harmonic note partial k sits at k&lambda;, so &omega;&#770;<sub>k</sub>/k is another measurement of the pitch &lambda;,
      with s.e. divided by k. Theorem 2 gives each partial's s.e. and says the errors are independent, so the best combination is
      the precision-weighted average
      <span class="eq">&lambda;&#770; = &Sigma; v<sub>k</sub> k &omega;&#770;<sub>k</sub> / &Sigma; v<sub>k</sub> k<sup>2</sup></span>, with
      v<sub>k</sub> = 1/s.e.<sub>k</sub><sup>2</sup> and s.e.(&lambda;&#770;) = (&Sigma; v<sub>k</sub> k<sup>2</sup>)<sup>&minus;1/2</sup>.
      This is weighted least squares for &omega;&#770;<sub>k</sub> = k&lambda; + error.`));
    const ctl = el("div", { class: "controls" });
    const btn = el("button", { type: "button" }, "Simulate 200 frames"), status = el("span", { class: "small" });
    ctl.append(btn, status);
    stage.append(el("h3", {}, "Simulation: a 441 Hz note with 12 partials"), ctl);
    const f1 = figure("", null), res = el("div");
    stage.append(el("div", { class: "pair" }, f1, res));
    f1.append(legend([["s2", "partial 1 only (the paper)", true], ["s1", "all partials (Extension 1)", true]]));
    async function sim() {
      btn.disabled = true;
      const K = 12, T = 1025, g = H.rng(2026 + Math.floor(Math.random() * 1000)), e1 = [], ep = [];
      let c1 = 0, cp = 0;
      for (let r = 0; r < 200; r++) {
        const om = [], A = [], B = [];
        for (let k = 1; k <= K; k++) { const ph = g.u() * TAU, a = noteAmp(k); om.push(TAU * k * 441 / 44100); A.push(a * Math.cos(ph)); B.push(-a * Math.sin(ph)); }
        const e = H.noise(1, T, "white", 0.02, 0, g), y = new Float64Array(T);
        for (let i = 0; i < T; i++) { let v = 0; for (let j = 0; j < K; j++) v += A[j] * Math.cos(om[j] * (i + 1)) + B[j] * Math.sin(om[j] * (i + 1)); y[i] = v + e[i]; }
        const F = H.fitHarmonics(y, K, { window: "hann", f0hint: om[0] });
        const fh = F.omega.map(hz), sh = F.se.omega.map(hz), p = H.pooledF0(fh, sh);
        e1.push(fh[0] - 441); ep.push(p.f0 - 441);
        c1 += Math.abs(fh[0] - 441) <= 1.96 * sh[0]; cp += Math.abs(p.f0 - 441) <= 1.96 * p.se;
        if (r % 20 === 19) { status.textContent = `${r + 1} / 200`; await sleep0(); }
      }
      status.textContent = "";
      const h1 = histogram(e1, -0.15, 0.15, 40), h2 = histogram(ep, -0.15, 0.15, 40);
      plot(f1.canvas, { x: { min: -0.15, max: 0.15, label: "pitch error (Hz)" }, y: { min: 0 }, layers: [
        { type: "hist", x: h1.edges, y: h1.dens, color: "s2", alpha: 0.55 },
        { type: "hist", x: h2.edges, y: h2.dens, color: "s1", alpha: 0.7 }] });
      const sd1 = Math.sqrt(variance(e1)), sdp = Math.sqrt(variance(ep));
      res.innerHTML = "";
      res.append(table(["", "partial 1", "all partials"], [
        ["spread of the error (mHz)", (1000 * sd1).toFixed(1), (1000 * sdp).toFixed(1)],
        ["95% intervals that cover 441 Hz", (c1 / 2).toFixed(1) + "%", (cp / 2).toFixed(1) + "%"]], "num"),
        el("p", { class: "small" }, `Gain this run: ${(sd1 / sdp).toFixed(2)}×. Formula: ${E.sim.gain_theory.toFixed(2)}×.
          With ${E.sim.R} frames in Python: ${E.sim.gain_measured.toFixed(2)}×, coverage ${(100 * E.sim.coverage_pooled).toFixed(1)}%.`));
      btn.disabled = false;
    }
    btn.onclick = sim;
    stage.append(el("h3", {}, "Real recordings (Python run, T = 1025)"));
    const tr = E.track, f2 = figure("", "Clarinet B♭4, one point per 23 ms frame. Light band: ω̂₁ ± 2 s.e. (the paper). Dark band: pooled pitch ± 2 s.e.");
    stage.append(f2);
    plot(f2.canvas, { x: { label: "time (s)" }, y: { label: "Hz", hfmt: v => v.toFixed(3) }, layers: [
      { type: "band", x: tr.t, lo: tr.f1.map((v, i) => v - 2 * tr.s1[i]), hi: tr.f1.map((v, i) => v + 2 * tr.s1[i]), color: "band-2", alpha: 0.9 },
      { type: "band", x: tr.t, lo: tr.fp.map((v, i) => v - 2 * tr.sp[i]), hi: tr.fp.map((v, i) => v + 2 * tr.sp[i]), color: "s1", alpha: 0.45 },
      { type: "line", x: tr.t, y: tr.fp, color: "s1", width: 1.2, name: "pooled pitch" }] });
    stage.append(table(["recording", "s.e. of ω̂₁ (mHz)", "pooled s.e. (mHz)", "gain", "frames failing the χ² test"],
      E.rows.map(r => [r.label, r.se_single_mhz.toFixed(1), r.se_pooled_mhz.toFixed(1), r.gain.toFixed(1) + "×", (100 * r.share_frames_chi2_rejects).toFixed(0) + "%"]), "num"),
      el("p", { class: "small" }, `The last column is the left-over χ² = Σ v_k(ω̂_k − kλ̂)², a per-frame harmonicity test at the 5% level.
        It rejects more often than 5% even for the clarinet, mainly because of very weak high partials where Theorem 2's
        error bars are too small. For the oboe (93%) the 1.7× gain should not be trusted.`));
    sim();
  },
});

/* ---------- Step 4: Extension 2, frame length ---------- */
step({
  title: "Extension 2: how long can a frame be?",
  render(stage) {
    const E = D.ext2;
    stage.append(para(`Theorem 2 makes the s.e. fall like T<sup>&minus;3/2</sup>, so longer frames look better. But it assumes the note is
      steady inside the frame. To see when that stops being true we use the clarinet's own harmonicity: z = (&omega;&#770;<sub>k</sub> &minus; k&omega;&#770;<sub>1</sub>)/s.e.
      should be standard normal, so about 4.55% of the z's should fall outside &plusmn;2. At each frame length we count them over all
      frames and strong partials, and test the count against Binomial(n, 0.0455) at the 1% level. We keep the longest length that passes,
      with every shorter one (from 768 samples) passing too.`));
    const names = { clarinet_Bb4: "Clarinet B♭4", clarinet_D4: "Clarinet D4", guitar_steel_D3: "Steel guitar D3" };
    const colors = { clarinet_Bb4: "s1", clarinet_D4: "s2", guitar_steel_D3: "s3" };
    const f = figure("tall", "Share of z outside ±2 at each frame length. Dashed: the 4.55% expected when the error bars are honest. Rings: the chosen length.");
    stage.append(f);
    const layers = [{ type: "hline", y: 4.55, color: "ink", dash: [5, 4], width: 1 }];
    for (const key of Object.keys(names)) {
      const c = E.curves[key];
      layers.push({ type: "line", x: c.map(r => r.ms), y: c.map(r => 100 * r.share), color: colors[key], width: 1.6, name: names[key] },
        { type: "dots", x: c.map(r => r.ms), y: c.map(r => 100 * r.share), color: colors[key], r: 3, hover: false });
      const best = c.find(r => r.T === E.best[key]);
      if (best) layers.push({ type: "dots", x: [best.ms], y: [100 * best.share], color: "ink", r: 6, alpha: 0.35, hover: false });
    }
    plot(f.canvas, { x: { label: "frame length (ms)" }, y: { log: true, min: 1.5, max: 100, label: "% outside ±2", ticks: [2, 5, 10, 20, 50, 100] }, layers });
    f.append(legend(Object.keys(names).map(k => [colors[k], names[k]])));
    let key = "clarinet_D4";
    const pick = segmented(Object.keys(names).map(k => [k, names[k]]), key, v => { key = v; fill(); });
    const holder = el("div");
    stage.append(el("div", { class: "controls" }, el("span", { class: "lbl" }, "Details for"), pick), holder);
    function fill() {
      const c = E.curves[key];
      holder.innerHTML = "";
      holder.append(table(["T", "ms", "frames", "z values", "outside ±2", "p", "pass", "s.e. ω̂₁ (mHz)", "pooled s.e. (mHz)"],
        c.map(r => [String(r.T), r.ms.toFixed(0), String(r.n_frames), String(r.n), (100 * r.share).toFixed(1) + "%",
          r.p < 1e-4 ? "<0.0001" : r.p.toFixed(4), el("span", { class: r.p > 0.01 ? "ok" : "no" }, r.p > 0.01 ? "yes" : "no"),
          r.se_single_mhz.toFixed(1), r.se_pooled_mhz.toFixed(2)]), "num"),
        el("p", { class: "small" }, E.best[key] ? `Chosen length: T = ${E.best[key]} (${(1000 * E.best[key] / 44100).toFixed(0)} ms).` :
          "No length passes: the plucked string fades inside every frame and its partials are not exact multiples."));
    }
    fill();
    stage.append(el("h3", {}, "Both extensions together"),
      table(["", "paper (ω̂₁, T = 1025)", "+ Extension 1", "+ Extensions 1 and 2"], [
        ["Clarinet B♭4", E.combined.clarinet_Bb4.paper_mhz.toFixed(0) + " mHz", E.combined.clarinet_Bb4.ext1_mhz.toFixed(0) + " mHz", `${E.combined.clarinet_Bb4.both_mhz.toFixed(0)} mHz (T = ${E.combined.clarinet_Bb4.ext2_T})`],
        ["Clarinet D4", E.combined.clarinet_D4.paper_mhz.toFixed(0) + " mHz", E.combined.clarinet_D4.ext1_mhz.toFixed(0) + " mHz", `${E.combined.clarinet_D4.both_mhz.toFixed(1)} mHz (T = ${E.combined.clarinet_D4.ext2_T}), ${E.combined.clarinet_D4.total_gain.toFixed(0)}× smaller`]], "num"),
      el("p", { class: "small" }, `A smooth pitch glide shared by all partials does not trigger the check: in simulation with glides of 0, 1 and 4 Hz/s
        the share outside ±2 stayed between ${(100 * E.glide_min).toFixed(0)}% and ${(100 * E.glide_max).toFixed(0)}%.`));
  },
});

/* ---------- Step 5: Extension 3, strings ---------- */
step({
  title: "Extension 3: a guitar",
  render(stage) {
    const E = D.ext3;
    stage.append(para(`A guitar string is stiff, so its partials are not exact multiples and Extensions 1 and 2 do not apply. We let every
      partial sit a fixed number of cents away from k&lambda;: <span class="eq">log(&omega;&#770;<sub>ik</sub>/k) = log &lambda;<sub>i</sub> + &gamma;<sub>k</sub> + error</span>
      for frame i and partial k, with &gamma;<sub>1</sub> = 0 and Theorem-2 weights. In R this is
      <code>lm(log(f/k) ~ 0 + factor(frame) + factor(partial), weights = (f/se)^2)</code>. The offsets &gamma;<sub>k</sub> soak up stiffness and anything
      else that stays the same over time; the frame effects &lambda;<sub>i</sub> are the pitch.`));
    const ctl = el("div", { class: "controls" });
    ctl.append(el("span", { class: "lbl" }, "Listen"),
      el("button", { type: "button", onclick: () => playSamples(AUDIO.guitar.x, AUDIO.guitar.fs) }, "steel guitar D3"),
      el("button", { type: "button", onclick: () => playSamples(AUDIO.clarinet.x, AUDIO.clarinet.fs) }, "clarinet B♭4"),
      el("button", { type: "button", onclick: stopAudio }, "stop"));
    stage.append(ctl);
    const k = E.offsets.steel.map((_, i) => i + 1);
    const f1 = figure("", `Estimated offsets ± 2 s.e. Dotted: a stiff string with B = ${E.B_steel.toExponential(1)}. Dashed: the same with partial 1 sitting ${E.shift_steel.toFixed(1)} cents flat.`);
    const kq = linspace(1, 12, 60), CENTS = 1200 / Math.log(2);
    plot(f1.canvas, { x: { label: "partial k", min: 0.5, max: 12.5 }, y: { label: "cents" }, layers: [
      { type: "line", x: kq, y: kq.map(q => CENTS / 2 * E.B_steel * (q * q - 1)), color: "muted", dash: [2, 3], width: 1.2, hover: false },
      { type: "line", x: kq, y: kq.map(q => E.shift_steel + CENTS / 2 * E.B_steel * (q * q - 1)), color: "muted", dash: [6, 4], width: 1.2, hover: false },
      { type: "errbar", x: k.map(v => v - 0.1), lo: E.offsets.steel.map((v, i) => v - 2 * E.offsets.steel_se[i]), hi: E.offsets.steel.map((v, i) => v + 2 * E.offsets.steel_se[i]), color: "s1" },
      { type: "dots", x: k.map(v => v - 0.1), y: E.offsets.steel, color: "s1", r: 3.5, name: "steel" },
      { type: "errbar", x: k.map(v => v + 0.1), lo: E.offsets.nylon.map((v, i) => v - 2 * E.offsets.nylon_se[i]), hi: E.offsets.nylon.map((v, i) => v + 2 * E.offsets.nylon_se[i]), color: "s2" },
      { type: "dots", x: k.map(v => v + 0.1), y: E.offsets.nylon, color: "s2", r: 3.5, name: "nylon" }] });
    f1.append(legend([["s1", "steel guitar"], ["s2", "nylon guitar"]]));
    const f2 = figure("", "Share of z outside ±2 (T = 3000) when the reference is exact harmonics, the stiff-string law, or the offset model.");
    const groups = ["harmonic", "stiff", "offsets"], series = [["sim_fixed", "simulated, fixed offsets", "s3"], ["sim_wander", "simulated, wandering offsets", "s4"], ["steel", "steel guitar", "s1"]];
    const layers = [{ type: "hline", y: 4.55, color: "ink", dash: [5, 4], width: 1 }];
    series.forEach(([key, name, c], j) => layers.push({ type: "bars", x: groups.map((_, i) => i + (j - 1) * 0.27), y: groups.map(g => 100 * E.checks[key][g]), width: 0.25, color: c, name }));
    plot(f2.canvas, { x: { min: -0.6, max: 2.6, ticks: [0, 1, 2], fmt: v => ["exact harmonics", "stiff string", "offsets"][Math.round(v)] || "" }, y: { min: 0, max: 100, label: "% outside ±2" }, layers });
    f2.append(legend(series.map(([, n, c]) => [c, n, true])));
    stage.append(el("div", { class: "pair" }, f1, f2));
    stage.append(table(["", "steel guitar", "nylon guitar"], [
      ["outside ±2: exact harmonics", pc(E.checks.steel.harmonic), pc(E.checks.nylon.harmonic)],
      ["outside ±2: stiff string", pc(E.checks.steel.stiff), pc(E.checks.nylon.stiff)],
      ["outside ±2: offset model", pc(E.checks.steel.offsets), pc(E.checks.nylon.offsets)],
      ["s.e. of ω̂₁ (mHz)", E.se.steel.omega1.toFixed(1), E.se.nylon.omega1.toFixed(1)],
      ["s.e. of pitch, offset model (mHz)", E.se.steel.model.toFixed(1), E.se.nylon.model.toFixed(1)],
      ["same, widened by √(χ²/df) (mHz)", E.se.steel.inflated.toFixed(1), E.se.nylon.inflated.toFixed(1)]], "num"));
    stage.append(para(`<b>What this shows.</b> In simulation, when the offsets really are fixed, the check sees ${pc(E.checks.sim_fixed.offsets)} outside ±2
      and the pitch intervals cover the truth ${pc(E.sim_coverage)} of the time, with a ${E.sim_gain.toFixed(1)}× smaller spread than ω̂₁.
      On both real guitars the offset model removes most of the misfit (${pc(E.checks.steel.harmonic)} → ${pc(E.checks.steel.offsets)} for steel),
      but not all of it: a few partials (4, 7, 8, 9, 11 on the steel guitar) drift relative to the others from frame to frame,
      which looks like the simulated "wandering offsets" case. So the guitar's pitch s.e. should be widened, and no frame length passes the check.
      Letting each partial fade inside the frame shrinks the s.e. further but makes the misfit worse (${pc(E.decay_offsets)} outside), so decay is not the missing piece.`));
  },
});
const pc = v => (100 * v).toFixed(0) + "%";

/* ---------- page shell ---------- */
let AUDIO = {};
function show(i) {
  CURRENT = Math.max(0, Math.min(STEPS.length - 1, i));
  stopAudio();
  document.querySelectorAll("#nav button").forEach((b, j) => b.setAttribute("aria-current", String(j === CURRENT)));
  const stage = $("#stage");
  stage.innerHTML = "";
  stage.append(el("h2", {}, `${CURRENT + 1}. ${STEPS[CURRENT].title}`));
  STEPS[CURRENT].render(stage);
  const prev = STEPS[CURRENT - 1], nxt = STEPS[CURRENT + 1];
  stage.append(el("div", { class: "pager" },
    prev ? el("button", { type: "button", onclick: () => show(CURRENT - 1) }, `← ${prev.title}`) : el("span"),
    nxt ? el("button", { type: "button", onclick: () => show(CURRENT + 1) }, `${nxt.title} →`) : el("span")));
  try { history.replaceState(null, "", "#" + (CURRENT + 1)); } catch (e) { /* file:// in some browsers */ }
  window.scrollTo(0, 0);
}
window.addEventListener("DOMContentLoaded", () => {
  readColors();
  for (const [name, b64] of Object.entries(D.audio)) AUDIO[name] = decodeWav(b64);
  $("#nav").append(...STEPS.map((s, i) => el("button", { type: "button", onclick: () => show(i) }, `${i + 1}. ${s.title.split(":")[0]}`)));
  document.addEventListener("keydown", e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") show(CURRENT + 1);
    if (e.key === "ArrowLeft") show(CURRENT - 1);
  });
  const start = parseInt((location.hash || "#1").slice(1), 10);
  show(isFinite(start) ? start - 1 : 0);
});
