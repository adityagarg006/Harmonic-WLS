"""E7. Extension 3: pooling the partials of a guitar.

Extensions 1 and 2 need a harmonic note. A guitar string is not harmonic,
so pooling under omega_k = k lambda is biased and the frame-length check
rejects every length. Two ways to model the guitar's partials:

  stiff string    omega_k = k lambda sqrt(1 + B k^2), fitted in every frame
                  (the paper's eq. 18 points this way; Fletcher and Rossing)
  partial offsets log(omega_ik / k) = log(lambda_i) + gamma_k, a weighted
                  two-way model over frames x partials: every partial may sit
                  a fixed number of cents away from k lambda, whatever the
                  reason, as long as the offset does not change over time.

 (a) simulation with a known truth: does the offset model give honest
     error bars when its assumption holds, and does the check catch it when
     the offsets wander over time?
 (b) the two real guitars at the paper's frame length (T = 3000)
 (c) the frame-length check with the offset model as reference (steel guitar)
 (d) the same with the fading-partial fit (method "wls-decay")
 (e) a clarinet: the offset model should find offsets of about 0
 (f) why the guitar needs the joint fit: spread of omega_hat_k over its
     Theorem-2 s.e., two-step estimator against joint WLS, in simulation
"""
from common import *  # noqa: F401,F403
import hwls
from hwls.simulate import synthetic_note

out = {}
CENTS = 1200 / np.log(2)
# For the guitar we use the paper's full weighted least squares fit (its nls()
# step) rather than the two-step periodogram estimator. A guitar note is very
# clean, and at that signal-to-noise ratio the strong first partial leaks into
# its neighbours' periodogram peaks enough to matter: in simulation the
# two-step estimator gives partial 2 a spread 1.8 times its Theorem-2 s.e.,
# the joint fit 1.0 times.
METHOD = "wls"


def three_checks(rec):
    """Share of z outside +-2 for the three references: exact harmonics,
    the stiff-string law, and the partial-offset model."""
    use = hwls.strong_partials(rec)
    use &= use[:, :1]
    model = hwls.partial_offset_model(rec.freq_hz, rec.freq_se_hz, use)
    return dict(harmonic=hwls.count_outside(hwls.harmonic_z(rec)),
                stiff=hwls.count_outside(hwls.stiff_z(rec)),
                offsets=hwls.count_outside(model["z"])), model


# (b) real guitars at T = 3000
real = {}
for key in ["guitar_steel_D3", "guitar_nylon_D3"]:
    cfg = RECORDINGS[key]
    x, fs = load_recording(key)
    rec = hwls.analyse_recording(x, fs, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"],
                                 n_frames=cfg["n_frames"], method=METHOD)
    checks, model = three_checks(rec)
    k = np.arange(1, cfg["K"] + 1)
    stiff_B = [hwls.stiff_string_fit(rec.freq_hz[i], rec.freq_se_hz[i], k)["B"] for i in range(len(rec.times))]
    # stiffness from the offsets: in cents, 1200 log2 sqrt(1 + B k^2) is about (CENTS / 2) B k^2.
    # Fit gamma_k = a + (CENTS / 2) B k^2 over k >= 2 by weighted least squares, so that a
    # measures how far partial 1 sits from the line through the others.
    g, gse = model["offset_cents"][1:], model["offset_se_cents"][1:]
    X = np.column_stack([np.ones(len(g)), CENTS / 2 * k[1:] ** 2])
    W = 1 / gse ** 2
    coef = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * g))
    # the line evaluated at k = 1 is where partial 1 "should" be; gamma_1 = 0 is where it is
    infl = np.sqrt(max(1.0, model["chi2"] / model["df"]))
    real[key] = dict(
        checks={m: dict(share=c["share"], n=c["n"], p=c["p"]) for m, c in checks.items()},
        se_omega1_mhz=1000 * np.median(rec.freq_se_hz[:, 0]),
        se_pitch_offset_model_mhz=1000 * np.nanmedian(model["pitch_se"]),
        chi2_over_df=model["chi2"] / model["df"], inflation=infl,
        se_pitch_inflated_mhz=1000 * np.nanmedian(model["pitch_se"]) * infl,
        offsets_cents=model["offset_cents"], offsets_se_cents=model["offset_se_cents"],
        B_stiff_median=float(np.median(stiff_B)), B_from_offsets=float(coef[1]),
        partial1_shift_cents=float(coef[0] + CENTS / 2 * coef[1]),
        pitch=model["pitch"], pitch_se=model["pitch_se"], f1=rec.freq_hz[:, 0], f1_se=rec.freq_se_hz[:, 0],
        times=rec.times)
    use = hwls.strong_partials(rec)
    use &= use[:, :1]
    zfull = np.full(use.shape, np.nan)
    for (i, j), zz in zip(model["where"], model["z"]):
        zfull[i, j] = zz
    real[key]["share_outside_by_partial"] = [float(np.nanmean(np.abs(zfull[:, j]) > 2)) if np.any(use[:, j]) else None
                                             for j in range(cfg["K"])]
    print(key, {m: round(100 * c["share"], 1) for m, c in checks.items()},
          f"s.e. omega1 {real[key]['se_omega1_mhz']:.1f}  offset-model {real[key]['se_pitch_offset_model_mhz']:.1f}"
          f"  (x{infl:.2f} -> {real[key]['se_pitch_inflated_mhz']:.1f}) mHz,  B {coef[1]:.2e},"
          f"  partial 1 {real[key]['partial1_shift_cents']:.2f} cents flat")
    print("   offsets (cents):", np.round(model["offset_cents"], 2))
    print("   % outside by partial:", [None if v is None else round(100 * v) for v in real[key]["share_outside_by_partial"]])
out["real"] = real

# (a) simulation calibrated on the steel guitar
steel = real["guitar_steel_D3"]
cfg = RECORDINGS["guitar_steel_D3"]
x, fs = load_recording("guitar_steel_D3")
rec = hwls.analyse_recording(x, fs, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"],
                             n_frames=cfg["n_frames"], method=METHOD)
amps = np.median(rec.rho, axis=0)
sigma = float(np.sqrt(2 * np.pi * np.median([f.f_noise for f in rec.fits])))
offsets = np.array(steel["offsets_cents"])
F0, T, K, NF, GLIDE = float(np.median(rec.freq_hz[:, 0])), cfg["T"], cfg["K"], cfg["n_frames"], 0.07
R = 10 if FAST else 40
sim = {}
for scenario, wander in [("fixed offsets", 0.0), ("wandering offsets", 0.1)]:
    rng = np.random.default_rng(7 if wander == 0 else 8)
    shares = {m: [] for m in ("harmonic", "stiff", "offsets")}
    err1, errl, cov1, covl = [], [], [], []
    for _ in range(R):
        off = offsets.copy()
        freq, se, rho, rho_se, truth = [], [], [], [], []
        for i in range(NF):
            if wander > 0:
                off[1:] += wander * rng.standard_normal(K - 1)       # random walk, cents per frame
            y, f_true = synthetic_note(rng, T, fs, F0, amps, sigma, offsets_cents=off,
                                       glide_hz_per_s=GLIDE, t0_s=i * T / fs)
            fit = hwls.fit_frame(y, K, hwls.hz_to_rad(f_true[0], fs), method=METHOD)
            freq.append(hwls.rad_to_hz(fit.omega, fs))
            se.append(hwls.rad_to_hz(fit.se["omega"], fs))
            rho.append(fit.rho)
            rho_se.append(fit.se["rho"])
            truth.append(f_true[0])
        r = hwls.Recording(fs=fs, T=T, K=K, times=np.arange(NF) * T / fs, freq_hz=np.array(freq),
                           freq_se_hz=np.array(se), rho=np.array(rho), rho_se=np.array(rho_se))
        checks, model = three_checks(r)
        for m in shares:
            shares[m].append(checks[m]["share"])
        truth = np.array(truth)
        err1.extend(r.freq_hz[:, 0] - truth)
        errl.extend(model["pitch"] - truth)
        cov1.extend(np.abs(r.freq_hz[:, 0] - truth) <= 1.96 * r.freq_se_hz[:, 0])
        covl.extend(np.abs(model["pitch"] - truth) <= 1.96 * model["pitch_se"])
    err1, errl = np.array(err1), np.array(errl)
    sim[scenario] = dict(share_outside={m: float(np.mean(v)) for m, v in shares.items()},
                         gain=float(err1.std() / errl.std()), coverage_omega1=float(np.mean(cov1)),
                         coverage_pitch=float(np.mean(covl)), sd_omega1_mhz=1000 * err1.std(),
                         sd_pitch_mhz=1000 * errl.std())
    print("simulation,", scenario, {m: round(100 * v, 1) for m, v in sim[scenario]["share_outside"].items()},
          f"gain {sim[scenario]['gain']:.2f}  coverage omega1 {sim[scenario]['coverage_omega1']:.3f}"
          f"  pitch {sim[scenario]['coverage_pitch']:.3f}")
out["simulation"] = dict(R=R, frames=NF, T=T, sigma=sigma, glide_hz_per_s=GLIDE, results=sim)

# (c) frame-length check with the offset model as reference (steel guitar)
LENGTHS = [1025, 1280, 1536, 2050, 2560, 3000, 4100, 5000, 6000]
rows = hwls.frame_length_check(x, fs, cfg["K"], cfg["nominal"], cfg["start"], 30 * 3000, LENGTHS,
                               z_function=hwls.offset_model_z, method=METHOD)
scan = []
for r in rows:
    rec_T = r.pop("rec")
    r["share_harmonic"] = hwls.count_outside(hwls.harmonic_z(rec_T))["share"]
    scan.append(r)
    print(f"   T = {r['T']:5d}: outside +-2  harmonic {100 * r['share_harmonic']:5.1f}%   offsets {100 * r['share']:5.1f}%  p = {r['p']:.2g}")
out["frame_length_scan_steel"] = dict(rows=scan, longest_honest_T=hwls.longest_honest_frame(scan, shortest=1025))

# (d) fading partials (steel guitar, T = 3000)
rec_d = hwls.analyse_recording(x, fs, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"],
                               n_frames=cfg["n_frames"], method="wls-decay")
checks_d, model_d = three_checks(rec_d)
drop_db = 20 / np.log(10) * np.median([f.decay for f in rec_d.fits], axis=0)
out["decay_model_steel"] = dict(checks={m: c["share"] for m, c in checks_d.items()},
                                se_omega1_mhz=1000 * np.median(rec_d.freq_se_hz[:, 0]),
                                se_pitch_offset_model_mhz=1000 * np.nanmedian(model_d["pitch_se"]),
                                amplitude_drop_db_per_frame=drop_db)
print("decay model:", {m: round(100 * v, 1) for m, v in out["decay_model_steel"]["checks"].items()},
      f"s.e. omega1 {out['decay_model_steel']['se_omega1_mhz']:.1f} mHz,",
      "amplitude drop per frame (dB):", np.round(drop_db, 1))

# (e) clarinet sanity check
cfg_c = RECORDINGS["clarinet_Bb4"]
xc, fsc = load_recording("clarinet_Bb4")
rec_c = hwls.analyse_recording(xc, fsc, cfg_c["T"], cfg_c["K"], cfg_c["nominal"], start_s=cfg_c["start"], n_frames=cfg_c["n_frames"])
use = hwls.strong_partials(rec_c)
use &= use[:, :1]
model_c = hwls.partial_offset_model(rec_c.freq_hz, rec_c.freq_se_hz, use)
out["clarinet_Bb4"] = dict(max_abs_offset_cents=float(np.max(np.abs(model_c["offset_cents"]))),
                           share_outside=hwls.count_outside(model_c["z"])["share"],
                           se_pitch_mhz=1000 * np.nanmedian(model_c["pitch_se"]))
print("clarinet:", out["clarinet_Bb4"])

# (f) two-step periodogram estimator against the joint WLS fit, one frame at a time
rng = np.random.default_rng(11)
ratio = {}
for method in ["periodogram", "wls"]:
    z = []
    for _ in range(100 if FAST else 200):
        y, f_true = synthetic_note(rng, T, fs, F0, amps, sigma, offsets_cents=offsets)
        fit = hwls.fit_frame(y, K, hwls.hz_to_rad(f_true[0], fs), method=method)
        z.append((hwls.rad_to_hz(fit.omega, fs) - f_true) / hwls.rad_to_hz(fit.se["omega"], fs))
    ratio[method] = np.array(z).std(axis=0)
    print(f"spread / s.e. by partial, {method:11s}:", np.round(ratio[method], 2))
out["two_step_vs_joint_fit"] = ratio
save_json("e7_ext3_strings", out)

# Figure
fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.95), gridspec_kw=dict(width_ratios=[1.1, 1, 1.1]))
kk = np.arange(1, cfg["K"] + 1)
for i, key in enumerate(["guitar_steel_D3", "guitar_nylon_D3"]):
    o, ose = np.array(real[key]["offsets_cents"]), np.array(real[key]["offsets_se_cents"])
    ax[0].errorbar(kk + 0.12 * (2 * i - 1), o, yerr=2 * ose, fmt="o", color=C[i], ms=2.8, elinewidth=0.8,
                   label=RECORDINGS[key]["label"])
kq = np.linspace(1, 12, 100)
ax[0].plot(kq, (CENTS / 2) * steel["B_from_offsets"] * (kq ** 2 - 1), color=GREY, lw=0.8, ls=":",
           label="stiff string only")
ax[0].plot(kq, steel["partial1_shift_cents"] + (CENTS / 2) * steel["B_from_offsets"] * (kq ** 2 - 1), color=GREY,
           lw=0.8, ls="--", label="stiff string, partial 1 flat")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_xticks([1, 3, 5, 7, 9, 11])
ax[0].set_xlabel("partial k")
ax[0].set_ylabel("cents")
ax[0].set_title(r"(a) offsets $\hat\gamma_k$ $\pm 2$ s.e.")
ax[0].set_ylim(-0.5, 8.5)
ax[0].legend(loc="upper left", fontsize=5.8)
labels = ["exact\nharmonics", "stiff\nstring", "partial\noffsets"]
xpos = np.arange(3)
for i, (name, vals) in enumerate([("simulated, fixed offsets", sim["fixed offsets"]["share_outside"]),
                                  ("simulated, wandering", sim["wandering offsets"]["share_outside"]),
                                  ("steel guitar", {m: c["share"] for m, c in real["guitar_steel_D3"]["checks"].items()})]):
    ax[1].bar(xpos + (i - 1) * 0.27, [100 * vals[m] for m in ("harmonic", "stiff", "offsets")], 0.25,
              color=[C[2], C[3], C[1]][i], label=name)
ax[1].axhline(4.55, color="black", lw=0.8, ls="--")
ax[1].set_xticks(xpos, labels)
ax[1].set_ylabel("% of z outside ±2")
ax[1].set_title("(b) the check, T = 3000")
ax[1].set_ylim(0, 135)
ax[1].legend(loc="upper right", fontsize=5.8)
ax[1].grid(axis="x", visible=False)
ms = [r["ms"] for r in scan]
ax[2].plot(ms, [100 * r["share_harmonic"] for r in scan], "o-", color=GREY, label="reference: exact harmonics")
ax[2].plot(ms, [100 * r["share"] for r in scan], "o-", color=C[1], label="reference: partial offsets")
ax[2].axhline(4.55, color="black", lw=0.8, ls="--")
ax[2].set_ylim(0, 100)
ax[2].set_ylabel("% of z outside ±2")
ax[2].set_xlabel("frame length (ms)")
ax[2].set_title("(c) steel guitar, all lengths")
ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.75), fontsize=5.8)
save_fig(fig, "e7_ext3")
