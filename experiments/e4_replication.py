"""E4. The paper's Section 4 on library recordings.

The paper's own recordings were never released, so we use a clarinet B-flat4
(stands in for the paper's clarinet A) and a steel-string guitar D3 (the
same note as the paper), with the paper's frame lengths and K.

  Q1  Is the clarinet out of tune?  omega_hat_1 per frame with +-2 s.e.,
      against the nominal pitch and the ear's +-0.03 semitone band.
  Q2  Are the partials harmonic?    d_k = omega_hat_k - k omega_hat_1 with
      +-2 s.e. limits, for the clarinet and the guitar.
  Also: the paper's full non-linear WLS (its nls() fit) against the faster
      periodogram estimator, which Section 2.1 says should agree.
"""
from common import *  # noqa: F401,F403
import hwls

out = {}

# Clarinet: Q1 and Q2
cfg = RECORDINGS["clarinet_Bb4"]
x, fs = load_recording("clarinet_Bb4")
cl = hwls.analyse_recording(x, fs, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"], n_frames=cfg["n_frames"])
cl_wls = hwls.analyse_recording(x, fs, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"],
                                n_frames=cfg["n_frames"], method="wls")
nom, jnd = cfg["nominal"], hwls.jnd_hz(cfg["nominal"])
f1, s1 = cl.freq_hz[:, 0], cl.freq_se_hz[:, 0]
d, dse = hwls.harmonic_deviations(cl.freq_hz, cl.freq_se_hz)
fit0 = cl.fits[0]
out["clarinet"] = dict(
    n_frames=len(f1), f1_range=[f1.min(), f1.max()], f1_se_median=np.median(s1),
    mean_cents_off=float(hwls.cents(f1.mean(), nom)),
    share_frames_off_nominal=float(np.mean(np.abs(f1 - nom) > 2 * s1)),
    share_intervals_inside_ear_band=float(np.mean((f1 + 2 * s1 <= nom + jnd) & (f1 - 2 * s1 >= nom - jnd))),
    jnd_hz=jnd,
    share_deviations_outside_2se=float(np.mean(np.abs(d[:, 1:]) > 2 * dse[:, 1:])),
    residual_over_signal_variance_frame1=float(np.var(fit0.residuals) / np.mean(fit0.y ** 2)),
    wls_vs_periodogram_max_diff_in_se=float(np.max(np.abs(cl_wls.freq_hz - cl.freq_hz) / cl.freq_se_hz)),
    wls_vs_periodogram_median_diff_hz=float(np.median(np.abs(cl_wls.freq_hz - cl.freq_hz))),
    first_five_frames_hz=cl.freq_hz[:5], first_five_frames_se_hz=cl.freq_se_hz[:5],
)

# Guitar: Q2
cfg_g = RECORDINGS["guitar_steel_D3"]
xg, fsg = load_recording("guitar_steel_D3")
gt = hwls.analyse_recording(xg, fsg, cfg_g["T"], cfg_g["K"], cfg_g["nominal"], start_s=cfg_g["start"],
                            n_frames=cfg_g["n_frames"])
dg, dgse = hwls.harmonic_deviations(gt.freq_hz, gt.freq_se_hz)
out["guitar"] = dict(
    n_frames=len(gt.times), share_deviations_outside_2se=float(np.mean(np.abs(dg[:, 1:]) > 2 * dgse[:, 1:])),
    median_deviation_k12_hz=float(np.median(dg[:, -1])), f1_se_median=float(np.median(gt.freq_se_hz[:, 0])),
)
for key, val in out.items():
    print(key, {k: v for k, v in val.items() if not isinstance(v, (list, np.ndarray))})
save_json("e4_replication", out)

# Figure: one clarinet frame and its tapered periodogram (the model in a picture)
fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.55), gridspec_kw=dict(width_ratios=[1, 1.3]))
tms = np.arange(cfg["T"]) / fs * 1000
ax[0].plot(tms, fit0.y, color=C[0], lw=1, label="recording")
ax[0].plot(tms, fit0.fitted(), color=C[1], lw=0.8, ls="--", label=f"fitted model, K = {cfg['K']}")
ax[0].set_xlabel("time (ms)")
ax[0].set_title("(a) one 23 ms frame of the clarinet")
ax[0].legend(loc="lower right", ncol=2)
N = 1 << 15
P = np.abs(np.fft.rfft(hwls.taper_weights("hann", cfg["T"]) * fit0.y, N)) ** 2
freq = np.fft.rfftfreq(N, 1 / fs)
ax[1].plot(freq / nom, 10 * np.log10(P / P.max()), color=C[0], lw=0.7)
ax[1].plot(cl.freq_hz[0] / nom, 20 * np.log10(fit0.rho / fit0.rho.max()), "o", color=C[1], ms=3)
ax[1].set_xlim(0, cfg["K"] + 1)
ax[1].set_ylim(-110, 5)
ax[1].set_xlabel(f"frequency / {nom} Hz")
ax[1].set_ylabel("dB")
ax[1].set_title("(b) Hann-tapered periodogram; dots = estimated partials")
save_fig(fig, "e4_clarinet_frame")

# Figure: Q1 and Q2
fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.8), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
tc = cl.times - cl.times[0] + cfg["T"] / fs / 2
ax[0].axhspan(nom - jnd, nom + jnd, color=BAND2, lw=0, label="ear cannot tell apart (±0.03 semitone)")
ax[0].axhline(nom, color=C[1], lw=0.8, ls="--")
ax[0].fill_between(tc, f1 - 2 * s1, f1 + 2 * s1, color=BAND, lw=0, label=r"$\hat\omega_1 \pm 2$ s.e.")
ax[0].plot(tc, f1, "o-", color=C[0], ms=2, lw=0.9)
ax[0].set_ylim(nom - 0.9, nom + 1.9)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("Hz")
ax[0].set_title(r"(a) Q1: clarinet pitch per frame")
ax[0].legend(loc="upper right", ncol=1, fontsize=6)
zc = (d / dse)[:, 1:]
zg = (dg / dgse)[:, 1:]
rng = np.random.default_rng(0)
for kk in range(2, cfg["K"] + 1):
    ax[1].plot(kk + rng.uniform(-0.2, 0.2, len(zc)), zc[:, kk - 2], ".", color=C[0], ms=1.5, alpha=0.6)
ax[1].axhspan(-2, 2, color=LIGHT, zorder=0)
ax[1].set_ylim(-6, 6)
ax[1].set_xlabel("partial k")
ax[1].set_ylabel("z = deviation / s.e.")
ax[1].set_title("(b) Q2: clarinet")
for kk in range(2, cfg_g["K"] + 1):
    ax[2].plot(kk + rng.uniform(-0.2, 0.2, len(zg)), zg[:, kk - 2], ".", color=C[1], ms=1.5, alpha=0.6)
ax[2].axhspan(-2, 2, color=LIGHT, zorder=0)
ax[2].set_yscale("symlog", linthresh=2)
ax[2].set_yticks([-10, -2, 0, 2, 10], ["-10", "-2", "0", "2", "10"])
ax[2].set_ylim(-12, 30)
ax[2].set_xlabel("partial k")
ax[2].set_title("(c) Q2: steel guitar")
save_fig(fig, "e4_q1_q2")
