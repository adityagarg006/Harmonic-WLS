"""E5. Extension 1: use every partial to measure the pitch.

 (a) Simulation with a known truth: a clarinet-like note at exactly 441 Hz
     (12 partials, odd ones strong, random phases, white noise), T = 1025,
     Hann. Is the pooled pitch really sharper, by the amount the formula
     predicts, and do its 95% intervals still cover the truth 95% of the time?
 (b) Real recordings: clarinet B-flat4 and D4, flute, oboe.
     Median s.e. of omega_hat_1 (the paper) and of the pooled pitch, and the
     per-frame chi-square harmonicity test.
"""
from common import *  # noqa: F401,F403
import hwls
from hwls.simulate import synthetic_note

out = {}

# (a) simulation
fs, T, K, F0, SIGMA = 44100, 1025, 12, 441.0, 0.02
k = np.arange(1, K + 1)
amps = np.where(k % 2 == 1, 1.0, 0.06) / k ** 1.1          # odd partials strong, like a clarinet
R = 500 if FAST else 2000
rng = np.random.default_rng(2026)
err1, errp, se1, sep = [], [], [], []
for _ in range(R):
    y, f_true = synthetic_note(rng, T, fs, F0, amps, SIGMA)
    fit = hwls.fit_frame(y, K, hwls.hz_to_rad(F0, fs))
    f_hat, se_hat = hwls.rad_to_hz(fit.omega, fs), hwls.rad_to_hz(fit.se["omega"], fs)
    p = hwls.pooled_pitch(f_hat, se_hat)
    err1.append(f_hat[0] - F0); se1.append(se_hat[0])
    errp.append(p["pitch"] - F0); sep.append(p["se"])
err1, errp, se1, sep = map(np.array, (err1, errp, se1, sep))
gain_theory = np.sqrt(np.sum(k ** 2 * amps ** 2)) / amps[0]          # white noise: f is the same at every k
out["simulation"] = dict(
    R=R, sd_single_mhz=1000 * err1.std(), sd_pooled_mhz=1000 * errp.std(),
    gain_measured=err1.std() / errp.std(), gain_theory=gain_theory,
    coverage_single=float(np.mean(np.abs(err1) <= 1.96 * se1)),
    coverage_pooled=float(np.mean(np.abs(errp) <= 1.96 * sep)),
    bias_pooled_mhz=1000 * errp.mean())
print("simulation:", {k_: round(v, 4) for k_, v in out["simulation"].items()})

# (b) real recordings
real = []
tracks = {}
for key in ["clarinet_Bb4", "clarinet_D4", "flute_A5", "oboe_Bb5"]:
    cfg = RECORDINGS[key]
    x, fs_ = load_recording(key)
    rec = hwls.analyse_recording(x, fs_, cfg["T"], cfg["K"], cfg["nominal"], start_s=cfg["start"], n_frames=cfg["n_frames"])
    pooled = [hwls.pooled_pitch(rec.freq_hz[i], rec.freq_se_hz[i]) for i in range(len(rec.times))]
    row = dict(recording=key, K=cfg["K"], se_single_mhz=1000 * np.median(rec.freq_se_hz[:, 0]),
               se_pooled_mhz=1000 * np.median([p["se"] for p in pooled]),
               share_frames_chi2_rejects=float(np.mean([p["p"] < 0.05 for p in pooled])))
    row["gain"] = row["se_single_mhz"] / row["se_pooled_mhz"]
    real.append(row)
    tracks[key] = dict(t=rec.times - rec.times[0] + cfg["T"] / fs_ / 2, f1=rec.freq_hz[:, 0], s1=rec.freq_se_hz[:, 0],
                       fp=np.array([p["pitch"] for p in pooled]), sp=np.array([p["se"] for p in pooled]))
    print(f"{key:13s} s.e. {row['se_single_mhz']:6.1f} -> {row['se_pooled_mhz']:5.1f} mHz  "
          f"gain {row['gain']:.2f}  chi2 rejects in {100 * row['share_frames_chi2_rejects']:.0f}% of frames")
out["recordings"] = real
out["track_clarinet_Bb4"] = tracks["clarinet_Bb4"]
save_json("e5_ext1_pooled_pitch", out)

fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.65), gridspec_kw=dict(width_ratios=[1.4, 1]))
tr = tracks["clarinet_Bb4"]
ax[0].fill_between(tr["t"], tr["f1"] - 2 * tr["s1"], tr["f1"] + 2 * tr["s1"], color=BAND2, lw=0,
                   label=r"paper: $\hat\omega_1 \pm 2$ s.e.")
ax[0].fill_between(tr["t"], tr["fp"] - 2 * tr["sp"], tr["fp"] + 2 * tr["sp"], color=C[0], alpha=0.45, lw=0,
                   label=r"Extension 1: $\hat\lambda \pm 2$ s.e.")
ax[0].plot(tr["t"], tr["fp"], color=C[0], lw=0.8)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("Hz")
ax[0].set_title(r"(a) clarinet B$\flat$4: pitch per 23 ms frame")
ax[0].legend(loc="upper right")
bins = np.linspace(-0.15, 0.15, 51)
ax[1].hist(err1, bins=bins, color=C[1], alpha=0.55, label=f"partial 1 only, sd {1000 * err1.std():.0f} mHz")
ax[1].hist(errp, bins=bins, color=C[0], alpha=0.7, label=f"all 12 partials, sd {1000 * errp.std():.0f} mHz")
ax[1].set_xlabel("pitch error (Hz)")
ax[1].set_title(f"(b) simulated note, {R} frames")
ax[1].legend(loc="upper left")
save_fig(fig, "e5_ext1")
