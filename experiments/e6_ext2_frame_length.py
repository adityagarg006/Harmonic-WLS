"""E6. Extension 2: how long can a frame be?

Theorem 2 says the s.e. of omega_hat falls like T^(-3/2), so longer frames
look better. But the theorem assumes the note does not change inside the
frame. When that fails, the formula keeps giving smaller s.e. while the real
error stops shrinking, and the error bars start to lie.

The check: for a harmonic note z = (omega_hat_k - k omega_hat_1) / s.e.
should be N(0, 1), so about 4.55% of the z's should fall outside +-2. At each
frame length we count them and test the count against Binomial(n, 0.0455)
at the 1% level. The chosen frame is the longest one that passes, with
every shorter one (from 768 samples) passing too.

 (a) clarinet B-flat4, clarinet D4, steel guitar: the check at 11 lengths
 (b) the two ideas together: pitch s.e. at the chosen length
 (c) simulation: a smooth pitch glide (all partials moving together) should
     NOT trigger the check, because the deviations stay at zero.
"""
from common import *  # noqa: F401,F403
import hwls
from hwls.simulate import synthetic_note

LENGTHS = [512, 768, 1025, 1280, 1536, 2050, 2560, 3000, 4100, 5000, 6000]
out = dict(lengths=LENGTHS)

curves = {}
for key, duration in [("clarinet_Bb4", 43 * 1025), ("clarinet_D4", 43 * 1025), ("guitar_steel_D3", 30 * 3000)]:
    cfg = RECORDINGS[key]
    x, fs = load_recording(key)
    rows = hwls.frame_length_check(x, fs, cfg["K"], cfg["nominal"], cfg["start"], duration, LENGTHS)
    best = hwls.longest_honest_frame(rows)
    table = []
    for r in rows:
        rec = r.pop("rec")
        pooled = [hwls.pooled_pitch(rec.freq_hz[i], rec.freq_se_hz[i])["se"] for i in range(len(rec.times))]
        r.update(se_single_mhz=1000 * np.median(rec.freq_se_hz[:, 0]), se_pooled_mhz=1000 * np.median(pooled))
        table.append(r)
    curves[key] = table
    out[key] = dict(rows=table, longest_honest_T=best)
    print(key, "longest honest frame:", best)
    for r in table:
        print(f"   T = {r['T']:5d} ({r['ms']:5.1f} ms) frames {r['n_frames']:3d}  n {r['n']:4d}  "
              f"outside {100 * r['share']:5.1f}%  p = {r['p']:.2g}  s.e. {r['se_single_mhz']:6.1f} / pooled {r['se_pooled_mhz']:6.2f} mHz")

# (b) the two ideas together
summary = {}
for key in ["clarinet_Bb4", "clarinet_D4"]:
    rows = {r["T"]: r for r in curves[key]}
    best = out[key]["longest_honest_T"]
    summary[key] = dict(paper_mhz=rows[1025]["se_single_mhz"], ext1_mhz=rows[1025]["se_pooled_mhz"],
                        ext2_T=best, ext2_single_mhz=rows[best]["se_single_mhz"],
                        both_mhz=rows[best]["se_pooled_mhz"],
                        total_gain=rows[1025]["se_single_mhz"] / rows[best]["se_pooled_mhz"])
    print(key, {k: round(v, 2) for k, v in summary[key].items()})
out["combined"] = summary

# (c) a smooth common glide does not trigger the check
# noise level chosen so that the s.e. of omega_hat_1 at T = 1025 is close to
# the real clarinet D4's (about 0.12 Hz)
fs, F0, K, SIGMA, DUR = 44100, 293.66, 15, 0.08, 43 * 1025
kk = np.arange(1, K + 1)
amps = np.where(kk % 2 == 1, 1.0, 0.06) / kk ** 1.1
glide = {}
for g in [0.0, 1.0, 4.0]:
    rng = np.random.default_rng(int(10 * g) + 3)
    shares, fails = [], 0
    for T in LENGTHS[1:]:
        z = []
        for i in range(DUR // T):
            y, _ = synthetic_note(rng, T, fs, F0, amps, SIGMA, glide_hz_per_s=g, t0_s=i * T / fs)
            fit = hwls.fit_frame(y, K, hwls.hz_to_rad(F0 + g * i * T / fs, fs))
            f, s = hwls.rad_to_hz(fit.omega, fs), hwls.rad_to_hz(fit.se["omega"], fs)
            d, dse = hwls.harmonic_deviations(f, s)
            strong = fit.rho / fit.se["rho"] >= 8
            z.extend((d / dse)[1:][strong[1:] & strong[0]])
        res = hwls.count_outside(z)
        shares.append(res["share"])
        fails += res["p"] <= 0.01
    glide[str(g)] = dict(share_outside=shares, lengths_failing_at_1pct=int(fails))
    print(f"glide {g} Hz/s: outside +-2 =", np.round(100 * np.array(shares), 1), " lengths failing:", fails)
out["glide_simulation"] = dict(lengths=LENGTHS[1:], sigma=SIGMA, results=glide)
save_json("e6_ext2_frame_length", out)

fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
names = {"clarinet_Bb4": r"clarinet B$\flat$4", "clarinet_D4": "clarinet D4", "guitar_steel_D3": "steel guitar D3"}
for i, key in enumerate(curves):
    ms = [r["ms"] for r in curves[key]]
    sh = [100 * r["share"] for r in curves[key]]
    ax[0].plot(ms, sh, "o-", color=C[i], label=names[key])
    best = out[key]["longest_honest_T"]
    if best:
        r = [r for r in curves[key] if r["T"] == best][0]
        ax[0].plot(r["ms"], 100 * r["share"], "o", ms=7, mfc="none", mec="black", mew=0.8)
ax[0].axhline(4.55, color="black", lw=0.8, ls="--")
ax[0].set_yscale("log")
ax[0].set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
ax[0].set_xlabel("frame length (ms)")
ax[0].set_ylabel("% of z outside ±2")
ax[0].set_title("(a) the check (circle = chosen length)")
ax[0].set_xlim(5, 175)
ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=6)
for i, key in enumerate(["clarinet_Bb4", "clarinet_D4"]):
    rr = [r for r in curves[key] if r["T"] >= 768]
    ms = [r["ms"] for r in rr]
    ax[1].loglog(ms, [r["se_single_mhz"] for r in rr], "o--", color=C[i], ms=2.5, lw=0.8)
    ax[1].loglog(ms, [r["se_pooled_mhz"] for r in rr], "o-", color=C[i], ms=2.5, label=names[key])
ax[1].plot([], [], "k--", lw=0.8, label=r"$\hat\omega_1$ only")
ax[1].plot([], [], "k-", label="all partials")
ax[1].set_xticks([20, 50, 100], ["20", "50", "100"])
ax[1].minorticks_off()
ax[1].set_xlabel("frame length (ms)")
ax[1].set_ylabel("median pitch s.e. (mHz)")
ax[1].set_title("(b) what a longer frame would give")
ax[1].legend(loc="upper right", fontsize=6)
save_fig(fig, "e6_ext2")
