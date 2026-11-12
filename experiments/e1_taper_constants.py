"""E1. The constants of eq. (15) for five tapers.

Checks:
  * eq. (15) typed in correctly: it must equal the weighted least squares
    sandwich covariance, computed a different way (tapers.sandwich_cov);
  * Remark 3: with no taper the constants are Walker's 12, 1, 4, 3, 6;
  * the price of tapering, sqrt(c0 / 12), and what it buys (side-lobe level).
"""
from common import *  # noqa: F401,F403
import hwls

rows, worst = [], 0.0
for name in hwls.TAPERS:
    c = hwls.taper_constants(name)
    for A, B in [(1, 0), (0.3, -1.2), (2, 0.7)]:
        a, b = hwls.theorem2_cov(name, A, B, 0.1), hwls.sandwich_cov(name, A, B, 0.1)
        worst = max(worst, np.abs(a - b).max() / np.abs(b).max())
    kern = hwls.leakage_kernel(name, np.linspace(0.01, 40 * np.pi, 3000))
    first_null = np.argmax(np.diff(kern) > 0)
    sidelobe = 20 * np.log10(kern[first_null:].max()) if first_null > 0 else float("nan")
    rows.append(dict(taper=name, **{k: c[k] for k in ["c0", "c1", "c2", "c3", "c4"]},
                     se_vs_no_taper=np.sqrt(c["c0"] / 12), sidelobe_db=sidelobe))
    print(f"{name:12s} c0 = {c['c0']:6.2f}  s.e. x{np.sqrt(c['c0'] / 12):.2f}  side-lobe {sidelobe:6.1f} dB")
print(f"largest relative difference, eq. (15) vs sandwich: {worst:.1e}")
save_json("e1_taper_constants", dict(rows=rows, max_rel_diff_vs_sandwich=worst))

fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.45))
s = np.linspace(0, 1, 400)
for i, name in enumerate(["rectangular", "hann", "blackman"]):
    ax[0].plot(s, hwls.TAPERS[name](s), color=C[i], label=hwls.LABELS[name].split(" (")[0])
    b = np.linspace(0.01, 2 * np.pi * 12, 2000)
    ax[1].plot(b / (2 * np.pi), 20 * np.log10(hwls.leakage_kernel(name, b) + 1e-9), color=C[i], lw=1)
ax[0].set_xlabel("s = t / T")
ax[0].set_title("taper w(s)")
ax[0].legend(loc="lower center", fontsize=6)
ax[1].set_ylim(-90, 3)
ax[1].set_xlabel("distance (Fourier bins)")
ax[1].set_ylabel("leakage (dB)")
ax[1].set_title("leakage kernel")
save_fig(fig, "e1_tapers")
