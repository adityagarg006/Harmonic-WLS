"""E2. Monte Carlo check of Theorems 1 and 2.

One sinusoid y_t = A cos(omega t) + B sin(omega t) + e_t with known A, B,
omega. For each setting we simulate R frames, estimate each one, and
compare the spread of the estimates with Theorem 2.

 (a) 5 tapers x 3 noise types (white, AR(1), Student t5) x R runs, T = 1024:
     empirical variance / Theorem-2 variance for A, B and omega.
 (b) the same ratio as T grows (does the asymptotic formula kick in?).
 (c) coverage of the plug-in interval omega_hat +- 1.96 s.e., where the
     noise spectrum is estimated from each frame's residuals, as on real data.
 (d) Theorem 1: T |omega_hat - omega| shrinks to 0.
"""
from common import *  # noqa: F401,F403
import hwls
from hwls.simulate import simulate_one_sinusoid, make_noise, estimate_one_sinusoid, plugin_omega_se
from scipy import stats

A0, B0, OMEGA0 = 1.0, 0.5, 2 * np.pi * 0.1234
R = 1500 if FAST else 5000
TAPERS = ["rectangular", "hann", "hamming", "blackman", "ramp"]
NOISES = ["white", "ar1", "t5"]
out = dict(settings=dict(A0=A0, B0=B0, omega0=OMEGA0, R=R))

# (a) variance ratios at T = 1024
T = 1024
table = []
for i, taper in enumerate(TAPERS):
    for j, noise in enumerate(NOISES):
        r = simulate_one_sinusoid(R, T, A0, B0, OMEGA0, taper, noise, seed=1000 + 10 * i + j)
        Z = np.vstack([np.sqrt(T) * (r["A"] - A0), np.sqrt(T) * (r["B"] - B0), T ** 1.5 * (r["omega"] - OMEGA0)])
        emp = np.cov(Z)
        th = hwls.theorem2_cov(taper, A0, B0, float(r["f"](OMEGA0)))
        row = dict(taper=taper, noise=noise, ratio_A=emp[0, 0] / th[0, 0], ratio_B=emp[1, 1] / th[1, 1],
                   ratio_omega=emp[2, 2] / th[2, 2])
        table.append(row)
        print(f"{taper:12s} {noise:6s} var ratio  A {row['ratio_A']:.3f}  B {row['ratio_B']:.3f}  omega {row['ratio_omega']:.3f}")
worst = max(abs(row[k] - 1) for row in table for k in ("ratio_A", "ratio_B", "ratio_omega"))
print(f"largest |ratio - 1| = {worst:.3f}")
out["table_T1024"] = table
out["max_abs_ratio_minus_1"] = worst

# standardised errors for the histogram (Hann, white noise, T = 512)
r = simulate_one_sinusoid(R, 512, A0, B0, OMEGA0, "hann", "white", seed=11)
th = hwls.theorem2_cov("hann", A0, B0, float(r["f"](OMEGA0)))
z_hist = 512 ** 1.5 * (r["omega"] - OMEGA0) / np.sqrt(th[2, 2])
out["hann_T512_shapiro_p"] = float(stats.shapiro(z_hist[:4999])[1])

# (b) variance ratio as T grows
Ts = [32, 64, 128, 256, 512, 1024, 2048]
conv = {}
for taper in ["rectangular", "hann", "blackman"]:
    conv[taper] = []
    for T_ in Ts:
        r = simulate_one_sinusoid(R, T_, A0, B0, OMEGA0, taper, "white", seed=T_ + 7)
        th = hwls.theorem2_cov(taper, A0, B0, float(r["f"](OMEGA0)))
        conv[taper].append(np.var(T_ ** 1.5 * (r["omega"] - OMEGA0)) / th[2, 2])
out["variance_ratio_vs_T"] = dict(T=Ts, ratio=conv)

# (c) coverage of plug-in 95% intervals
rng = np.random.default_rng(5)
cover = []
for taper in ["rectangular", "hann", "blackman"]:
    for noise in ["white", "ar1"]:
        for T_ in [64, 128, 256, 512, 1024]:
            t = np.arange(1, T_ + 1)
            e, _ = make_noise(rng, R // 2, T_, noise)
            Y = A0 * np.cos(OMEGA0 * t) + B0 * np.sin(OMEGA0 * t) + e
            om, A, B = estimate_one_sinusoid(Y, taper)
            se = plugin_omega_se(Y, om, A, B, taper)
            cover.append(dict(taper=taper, noise=noise, T=T_, coverage=float(np.mean(np.abs(om - OMEGA0) <= 1.96 * se))))
            print(f"coverage {taper:11s} {noise:5s} T = {T_:5d}: {cover[-1]['coverage']:.3f}")
out["coverage"] = cover
big = [c["coverage"] for c in cover if c["T"] >= 256]
out["coverage_range_T_ge_256"] = [min(big), max(big)]

# (d) Theorem 1: consistency
med = []
for T_ in Ts:
    r = simulate_one_sinusoid(2000, T_, A0, B0, OMEGA0, "hann", "white", seed=T_)
    med.append(float(np.median(T_ * np.abs(r["omega"] - OMEGA0))))
slope = np.polyfit(np.log(Ts[2:]), np.log(med[2:]), 1)[0]
out["theorem1"] = dict(T=Ts, median_T_abs_error=med, loglog_slope=slope)
print("Theorem 1: median T|error| =", np.round(med, 4), " slope", round(slope, 2))
save_json("e2_theorem2_montecarlo", out)

fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.75))
x = np.linspace(-4, 4, 300)
ax[0].hist(z_hist, bins=50, density=True, color=C[0], alpha=0.8, edgecolor="white", linewidth=0.3)
ax[0].plot(x, stats.norm.pdf(x), color="black", lw=1)
ax[0].set_title("(a) Hann, T = 512")
ax[0].set_xlabel(r"$T^{3/2}(\hat\omega-\omega)/\mathrm{sd}_{\mathrm{Thm\,2}}$")
for i, taper in enumerate(conv):
    ax[1].semilogx(Ts, conv[taper], "o-", color=C[i], label=hwls.LABELS[taper].split(" (")[0])
ax[1].axhspan(0.95, 1.05, color=LIGHT, zorder=0)
ax[1].axhline(1, color="black", lw=0.8)
ax[1].set_yscale("log")
ax[1].set_ylim(0.7, 300)
ax[1].set_xlabel("frame length T")
ax[1].set_title("(b) variance: simulated / theory")
ax[1].legend(loc="upper right")
for i, taper in enumerate(["rectangular", "hann", "blackman"]):
    for noise, ls in [("white", "-"), ("ar1", "--")]:
        rows_ = [c for c in cover if c["taper"] == taper and c["noise"] == noise]
        ax[2].semilogx([c["T"] for c in rows_], [c["coverage"] for c in rows_], "o" + ls, color=C[i], ms=2.5)
ax[2].plot([], [], "k-", label="white noise")
ax[2].plot([], [], "k--", label="AR(1) noise")
ax[2].axhline(0.95, color="black", lw=0.8)
ax[2].set_ylim(0.5, 1.0)
ax[2].set_xlabel("frame length T")
ax[2].set_title("(c) coverage of 95% CIs")
ax[2].legend(loc="lower right")
save_fig(fig, "e2_theorem2")
