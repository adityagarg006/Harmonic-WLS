"""Simulation tools. On a real recording the true frequency is unknown, so
Theorem 2 cannot be checked there. Here we make signals whose truth we
know, estimate them many times, and compare the spread of the errors with
the formula.

estimate_one_sinusoid() is a fast version of fit_frame() for K = 1: it
estimates many simulated frames at once, one per row of a matrix (the way
apply() over rows would in R, but vectorised).
"""
import numpy as np
from scipy import signal

from .tapers import taper_weights


def make_noise(rng, R, T, kind="white", sigma=1.0, phi=0.6, burn=200):
    """R independent noise series of length T (an R x T matrix) and the
    function f(omega) giving their spectrum, in the paper's convention
    f(omega) = (1/2pi) sum_u c(u) exp(-i omega u).

    white  Gaussian white noise, f = sigma^2 / (2 pi)
    ar1    e_t = phi e_{t-1} + u_t, like arima.sim(list(ar = 0.6), ...) in R
    t5     Student t with 5 df scaled to variance sigma^2: heavy tails, so
           Assumption 1 (all moments exist) fails. A robustness check."""
    if kind == "white":
        e = sigma * rng.standard_normal((R, T))
        f = lambda w: sigma ** 2 / (2 * np.pi) * np.ones_like(np.asarray(w, float))
    elif kind == "ar1":
        u = sigma * rng.standard_normal((R, T + burn))
        e = signal.lfilter([1.0], [1.0, -phi], u, axis=1)[:, burn:]
        f = lambda w: sigma ** 2 / (2 * np.pi * np.abs(1 - phi * np.exp(-1j * np.asarray(w))) ** 2)
    elif kind == "t5":
        e = sigma * rng.standard_t(5, (R, T)) / np.sqrt(5 / 3)
        f = lambda w: sigma ** 2 / (2 * np.pi) * np.ones_like(np.asarray(w, float))
    else:
        raise ValueError(kind)
    return e, f


def estimate_one_sinusoid(Y, taper, pad=8, newton_steps=8):
    """Steps 2-5 of the pipeline for K = 1, applied to every row of Y."""
    R, T = Y.shape
    t = np.arange(1, T + 1, dtype=float)
    w = taper_weights(taper, T)
    Z = Y * w
    N = 1 << int(np.ceil(np.log2(pad * T)))
    F = np.abs(np.fft.rfft(Z, N, axis=1))
    F[:, 0] = 0
    grid = 2 * np.pi * np.arange(F.shape[1]) / N
    om = grid[np.argmax(F, axis=1)]
    lo, hi = om - 2 * np.pi / N, om + 2 * np.pi / N
    for _ in range(newton_steps):
        E = np.exp(-1j * np.outer(om, t))
        Z0 = (Z * E).sum(1)
        Z1 = (Z * E * (-1j * t)).sum(1)
        Z2 = (Z * E * (-(t ** 2))).sum(1)
        grad = 2 * np.real(np.conj(Z0) * Z1)
        hess = 2 * np.real(np.conj(Z1) * Z1 + np.conj(Z0) * Z2)
        step = np.where(hess < 0, -grad / np.where(hess == 0, -1, hess), 0.0)
        om = np.clip(om + step, lo, hi)
    sw = w.sum()
    A = 2 * (Z * np.cos(np.outer(om, t))).sum(1) / sw
    B = 2 * (Z * np.sin(np.outer(om, t))).sum(1) / sw
    return om, A, B


def simulate_one_sinusoid(R, T, A0, B0, omega0, taper, noise="white", sigma=1.0, seed=0, chunk=500):
    """R simulated frames of y_t = A0 cos(omega0 t) + B0 sin(omega0 t) + e_t,
    each estimated. Returns the R estimates of omega, A, B and the true
    noise spectrum function f."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1)
    s = A0 * np.cos(omega0 * t) + B0 * np.sin(omega0 * t)
    out = {"omega": [], "A": [], "B": []}
    f = None
    for i in range(0, R, chunk):
        e, f = make_noise(rng, min(chunk, R - i), T, noise, sigma)
        om, A, B = estimate_one_sinusoid(s + e, taper)
        out["omega"].append(om)
        out["A"].append(A)
        out["B"].append(B)
    res = {key: np.concatenate(v) for key, v in out.items()}
    res["f"] = f
    return res


def plugin_omega_se(Y, om, A, B, taper):
    """Theorem-2 standard error of omega_hat for each row of Y, with the
    noise spectrum estimated from the residuals exactly as in step 6
    (bins 3 to 12 away from the partial). Vectorised version of
    inference.attach_theorem2_se for K = 1."""
    from .tapers import taper_constants
    R, T = Y.shape
    t = np.arange(1, T + 1)
    res = Y - (A[:, None] * np.cos(np.outer(om, t)) + B[:, None] * np.sin(np.outer(om, t)))
    I = np.abs(np.fft.rfft(res, axis=1)) ** 2 / (2 * np.pi * T)
    lam = 2 * np.pi * np.arange(I.shape[1]) / T
    dist = np.abs(lam[None] - om[:, None]) / (2 * np.pi / T)
    use = (dist <= 12) & (dist > 2)
    f = (I * use).sum(1) / use.sum(1)
    c0 = taper_constants(taper)["c0"]
    return np.sqrt(4 * np.pi * f * c0 / (T ** 3 * (A ** 2 + B ** 2)))


def synthetic_note(rng, T, fs, f0_hz, amps, sigma, inharm_B=0.0, offsets_cents=None,
                   glide_hz_per_s=0.0, t0_s=0.0):
    """One frame of a synthetic note: sum of len(amps) partials with random
    phases plus white noise of standard deviation sigma.

    Partial k sits at k f0 sqrt(1 + B k^2) * 2^(offset_k / 1200). With
    B = 0 and no offsets the note is exactly harmonic. glide_hz_per_s makes
    the pitch move linearly in time (all partials together); t0_s is the
    time at which this frame starts. Returns the frame and the true
    frequencies (Hz) at the frame centre."""
    K = len(amps)
    k = np.arange(1, K + 1)
    off = np.zeros(K) if offsets_cents is None else np.asarray(offsets_cents)
    ratio = k * np.sqrt(1 + inharm_B * k ** 2) * 2 ** (off / 1200)
    t = np.arange(1, T + 1)
    time_s = t0_s + t / fs
    f0_t = f0_hz + glide_hz_per_s * time_s                          # pitch at each sample
    phase_inc = 2 * np.pi * np.outer(ratio, f0_t) / fs
    phase = np.cumsum(phase_inc, axis=1) + rng.uniform(0, 2 * np.pi, K)[:, None]
    y = (np.asarray(amps)[:, None] * np.cos(phase)).sum(0) + sigma * rng.standard_normal(T)
    f_centre = ratio * (f0_hz + glide_hz_per_s * (t0_s + (T + 1) / 2 / fs))
    return y, f_centre
