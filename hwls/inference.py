"""Standard errors (Theorem 2) and the paper's two questions.

Q1  Is the player out of tune? Compare omega_hat_1 with the target note,
    using omega_hat_1 +- 2 s.e.
Q2  Are the partials exact multiples of the pitch? Look at the deviations
    d_k = omega_hat_k - k * omega_hat_1 with +- 2 s.e. limits.

All the standard errors come from Theorem 2:

    var(omega_hat_k) ~ 4 pi c0 f(omega_k) / (T^3 rho_k^2)
    var(rho_hat_k)   ~ 4 pi c1 f(omega_k) / T                 (delta method)

where f is the noise spectrum. f is unknown, so step 6 of the pipeline
estimates it from the residuals.
"""
import numpy as np

from .tapers import taper_constants, taper_weights


def noise_spectrum(resid, omegas, half_width=12, notch=2):
    """Step 6: estimate the noise spectrum f(omega) at each partial.

    Take the periodogram of the residuals, I(lambda_j) = |sum e_t exp(-i lambda_j t)|^2 / (2 pi T),
    and average it over the Fourier frequencies within +-half_width bins of
    each partial (a "Daniell" moving average). Bins within +-notch of any
    fitted partial are skipped: the fit has removed some noise there, so
    those bins are too small. If too few bins are left (partials crowded
    together) the window is widened."""
    T = len(resid)
    I = np.abs(np.fft.rfft(resid)) ** 2 / (2 * np.pi * T)
    lam = 2 * np.pi * np.arange(len(I)) / T
    df = 2 * np.pi / T
    excluded = np.zeros(len(I), dtype=bool)
    for w in omegas:
        excluded |= np.abs(lam - w) <= notch * df
    excluded[0] = True                      # the mean
    out = np.empty(len(omegas))
    for i, w in enumerate(omegas):
        hw = half_width
        use = (np.abs(lam - w) <= hw * df) & ~excluded
        while use.sum() < 4:
            hw *= 2
            use = (np.abs(lam - w) <= hw * df) & ~excluded
        out[i] = I[use].mean()
    return out


def attach_theorem2_se(fit):
    """Step 7: plug the estimates into Theorem 2 to get standard errors.
    Results are stored in fit.se (radians per sample for omega)."""
    c = taper_constants(fit.taper)
    T = fit.T
    f = noise_spectrum(fit.residuals, fit.omega)
    A, B = fit.A, fit.B
    r2 = A ** 2 + B ** 2
    k = 4 * np.pi * f
    fit.f_noise = f
    fit.se = {
        "omega": np.sqrt(k * c["c0"] / (T ** 3 * r2)),
        "A": np.sqrt(k * (c["c1"] * A ** 2 + c["c2"] * B ** 2) / (T * r2)),
        "B": np.sqrt(k * (c["c2"] * A ** 2 + c["c1"] * B ** 2) / (T * r2)),
        "rho": np.sqrt(k * c["c1"] / T),
    }
    return fit


def attach_sandwich_se(fit):
    """Standard errors for the fading-partial model (method "wls-decay").

    Theorem 2 has no closed form for this model, so we use the weighted
    least squares sandwich M^-1 N M^-1 one partial at a time, with
        G_t = derivative of partial k's signal w.r.t. (A, B, omega, a),
        M   = sum_t w_t G_t G_t',      N = 2 pi f(omega_k) sum_t w_t^2 G_t G_t'.
    With a = 0 this is the finite-T version of eq. (14), so it agrees with
    attach_theorem2_se() for the paper's model."""
    T = fit.T
    t = np.arange(1, T + 1, dtype=float)
    w = taper_weights(fit.taper, T)
    f = noise_spectrum(fit.residuals, fit.omega)
    env = fit.envelope()
    se = {key: np.empty(len(fit.omega)) for key in ("A", "B", "omega", "decay", "rho")}
    for k, om in enumerate(fit.omega):
        c, s = np.cos(om * t), np.sin(om * t)
        A, B = fit.A[k], fit.B[k]
        G = env[k] * np.vstack([c, s, t * (-A * s + B * c), -(t - (T + 1) / 2) / T * (A * c + B * s)])
        Minv = np.linalg.inv((G * w) @ G.T)
        V = Minv @ (2 * np.pi * f[k] * (G * w ** 2) @ G.T) @ Minv
        g = np.array([A, B]) / np.hypot(A, B)
        se["A"][k], se["B"][k], se["omega"][k], se["decay"][k] = np.sqrt(np.diag(V))
        se["rho"][k] = np.sqrt(g @ V[:2, :2] @ g)
    fit.f_noise = f
    fit.se = se
    return fit


def harmonic_deviations(freq_hz, se_hz):
    """Q2: d_k = omega_hat_k - k omega_hat_1 and its standard error.

    Theorem 2 says different partials are independent, so
        var(d_k) = var(omega_hat_k) + k^2 var(omega_hat_1).
    Works on one frame (vectors) or many frames (matrices, one row each)."""
    K = freq_hz.shape[-1]
    k = np.arange(1, K + 1)
    d = freq_hz - k * freq_hz[..., :1]
    se = np.sqrt(se_hz ** 2 + k ** 2 * se_hz[..., :1] ** 2)
    return d, se


def jnd_hz(f_hz, semitones=0.03):
    """Half-width of the band the ear cannot tell apart: 0.03 semitone
    (Pierce 1992, as used in the paper). At 441 Hz this is 0.76 Hz."""
    return f_hz * (2 ** (semitones / 12) - 1)


def cents(f, ref):
    """Distance from f to ref in cents (100 cents = 1 semitone)."""
    return 1200 * np.log2(np.asarray(f) / ref)


def strong_partials(rec, threshold=8.0):
    """Partials whose amplitude is at least `threshold` standard errors.
    Very weak partials sit near the noise floor, where Theorem 2 (a
    large-sample result) is not reliable. Returns a frames x K logical
    matrix."""
    return rec.rho / rec.rho_se >= threshold
