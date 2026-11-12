"""The estimator of Irizarry (2002), Section 2.1, applied one frame at a time.

Model (eq. 2), for one frame y_1, ..., y_T:

    y_t = sum_{k=1}^{K} { A_k cos(omega_k t) + B_k sin(omega_k t) } + e_t

Frequencies are in radians per sample inside this package
(Hz = omega * fs / (2 pi)); use hz_to_rad() and rad_to_hz() to convert.

fit_frame() runs the seven steps used in the slides and the demo site:

    1. take the frame y_t
    2. taper it:                     z_t = w(t/T) y_t
    3. tapered periodogram:          |sum_t z_t exp(-i omega t)|^2 on an FFT grid
    4. refine each peak:             Newton steps to the exact maximum (eq. 9)
    5. amplitudes:                   A_k, B_k by weighted projection (eqs. 7-8)
    6. noise level near each partial (inference.noise_spectrum)
    7. standard errors from Theorem 2 (inference.attach_theorem2_se)

Two other fitting methods are available through `method=`:

  "wls"        the paper's full weighted non-linear least squares, eq. (3).
               The paper ran this with S-Plus nls(); here it is
               scipy's least_squares(), the analogue of R's nls(). It starts
               from the periodogram answer. Section 2.1 says both methods
               agree for large T; experiment e4 checks that they do.
  "wls-decay"  our addition for plucked strings (Extension 3): the same
               fit, but each partial may fade inside the frame.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .tapers import taper_weights


def hz_to_rad(f_hz, fs):
    return 2 * np.pi * np.asarray(f_hz) / fs


def rad_to_hz(omega, fs):
    return np.asarray(omega) * fs / (2 * np.pi)


@dataclass
class FrameFit:
    """Everything estimated from one frame (think of it as an R list).

    omega, A, B : one value per partial, omega sorted (constraint 16)
    se          : dict of standard errors, same units ("omega", "A", "B", "rho")
    f_noise     : estimated noise spectrum at each partial (step 6)
    decay       : only for method "wls-decay", the log-amplitude drop of each
                  partial across the frame"""
    omega: np.ndarray
    A: np.ndarray
    B: np.ndarray
    T: int
    taper: str
    method: str
    y: np.ndarray = field(repr=False)
    f_noise: np.ndarray = None
    se: dict = None
    decay: np.ndarray = None

    @property
    def rho(self):
        """Amplitude of each partial, sqrt(A^2 + B^2)."""
        return np.hypot(self.A, self.B)

    def envelope(self):
        """Amplitude envelope of each partial over the frame (all ones unless
        the decay model was fitted). Time is measured from the frame centre."""
        t = np.arange(1, self.T + 1)
        if self.decay is None:
            return np.ones((len(self.omega), self.T))
        return np.exp(-np.outer(self.decay, (t - (self.T + 1) / 2) / self.T))

    def fitted(self):
        """The fitted sum of sinusoids, s(t, beta_hat), for t = 1..T."""
        t = np.arange(1, self.T + 1)
        waves = (self.A[:, None] * np.cos(np.outer(self.omega, t))
                 + self.B[:, None] * np.sin(np.outer(self.omega, t)))
        return (self.envelope() * waves).sum(axis=0)

    @property
    def residuals(self):
        return self.y - self.fitted()


def tapered_dft(z, omega, t):
    """Z(omega) = sum_t z_t exp(-i omega t) and its first two derivatives in
    omega. Step 4 needs the derivatives for Newton's method."""
    e = np.exp(-1j * omega * t)
    return np.sum(z * e), np.sum(-1j * t * z * e), np.sum(-(t ** 2) * z * e)


def refine_peak(z, t, omega_start, half_width):
    """Step 4: climb from the FFT-grid peak to the exact maximum of |Z(omega)|^2.

    Newton's method on the analytic gradient and Hessian, as long as each
    step stays inside [omega_start - half_width, omega_start + half_width].
    If Newton misbehaves (for example the curve is not concave there), fall
    back to a bounded one-dimensional search, the analogue of R's optimize()."""
    lo, hi = omega_start - half_width, omega_start + half_width
    w = omega_start
    for _ in range(30):
        Z0, Z1, Z2 = tapered_dft(z, w, t)
        grad = 2 * np.real(np.conj(Z0) * Z1)
        hess = 2 * np.real(np.conj(Z1) * Z1 + np.conj(Z0) * Z2)
        if hess >= 0:
            break
        step = -grad / hess
        if not (lo <= w + step <= hi):
            break
        w = w + step
        if abs(step) < 1e-13:
            return w
    res = optimize.minimize_scalar(lambda x: -abs(np.sum(z * np.exp(-1j * x * t))) ** 2,
                                   bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return float(res.x)


def fit_frame(y, K, f0_hint, taper="hann", method="periodogram", search_frac=0.45, pad=8,
              with_se=True):
    """Fit K partials to one frame y (a numeric vector of length T).

    f0_hint (rad/sample) is a rough pitch. Partial k is looked for inside
    the band [(k - 0.45) f0_hint, (k + 0.45) f0_hint]. This is a practical
    form of the paper's separation condition (13) and ordering (16).

    Returns a FrameFit."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)

    # Steps 1-2: taper the frame.
    w = taper_weights(taper, T)
    z = w * y

    # Step 3: tapered periodogram on a zero-padded FFT grid (pad = 8 times
    # finer than the natural 2 pi / T spacing). R's fft() would do the same.
    N = 1 << int(np.ceil(np.log2(pad * T)))
    mag = np.abs(np.fft.rfft(z, N))
    grid = 2 * np.pi * np.arange(len(mag)) / N
    cell = 2 * np.pi / N

    start = []
    for k in range(1, K + 1):
        lo, hi = max((k - search_frac) * f0_hint, 1e-3), min((k + search_frac) * f0_hint, np.pi - 1e-3)
        band = np.where((grid >= lo) & (grid <= hi))[0]
        if len(band) == 0:
            raise ValueError(f"the search band of partial {k} lies outside (0, pi)")
        start.append(grid[band[np.argmax(mag[band])]])

    # Step 4: refine each grid peak to the true maximum.
    omega = np.sort([refine_peak(z, t, s, 1.5 * cell) for s in start])

    # Step 5: amplitudes by weighted projection (eqs. 7 and 8).
    sum_w = w.sum()
    A = 2 * (np.cos(np.outer(omega, t)) * z).sum(axis=1) / sum_w
    B = 2 * (np.sin(np.outer(omega, t)) * z).sum(axis=1) / sum_w

    decay = None
    if method == "wls":
        omega, A, B = _wls_refine(y, t, w, omega, A, B)
    elif method == "wls-decay":
        omega, A, B, decay = _wls_decay_refine(y, t, w, omega, A, B)
    elif method != "periodogram":
        raise ValueError("method must be 'periodogram', 'wls' or 'wls-decay'")

    fit = FrameFit(omega=omega, A=A, B=B, T=T, taper=taper, method=method, y=y, decay=decay)

    # Steps 6-7: noise level and standard errors (see inference.py).
    if with_se:
        from .inference import attach_theorem2_se, attach_sandwich_se
        if decay is None:
            attach_theorem2_se(fit)
        else:
            attach_sandwich_se(fit)
    return fit


def _wls_refine(y, t, w, omega, A, B):
    """Minimise the paper's criterion (3),
        S_T(beta) = sum_t w(t/T) { y_t - s(t, beta) }^2,
    over all 3K parameters at once. Multiplying each residual by sqrt(w_t)
    turns it into ordinary non-linear least squares, which is what R's
    nls(..., weights = w) does internally."""
    K, T = len(omega), len(t)
    sq = np.sqrt(w)

    def unpack(p):                      # frequencies are stored as omega * T
        return p[:K], p[K:2 * K], p[2 * K:] / T     # so all parameters have similar size

    def resid(p):
        a, b, om = unpack(p)
        ph = np.outer(om, t)
        return sq * (y - (a[:, None] * np.cos(ph) + b[:, None] * np.sin(ph)).sum(0))

    def jac(p):
        a, b, om = unpack(p)
        ph = np.outer(om, t)
        c, s = np.cos(ph), np.sin(ph)
        return np.hstack([-(sq * c).T, -(sq * s).T,
                          -(sq * (t / T) * (-a[:, None] * s + b[:, None] * c)).T])

    p0 = np.concatenate([A, B, omega * T])
    res = optimize.least_squares(resid, p0, jac=jac, method="lm", xtol=1e-14, ftol=1e-14,
                                 gtol=1e-14, max_nfev=200)
    a, b, om = unpack(res.x)
    order = np.argsort(om)
    return om[order], a[order], b[order]


def _wls_decay_refine(y, t, w, omega, A, B):
    """Criterion (3) with fading partials (Extension 3):
        s(t) = sum_k exp(-a_k (t - t_c)/T) { A_k cos(omega_k t) + B_k sin(omega_k t) },
    with t_c the frame centre. a_k is the log-amplitude drop of partial k
    across the frame; a_k = 0 gives back the paper's model."""
    K, T = len(omega), len(t)
    sq = np.sqrt(w)
    tc = (t - (T + 1) / 2) / T

    def parts(p):
        a, b, om, dec = p[:K], p[K:2 * K], p[2 * K:3 * K] / T, p[3 * K:]
        env = np.exp(-np.outer(dec, tc))
        ph = np.outer(om, t)
        return a, b, env, np.cos(ph), np.sin(ph)

    def resid(p):
        a, b, env, c, s = parts(p)
        return sq * (y - (env * (a[:, None] * c + b[:, None] * s)).sum(0))

    def jac(p):
        a, b, env, c, s = parts(p)
        wave = a[:, None] * c + b[:, None] * s
        return np.hstack([-(sq * env * c).T, -(sq * env * s).T,
                          -(sq * env * (t / T) * (-a[:, None] * s + b[:, None] * c)).T,
                          (sq * env * tc * wave).T])

    p0 = np.concatenate([A, B, omega * T, np.zeros(K)])
    res = optimize.least_squares(resid, p0, jac=jac, method="lm", xtol=1e-12, ftol=1e-12,
                                 gtol=1e-12, max_nfev=300)
    a, b, om, dec = res.x[:K], res.x[K:2 * K], res.x[2 * K:3 * K] / T, res.x[3 * K:]
    order = np.argsort(om)
    return om[order], a[order], b[order], dec[order]


@dataclass
class Recording:
    """Results for every frame of one recording. Each array has one row per
    frame and one column per partial, like a matrix in R."""
    fs: float
    T: int
    K: int
    times: np.ndarray          # start time of each frame, seconds
    freq_hz: np.ndarray        # omega_hat in Hz
    freq_se_hz: np.ndarray     # its standard error in Hz
    rho: np.ndarray            # amplitude estimates
    rho_se: np.ndarray
    fits: list = field(repr=False, default_factory=list)


def analyse_recording(x, fs, T, K, f0_hz, start_s=0.0, n_frames=None, taper="hann",
                      method="periodogram"):
    """The paper's Section 4 procedure: cut the signal into contiguous,
    non-overlapping frames of T samples and fit model (2) to each frame
    separately. The pitch hint for each frame is the previous frame's
    omega_1, so slow drift is followed."""
    x = np.asarray(x, dtype=float)
    i0 = int(round(start_s * fs))
    max_frames = (len(x) - i0) // T
    n = max_frames if n_frames is None else min(n_frames, max_frames)
    f0 = hz_to_rad(f0_hz, fs)
    fits = []
    for j in range(n):
        fit = fit_frame(x[i0 + j * T: i0 + (j + 1) * T], K, f0, taper=taper, method=method)
        f0 = fit.omega[0]
        fits.append(fit)
    return Recording(
        fs=fs, T=T, K=K,
        times=(i0 + np.arange(n) * T) / fs,
        freq_hz=np.array([rad_to_hz(f.omega, fs) for f in fits]),
        freq_se_hz=np.array([rad_to_hz(f.se["omega"], fs) for f in fits]),
        rho=np.array([f.rho for f in fits]),
        rho_se=np.array([f.se["rho"] for f in fits]),
        fits=fits,
    )
