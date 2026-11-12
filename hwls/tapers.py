"""Taper (weight) functions and the constants of Irizarry (2002).

A taper w(s) is a function on [0, 1]. Sample t of a frame of length T gets
weight w(t/T). The paper needs a handful of integrals of the taper:

    W_n = integral_0^1 s^n w(s)   ds      (eq. 4)
    U_n = integral_0^1 s^n w(s)^2 ds      (just below eq. 15)

and builds the five constants c0, ..., c4 of eq. (15) from them. Only c0
matters for frequencies: var(omega_hat) is proportional to c0.

R users: every function here is the equivalent of a small R function that
calls integrate(); scipy's quad() plays the role of integrate().
"""
import numpy as np
from scipy import integrate

# Each taper is a function of s in [0, 1]. These five are the ones used in
# the Monte Carlo study. "ramp" is deliberately asymmetric, to check that the
# general (non-symmetric) formula in eq. (15) is right.
TAPERS = {
    "rectangular": lambda s: np.ones_like(np.asarray(s, dtype=float)),
    "hann": lambda s: np.sin(np.pi * np.asarray(s)) ** 2,
    "hamming": lambda s: 0.54 - 0.46 * np.cos(2 * np.pi * np.asarray(s)),
    "blackman": lambda s: (0.42 - 0.5 * np.cos(2 * np.pi * np.asarray(s))
                           + 0.08 * np.cos(4 * np.pi * np.asarray(s))),
    "ramp": lambda s: np.asarray(s, dtype=float),
}

LABELS = {"rectangular": "Rectangular (no taper)", "hann": "Hann", "hamming": "Hamming",
          "blackman": "Blackman", "ramp": "Ramp w(s) = s"}


def taper_weights(taper, T):
    """The weights w(1/T), w(2/T), ..., w(T/T) for one frame of length T.

    Same as w(seq_len(T) / T) in R."""
    t = np.arange(1, T + 1)
    return TAPERS[taper](t / T).astype(float)


def taper_moments(taper):
    """W0, W1, W2 and U0, U1, U2 by numerical integration (eq. 4)."""
    w = TAPERS[taper]

    def quad(g):
        return integrate.quad(g, 0.0, 1.0, limit=200, epsabs=1e-13, epsrel=1e-12)[0]

    m = {}
    for n in range(3):
        m[f"W{n}"] = quad(lambda s, n=n: s ** n * w(s))
        m[f"U{n}"] = quad(lambda s, n=n: s ** n * w(s) ** 2)
    return m


def taper_constants(taper):
    """The constants c0, c1, c2, c3, c4 of eq. (15), copied term by term from
    the paper. For the rectangular taper they reduce to Walker's values
    c0 = 12, c1 = 1, c2 = 4, c3 = 3, c4 = 6 (Remark 3)."""
    m = taper_moments(taper)
    W0, W1, W2 = m["W0"], m["W1"], m["W2"]
    U0, U1, U2 = m["U0"], m["U1"], m["U2"]
    W = [W0, W1, W2]
    a0 = (W0 * W2 - W1 ** 2) ** -2

    def b(n):
        return W[n] ** 2 * U2 + W[n + 1] * (W[n + 1] * U0 - 2 * W[n] * U1)

    c = {
        "c0": a0 * b(0),
        "c1": U0 / W0 ** 2,
        "c2": a0 * b(1),
        "c3": a0 * W1 * W0 ** -2 * (W0 ** 2 * W1 * U2 - W1 ** 3 * U0
                                    - 2 * W0 ** 2 * W2 * U1 + 2 * W0 * W1 * W2 * U0),
        "c4": a0 * (W0 * W1 * U2 - W1 ** 2 * U1 - W0 * W2 * U1 + W1 * W2 * U0),
    }
    c.update(m)
    return c


def theorem2_cov(taper, A, B, f_at_omega):
    """Theorem 2: the limiting 3 x 3 covariance matrix of

        ( sqrt(T) (A_hat - A),  sqrt(T) (B_hat - B),  T^(3/2) (omega_hat - omega) )

    for one partial, i.e. 4 pi f(omega) / (A^2 + B^2) * V with V from eq. (14).
    f_at_omega is the noise spectrum at the partial's frequency."""
    c = taper_constants(taper)
    V = np.array([
        [c["c1"] * A ** 2 + c["c2"] * B ** 2, -c["c3"] * A * B, -c["c4"] * B],
        [-c["c3"] * A * B, c["c2"] * A ** 2 + c["c1"] * B ** 2, c["c4"] * A],
        [-c["c4"] * B, c["c4"] * A, c["c0"]],
    ])
    return 4 * np.pi * f_at_omega / (A ** 2 + B ** 2) * V


def sandwich_cov(taper, A, B, f_at_omega):
    """An independent route to the same matrix, used only as a check on our
    reading of eq. (15). Weighted least squares has the usual "sandwich"
    covariance M^-1 N M^-1 (the same idea as sandwich() in R's sandwich
    package), where M uses the integrals W_n and N uses U_n. If eq. (15) is
    typed in correctly, this matches theorem2_cov() to rounding error."""
    m = taper_moments(taper)
    r2 = A * A + B * B

    def mat(X0, X1, X2):
        return 0.5 * np.array([[X0, 0, B * X1], [0, X0, -A * X1], [B * X1, -A * X1, r2 * X2]])

    M = mat(m["W0"], m["W1"], m["W2"])
    N = mat(m["U0"], m["U1"], m["U2"])
    Mi = np.linalg.inv(M)
    return 2 * np.pi * f_at_omega * Mi @ N @ Mi


def leakage_kernel(taper, b, n_grid=4001):
    """|integral_0^1 w(s) exp(i b s) ds| / W0: how much of a partial leaks to
    a frequency b/(2 pi) Fourier bins away. The first side-lobe of this
    curve is the "-13 dB" (rectangular) and "-31 dB" (Hann) in the slides."""
    s = np.linspace(0, 1, n_grid)
    w = TAPERS[taper](s)
    k = np.abs(np.trapezoid(w[None, :] * np.exp(1j * np.outer(b, s)), s, axis=1))
    return k / np.trapezoid(w, s)
