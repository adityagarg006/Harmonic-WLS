"""Our three extensions. None of them needs new theory: each one only uses
what Theorem 2 already gives, namely a standard error for every partial and
the fact that different partials have independent errors.

Extension 1  pooled_pitch()         use every partial to measure the pitch
Extension 2  harmonic_z(), frame_length_check(), longest_honest_frame()
                                    how long can a frame be before the
                                    error bars stop being honest?
Extension 3  partial_offset_model(), stiff_string_fit()
                                    the same pooling idea for a guitar,
                                    whose partials are not exact multiples
"""
import numpy as np
from scipy import optimize, stats

from .estimator import analyse_recording
from .inference import harmonic_deviations, strong_partials

P_OUTSIDE_2SE = 2 * stats.norm.sf(2.0)      # P(|Z| > 2) = 0.0455 for Z ~ N(0, 1)


def pooled_pitch(freq_hz, se_hz):
    """Extension 1: combine all partials of one frame into one pitch estimate.

    For a harmonic note omega_k = k * lambda, so omega_hat_k / k measures
    lambda with standard error se_k / k. The best linear combination of
    independent unbiased measurements weights each by 1 / variance. That
    gives

        lambda_hat     = sum(v_k k omega_hat_k) / sum(v_k k^2),   v_k = 1 / se_k^2
        se(lambda_hat) = 1 / sqrt(sum(v_k k^2))

    which is exactly weighted least squares through the origin:
    lm(freq ~ 0 + k, weights = 1/se^2) in R.

    The left-over chi-square, sum v_k (omega_hat_k - k lambda_hat)^2, has a
    chi-square distribution on K - 1 df if the note is harmonic, so it is
    also a per-frame test of harmonicity."""
    f, se = np.asarray(freq_hz, float), np.asarray(se_hz, float)
    k = np.arange(1, len(f) + 1)
    v = 1 / se ** 2
    lam = np.sum(v * k * f) / np.sum(v * k ** 2)
    chi2 = float(np.sum(v * (f - k * lam) ** 2))
    df = len(f) - 1
    return dict(pitch=lam, se=1 / np.sqrt(np.sum(v * k ** 2)), chi2=chi2, df=df,
                p=float(stats.chi2.sf(chi2, df)))


def harmonic_z(rec, threshold=8.0):
    """Extension 2, steps 1-2: for a harmonic note the deviation
    d_k = omega_hat_k - k omega_hat_1 is pure estimation noise, so
    z = d_k / se(d_k) should be N(0, 1). Returns the z values of all
    partials k >= 2, over all frames, for which both partial k and partial 1
    are strong (amplitude >= threshold standard errors)."""
    d, se = harmonic_deviations(rec.freq_hz, rec.freq_se_hz)
    strong = strong_partials(rec, threshold)
    use = strong[:, 1:] & strong[:, :1]
    return (d / se)[:, 1:][use]


def count_outside(z):
    """Extension 2, steps 3-5: what share of z lies outside +-2, and is that
    more than the 4.55% expected when the error bars are honest?

    If the s.e. are right, the count is Binomial(n, 0.0455); p is the chance
    of a count this high or higher (binom.test(..., alternative = "greater")
    in R)."""
    z = np.asarray(z)
    n, c = len(z), int(np.sum(np.abs(z) > 2))
    return dict(n=n, outside=c, share=c / n if n else np.nan,
                p=float(stats.binom.sf(c - 1, n, P_OUTSIDE_2SE)) if n else np.nan)


def frame_length_check(x, fs, K, f0_hz, start_s, duration, lengths, z_function=harmonic_z,
                       method="periodogram"):
    """Extension 2: re-cut the same stretch of recording into frames of each
    length in `lengths`, run the estimator, and count how many standardised
    deviations fall outside +-2. `duration` is in samples, so every frame
    length looks at the same part of the note.

    z_function decides what "should be zero" means: harmonic_z for a
    clarinet; for a guitar Extension 3 supplies offset_model_z instead."""
    rows = []
    for T in lengths:
        rec = analyse_recording(x, fs, T, K, f0_hz, start_s=start_s, n_frames=duration // T,
                                method=method)
        res = count_outside(z_function(rec))
        res.update(T=T, ms=1000 * T / fs, n_frames=len(rec.times), rec=rec)
        rows.append(res)
    return rows


def longest_honest_frame(rows, alpha=0.01, shortest=768):
    """Extension 2, step 6: the longest frame length that passes (p > alpha),
    with every shorter length from `shortest` upwards passing too. alpha is
    1% rather than 5% because about ten lengths are tested. Frames shorter
    than `shortest` are skipped: at 12 ms the partials of a low note cannot
    be separated, which is a different problem.
    Returns None if even the shortest length fails."""
    best = None
    for r in sorted(rows, key=lambda r: r["T"]):
        if r["T"] < shortest:
            continue
        if r["p"] > alpha:
            best = r["T"]
        else:
            break
    return best


def stiff_string_fit(freq_hz, se_hz, k):
    """Physics of a stiff string (Fletcher and Rossing 1991):
        f_k = k f0 sqrt(1 + B k^2),
    fitted by weighted non-linear least squares with Theorem-2 weights,
    like nls(f ~ k * f0 * sqrt(1 + B * k^2), weights = 1/se^2) in R.
    Returns f0, B, their standard errors and standardised residuals."""
    f, se, k = np.asarray(freq_hz), np.asarray(se_hz), np.asarray(k)

    def resid(p):                           # B is carried as B * 1e4 for scaling
        return (f - k * p[0] * np.sqrt(1 + p[1] * 1e-4 * k ** 2)) / se

    r = optimize.least_squares(resid, [f[0] / k[0], 0.4], method="lm")
    J = r.jac
    cov = np.linalg.inv(J.T @ J)
    lev = np.diag(J @ cov @ J.T)             # leverages, as hatvalues() in R
    return dict(f0=r.x[0], B=r.x[1] * 1e-4, se_f0=np.sqrt(cov[0, 0]), se_B=np.sqrt(cov[1, 1]) * 1e-4,
                z=r.fun / np.sqrt(np.clip(1 - lev, 1e-9, None)), chi2=float(np.sum(r.fun ** 2)),
                df=len(f) - 2)


def stiff_z(rec, threshold=8.0):
    """z values for the frame-length check when the reference is the
    stiff-string law, fitted separately in every frame."""
    strong = strong_partials(rec, threshold)
    k = np.arange(1, rec.K + 1)
    out = []
    for i in range(len(rec.times)):
        m = strong[i]
        if m.sum() >= 4 and m[0]:
            out.append(stiff_string_fit(rec.freq_hz[i, m], rec.freq_se_hz[i, m], k[m])["z"])
    return np.concatenate(out) if out else np.array([])


def partial_offset_model(freq_hz, se_hz, use):
    """Extension 3: a frame x partial model for notes that are not harmonic.

    For every frame i and partial k we model

        log(omega_hat_ik / k) = log(lambda_i) + gamma_k + error,   gamma_1 = 0,

    with weights (omega_hat_ik / se_ik)^2 from Theorem 2 (the delta method
    gives var(log omega_hat) = se^2 / omega^2). In R this is

        lm(log(f / k) ~ 0 + factor(frame) + factor(partial), weights = (f / se)^2)

    lambda_i is the pitch of frame i, measured on partial 1's scale, and
    gamma_k is a fixed offset of partial k that is the same in every frame.
    The offsets absorb whatever makes the string inharmonic: stiffness,
    coupling to the guitar body, anything that does not change over time.
    For a harmonic note all gamma_k are 0 and this reduces to Extension 1.

    use is a frames x K logical matrix: which estimates to include.

    Returns the pitch of each frame with its s.e., the offsets in cents
    with their s.e., and standardised residuals z (residual divided by
    sqrt(1 - leverage), with the Theorem-2 variance taken as known). If the
    model and the error bars are right, z ~ N(0, 1)."""
    n, K = freq_hz.shape
    rows, y, wt, where = [], [], [], []
    for i in range(n):
        for j in range(K):
            if use[i, j]:
                x = np.zeros(n + K - 1)
                x[i] = 1                        # frame dummy
                if j > 0:
                    x[n + j - 1] = 1            # partial dummy (partial 1 is the baseline)
                rows.append(x)
                y.append(np.log(freq_hz[i, j] / (j + 1)))
                wt.append((freq_hz[i, j] / se_hz[i, j]) ** 2)
                where.append((i, j))
    X, y, wt = np.array(rows), np.array(y), np.array(wt)
    keep = X.sum(axis=0) > 0                    # drop frames / partials with no data
    X = X[:, keep]
    Xw, yw = X * np.sqrt(wt)[:, None], y * np.sqrt(wt)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.pinv(Xw.T @ Xw)             # (X'WX)^-1, as vcov() with sigma = 1
    lev = np.einsum("ij,jk,ik->i", Xw, cov, Xw)
    resid = yw - Xw @ beta
    z = resid / np.sqrt(np.clip(1 - lev, 1e-12, None))
    se_beta = np.sqrt(np.diag(cov))

    frame_cols = keep[:n]
    n_frames_used = int(frame_cols.sum())
    pitch = np.full(n, np.nan)
    pitch_se = np.full(n, np.nan)
    pitch[frame_cols] = np.exp(beta[:n_frames_used])
    pitch_se[frame_cols] = pitch[frame_cols] * se_beta[:n_frames_used]
    gamma = np.zeros(K)
    gamma_se = np.zeros(K)
    partial_cols = np.where(keep[n:])[0] + 1
    gamma[partial_cols] = beta[n_frames_used:]
    gamma_se[partial_cols] = se_beta[n_frames_used:]
    df = len(y) - X.shape[1]
    return dict(pitch=pitch, pitch_se=pitch_se, offset_cents=1200 / np.log(2) * gamma,
                offset_se_cents=1200 / np.log(2) * gamma_se, z=z, where=where,
                chi2=float(np.sum(resid ** 2)), df=df)


def offset_model_z(rec, threshold=8.0):
    """z values for the frame-length check with Extension 3 as reference."""
    use = strong_partials(rec, threshold)
    use &= use[:, :1]
    return partial_offset_model(rec.freq_hz, rec.freq_se_hz, use)["z"]
