"""E3. Why taper at all?

Tapering makes the frequency s.e. larger (Hann: 1.53 times, see E1). What it
buys is protection against leakage: without a taper, a loud partial's
side-lobes (-13 dB) can be taller than a quiet neighbour, and the search
then locks onto a side-lobe instead of the real partial.

Set-up: a strong partial and one 30 dB weaker, `sep` Fourier bins apart,
little noise. We look for the two tallest separated peaks of the tapered
periodogram and record how often the weaker one is missed (no peak within
1 bin of it)."""
from common import *  # noqa: F401,F403
import hwls

T, sigma = 512, 0.01
R = 150 if FAST else 400
t = np.arange(1, T + 1)
w_strong = 2 * np.pi * 0.08
weak = 10 ** (-30 / 20)
seps = [3, 4, 6, 8, 12, 16, 24, 32]
rng = np.random.default_rng(4)


def two_tallest_peaks(y, taper, pad=8):
    """Frequencies of the two tallest local maxima of the tapered
    periodogram that are at least 1.5 bins apart."""
    z = hwls.taper_weights(taper, T) * y
    N = pad * T
    mag = np.abs(np.fft.rfft(z, N))
    grid = 2 * np.pi * np.arange(len(mag)) / N
    peaks = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    peaks = peaks[np.argsort(-mag[peaks])]
    chosen = []
    for p in peaks:
        if all(abs(grid[p] - grid[c]) >= 1.5 * 2 * np.pi / T for c in chosen):
            chosen.append(p)
        if len(chosen) == 2:
            break
    return grid[chosen]


miss = {taper: [] for taper in ["rectangular", "hann", "blackman"]}
for sep in seps:
    w_weak = w_strong + 2 * np.pi * sep / T
    count = {taper: 0 for taper in miss}
    for _ in range(R):
        ph = rng.uniform(0, 2 * np.pi, 2)
        y = np.cos(w_strong * t + ph[0]) + weak * np.cos(w_weak * t + ph[1]) + sigma * rng.standard_normal(T)
        for taper in miss:
            found = two_tallest_peaks(y, taper)
            if np.min(np.abs(found - w_weak)) > 2 * np.pi / T:
                count[taper] += 1
    for taper in miss:
        miss[taper].append(count[taper] / R)
    print(f"separation {sep:2d} bins: missed", {k: v[-1] for k, v in miss.items()})
save_json("e3_leakage", dict(T=T, weak_db=-30, sigma=sigma, R=R, separation_bins=seps, miss_rate=miss))
