# Weighted estimation of harmonic components: implementation and extensions

Code for our DA-3041 project on

> R. A. Irizarry, *Weighted estimation of harmonic components in a musical sound signal*,
> Journal of Time Series Analysis 23(1), 29–48, 2002.

Group 3: Rachit Gupta, Nilay Mittal, Aditya Garg, Freya Sheth

The code is in Python. It is written to be read by someone who knows R: every function says what
it computes, and where there is a direct R equivalent (`lm()`, `nls()`, `integrate()`,
`binom.test()`, …) the comment names it.

## What is in the folder

| Path | Contents |
|---|---|
| `hwls/tapers.py` | Taper functions w(s), the integrals W<sub>n</sub>, U<sub>n</sub> (eq. 4) and the constants c<sub>0</sub>…c<sub>4</sub> (eq. 15); the Theorem-2 covariance matrix (eq. 14) |
| `hwls/estimator.py` | The estimator for one frame (eqs. 7–9), in the seven steps used in our slides; the paper's full weighted non-linear least squares (eq. 3); a whole recording cut into frames (Section 4) |
| `hwls/inference.py` | Noise-spectrum estimate, Theorem-2 standard errors, the paper's two questions (Q1: out of tune? Q2: harmonic?) |
| `hwls/extensions.py` | Our Extensions 1, 2 and 3 |
| `hwls/simulate.py` | Simulated signals with a known truth, for checking the theorems |
| `experiments/e1` … `e7` | One script per study; each writes its numbers to `results/` and its figures to `figures/` |
| `data/` | The six recordings we used (see below) |
| `web/index.html` | The demo page: one file, works offline in any modern browser |
| `web/src/`, `web/build.py` | Sources of the demo page (`hwls.js` is a line-by-line JavaScript port of the estimator) |

## Running it

```bash
pip install numpy scipy matplotlib soundfile
cd experiments
./run_all.sh              # all studies, about 4 minutes
FAST=1 ./run_all.sh       # fewer Monte Carlo runs, for a quick check
python3 ../web/build.py   # rebuild the demo page from the new results
```

Each script can also be run on its own, e.g. `python3 e5_ext1_pooled_pitch.py`.
All random numbers are seeded, so two runs give the same numbers.

To fit one frame yourself:

```python
import hwls, soundfile as sf
x, fs = sf.read("data/clarinet_Bb4.wav")
frame = x[44100:44100 + 1025]                       # 23 ms, starting at 1 s
fit = hwls.fit_frame(frame, K=15, f0_hint=hwls.hz_to_rad(466.16, fs))
hwls.rad_to_hz(fit.omega, fs)                       # partial frequencies (Hz)
hwls.rad_to_hz(fit.se["omega"], fs)                 # their Theorem-2 standard errors
```

## Where each part of the paper is

| Paper | Code |
|---|---|
| Model (2), criterion (3) | `estimator._wls_refine` (the paper's S-Plus `nls()` step) |
| Eqs. (7)–(9): tapered periodogram estimator | `estimator.fit_frame`, steps 2–5 |
| Eq. (4), eq. (15): W<sub>n</sub>, U<sub>n</sub>, c<sub>0</sub>…c<sub>4</sub> | `tapers.taper_moments`, `tapers.taper_constants` |
| Theorem 2, eq. (14) | `tapers.theorem2_cov`; plug-in version `inference.attach_theorem2_se` |
| Noise spectrum estimate f̂ | `inference.noise_spectrum` |
| Section 4, Fig. 2 (Q1) and Figs. 3, 5 (Q2) | `experiments/e4_replication.py` |

## The studies and what they found

| Script | What it checks | Result |
|---|---|---|
| `e1_taper_constants.py` | eq. (15) against an independent sandwich formula; Remark 3 | agree to 3×10<sup>−15</sup>; no taper gives c<sub>0</sub> = 12, Hann 28.1 |
| `e2_theorem2_montecarlo.py` | Theorem 2 by simulation: 5 tapers × 3 noise types × 5000 runs | variances within 5.4% of the formula; 95% intervals cover 91–95% for T ≥ 256 |
| `e3_leakage.py` | why taper | a partial 30 dB below its neighbour is missed in 100% of runs without a taper, 0% with Hann |
| `e4_replication.py` | the paper's Section 4 on library recordings | clarinet off the note in 67% of frames but always inside the ear's ±0.03 semitone; harmonic (5.6% of deviations outside ±2 s.e.); guitar inharmonic (84%) |
| `e5_ext1_pooled_pitch.py` | Extension 1: pool all partials | pitch s.e. 4.1× smaller on clarinet B♭4, 3.0× on D4; simulation 2.10× (theory 2.12×) with 94% coverage |
| `e6_ext2_frame_length.py` | Extension 2: longest honest frame | B♭4: 1025 samples (the paper's choice); D4: 4100 (4× longer); guitar: none. Both together: D4 pitch s.e. 122 → 4.9 mHz |
| `e7_ext3_strings.py` | Extension 3: frame × partial model for strings | honest in simulation (4.2% outside ±2, 94% coverage); on real guitars the misfit falls from 89% to 19% but not to 5% |

## Data

All recordings are 44.1 kHz single notes from open sample libraries; the paper's own recordings were never released.

| File | Note | Source |
|---|---|---|
| `clarinet_Bb4.wav` | B♭4, 466.16 Hz | VSCO-2 Community Edition (CC0), via nbrosowsky/tonejs-instruments |
| `clarinet_D4.wav` | D4, 293.66 Hz | same |
| `flute_A5.wav` | A5, 880 Hz | sgossner/VSCO-2-CE (CC0) |
| `oboe_Bb5.wav` | B♭5, 932.33 Hz | sgossner/VSCO-2-CE (CC0) |
| `guitar_acoustic_D3.wav` | D3, 146.83 Hz (steel strings) | University of Iowa Musical Instrument Samples, via tonejs-instruments |
| `guitar_nylon_D3.wav` | D3, 146.83 Hz (nylon strings) | Freesound pack 11573, via tonejs-instruments |

Winds: 43 frames of 1025 samples starting at 1.0 s (the attack is skipped). Guitars: 30 frames of 3000 samples starting at 0.1 s.
The libraries have no sustained clarinet A, so B♭4 stands in for the paper's A = 441 Hz; the guitar note is the paper's.

## Conventions

Inside `hwls`, frequencies are in radians per sample (Hz = ω·f<sub>s</sub>/2π) and time runs t = 1, …, T, as in the paper.
The noise spectrum follows the paper: f(λ) = (1/2π) Σ<sub>u</sub> c(u) e<sup>−iλu</sup>, so white noise of variance σ² has f = σ²/2π.
