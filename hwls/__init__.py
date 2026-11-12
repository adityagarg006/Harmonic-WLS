"""hwls: weighted estimation of harmonic components (Irizarry, JTSA 2002),
with our three extensions.

    tapers.py      taper functions, the constants c0..c4 of eq. (15)
    estimator.py   the 7-step estimator for one frame, and for a whole recording
    inference.py   Theorem-2 standard errors, the paper's questions Q1 and Q2
    extensions.py  Extension 1 (pooled pitch), 2 (frame length), 3 (strings)
    simulate.py    simulated signals with a known truth
"""
from .tapers import (TAPERS, LABELS, taper_weights, taper_moments, taper_constants,
                     theorem2_cov, sandwich_cov, leakage_kernel)
from .estimator import fit_frame, analyse_recording, hz_to_rad, rad_to_hz, FrameFit, Recording
from .inference import (noise_spectrum, attach_theorem2_se, attach_sandwich_se,
                        harmonic_deviations, jnd_hz, cents, strong_partials)
from .extensions import (pooled_pitch, harmonic_z, count_outside, frame_length_check,
                         longest_honest_frame, stiff_string_fit, stiff_z, partial_offset_model,
                         offset_model_z, P_OUTSIDE_2SE)
from . import simulate
