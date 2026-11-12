"""Things every experiment script needs: file locations, reading a
recording, a consistent plot style, and saving results as JSON."""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import soundfile as sf                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # so "import hwls" finds the package
FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# FAST=1 runs fewer Monte Carlo replications (a quick check that things work).
FAST = os.environ.get("FAST") == "1"

# Recordings used, with the paper's settings: frame length T, number of
# partials K, where the analysed stretch starts and how many frames.
RECORDINGS = {
    "clarinet_Bb4": dict(file="clarinet_Bb4.wav", label="Clarinet B$\\flat$4", nominal=466.16,
                         T=1025, K=15, start=1.0, n_frames=43),
    "clarinet_D4": dict(file="clarinet_D4.wav", label="Clarinet D4", nominal=293.66,
                        T=1025, K=15, start=1.0, n_frames=43),
    "flute_A5": dict(file="flute_A5.wav", label="Flute A5", nominal=880.0,
                     T=1025, K=10, start=1.0, n_frames=43),
    "oboe_Bb5": dict(file="oboe_Bb5.wav", label="Oboe B$\\flat$5", nominal=932.33,
                     T=1025, K=10, start=1.0, n_frames=43),
    "guitar_steel_D3": dict(file="guitar_acoustic_D3.wav", label="Steel guitar D3", nominal=146.83,
                            T=3000, K=12, start=0.1, n_frames=30),
    "guitar_nylon_D3": dict(file="guitar_nylon_D3.wav", label="Nylon guitar D3", nominal=146.83,
                            T=3000, K=12, start=0.1, n_frames=30),
}

# Colours (blue, orange, green, then a neutral grey) and a plain style sized
# for a two-column paper: one column is 3.25 in wide, the full page 6.75 in.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
GREY, LIGHT = "#8a8984", "#e4e3df"
BAND = "#cde2fb"
BAND2 = "#fde0d2"
COL1, COL2 = 3.25, 6.75
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6, "axes.titlelocation": "left",
    "lines.linewidth": 1.3, "lines.markersize": 3.5, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.02, "figure.constrained_layout.wspace": 0.04,
})


def load_recording(key):
    """Read a .wav file, average the channels and scale the peak to 1
    (like tuneR::readWave followed by normalisation in R)."""
    x, fs = sf.read(DATA / RECORDINGS[key]["file"], always_2d=True)
    x = x.mean(axis=1)
    return x / np.max(np.abs(x)), fs


def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  figure:", name)


def save_json(name, obj):
    """Write results to results/<name>.json (numpy types converted)."""
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(RES / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=convert)
    print("  results:", name)
