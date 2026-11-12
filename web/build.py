"""Build web/index.html: one self-contained file with the scripts, the
style sheet, the numbers from results/*.json and two short audio clips.

Run the experiments first (experiments/run_all.sh), then
    python3 web/build.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
RES = ROOT / "results"


def load(name):
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def clip_b64(file, start_s, seconds):
    """A mono 16-bit WAV clip of a recording, base64-encoded."""
    x, fs = sf.read(ROOT / "data" / file, always_2d=True)
    x = x.mean(1)
    seg = x[int(start_s * fs): int((start_s + seconds) * fs)]
    seg = 0.9 * seg / np.max(np.abs(seg))
    buf = io.BytesIO()
    sf.write(buf, seg, fs, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


e2, e5, e6, e7 = (load(n) for n in ["e2_theorem2_montecarlo", "e5_ext1_pooled_pitch",
                                     "e6_ext2_frame_length", "e7_ext3_strings"])
labels = {"clarinet_Bb4": "Clarinet B♭4", "clarinet_D4": "Clarinet D4", "flute_A5": "Flute A5", "oboe_Bb5": "Oboe B♭5"}
glide = [v for g in e6["glide_simulation"]["results"].values() for v in g["share_outside"]]
real = e7["real"]
data = {
    "mc": dict(R=e2["settings"]["R"], worst=e2["max_abs_ratio_minus_1"], coverage=e2["coverage_range_T_ge_256"]),
    "ext1": dict(sim=e5["simulation"], track=e5["track_clarinet_Bb4"],
                 rows=[dict(r, label=labels[r["recording"]]) for r in e5["recordings"]]),
    "ext2": dict(curves={k: e6[k]["rows"] for k in ["clarinet_Bb4", "clarinet_D4", "guitar_steel_D3"]},
                 best={k: e6[k]["longest_honest_T"] for k in ["clarinet_Bb4", "clarinet_D4", "guitar_steel_D3"]},
                 combined=e6["combined"], glide_min=min(glide), glide_max=max(glide)),
    "ext3": dict(
        offsets=dict(steel=real["guitar_steel_D3"]["offsets_cents"], steel_se=real["guitar_steel_D3"]["offsets_se_cents"],
                     nylon=real["guitar_nylon_D3"]["offsets_cents"], nylon_se=real["guitar_nylon_D3"]["offsets_se_cents"]),
        B_steel=real["guitar_steel_D3"]["B_from_offsets"], shift_steel=real["guitar_steel_D3"]["partial1_shift_cents"],
        checks=dict(steel={m: c["share"] for m, c in real["guitar_steel_D3"]["checks"].items()},
                    nylon={m: c["share"] for m, c in real["guitar_nylon_D3"]["checks"].items()},
                    sim_fixed=e7["simulation"]["results"]["fixed offsets"]["share_outside"],
                    sim_wander=e7["simulation"]["results"]["wandering offsets"]["share_outside"]),
        se={k.split("_")[1]: dict(omega1=real[k]["se_omega1_mhz"], model=real[k]["se_pitch_offset_model_mhz"],
                                  inflated=real[k]["se_pitch_inflated_mhz"]) for k in ["guitar_steel_D3", "guitar_nylon_D3"]},
        sim_coverage=e7["simulation"]["results"]["fixed offsets"]["coverage_pitch"],
        sim_gain=e7["simulation"]["results"]["fixed offsets"]["gain"],
        decay_offsets=e7["decay_model_steel"]["checks"]["offsets"]),
    "audio": dict(clarinet=clip_b64("clarinet_Bb4.wav", 1.0, 1.2), guitar=clip_b64("guitar_acoustic_D3.wav", 0.0, 2.0)),
}

src = WEB / "src"
page = (src / "page.html").read_text()
for tag, text in [("/*STYLE*/", (src / "style.css").read_text()), ("/*DATA*/", json.dumps(data)),
                  ("/*HWLS*/", (src / "hwls.js").read_text()), ("/*PLOT*/", (src / "plot.js").read_text()),
                  ("/*APP*/", (src / "app.js").read_text())]:
    page = page.replace(tag, text)
(WEB / "index.html").write_text(page)
print(f"wrote {WEB / 'index.html'} ({len(page) / 1e6:.2f} MB)")
