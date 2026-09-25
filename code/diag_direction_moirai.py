"""Moirai direction at H=128 with the default patch_size='auto', on the same
contexts as the Chronos/FinCast direction runs (seed 4000), so Table tab:dir
can carry a Moirai row. The freq stage covers the five fixed patch sizes; this
adds the model's own default. Runs inside the moirai env."""
import json, os, numpy as np
from pilot_a_moirai import forecast_moirai, log, MODEL_TAG
from nulls import RUNGS
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
H, n, S = 128, 128, 100
recs = []
for rung in ("N1", "N2", "N4"):
    y, info = RUNGS[rung](n, 512, H, np.random.default_rng(4000))
    ctx = y[:, :512]; sig = info["sigma"]
    s = forecast_moirai(ctx, H, S, "auto", batch=16, seed=0)
    dep = s.mean(1) - ctx[:, -1:]
    s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
    np.savez_compressed(f"{OUT}/departures_moirai_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
    r = {"model": MODEL_TAG, "patch": "auto", "rung": rung, "H": H, "n": n, "S": S,
         "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
         "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
         "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
         "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
    recs.append(r)
    log(f"{rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h=128 mean {r['mean_dep_sigma'][128]:+.2f} up {r['frac_up'][128]:.2f} corr(slope) {r['corr_slope64'][128]:+.2f}")
json.dump(recs, open(f"{OUT}/diag_direction_moirai.json", "w"), indent=1)
log("wrote diag_direction_moirai.json")
