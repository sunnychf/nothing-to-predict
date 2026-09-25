"""FinCast declared-frequency test, re-run to record the DIRECTION of the
drift under each label (the first run predates the direction metric)."""
import json, os, numpy as np
from pilot_a_fincast import load_fincast, forecast_fincast, log, FREQ_NAMES
from nulls import RUNGS
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
H, n = 128, 128
api = load_fincast(H)
y, info = RUNGS["N1"](n, 512, H, np.random.default_rng(3000))     # same seed as the freq stage
ctx = y[:, :512]; sig = info["sigma"]
s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
recs = []
for f in (0, 1, 2):
    yhat, _ = forecast_fincast(api, ctx, H, f, batch=32)
    dep = yhat - ctx[:, -1:]
    np.savez_compressed(f"{OUT}/departures_fincast_freq{f}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
    r = {"freq": f, "freq_name": FREQ_NAMES[f],
         "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
         "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
         "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
         "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
    recs.append(r)
    log(f"f{f} {FREQ_NAMES[f]:>15}: h=128 mean {r['mean_dep_sigma'][128]:+.3f}σ rms {r['rms_dep_sigma'][128]:.3f} "
        f"up {r['frac_up'][128]:.2f} corr(slope) {r['corr_slope64'][128]:+.2f}")
json.dump(recs, open(f"{OUT}/diag_freq_direction_fincast.json", "w"), indent=1)
log("wrote diag_freq_direction_fincast.json")
