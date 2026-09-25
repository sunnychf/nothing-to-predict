"""Does FinCast share Chronos's upward drift on zero-drift random walks?
Same contexts as the Chronos spectral run (seed 4000), H=128 and 256, so the
two models are compared on identical inputs. Saves raw departures."""
import json, os, sys, numpy as np
from pilot_a_fincast import load_fincast, forecast_fincast, log
from nulls import RUNGS
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
api = load_fincast(256)
recs = []
for rung in ("N1", "N2", "N4"):
    for H in (128, 256):
        y, info = RUNGS[rung](128, 512, H, np.random.default_rng(4000))
        ctx, truth = y[:, :512], y[:, 512:]
        yhat, _ = forecast_fincast(api, ctx, H, 0, batch=32)
        dep = yhat - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{OUT}/departures_fincast_{rung}_H{H}.npz", dep=dep.astype(np.float32),
                            yhat=yhat.astype(np.float32), last=ctx[:, -1].astype(np.float32),
                            truth=truth.astype(np.float32), sigma=info["sigma"])
        sig = info["sigma"]
        rec = {"model": "fincast_v1", "rung": rung, "H": H, "n": 128,
               "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, H - 1)},
               "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, H - 1)},
               "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, H - 1)},
               "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, H - 1)}}
        recs.append(rec)
        log(f"{rung} H={H}: h=1 mean {rec['mean_dep_sigma'][1]:+.3f} up {rec['frac_up'][1]:.2f} | "
            f"h={H} mean {rec['mean_dep_sigma'][H]:+.3f} up {rec['frac_up'][H]:.2f} "
            f"corr(slope64) {rec['corr_slope64'][H]:+.2f}")
json.dump(recs, open(f"{OUT}/diag_direction_fincast.json", "w"), indent=1)
log("wrote diag_direction_fincast.json")
