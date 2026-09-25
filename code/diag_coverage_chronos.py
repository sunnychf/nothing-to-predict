"""Calibration of Chronos-T5's 80 percent interval on the ladder (the other drivers record
`coverage_80` and `interval80_width_over_sigma` in their ladder rows; pilot_a.py did not, so
this fills the gap for the coverage table). Same contexts and futures as the ladder
(probe_metrics.make_data, seeds 1000-1002), H=16, S=100 sample paths; the interval is the
0.1-0.9 sample quantile band, as for Moirai and Sundial.

Usage (inside env.sh):  python diag_coverage_chronos.py [amazon/chronos-t5-small]
Writes <SPEC_OUT>/diag_coverage_chronos.json: per rung, per seed, coverage and mean width over
sigma at every horizon.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from pilot_a import load_pipeline, forecast, log
from probe_metrics import make_data
from paths import RESULTS_DIR

OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
model = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-t5-small"
n, T, H, S = 512, 512, 16, 100
pipe = load_pipeline(model, "pretrained", "cuda")
recs = []
for rung in ("N1", "N2", "N3_nu3", "N3_nu5", "N4"):
    for seed in range(3):
        y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
        t0 = time.time(); s = forecast(pipe, y_ctx, H, S, batch=64, seed=seed)
        lo, hi = np.quantile(s, [0.1, 0.9], axis=1)
        rec = {"model": model, "rung": rung, "seed": seed, "n": n, "H": H, "num_samples": S,
               "coverage_80": np.mean((y_true >= lo) & (y_true <= hi), 0).tolist(),
               "interval80_width_over_sigma": (np.mean(hi - lo, 0) / info["sigma"]).tolist()}
        recs.append(rec)
        log(f"{rung} s{seed}: cov80 h1={rec['coverage_80'][0]:.3f} h16={rec['coverage_80'][-1]:.3f} "
            f"width/sigma h1={rec['interval80_width_over_sigma'][0]:.2f} h16={rec['interval80_width_over_sigma'][-1]:.2f} ({time.time() - t0:.0f}s)")
json.dump(recs, open(os.path.join(OUT, "diag_coverage_chronos.json"), "w"), indent=1)
log("wrote diag_coverage_chronos.json")
