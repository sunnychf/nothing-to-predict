"""Does the magnitude of Chronos's upward drift depend on the price level?

The four-condition control (diag_direction_control.py) found every condition
drifts up but by different amounts: +6.4 sigma at level -100, +3.4 at +100,
+2.1 centred on zero. This sweeps seven offsets on ONE set of increments
(sigma = 1 explicit, H=128, n=128, S=100) to see whether the bias is a
function of |level|, of sign, or neither.
"""
import json, os, sys, numpy as np
from pilot_a import load_pipeline, forecast, log
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
H, n, S, T = 128, 128, 100, 512
LEVELS = [-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0]
J = f"{OUT}/diag_direction_level.jsonl"
done = set()
if os.path.exists(J):
    for l in open(J):
        try: done.add(json.loads(l)["level"])
        except Exception: pass
rng = np.random.default_rng(4000)
z = rng.standard_normal((n, T + H))                     # sigma = 1
base = np.cumsum(z, axis=1)
pipe = load_pipeline("amazon/chronos-t5-small", "pretrained", "cuda")
for lv in LEVELS:
    if lv in done: log(f"skip level {lv:g}"); continue
    ctx = (lv + base)[:, :T]
    s = forecast(pipe, ctx, H, S, batch=8, seed=0)
    dep = s.mean(1) - ctx[:, -1:]
    np.savez_compressed(f"{OUT}/departures_level_{lv:g}_H{H}.npz", dep=dep.astype(np.float32), sigma=1.0)
    scale = float(np.abs(ctx).mean())                    # what Chronos divides by
    r = {"level": lv, "H": H, "n": n, "S": S, "mean_abs_context": scale,
         "grid_step_sigma": 30.0 / 4093 * scale,
         "mean_dep_sigma": {h + 1: float(dep[:, h].mean()) for h in (0, 15, 63, H - 1)},
         "median_dep_sigma": {h + 1: float(np.median(dep[:, h])) for h in (0, 15, 63, H - 1)},
         "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean())) for h in (0, 15, 63, H - 1)},
         "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)}}
    with open(J, "a") as f: f.write(json.dumps(r) + "\n")
    log(f"level {lv:>6g} (scale {scale:7.1f}, grid {r['grid_step_sigma']:.3f}σ): h=1 mean {r['mean_dep_sigma'][1]:+.3f} up {r['frac_up'][1]:.2f} | "
        f"h=128 mean {r['mean_dep_sigma'][128]:+.2f} median {r['median_dep_sigma'][128]:+.2f} up {r['frac_up'][128]:.2f}")
log("=== LEVEL SWEEP COMPLETE ===")
