"""Level / shape decoupling: is the drift a function of where the input sits after the model's own
scaling, or of the trajectory's shape?

One set of 128 zero-drift random-walk increments (seed 4000, sigma = 1, the construction of
diag_direction_level.py) is placed at several raw levels: y = level + cumsum(z). The z-scored
trajectory is identical at every level, so a model that centres and scales its input by the
context's mean and standard deviation sees the same input at every level and must return the
same departure in sigma units, up to numerical noise; a model that scales without centring
(Chronos-T5 divides by the mean absolute value and quantises) sees a different token sequence at
every level. For each model and level the departure at h = 16 and h = 128 is recorded, and the
per-series difference from the level-100 forecast (the paper's setting), whose maximum over
series and horizons is the model's level sensitivity.

Usage (inside the model's env script):
    python diag_level_all.py --model timesfm [--levels -100,-10,0,10,100 --n 128 --horizon 128]
Writes <out-dir>/diag_level_<model>.json.
"""
import argparse, json, os, time
import numpy as np
from model_adapters import get_forecaster, SAMPLED
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["chronos", "fincast", "timesfm", "timemoe", "moirai", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial"])
ap.add_argument("--levels", default="-100,-10,0,10,100")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--horizon", type=int, default=128)
ap.add_argument("--seed", type=int, default=4000)
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out-dir", default=os.environ.get("SPEC_OUT", RESULTS_DIR))
args = ap.parse_args()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

n, T, H = args.n, 512, args.horizon
levels = [float(v) for v in args.levels.split(",")]
rng = np.random.default_rng(args.seed)
z = rng.standard_normal((n, T + H)); base = np.cumsum(z, axis=1)               # sigma = 1
fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)
deps = {}; recs = []
for lv in levels:
    ctx = (lv + base)[:, :T]
    t0 = time.time(); yh, mcv = fn(ctx)
    dep = yh - ctx[:, -1:]                                                       # (n, H), sigma = 1
    deps[lv] = dep
    r = {"level": lv, "n": n, "H": H, "seconds": round(time.time() - t0, 1),
         "mean_dep": {str(h): float(dep[:, h - 1].mean()) for h in (1, 16, 64, 128)},
         "mean_dep_se": {str(h): float(dep[:, h - 1].std(ddof=1) / np.sqrt(n)) for h in (1, 16, 64, 128)},
         "frac_up": {str(h): float((dep[:, h - 1] > 0).mean()) for h in (1, 16, 64, 128)},
         "rms_dep": {str(h): float(np.sqrt((dep[:, h - 1] ** 2).mean())) for h in (1, 16, 64, 128)}}
    recs.append(r)
    log(f"  level {lv:+6g}: h=16 dep {r['mean_dep']['16']:+.2f} up {r['frac_up']['16']:.2f} | h=128 dep {r['mean_dep']['128']:+.2f}±{r['mean_dep_se']['128']:.2f} up {r['frac_up']['128']:.2f} ({r['seconds']}s)")
ref = deps[100.0] if 100.0 in deps else deps[levels[-1]]
for r in recs:
    d = deps[r["level"]] - ref
    r["max_abs_diff_from_level100"] = float(np.abs(d).max())                   # over series and horizons, sigma units
    r["rms_diff_from_level100_h128"] = float(np.sqrt((d[:, -1] ** 2).mean()))
os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"diag_level_{args.model}.json")
json.dump({"model": args.model, "rung": "N1", "seed": args.seed, "sigma": 1.0, "levels": levels, "samples": args.samples if args.model in SAMPLED else 0,
           "records": recs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(path, "w"), indent=1)
log(f"level sensitivity (max |dep - dep at level 100| over series and horizons): " + ", ".join(f"{r['level']:+g}: {r['max_abs_diff_from_level100']:.3g}" for r in recs))
log(f"wrote {path}")
