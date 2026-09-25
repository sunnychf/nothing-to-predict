"""Does the imported prior shrink when the model sees more evidence of no drift?

One set of 128 zero-drift random walks (N1, seed 4000, level 100, step 1 percent) of length
2048 + 128 is drawn once; for each context length T in --lengths the model forecasts the LAST
T points of every series, so contexts are nested suffixes and the futures are identical across
T. A forecaster that updates on evidence should move towards persistence as T grows; a prior
that is applied regardless of the evidence does not. Chronos-T5 reads at most 512 points, so
for it the sweep stops there; the others take the input's length up to their own maximum
(Chronos-Bolt and TimesFM 2048, Time-MoE 4096, Chronos-2 8192, Moirai rebuilt at each length).

Usage (inside the model's env script):
    python diag_context.py --model chronosbolt [--lengths 128,256,512,1024,2048 --n 128 --horizon 128]
Writes <out-dir>/diag_context_<model>.json: per T, mean signed departure at h=16 and h=128 in
sigma units (with the standard error over series), fraction of forecasts above the last value,
rms departure, and skill against persistence (Monte Carlo term removed for the sampled models).
The same model/forecast defaults as every other run (model_adapters.py).
"""
import argparse, json, os, time
import numpy as np
from nulls import RUNGS
from model_adapters import get_forecaster, SAMPLED
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["chronos", "fincast", "timesfm", "timemoe", "moirai", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial"])
ap.add_argument("--lengths", default="128,256,512,1024,2048")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--horizon", type=int, default=128)
ap.add_argument("--seed", type=int, default=4000)
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out-dir", default=os.environ.get("SPEC_OUT", RESULTS_DIR))
args = ap.parse_args()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

lengths = [int(t) for t in args.lengths.split(",")]
T_MAX, H = max(lengths), args.horizon
y, info = RUNGS["N1"](args.n, T_MAX, H, np.random.default_rng(args.seed))
sigma = float(info["sigma"]); fut = y[:, T_MAX:]
log(f"{args.model}: {args.n} series, T_max {T_MAX}, H {H}, sigma {sigma:.3f}, lengths {lengths}")
recs = []
for T in lengths:
    ctx = y[:, T_MAX - T:T_MAX]
    fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch, ctx_len=T)
    t0 = time.time(); yh, mcv = fn(ctx)
    last = ctx[:, -1:]
    dep = (yh - last) / sigma                                        # (n, H) in sigma units
    err = (fut - yh) ** 2 - mcv; err_p = (fut - last) ** 2
    r = {"T": int(T), "n": int(args.n), "H": int(H), "seconds": round(time.time() - t0, 1),
         "mean_dep": {str(h): float(dep[:, h - 1].mean()) for h in (1, 16, 64, 128)},
         "mean_dep_se": {str(h): float(dep[:, h - 1].std(ddof=1) / np.sqrt(args.n)) for h in (1, 16, 64, 128)},
         "frac_up": {str(h): float((dep[:, h - 1] > 0).mean()) for h in (1, 16, 64, 128)},
         "rms_dep": {str(h): float(np.sqrt((dep[:, h - 1] ** 2).mean())) for h in (1, 16, 64, 128)},
         "skill": {str(h): float(1.0 - err[:, h - 1].mean() / err_p[:, h - 1].mean()) for h in (1, 16, 64, 128)}}
    recs.append(r)
    log(f"  T={T:5d}: h=16 dep {r['mean_dep']['16']:+.2f} up {r['frac_up']['16']:.2f} skill {r['skill']['16']:+.3f} | "
        f"h=128 dep {r['mean_dep']['128']:+.2f}±{r['mean_dep_se']['128']:.2f} up {r['frac_up']['128']:.2f} skill {r['skill']['128']:+.3f} ({r['seconds']}s)")
os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"diag_context_{args.model}.json")
json.dump({"model": args.model, "rung": "N1", "seed": args.seed, "sigma": sigma, "samples": args.samples if args.model in SAMPLED else 0,
           "nested_suffixes": True, "lengths": lengths, "records": recs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(path, "w"), indent=1)
log(f"wrote {path}")
