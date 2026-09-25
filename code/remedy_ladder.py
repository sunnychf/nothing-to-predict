"""The sign-averaged correction on the synthetic ladder: forecasts on each rung's contexts
and on K sign-randomised copies of every context, saved for remedy_summary.py.

For a context y_0..y_T with increments d_t = y_t - y_{t-1}, a sign-randomised copy is
    ~y_0 = y_0,   ~y_t = ~y_{t-1} + eps_t d_t,   eps_t i.i.d. +-1,
additive because the ladder's generators are additive (level + cumulated steps). On
N1-N3 the signs of the data's own increments are i.i.d. given their magnitudes, so the
copy has the data's distribution and the model's mean departure over copies is exactly
its prior on that magnitude path; subtracting it (remedy_summary.py) leaves a forecast
whose departure has zero mean under the null. On N4 the bounce makes signs dependent,
the copy destroys the MA(1) structure, and the correction subtracts only what the model
would do without it: that rung asks whether the structure the model does capture
survives the subtraction.

Usage (inside the model's env script):
    python remedy_ladder.py --model chronos [--n 256 --k 4 --horizon 128 --rungs N1,N2,N3_nu5,N4]
Writes <out-dir>/remedy_<model>.npz with, per rung R:
    R/ctx (n, T), R/fut (n, H), R/sigma, R/oracle (n, H),
    R/yhat_raw (n, H), R/mcvar_raw (n, H), R/yhat_sur (n, K, H), R/mcvar_sur (n, K, H)
Contexts come from probe_metrics.make_data with seed 6000 (fixed across models), and the
signs from numpy generator seed 7000 + rung index, so every model sees the same windows.
The contexts are antithetic pairs: n/2 draws and their mirrors about y_0 (every increment
negated, future included), so the sample's mean move and mean slope are exactly zero. Without
this, a model that follows or reverts against the realised slope leaves a residual mean
departure after the correction that is the finite sample's drift, not a failure of the
correction (seed 6000's first 256 draws end 2.6 sigma above their start on average).
"""
import argparse, json, os, time
import numpy as np
from probe_metrics import make_data
from nulls import oracle_forecast
from model_adapters import get_forecaster, SAMPLED
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["chronos", "fincast", "timesfm", "timemoe", "moirai", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial"])
ap.add_argument("--out-dir", default=os.environ.get("REAL_OUT", RESULTS_DIR))
ap.add_argument("--n", type=int, default=256, help="contexts per rung, half of them mirrors of the other half")
ap.add_argument("--no-antithetic", action="store_true")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--horizon", type=int, default=128)
ap.add_argument("--rungs", default="N1,N2,N3_nu5,N4")
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

n, K, T, H = args.n, args.k, args.ctx, args.horizon
fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)
out, done = {}, {}
log(f"antithetic pairs: {not args.no_antithetic}")
for ri, rung in enumerate(args.rungs.split(",")):
    if args.no_antithetic:
        ctx, fut, info = make_data(rung, n, T, H, seed=6000)
    else:
        ctx0, fut0, info = make_data(rung, n // 2, T, H, seed=6000)
        ctx = np.concatenate([ctx0, 2 * ctx0[:, :1] - ctx0]); fut = np.concatenate([fut0, 2 * ctx0[:, :1] - fut0])
    rng = np.random.default_rng(7000 + ri)
    d = np.diff(ctx, axis=1)                                          # (n, T-1)
    eps = rng.choice(np.array([-1.0, 1.0]), size=(n, K, T - 1))
    sur = np.empty((n, K, T)); sur[:, :, 0] = ctx[:, :1]; sur[:, :, 1:] = ctx[:, None, :1] + np.cumsum(eps * d[:, None, :], axis=2)
    log(f"{rung}: {n} contexts, K={K}, sigma {info['sigma']:.3f}")
    t0 = time.time(); yh, mv = fn(ctx)
    log(f"  raw: {time.time()-t0:.0f}s; mean signed dep at h={H}: {np.mean((yh[:, -1] - ctx[:, -1]) / info['sigma']):+.2f} sigma")
    t0 = time.time(); ys, ms = fn(sur.reshape(n * K, T))
    log(f"  sur: {time.time()-t0:.0f}s ({n * K} contexts); mean signed dep at h={H}: {np.mean((ys[:, -1] - sur.reshape(n * K, T)[:, -1]) / info['sigma']):+.2f} sigma")
    out[f"{rung}/ctx"], out[f"{rung}/fut"], out[f"{rung}/sigma"] = ctx, fut, np.float64(info["sigma"])
    out[f"{rung}/oracle"] = oracle_forecast(rung, ctx, info, H)
    out[f"{rung}/yhat_raw"], out[f"{rung}/mcvar_raw"] = yh.astype(np.float32), mv.astype(np.float32)
    out[f"{rung}/yhat_sur"], out[f"{rung}/mcvar_sur"] = ys.reshape(n, K, H).astype(np.float32), ms.reshape(n, K, H).astype(np.float32)
    done[rung] = {"n": n, "K": K, "sigma": float(info["sigma"]), "antithetic": not args.no_antithetic}

os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"remedy_{args.model}.npz")
np.savez_compressed(path, **out)
json.dump({"model": args.model, "rungs": done, "T": T, "H": H, "samples": args.samples if args.model in SAMPLED else 0,
           "seed_data": 6000, "seed_signs": "7000 + rung index", "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(path.replace(".npz", ".json"), "w"), indent=1)
log(f"wrote {path}")
