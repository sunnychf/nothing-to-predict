"""Run one model on the real-data windows (raw and martingale surrogates) and save its
point forecasts; metrics are computed afterwards by real_summary.py / remedy_summary.py
so every model's numbers come from one code path.

Usage (inside the model's env script):
    python real_probe.py --model chronos   # tsfmfin env,  env.sh
    python real_probe.py --model fincast   # fincast_v1,   env_fincast.sh
    python real_probe.py --model timesfm   # timesfm,      env_timesfm.sh
    python real_probe.py --model timemoe   # fincast_v1,   env_timemoe.sh
    python real_probe.py --model moirai    # moirai,       env_moirai.sh
Reads  the windows file from real_data.py; writes <out-dir>/real_<model>.npz with
  yhat_raw  (n, H)        point forecast on the raw window (sampled models: mean of S trajectories)
  yhat_sur  (n, K, H)     point forecasts on the K sign-randomised surrogates of each window
  mcvar_raw (n, H), mcvar_sur (n, K, H)   Var(samples)/S for sampled models (zeros otherwise)
A windows file with a single surrogate per window (ctx_sur of shape (n, T)) is handled as K=1.
--sur-from N forecasts only surrogate copies N.. (no raw window) and writes real_<model>_fromN.npz,
so a K=16 file whose first four draws equal the K=4 file's can be completed without redoing them.
--mirror forecasts the multiplicative mirror of every raw window instead (x_i = x_{i-1} (1 - r_i):
every return negated, level kept positive) and writes real_<model>_mirror.npz with yhat_mirror,
mcvar_mirror; the mirror-averaged correction of Section 5 needs exactly this one extra pass.
--mirror-sur does the same for every surrogate copy (the null test of the mirror correction on
real increments) and writes real_<model>_mirror_sur.npz with yhat_mirror_sur (n, K, H), mcvar_mirror_sur.
--prefix intraday reads the intraday windows (intraday_data.py; same keys, H=16) and writes intraday_<model>*.npz.
Each model is used exactly as in its Pilot A driver (see model_adapters.py).
"""
import argparse, json, os, time
import numpy as np
from model_adapters import get_forecaster, SAMPLED
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["chronos", "fincast", "timesfm", "timemoe", "moirai", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial"])
ap.add_argument("--windows", default=os.environ.get("REAL_WINDOWS", os.path.join(RESULTS_DIR, "real_windows.npz")))
ap.add_argument("--out-dir", default=os.environ.get("REAL_OUT", RESULTS_DIR))
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--sur-from", type=int, default=0, help="forecast surrogate copies from this index on, and skip the raw window")
ap.add_argument("--mirror", action="store_true", help="forecast the mirrors of the raw windows only")
ap.add_argument("--mirror-sur", action="store_true", help="forecast the mirrors of the surrogate copies only")
ap.add_argument("--prefix", default="real", help="output file prefix: real_<model>... (the daily anchor) or intraday_<model>... (intraday_data.py windows)")
args = ap.parse_args()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

z = np.load(args.windows)
ctx_raw, ctx_sur = z["ctx_raw"], z["ctx_sur"]
if ctx_sur.ndim == 2:
    ctx_sur = ctx_sur[:, None, :]
if args.sur_from:
    ctx_sur = ctx_sur[:, args.sur_from:]
n, K, T = ctx_sur.shape
H = z["fut_raw"].shape[1]
log(f"{n} windows, ctx {T}, H {H}, K {K} surrogates per window{f' (copies {args.sur_from}..)' if args.sur_from else ''}, model {args.model}")
fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)

def dep_sigma(yh, c):
    return np.mean((yh[:, -1] - c[:, -1]) / np.std(np.diff(c, axis=1), axis=1))

out = {}
def mirror_of(c):
    r = c[:, 1:] / c[:, :-1] - 1.0
    m = np.empty_like(c); m[:, 0] = c[:, 0]; m[:, 1:] = c[:, :1] * np.cumprod(1.0 - r, axis=1); return m
if args.mirror_sur:
    flat = mirror_of(ctx_sur.reshape(n * K, T))
    t0 = time.time(); yh, mv = fn(flat)
    out["yhat_mirror_sur"], out["mcvar_mirror_sur"] = yh.reshape(n, K, H).astype(np.float32), mv.reshape(n, K, H).astype(np.float32)
    log(f"  mirror-sur: done in {time.time()-t0:.0f}s ({n * K} windows); mean signed dep at h={H}: {dep_sigma(yh, flat):+.2f} sigma")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.prefix}_{args.model}_mirror_sur.npz"); np.savez_compressed(path, **out)
    json.dump({"model": args.model, "windows": os.path.basename(args.windows), "n": int(n), "K": int(K), "H": int(H), "mirror_sur": True,
               "samples": args.samples if args.model in SAMPLED else 0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(path.replace(".npz", ".json"), "w"), indent=1)
    log(f"wrote {path}"); raise SystemExit(0)
if args.mirror:
    mir = mirror_of(ctx_raw)
    t0 = time.time(); yh, mv = fn(mir)
    out["yhat_mirror"], out["mcvar_mirror"] = yh.astype(np.float32), mv.astype(np.float32)
    log(f"  mirror: done in {time.time()-t0:.0f}s; mean signed dep at h={H}: {dep_sigma(yh, mir):+.2f} sigma")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.prefix}_{args.model}_mirror.npz"); np.savez_compressed(path, **out)
    json.dump({"model": args.model, "windows": os.path.basename(args.windows), "n": int(n), "H": int(H), "mirror": True,
               "samples": args.samples if args.model in SAMPLED else 0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(path.replace(".npz", ".json"), "w"), indent=1)
    log(f"wrote {path}"); raise SystemExit(0)
if not args.sur_from:
    t0 = time.time(); yh, mv = fn(ctx_raw)
    out["yhat_raw"], out["mcvar_raw"] = yh.astype(np.float32), mv.astype(np.float32)
    log(f"  raw: done in {time.time()-t0:.0f}s; mean signed dep at h={H}: {dep_sigma(yh, ctx_raw):+.2f} sigma")
flat = ctx_sur.reshape(n * K, T)
t0 = time.time(); yh, mv = fn(flat)
out["yhat_sur"], out["mcvar_sur"] = yh.reshape(n, K, H).astype(np.float32), mv.reshape(n, K, H).astype(np.float32)
log(f"  sur: done in {time.time()-t0:.0f}s ({n * K} windows); mean signed dep at h={H}: {dep_sigma(yh, flat):+.2f} sigma")

os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"{args.prefix}_{args.model}{f'_from{args.sur_from}' if args.sur_from else ''}.npz")
np.savez_compressed(path, **out)
json.dump({"model": args.model, "windows": os.path.basename(args.windows), "n": int(n), "K": int(K), "sur_from": args.sur_from, "H": int(H),
           "samples": args.samples if args.model in SAMPLED else 0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(path.replace(".npz", ".json"), "w"), indent=1)
log(f"wrote {path}")
