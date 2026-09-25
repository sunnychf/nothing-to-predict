"""Raw or sign-randomised pass of real_probe.py for a slice of windows (K = 1 windows files), so a slow sampled model
can be split across processes. Same forecaster (model_adapters.get_forecaster) and outputs as real_probe.py; only
the sampling seed stream differs by shard (each shard seeds 0 at its start). --merge writes the standard
<prefix>_<model>.npz/.json (yhat_raw, mcvar_raw, yhat_sur (n, 1, H), mcvar_sur) from the raw and sur parts."""
import argparse, glob, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--windows", required=True)
ap.add_argument("--out-dir", required=True); ap.add_argument("--prefix", default="falling")
ap.add_argument("--part", choices=["raw", "sur"]); ap.add_argument("--lo", type=int); ap.add_argument("--hi", type=int)
ap.add_argument("--samples", type=int, default=100); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


sd = os.path.join(args.out_dir, "shards"); os.makedirs(sd, exist_ok=True)
z = np.load(args.windows); ctx_raw, ctx_sur = z["ctx_raw"], z["ctx_sur"]
if ctx_sur.ndim == 3:
    assert ctx_sur.shape[1] == 1, "K = 1 windows only"; ctx_sur = ctx_sur[:, 0]
n, H = len(ctx_raw), z["fut_raw"].shape[1]


def collect(part):
    parts = glob.glob(os.path.join(sd, "%s_%s_%s_part*.npz" % (args.prefix, args.model, part)))
    rng = sorted((int(p.rsplit("_part", 1)[1].split("_")[0]), int(p.rsplit("_", 1)[1].split(".")[0]), p) for p in parts)
    assert rng and rng[0][0] == 0 and rng[-1][1] == n and all(a[1] == b[0] for a, b in zip(rng[:-1], rng[1:])), (part, rng)
    Z = [np.load(p) for _, _, p in rng]
    return np.concatenate([q["yhat"] for q in Z]), np.concatenate([q["mcvar"] for q in Z]), [[lo, hi] for lo, hi, _ in rng]


if args.merge:
    yr, mr, sr = collect("raw"); ys, ms, ss = collect("sur")
    out = {"yhat_raw": yr, "mcvar_raw": mr, "yhat_sur": ys[:, None], "mcvar_sur": ms[:, None]}
    path = os.path.join(args.out_dir, "%s_%s.npz" % (args.prefix, args.model)); np.savez_compressed(path, **out)
    json.dump({"model": args.model, "windows": os.path.basename(args.windows), "n": int(n), "K": 1, "sur_from": 0, "H": int(H),
               "samples": args.samples, "sharded": {"raw": sr, "sur": ss}, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(path.replace(".npz", ".json"), "w"), indent=1)
    log("merged -> %s" % path); raise SystemExit(0)

from model_adapters import get_forecaster

fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)
c = (ctx_raw if args.part == "raw" else ctx_sur)[args.lo:args.hi]
t0 = time.time(); yh, mv = fn(c)
path = os.path.join(sd, "%s_%s_%s_part%d_%d.npz" % (args.prefix, args.model, args.part, args.lo, args.hi))
np.savez_compressed(path, yhat=yh.astype(np.float32), mcvar=mv.astype(np.float32))
log("%s windows %d:%d done in %.0fs -> %s" % (args.part, args.lo, args.hi, time.time() - t0, path))
