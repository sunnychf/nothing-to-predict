"""Mirror pass of real_probe.py for a slice of windows, so a slow sampled model (Chronos-T5 on CPU) can be split
across processes. Same forecaster (model_adapters.get_forecaster), same multiplicative mirror, same outputs;
only the sampling seed stream differs by shard (each shard seeds 0 at its start). --merge writes the standard
<prefix>_<model>_mirror.npz/.json from the parts."""
import argparse, glob, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--windows", required=True)
ap.add_argument("--out-dir", required=True); ap.add_argument("--prefix", default="falling")
ap.add_argument("--lo", type=int); ap.add_argument("--hi", type=int)
ap.add_argument("--samples", type=int, default=100); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


sd = os.path.join(args.out_dir, "shards"); os.makedirs(sd, exist_ok=True)
z = np.load(args.windows); ctx_raw = z["ctx_raw"]; n = len(ctx_raw); H = z["fut_raw"].shape[1]
stem = "%s_%s_mirror" % (args.prefix, args.model)
if args.merge:
    parts = glob.glob(os.path.join(sd, stem + "_part*.npz"))
    rng = sorted((int(p.rsplit("_part", 1)[1].split("_")[0]), int(p.rsplit("_", 1)[1].split(".")[0]), p) for p in parts)
    assert rng and rng[0][0] == 0 and rng[-1][1] == n and all(a[1] == b[0] for a, b in zip(rng[:-1], rng[1:])), rng
    Z = [np.load(p) for _, _, p in rng]
    out = {"yhat_mirror": np.concatenate([q["yhat_mirror"] for q in Z]), "mcvar_mirror": np.concatenate([q["mcvar_mirror"] for q in Z])}
    path = os.path.join(args.out_dir, stem + ".npz"); np.savez_compressed(path, **out)
    json.dump({"model": args.model, "windows": os.path.basename(args.windows), "n": int(n), "H": int(H), "mirror": True,
               "samples": args.samples, "sharded": [[lo, hi] for lo, hi, _ in rng], "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(path.replace(".npz", ".json"), "w"), indent=1)
    log("merged %d parts -> %s" % (len(rng), path)); raise SystemExit(0)

from model_adapters import get_forecaster


def mirror_of(c):
    r = c[:, 1:] / c[:, :-1] - 1.0
    m = np.empty_like(c); m[:, 0] = c[:, 0]; m[:, 1:] = c[:, :1] * np.cumprod(1.0 - r, axis=1); return m


fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)
mir = mirror_of(ctx_raw)[args.lo:args.hi]
t0 = time.time(); yh, mv = fn(mir)
path = os.path.join(sd, "%s_part%d_%d.npz" % (stem, args.lo, args.hi))
np.savez_compressed(path, yhat_mirror=yh.astype(np.float32), mcvar_mirror=mv.astype(np.float32))
log("windows %d:%d done in %.0fs -> %s" % (args.lo, args.hi, time.time() - t0, path))
