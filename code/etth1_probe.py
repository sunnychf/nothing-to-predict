"""A real positive control outside finance: ETTh1 (the seven hourly transformer-load and oil-temperature
series the models' own benchmarks use), forecast by every model exactly as the ladder is, so that the
models that depart on martingales are shown finding structure where it exists.

Windows: every column of data/etth1/ETTh1.csv (17420 hourly points), contexts of 512 and futures of 128
at stride 128 counted back from the last observation (non-overlapping futures), in the level units of
the series. Each model forecasts the raw context (the same adapter as diag_level_all.py) and its
additive mirror about the first value. Metrics are computed by etth1_summary.py.

    python etth1_probe.py --model chronos [--csv data/etth1/ETTh1.csv]
writes <out-dir>/etth1_<model>.npz (yhat_raw, yhat_mirror, mcvar) and, once, etth1_windows.npz.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from model_adapters import get_forecaster, SAMPLED
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["chronos", "fincast", "timesfm", "timemoe", "moirai", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial"])
ap.add_argument("--csv", default=os.path.join(os.path.dirname(RESULTS_DIR), "data", "etth1", "ETTh1.csv"))
ap.add_argument("--samples", type=int, default=100); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out-dir", default=os.environ.get("SPEC_OUT", RESULTS_DIR))
args = ap.parse_args()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
T, H, L = 512, 128, 640
df = pd.read_csv(args.csv); cols = list(df.columns[1:]); vals = df[cols].values.astype(np.float64)
ctx, fut, meta = [], [], []
for j, c in enumerate(cols):
    y = vals[:, j]; n = len(y); n_win = (n - L) // H + 1
    for k in range(n_win):
        e = n - 1 - k * H; st = e + 1 - L
        ctx.append(y[st:st + T]); fut.append(y[st + T:e + 1]); meta.append({"column": c, "end_row": int(e)})
ctx, fut = np.array(ctx), np.array(fut)
os.makedirs(args.out_dir, exist_ok=True)
wp = os.path.join(args.out_dir, "etth1_windows.npz")
if not os.path.exists(wp):
    np.savez_compressed(wp, ctx=ctx, fut=fut); json.dump(meta, open(wp.replace(".npz", "_meta.json"), "w"), indent=1)
log(f"{len(ctx)} windows over {len(cols)} columns, ctx {T}, H {H}; model {args.model}")
fn = get_forecaster(args.model, H, samples=args.samples, batch=args.batch)
t0 = time.time(); yh, mcv = fn(ctx); log(f"raw done ({time.time()-t0:.0f}s)")
mir = 2 * ctx[:, :1] - ctx
t0 = time.time(); ym, _ = fn(mir); log(f"mirror done ({time.time()-t0:.0f}s)")
np.savez_compressed(os.path.join(args.out_dir, f"etth1_{args.model}.npz"), yhat_raw=yh.astype(np.float32), yhat_mirror=ym.astype(np.float32), mcvar=mcv.astype(np.float32))
sig = np.std(np.diff(ctx, axis=1), axis=1); last = ctx[:, -1:]
skill = [float(1 - ((yh[:, h-1] - fut[:, h-1]) ** 2 / sig ** 2).mean() / ((last[:, 0] - fut[:, h-1]) ** 2 / sig ** 2).mean()) for h in (1, 16, 64, 128)]
log(f"wrote etth1_{args.model}.npz; skill vs persistence (sigma-normalised, pooled) at h=1/16/64/128: " + " / ".join(f"{s:+.3f}" for s in skill))
