"""Validate the Time-MoE decoding harness of pilot_a_timemoe.py (the greedy multi-horizon loop
written because the packaged generate() needs transformers 4.40 internals) by reproducing the
model's own published zero-shot benchmark: ETTh1 in the protocol of the Time-MoE repository's
BenchmarkEvalDataset / run_eval.py (Shi et al., 2024, Table "Full results of zero-shot
forecasting experiments").

Protocol, copied from the official evaluation code: columns after `date`; train rows
[0, 12*30*24), test rows [16*30*24 - context, 20*30*24); every column standardised by the
training rows' mean and std; windows of context + horizon at stride one over each column of the
test segment (offsets window_length .. n_points - 1); the model receives the standardised
context as is (no per-window normalisation in run_eval.py) and its last `horizon` outputs are
the forecast; MSE and MAE are means over every forecast element. Context lengths per horizon as
in the paper: 512 / 1024 / 2048 / 3072 for 96 / 192 / 336 / 720.

Two decodings are run: (a) exactly the official protocol through our loop (standardised input,
no per-window normalisation) and (b) the paper's own setting, per-window mean/std normalisation
around the same loop (forecast_tm), so that the difference the normalisation makes is on record.

Usage (inside env_timemoe.sh):  python validate_timemoe.py --csv data/etth1/ETTh1.csv [--horizons 96,192,336,720] [--tag _h96]
Writes <SPEC_OUT>/validate_timemoe_etth1<tag>.json after every decoding; --official-only runs (a) alone; one horizon per process (--horizons 720 --tag _h720)
lets the four run in parallel, and make_paper_tables.py merges validate_timemoe_etth1_h*.json.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from pilot_a_timemoe import load_model, decode, forecast_tm, log
from paths import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--horizons", default="96,192,336,720")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out-dir", default=os.environ.get("SPEC_OUT", RESULTS_DIR))
ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _h96 when one horizon is run per process")
ap.add_argument("--official-only", action="store_true", help="skip decoding (b); the JSON then carries the official protocol alone")
args = ap.parse_args()
CONTEXT = {96: 512, 192: 1024, 336: 2048, 720: 3072}
PAPER = {96: (0.350, 0.382), 192: (0.388, 0.412), 336: (0.411, 0.430), 720: (0.427, 0.455), "avg": (0.394, 0.419)}   # Time-MoE_large (200M activated), ETTh1

df = pd.read_csv(args.csv); vals = df[df.columns[1:]].values.astype(np.float64)
model = load_model()
out = {"csv": os.path.basename(args.csv), "n_rows": int(len(df)), "columns": list(df.columns[1:]), "paper_timemoe_large": {str(k): v for k, v in PAPER.items()}, "horizons": {}}
for H in [int(h) for h in args.horizons.split(",")]:
    T = CONTEXT[H]; L = T + H
    b1 = [0, 12 * 30 * 24 - T, 12 * 30 * 24 + 4 * 30 * 24 - T]; b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    train = vals[b1[0]:b2[0]]; test = vals[b1[2]:b2[2]]
    mu, sd = train.mean(0), train.std(0)                                          # sklearn StandardScaler: population std
    scaled = ((test - mu) / sd).T                                                  # (n_columns, n_points)
    wins = [(c, o) for c in range(scaled.shape[0]) for o in range(L, scaled.shape[1])]
    X = np.stack([scaled[c, o - L:o - H] for c, o in wins]).astype(np.float32); Y = np.stack([scaled[c, o - H:o] for c, o in wins]).astype(np.float32)
    log(f"H={H}, context {T}: {len(wins)} windows over {scaled.shape[0]} columns")
    # (a) official protocol through our decode loop: standardised input as is
    t0 = time.time(); preds = []; i, bs = 0, args.batch
    while i < len(X):
        x = torch.tensor(X[i:i + bs], device="cuda")
        try: y = decode(model, x, H)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        preds.append(y.float().cpu().numpy()); i += bs
    P = np.concatenate(preds, 0)
    mse, mae = float(np.mean((P - Y) ** 2)), float(np.mean(np.abs(P - Y)))
    t1 = time.time()
    out["horizons"][str(H)] = {"context": T, "n_windows": len(wins), "official_protocol": {"mse": mse, "mae": mae, "seconds": round(t1 - t0)},
                               "paper": {"mse": PAPER[H][0], "mae": PAPER[H][1]}}
    log(f"  official protocol: MSE {mse:.3f} MAE {mae:.3f} (paper {PAPER[H][0]:.3f} / {PAPER[H][1]:.3f})")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, f"validate_timemoe_etth1{args.tag}.json"), "w"), indent=1)      # written after decoding (a), so a kill during (b) keeps it
    if args.official_only: continue
    # (b) the paper's setting: per-window normalisation around the same loop
    Pn = forecast_tm(model, X, H, args.batch)
    mse_n, mae_n = float(np.mean((Pn - Y) ** 2)), float(np.mean(np.abs(Pn - Y)))
    out["horizons"][str(H)]["per_window_normalisation"] = {"mse": mse_n, "mae": mae_n, "seconds": round(time.time() - t1)}
    log(f"  per-window normalisation: MSE {mse_n:.3f} MAE {mae_n:.3f}")
    json.dump(out, open(os.path.join(args.out_dir, f"validate_timemoe_etth1{args.tag}.json"), "w"), indent=1)      # written after every horizon
hs = [h for h in out["horizons"]]
out["average"] = {"official_protocol": {"mse": float(np.mean([out["horizons"][h]["official_protocol"]["mse"] for h in hs])), "mae": float(np.mean([out["horizons"][h]["official_protocol"]["mae"] for h in hs]))},
                  "paper": {"mse": PAPER["avg"][0], "mae": PAPER["avg"][1]}, "horizons": hs}
if all("per_window_normalisation" in out["horizons"][h] for h in hs):
    out["average"]["per_window_normalisation"] = {"mse": float(np.mean([out["horizons"][h]["per_window_normalisation"]["mse"] for h in hs])), "mae": float(np.mean([out["horizons"][h]["per_window_normalisation"]["mae"] for h in hs]))}
os.makedirs(args.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(args.out_dir, f"validate_timemoe_etth1{args.tag}.json"), "w"), indent=1)
log(f"average: official protocol MSE {out['average']['official_protocol']['mse']:.3f} MAE {out['average']['official_protocol']['mae']:.3f} (paper {PAPER['avg'][0]:.3f} / {PAPER['avg'][1]:.3f}); wrote validate_timemoe_etth1.json")
