"""Pilot A driver for the 2025 members of the Chronos family: Chronos-Bolt-small (47M) and
Chronos-2 (119M), via chronos-forecasting 2.3.2 in the `tsfmfin` env.

Shares generators and metrics with the other drivers through nulls.py / probe_metrics.py;
same JSONL schema and key format.

Particulars (checked in the smoke run of 2026-09-17):
  - Both are quantile models: predict_quantiles returns nine quantiles (0.1 .. 0.9) and a point
    forecast that IS the median (max|point - q0.5| = 0). There is no mean output, so the point
    forecast used here is the median, the model's own designated summary; the average of the
    nine quantiles differs from it by up to 0.9 sigma on a random walk and is not a mean either.
  - Deterministic, no sampling, no Monte Carlo term; continuous scaling, floor = norm_floor.
  - Bolt takes (n, T) inputs and returns tensors; Chronos-2 takes (n, 1, T) and returns lists.
  - Bolt is horizon-prefix-equivalent (H=16 vs H=128 rms diff 0); Chronos-2 is NOT (0.10 sigma),
    so every stage forecasts at its own horizon, as for Moirai and Time-MoE.
  - Neither takes a frequency declaration: the declared-frequency test is undefined for them.

Usage (inside tsfmfin env):  python pilot_a_chronosx.py --model bolt --stage all
                             python pilot_a_chronosx.py --model chronos2 --stage all
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor

from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODELS = {"bolt": ("amazon/chronos-bolt-small", "chronos-bolt-small"), "chronos2": ("amazon/chronos-2", "chronos-2")}
CTX = 512
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def append(rec):
    with open(RESULTS, "a") as f: f.write(json.dumps(rec, default=float) + "\n"); f.flush(); os.fsync(f.fileno())
def done_keys():
    ks = set()
    if os.path.exists(RESULTS):
        for l in open(RESULTS):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if "key" in r and not r.get("error"): ks.add(r["key"])
    return ks

def load_pipe(model_id):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map=__import__("os").environ.get("TSFM_DEVICE","cuda"), torch_dtype=torch.float32)

_checked = [False]
@torch.no_grad()
def forecast_cx(pipe, ctx_np, H, batch=64):
    """(median (n,H), quantiles (n,H,9)); chunked, OOM-adaptive; both pipelines."""
    is_c2 = type(pipe).__name__ == "Chronos2Pipeline"
    pts, qs = [], []; i, bs = 0, batch
    while i < len(ctx_np):
        x = torch.tensor(np.asarray(ctx_np[i:i + bs], dtype=np.float32))
        if is_c2: x = x[:, None, :]
        try:
            q, m = pipe.predict_quantiles(x, prediction_length=H, quantile_levels=QL, limit_prediction_length=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        if is_c2: q, m = torch.stack(list(q))[:, 0], torch.stack(list(m))[:, 0]
        q, m = q.float().cpu().numpy()[:, :H, :], m.float().cpu().numpy()[:, :H]
        if not _checked[0]:
            log(f"  point vs q0.5 max|diff| = {np.max(np.abs(m - q[..., 4])):.2e} (the point forecast is the median); "
                f"point vs quantile-average max|diff| = {np.max(np.abs(m - q.mean(-1))):.2f}")
            _checked[0] = True
        pts.append(m); qs.append(q); i += bs
    return np.concatenate(pts, 0), np.concatenate(qs, 0)

def stage_ladder(args, pipe, done):
    n, T, H = args.n, CTX, args.horizon; TAG = MODELS[args.model][1]
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|median"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat, q = forecast_cx(pipe, y_ctx, H, args.batch); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": TAG, "init": "pretrained", "rung": rung, "seed": seed,
                   "H": H, "n": n, "num_samples": 0, "point": "median", "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=None),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            lo, hi = q[:, :, 0], q[:, :, 8]                      # 10% and 90%
            rec["interval80_width_over_sigma"] = (np.mean(hi - lo, 0) / info["sigma"]).tolist()
            rec["coverage_80"] = np.mean((y_true >= lo) & (y_true <= hi), 0).tolist()
            if rung == "N2":
                cv_last = info["cond_vol"][:, T - 1]
                rec["corr_width_last_condvol"] = [float(np.corrcoef((hi - lo)[:, h], cv_last)[0, 1]) for h in range(H)]
            append(rec)
            log(f"{rung} s{seed}: dep h1={rec['departure_in_sigma'][0]:.3f} h{H}={rec['departure_in_sigma'][-1]:.3f} "
                f"skill_h1={rec['skill_vs_persistence'][0]:+.3f} up_h{H}={rec['frac_departure_up'][-1]:.2f}")

def stage_direction(args, pipe, done):
    """Default setting at H=128 on the shared direction contexts (seed 4000)."""
    H, n = args.horizon_long, args.n_freq; TAG = MODELS[args.model][1]; short = args.model
    key = f"direction|{TAG}|pretrained|H{H}|n{n}|median"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        y_hat, _ = forecast_cx(pipe, ctx, H, args.batch); dep = y_hat - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{os.path.dirname(RESULTS)}/departures_{short}_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        r = {"model": TAG, "rung": rung, "H": H, "n": n,
             "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
             "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
             "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
             "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
        recs.append(r)
        log(f"direction {rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h={H} mean {r['mean_dep_sigma'][H]:+.2f} up {r['frac_up'][H]:.2f} corr_slope {r['corr_slope64'][H]:+.2f}")
    json.dump(recs, open(f"{os.path.dirname(RESULTS)}/diag_direction_{short}.json", "w"), indent=1)
    append({"key": key, "stage": "direction", "model": TAG, "init": "pretrained", "H": H, "n": n, "file": f"diag_direction_{short}.json"})
    append({"key": f"freq|{TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq", "model": TAG, "init": "pretrained", "applicable": False,
            "note": "no frequency, timestamp or calendar input: the declared-frequency test is undefined for this model"})

STAGES = {"ladder": stage_ladder, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    mid, tag = MODELS[args.model]
    log(f"loading {mid}"); pipe = load_pipe(mid); log(f"{type(pipe).__name__}: {sum(p.numel() for p in pipe.model.parameters()) // 10**6}M params")
    done = done_keys(); log(f"{len(done)} configs already done"); failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try: STAGES[st](args, pipe, done)
        except Exception:
            failed = True; tb = traceback.format_exc()
            print("!" * 70 + f"\nSTAGE {st} FAILED\n{tb}\n" + "!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{tag}|{time.time()}", "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES"); sys.exit(1 if failed else 0)
if __name__ == "__main__": main()
