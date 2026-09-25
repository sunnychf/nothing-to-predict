"""Pilot A driver for TimesFM 2.0-500m (Das et al. 2024), via timesfm==1.3.0.

Runs in the `timesfm` conda env. Shares generators and metrics with the other
drivers through nulls.py / probe_metrics.py; same JSONL schema and key format.

TimesFM particulars:
  - Deterministic: a mean plus nine quantiles (channel 0 = mean, 1..9 = deciles);
    the library's point_forecast is the median by default, so we take q[...,0].
    No sampling, no MC term.
  - Continuous, context-normalised, no token grid: floor is norm_floor (~0).
  - Decoder-only with output patch 128: an H=128 forecast's first steps equal an
    H=16 forecast's (checked in smoke_timesfm.py), so one load serves all H.
  - Takes a frequency indicator freq in {0, 1, 2}: 0 covers everything up to
    daily, 1 weekly/monthly, 2 quarterly/yearly. Three levels, hourly and
    daily indistinguishable, the same scheme FinCast inherited from it.
    timesfm 2.5 removed the indicator, which is why 2.0 is used here.

Usage (inside timesfm env):  python pilot_a_timesfm.py --stage all
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor
import timesfm

from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = "google/timesfm-2.0-500m-pytorch"; MODEL_TAG = "timesfm-2.0-500m"
CTX = 512
FREQ_NAMES = {0: "le_daily", 1: "weekly_monthly", 2: "quarterly_yearly"}
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
def load_timesfm(H):
    return timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend=__import__("os").environ.get("TSFM_BACKEND","gpu"), per_core_batch_size=32, horizon_len=H,
                                                          num_layers=50, use_positional_embedding=False, context_len=2048),
                           checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=MODEL_ID))
_median_checked = [False]
def forecast_tfm(tfm, ctx_np, H, freq, batch=64):
    """(mean (n,H), quantiles (n,H,10)); chunked, OOM-adaptive.

    TimesFM 2.0's point_forecast defaults to the MEDIAN (point_forecast_mode);
    the L2-optimal summary, and the one every other driver uses, is the mean,
    which sits at quantile channel 0. We return q[..., 0] as the point forecast
    and log once that the library's point output equals channel 5 (median)."""
    pts, qs = [], []; i, bs = 0, batch
    while i < len(ctx_np):
        chunk = [np.asarray(x, dtype=np.float32) for x in ctx_np[i:i + bs]]
        try: p, q = tfm.forecast(chunk, freq=[freq] * len(chunk))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        p, q = np.asarray(p)[:, :H], np.asarray(q)[:, :H, :]
        if not _median_checked[0]:
            log(f"  point vs q[...,5] (median) max|diff| = {np.max(np.abs(p - q[..., 5])):.2e}; "
                f"point vs q[...,0] (mean) = {np.max(np.abs(p - q[..., 0])):.2e}; using q[...,0] as the point forecast")
            _median_checked[0] = True
        pts.append(q[..., 0]); qs.append(q); i += bs
    return np.concatenate(pts, 0), np.concatenate(qs, 0)

def stage_ladder(args, tfm, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|f{args.freq}"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat, q = forecast_tfm(tfm, y_ctx, H, args.freq, args.batch); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung, "seed": seed,
                   "H": H, "n": n, "num_samples": 0, "freq": args.freq, "freq_name": FREQ_NAMES[args.freq],
                   "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=None),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            lo, hi = q[:, :, 1], q[:, :, 9]                      # 10% and 90%
            rec["interval80_width_over_sigma"] = (np.mean(hi - lo, 0) / info["sigma"]).tolist()
            rec["coverage_80"] = np.mean((y_true >= lo) & (y_true <= hi), 0).tolist()
            if rung == "N2":
                cv_last = info["cond_vol"][:, T - 1]
                rec["corr_width_last_condvol"] = [float(np.corrcoef((hi - lo)[:, h], cv_last)[0, 1]) for h in range(H)]
            append(rec)
            log(f"{rung} s{seed} f{args.freq}: dep h1={rec['departure_in_sigma'][0]:.3f} h{H}={rec['departure_in_sigma'][-1]:.3f} "
                f"skill_h1={rec['skill_vs_persistence'][0]:+.3f} up_h{H}={rec['frac_departure_up'][-1]:.2f}")

def stage_freq(args, tfm, done):
    n, T, H = args.n_freq, CTX, args.horizon_long
    y_ctx, y_true, info = make_data("N1", n, T, H, seed=3000); outs = {}
    for f in (0, 1, 2):
        key = f"freq|{MODEL_TAG}|pretrained|N1|f{f}|H{H}|n{n}"
        if key in done: log(f"skip {key}"); continue
        y_hat, _ = forecast_tfm(tfm, y_ctx, H, f, args.batch); outs[f] = y_hat; fl = norm_floor(y_ctx, H)
        rec = {"key": key, "stage": "freq", "model": MODEL_TAG, "init": "pretrained", "applicable": True, "rung": "N1",
               "freq": f, "freq_name": FREQ_NAMES[f], "H": H, "n": n, "num_samples": 0, "declaration_kind": "frequency indicator",
               **summarise(y_hat, y_ctx, y_true, info, "N1", H, floor=fl, samples=None),
               **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
        append(rec)
        log(f"freq f{f} ({FREQ_NAMES[f]}): dep h1={rec['departure_in_sigma'][0]:.3f} mean_h{H}={rec['departure_mean_signed_sigma'][-1]:+.2f}σ "
            f"up={rec['frac_departure_up'][-1]:.2f} flat={rec['spec_flatness_detrended']:.3f}")
    if len(outs) == 3:
        sig = info["sigma"]
        d = {f"rms_diff_f{a}_f{b}_sigma": float(np.sqrt(np.mean((outs[a] - outs[b]) ** 2)) / sig) for a, b in ((0, 1), (0, 2), (1, 2))}
        append({"key": f"freq_delta|{MODEL_TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq_delta", "model": MODEL_TAG,
                "init": "pretrained", "H": H, "n": n, "declaration_kind": "frequency indicator", **d,
                "note": "same synthetic data, only the declared frequency changed"})
        log("freq deltas (σ): " + " ".join(f"{k[9:-6]}={v:.2f}" for k, v in d.items()))

def stage_direction(args, tfm, done):
    """Default setting (freq 0) at H=128 on the shared direction contexts (seed 4000)."""
    H, n = args.horizon_long, args.n_freq
    key = f"direction|{MODEL_TAG}|pretrained|H{H}|n{n}|f0"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        y_hat, _ = forecast_tfm(tfm, ctx, H, 0, args.batch); dep = y_hat - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{os.path.dirname(RESULTS)}/departures_timesfm_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        r = {"model": MODEL_TAG, "rung": rung, "H": H, "n": n,
             "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
             "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
             "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
             "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
        recs.append(r)
        log(f"direction {rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h={H} mean {r['mean_dep_sigma'][H]:+.2f} up {r['frac_up'][H]:.2f} corr(slope) {r['corr_slope64'][H]:+.2f}")
    json.dump(recs, open(f"{os.path.dirname(RESULTS)}/diag_direction_timesfm.json", "w"), indent=1)
    append({"key": key, "stage": "direction", "model": MODEL_TAG, "init": "pretrained", "H": H, "n": n, "file": "diag_direction_timesfm.json"})

STAGES = {"ladder": stage_ladder, "freq": stage_freq, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--freq", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    Hmax = max(args.horizon, args.horizon_long)
    log(f"loading {MODEL_ID} with horizon_len={Hmax} (decoder-only, one load serves all H)")
    tfm = load_timesfm(Hmax); done = done_keys(); log(f"{len(done)} configs already done")
    failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try: STAGES[st](args, tfm, done)
        except Exception:
            failed = True; tb = traceback.format_exc()
            print("!" * 70 + f"\nSTAGE {st} FAILED\n{tb}\n" + "!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{MODEL_TAG}|{time.time()}", "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES"); sys.exit(1 if failed else 0)
if __name__ == "__main__": main()
