"""Pilot A driver for TimesFM-2.5-200M (Google, 2025; timesfm 3.x API) in the `tsfm25` env.

Shares generators and metrics with the other drivers through nulls.py / probe_metrics.py;
same JSONL schema and key format.

Particulars (smoke_timesfm25.py, 2026-09-18): decoder-only, 200M; the released ForecastConfig
defaults to `force_flip_invariance=True`, a symmetry averaging under value negation built into
the model (forecast the negated series and negate back, average), and `infer_is_positive=True`.
The default run uses those defaults, i.e. the model as released; --flip off switches the
built-in averaging off and tags the rows `timesfm25-noflip`, which measures what the vendor's
averaging removes. The point output equals the 0.5 quantile of the continuous quantile head
exactly, so the point forecast is the median, as for the other 2025 quantile models; the
frequency indicator of timesfm 1.x no longer exists, so the declared-frequency test is
undefined. Deterministic, no Monte Carlo term; continuous scaling, floor = norm_floor.
Horizon-prefix-equivalent (rms 0). Contexts up to the compiled max_context (2048 here).
per_core_batch_size is raised from the default 1, which made the released code forecast one
series at a time.

Usage (inside env_tsfm25.sh):  python pilot_a_timesfm25.py --stage all [--flip off]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor
from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = os.environ.get("TSFM25_MODEL", "google/timesfm-2.5-200m-pytorch"); MODEL_TAG = "timesfm25"
CTX = 512
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

def load_timesfm25(flip=True, max_context=2048, batch=64):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=256, normalize_inputs=True, per_core_batch_size=batch,
                                         use_continuous_quantile_head=True, force_flip_invariance=flip, infer_is_positive=True, fix_quantile_crossing=True))
    return model

_checked = [False]
def _np(a):
    return np.asarray(a.float().cpu() if torch.is_tensor(a) else a)
@torch.no_grad()
def forecast_timesfm25(model, ctx_np, H, batch=64):
    """(median (n,H), quantiles (n,H,10)); the compiled model batches internally (per_core_batch_size)."""
    p, q = model.forecast(horizon=H, inputs=[np.asarray(c, dtype=np.float32) for c in ctx_np])
    p, q = np.asarray(p)[:, :H], np.asarray(q)[:, :H, :]
    if not _checked[0]:
        log(f"  point vs q[...,5] (median) max|diff| = {np.max(np.abs(p - q[..., 5])):.2e} (the point forecast is the median); point vs q[...,0] = {np.max(np.abs(p - q[..., 0])):.2e}")
        _checked[0] = True
    return p, q

def stage_ladder(args, model, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|median"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat, q = forecast_timesfm25(model, y_ctx, H, args.batch); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung, "seed": seed,
                   "H": H, "n": n, "num_samples": 0, "point": "median", "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=None),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            lo, hi = q[:, :, 1], q[:, :, 9]                      # 10% and 90% (channel 0 is the head's first output, not a quantile)
            rec["interval80_width_over_sigma"] = (np.mean(hi - lo, 0) / info["sigma"]).tolist()
            rec["coverage_80"] = np.mean((y_true >= lo) & (y_true <= hi), 0).tolist()
            if rung == "N2":
                cv_last = info["cond_vol"][:, T - 1]
                rec["corr_width_last_condvol"] = [float(np.corrcoef((hi - lo)[:, h], cv_last)[0, 1]) for h in range(H)]
            append(rec)
            log(f"{rung} s{seed}: dep h1={rec['departure_in_sigma'][0]:.3f} h{H}={rec['departure_in_sigma'][-1]:.3f} "
                f"skill_h1={rec['skill_vs_persistence'][0]:+.3f} up_h{H}={rec['frac_departure_up'][-1]:.2f}")

def stage_direction(args, model, done):
    """Default setting at H=128 on the shared direction contexts (seed 4000)."""
    H, n = args.horizon_long, args.n_freq
    key = f"direction|{MODEL_TAG}|pretrained|H{H}|n{n}|median"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        y_hat, _ = forecast_timesfm25(model, ctx, H, args.batch); dep = y_hat - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{os.path.dirname(RESULTS)}/departures_{MODEL_TAG}_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        r = {"model": MODEL_TAG, "rung": rung, "H": H, "n": n,
             "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
             "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
             "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
             "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
        recs.append(r)
        log(f"direction {rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h={H} mean {r['mean_dep_sigma'][H]:+.2f} up {r['frac_up'][H]:.2f} corr_slope {r['corr_slope64'][H]:+.2f}")
    json.dump(recs, open(f"{os.path.dirname(RESULTS)}/diag_direction_{MODEL_TAG}.json", "w"), indent=1)
    append({"key": key, "stage": "direction", "model": MODEL_TAG, "init": "pretrained", "H": H, "n": n, "file": f"diag_direction_{MODEL_TAG}.json"})
    append({"key": f"freq|{MODEL_TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq", "model": MODEL_TAG, "init": "pretrained", "applicable": False,
            "note": "timesfm 3.x has no frequency indicator: the declared-frequency test of TimesFM-2.0 is undefined for this model"})

STAGES = {"ladder": stage_ladder, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--flip", default="on", choices=["on", "off"], help="the model's built-in flip-invariance averaging (released default: on)")
    args = ap.parse_args()
    global MODEL_TAG
    if args.flip == "off": MODEL_TAG = "timesfm25-noflip"
    log(f"loading {MODEL_ID} (flip invariance {args.flip})"); model = load_timesfm25(flip=(args.flip == "on"), batch=args.batch); log(f"{type(model).__name__} loaded")
    done = done_keys(); log(f"{len(done)} configs already done"); failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try: STAGES[st](args, model, done)
        except Exception:
            failed = True; tb = traceback.format_exc()
            print("!" * 70 + f"\nSTAGE {st} FAILED\n{tb}\n" + "!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{MODEL_TAG}|{time.time()}", "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES"); sys.exit(1 if failed else 0)
if __name__ == "__main__": main()
