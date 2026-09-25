"""Pilot A driver for Moirai-2.0-R-small (Salesforce, 2025): the decoder-only successor of
Moirai-1.1 (a masked encoder), through uni2ts 2.0.0's moirai2 module in the `moirai` env.

Shares generators and metrics with the other drivers through nulls.py / probe_metrics.py;
same JSONL schema and key format.

Particulars (smoke_moirai2.py, 2026-09-18): 11.4M parameters, fixed patch size 16 (no patch-size
declaration, so the declared-frequency test is undefined for it, unlike Moirai-1.1); the model
forecasts nine quantiles (0.1 .. 0.9) in one forward call and has no mean or sample output, so
the point forecast is the median, the model's own designated summary, as for the other 2025
quantile models (the average of the nine quantiles differs from it by up to 1.3 sigma on a
random walk). Deterministic, no Monte Carlo term; continuous scaling, floor = norm_floor.
Horizon-prefix-equivalent (rms 0 between H=16 and the first 16 of H=128). Accepts contexts of
512 to 4096 points; the forecaster is rebuilt with context_length = the input's length. The
forward call is used directly with (n, T, 1) tensors instead of the gluonts predictor.

Usage (inside env_moirai.sh):  python pilot_a_moirai2.py --stage all
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor
from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = os.environ.get("MOIRAI2_MODEL", "Salesforce/moirai-2.0-R-small"); MODEL_TAG = "moirai2"
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

def load_moirai2():
    from uni2ts.model.moirai2 import Moirai2Module
    return Moirai2Module.from_pretrained(MODEL_ID)

_checked = [False]
def _np(a):
    return np.asarray(a.float().cpu() if torch.is_tensor(a) else a)
_fc = {}
def _forecaster(module, H, T):
    from uni2ts.model.moirai2 import Moirai2Forecast
    key = (H, T)
    if key not in _fc:
        _fc[key] = Moirai2Forecast(module=module, prediction_length=H, context_length=T, target_dim=1,
                                   feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(__import__("os").environ.get("TSFM_DEVICE","cuda")).eval()
    return _fc[key]
@torch.no_grad()
def forecast_moirai2(module, ctx_np, H, batch=64):
    """(median (n,H), quantiles (n,H,9)); chunked, OOM-adaptive; direct forward call."""
    T = np.asarray(ctx_np).shape[1]; model = _forecaster(module, H, T)
    pts, qs = [], []; i, bs = 0, batch
    while i < len(ctx_np):
        x = torch.tensor(np.asarray(ctx_np[i:i + bs], dtype=np.float32), device=__import__("os").environ.get("TSFM_DEVICE","cuda"))[:, :, None]
        try:
            out = model(past_target=x, past_observed_target=torch.ones_like(x, dtype=torch.bool),
                        past_is_pad=torch.zeros(x.shape[:2], dtype=torch.bool, device=__import__("os").environ.get("TSFM_DEVICE","cuda")))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        out = _np(out)
        if out.ndim == 4: out = out[..., 0]                       # (n, 9, H)
        q = np.transpose(out, (0, 2, 1))[:, :H, :]               # (n, H, 9)
        m = q[..., 4]
        if not _checked[0]:
            log(f"  point forecast = the 0.5 quantile (no mean output); quantile-average minus median max|diff| = {np.max(np.abs(q.mean(-1) - m)):.2e}")
            _checked[0] = True
        pts.append(m); qs.append(q); i += bs
    return np.concatenate(pts, 0), np.concatenate(qs, 0)

def stage_ladder(args, model, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|median"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat, q = forecast_moirai2(model, y_ctx, H, args.batch); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung, "seed": seed,
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

def stage_direction(args, model, done):
    """Default setting at H=128 on the shared direction contexts (seed 4000)."""
    H, n = args.horizon_long, args.n_freq
    key = f"direction|{MODEL_TAG}|pretrained|H{H}|n{n}|median"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        y_hat, _ = forecast_moirai2(model, ctx, H, args.batch); dep = y_hat - ctx[:, -1:]
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
            "note": "fixed patch size 16 and no frequency input: the declared-frequency (patch-size) test of Moirai-1.1 is undefined for this model"})

STAGES = {"ladder": stage_ladder, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    log(f"loading {MODEL_ID}"); model = load_moirai2(); log(f"{type(model).__name__} loaded")
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
