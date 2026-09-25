"""Pilot A driver for Moirai 1.1-R (Woo et al. 2024), the LOTSA-corpus model.

Runs in the `moirai` conda env. Shares generators and metrics with the other
drivers through nulls.py / probe_metrics.py; writes the same schema and key
format to the same JSONL.

Moirai particulars, recorded rather than hidden:
  - Probabilistic: S sampled trajectories, so the Monte Carlo term is estimated
    and removed exactly like Chronos.
  - Continuous, instance-normalised, no token grid: the floor is the affine
    round trip (norm_floor), ~1e-7 sigma.
  - It receives NO frequency. Its forward pass takes a patch size, and each of
    its five patch sizes was trained on a band of frequencies (Woo et al. 2024,
    App. B.1: 8 yearly/quarterly, 16 weekly/daily, 32 daily/hourly, 64
    hourly/minute, 128 minute/second). The declared-frequency test for Moirai
    therefore varies patch_size, five ordered levels. The ladder uses
    patch_size="auto", the documented default, which picks per series by
    in-context validation loss.
  - NOT prefix-equivalent across horizons. Moirai is a masked encoder that
    predicts all H steps in one pass, so an H=128 forecast carries more masked
    future patches in the attention window than an H=16 one and its first
    patch differs (smoke test: 0.33 sigma rms at patch 32 against ~0.22 of
    sampling noise). Unlike the autoregressive Chronos and FinCast, one model
    cannot serve every horizon, so forecast_moirai() builds a fresh
    MoiraiForecast for the horizon it is asked for. Every stage does this.

Usage (inside moirai env):  python pilot_a_moirai.py --stage all
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, pandas as pd, torch
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset

from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = os.environ.get("MOIRAI_MODEL", "Salesforce/moirai-1.1-R-small")
MODEL_TAG = "moirai-1.1-R-small"
CTX = 512
PATCH_BAND = {8: "yearly_quarterly", 16: "weekly_daily", 32: "daily_hourly", 64: "hourly_minute", 128: "minute_second"}

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

_module = None
def module():
    global _module
    if _module is None: _module = MoiraiModule.from_pretrained(MODEL_ID)
    return _module

def forecast_moirai(ctx_np, H, S, patch, batch=16, seed=0, ctx_len=None):
    """Sampled trajectories (n, S, H). OOM-adaptive like the other drivers. ctx_len sets the
    model's context_length (default CTX=512, the setting of every run in the paper); the
    context-length sweep passes the input's own length."""
    torch.manual_seed(seed)
    model = MoiraiForecast(module=module(), prediction_length=H, context_length=CTX if ctx_len is None else int(ctx_len), patch_size=patch,
                           num_samples=S, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    bs = batch
    while True:
        try:
            pred = model.create_predictor(batch_size=bs, device=__import__("os").environ.get("TSFM_DEVICE","cuda"))
            ds = ListDataset([{"start": pd.Period("2000-01-01", freq="D"), "target": c.astype(np.float32)} for c in ctx_np], freq="D")
            return np.stack([f.samples for f in pred.predict(ds)])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}")

def stage_ladder(args, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|ns{args.num_samples}|patch{args.patch}"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            s = forecast_moirai(y_ctx, H, args.num_samples, args.patch, args.batch, seed)
            y_hat = s.mean(1); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung,
                   "seed": seed, "H": H, "n": n, "num_samples": args.num_samples, "patch_size": args.patch,
                   "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=s),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            lo, hi = np.quantile(s, [0.1, 0.9], axis=1)
            rec["interval80_width_over_sigma"] = (np.mean(hi - lo, 0) / info["sigma"]).tolist()
            rec["coverage_80"] = np.mean((y_true >= lo) & (y_true <= hi), 0).tolist()
            if rung == "N2":
                cv_last = info["cond_vol"][:, T - 1]
                rec["corr_width_last_condvol"] = [float(np.corrcoef((hi - lo)[:, h], cv_last)[0, 1]) for h in range(H)]
            append(rec)
            log(f"{rung} s{seed}: dep-MC h1={rec['departure_in_sigma_mc_corrected'][0]:.3f} h{H}={rec['departure_in_sigma_mc_corrected'][-1]:.3f} "
                f"skill_h1={rec['skill_vs_persistence'][0]:+.3f} up_h{H}={rec['frac_departure_up'][-1]:.2f}")

def stage_freq(args, done):
    """Declared-frequency test: same N1 data, five patch sizes."""
    n, T, H = args.n_freq, CTX, args.horizon_long
    y_ctx, y_true, info = make_data("N1", n, T, H, seed=3000)
    outs = {}
    for ps in (8, 16, 32, 64, 128):
        key = f"freq|{MODEL_TAG}|pretrained|N1|patch{ps}|H{H}|n{n}|ns{args.num_samples}"
        if key in done: log(f"skip {key}"); continue
        s = forecast_moirai(y_ctx, H, args.num_samples, ps, max(4, args.batch // 4), 0)
        y_hat = s.mean(1); outs[ps] = y_hat; fl = norm_floor(y_ctx, H)
        rec = {"key": key, "stage": "freq", "model": MODEL_TAG, "init": "pretrained", "applicable": True,
               "rung": "N1", "freq": ps, "freq_name": f"patch{ps}_{PATCH_BAND[ps]}", "H": H, "n": n,
               "num_samples": args.num_samples, "declaration_kind": "patch_size",
               **summarise(y_hat, y_ctx, y_true, info, "N1", H, floor=fl, samples=s),
               **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
        append(rec)
        log(f"patch {ps:>3} ({PATCH_BAND[ps]:>16}): dep-MC h1={rec['departure_in_sigma_mc_corrected'][0]:.3f} "
            f"h{H}={rec['departure_in_sigma_mc_corrected'][-1]:.3f} mean_h{H}={rec['departure_mean_signed_sigma'][-1]:+.2f}σ "
            f"up={rec['frac_departure_up'][-1]:.2f} flat={rec['spec_flatness_detrended']:.3f}")
    if len(outs) == 5:
        ks = sorted(outs); sig = info["sigma"]
        d = {f"rms_diff_p{a}_p{b}_sigma": float(np.sqrt(np.mean((outs[a] - outs[b]) ** 2)) / sig) for i, a in enumerate(ks) for b in ks[i + 1:]}
        append({"key": f"freq_delta|{MODEL_TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq_delta", "model": MODEL_TAG,
                "init": "pretrained", "H": H, "n": n, "declaration_kind": "patch_size", **d,
                "note": "same synthetic data, only the patch size (Moirai's frequency proxy) changed"})
        log("patch-size deltas (σ): " + " ".join(f"{k.split('_',2)[2][:-6]}={v:.2f}" for k, v in d.items()))

STAGES = {"ladder": stage_ladder, "freq": stage_freq}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--patch", default="auto", help="ladder patch size: auto or 8/16/32/64/128")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if args.patch != "auto": args.patch = int(args.patch)
    log(f"Moirai {MODEL_ID}; ladder patch={args.patch}")
    done = done_keys(); log(f"{len(done)} configs already done")
    failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try: STAGES[st](args, done)
        except Exception:
            failed = True; tb = traceback.format_exc()
            print("!" * 70 + f"\nSTAGE {st} FAILED\n{tb}\n" + "!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{MODEL_TAG}|{time.time()}", "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES"); sys.exit(1 if failed else 0)

if __name__ == "__main__": main()
