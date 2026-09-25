"""Pilot A driver for FinCast (Zhu et al. 2025, CIKM), the finance-native TSFM.

Lives in the fincast_v1 conda env, which pins its own torch. Shares the data
generators and metrics with pilot_a.py through nulls.py / probe_metrics.py, and
writes rows in the same schema and key format to the same JSONL, so the report
and the results generator read them unchanged.

Three things differ from the Chronos driver, each recorded rather than hidden:

  - FinCast returns a mean and nine quantiles deterministically. There are no
    sampled trajectories, so no Monte Carlo component and no mc_component field.
  - It is a continuous patched model with no token vocabulary. The artefact
    floor analogous to Chronos's codec is the round trip of y_T through an
    affine normalisation, which is float precision. It is computed and stored
    as codec_floor so the report's columns line up, and it is ~1e-7 sigma.
  - It accepts a frequency declaration, so the declared-frequency test of
    Section 4.1 IS defined for it, at the three levels its reader distinguishes:
    0 for anything up to daily, 1 for weekly/monthly, 2 for yearly. Hourly and
    daily are the same declaration to it, as they are to TimesFM.

Usage (inside fincast_v1):
  python pilot_a_fincast.py --stage ladder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor

from paths import RESULTS_DIR, WEIGHTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
WEIGHTS = os.environ.get("FINCAST_WEIGHTS", os.path.join(WEIGHTS_DIR, "fincast_v1.pth"))
MODEL_TAG = "fincast_v1"
CTX = 512
FREQ_NAMES = {0: "le_daily", 1: "weekly_monthly", 2: "yearly"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n"); f.flush(); os.fsync(f.fileno())


def done_keys():
    if not os.path.exists(RESULTS):
        return set()
    ks = set()
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" in r and not r.get("error"):
            ks.add(r["key"])
    return ks


# -------------------------------------------------------------------- model

def load_fincast(horizon_len, context_len=CTX):
    from tools.inference_utils import get_model_api
    cfg = SimpleNamespace(backend=__import__("os").environ.get("TSFM_BACKEND","gpu"), model_path=WEIGHTS,
                          horizon_len=horizon_len, context_len=context_len,
                          num_experts=4, gating_top_n=2, load_from_compile=True,
                          forecast_mode="mean")
    return get_model_api(cfg)


@torch.no_grad()
def forecast_fincast(api, ctx_np, H, freq_int, batch=64):
    """Returns (mean, full) with mean (n, H) and full (n, H, D), D = 1 + n_quantiles."""
    means, fulls = [], []
    i, bs = 0, batch
    while i < len(ctx_np):
        chunk = [np.asarray(x, dtype=np.float32) for x in ctx_np[i:i + bs]]
        try:
            out, full = api.forecast(chunk, [freq_int] * len(chunk))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        out = np.asarray(out)[:, :H]
        full = np.asarray(full)[:, :H, :]
        means.append(out); fulls.append(full); i += bs
    return np.concatenate(means, 0), np.concatenate(fulls, 0)


# ------------------------------------------------------------------- stages

def stage_ladder(args, api, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|f{args.freq}"
            if key in done:
                log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat, full = forecast_fincast(api, y_ctx, H, args.freq, args.batch)
            fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained",
                   "rung": rung, "seed": seed, "H": H, "n": n, "num_samples": 0,
                   "freq": args.freq, "freq_name": FREQ_NAMES[args.freq],
                   "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=None),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            # quantile interval (q1..q9 in full[..., 1:]) for the N2 remark
            if full.shape[-1] >= 10:
                q = full[:, :, 1:]                       # (n, H, 9)
                lo, hi = q[:, :, 0], q[:, :, 8]          # 10% and 90%
                rec["interval80_width_over_sigma"] = (np.mean(hi - lo, 0) / info["sigma"]).tolist()
                rec["coverage_80"] = np.mean((y_true >= lo) & (y_true <= hi), 0).tolist()
                if rung == "N2":
                    cv_last = info["cond_vol"][:, T - 1]
                    rec["corr_width_last_condvol"] = [
                        float(np.corrcoef((hi - lo)[:, h], cv_last)[0, 1]) for h in range(H)]
            append(rec)
            log(f"{rung} s{seed} f{args.freq}: dep/sigma h1={rec['departure_in_sigma'][0]:.3f} "
                f"h{H}={rec['departure_in_sigma'][-1]:.3f} skill_h1={rec['skill_vs_persistence'][0]:+.3f} "
                f"flat={rec['spec_flatness_detrended']:.3f}")


def stage_freq(args, api, done):
    """Declared-frequency test: same data, three declarations."""
    n, T, H = args.n_freq, CTX, args.horizon_long
    y_ctx, y_true, info = make_data("N1", n, T, H, seed=3000)
    outs = {}
    for f in (0, 1, 2):
        key = f"freq|{MODEL_TAG}|pretrained|N1|f{f}|H{H}|n{n}"
        if key in done:
            log(f"skip {key}"); continue
        y_hat, _ = forecast_fincast(api, y_ctx, H, f, max(8, args.batch // 4))
        outs[f] = y_hat
        fl = norm_floor(y_ctx, H)
        rec = {"key": key, "stage": "freq", "model": MODEL_TAG, "init": "pretrained",
               "applicable": True, "rung": "N1", "freq": f, "freq_name": FREQ_NAMES[f],
               "H": H, "n": n, "num_samples": 0,
               **summarise(y_hat, y_ctx, y_true, info, "N1", H, floor=fl, samples=None),
               **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
        append(rec)
        log(f"freq f{f} ({FREQ_NAMES[f]}): dep/sigma h1={rec['departure_in_sigma'][0]:.3f} "
            f"flat={rec['spec_flatness_detrended']:.3f} band7={rec['spec_band_power']['p7']:.4f} "
            f"band24={rec['spec_band_power']['p24']:.4f}")
    # the discriminating quantity: does the forecast MOVE with the declaration?
    if len(outs) == 3:
        d01 = float(np.sqrt(np.mean((outs[0] - outs[1]) ** 2)) / info["sigma"])
        d02 = float(np.sqrt(np.mean((outs[0] - outs[2]) ** 2)) / info["sigma"])
        d12 = float(np.sqrt(np.mean((outs[1] - outs[2]) ** 2)) / info["sigma"])
        append({"key": f"freq_delta|{MODEL_TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq_delta",
                "model": MODEL_TAG, "init": "pretrained", "H": H, "n": n,
                "rms_diff_f0_f1_sigma": d01, "rms_diff_f0_f2_sigma": d02,
                "rms_diff_f1_f2_sigma": d12,
                "note": "same synthetic data, only the declared frequency changed; a model "
                        "reading the data correctly gives identical forecasts"})
        log(f"freq deltas (sigma): f0-f1={d01:.4f} f0-f2={d02:.4f} f1-f2={d12:.4f}")


STAGES = {"ladder": stage_ladder, "freq": stage_freq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128)
    ap.add_argument("--freq", type=int, default=0, help="0 le-daily, 1 weekly/monthly, 2 yearly")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    # one load at the longest horizon needed; shorter horizons are prefixes
    Hmax = max(args.horizon, args.horizon_long if args.stage in ("freq", "all") else 0)
    log(f"loading FinCast v1 from {WEIGHTS} with horizon_len={Hmax}")
    api = load_fincast(Hmax)
    done = done_keys()
    log(f"{len(done)} configs already done")
    failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try:
            STAGES[st](args, api, done)
        except Exception:
            failed = True
            tb = traceback.format_exc()
            print("!" * 70, flush=True); print(f"STAGE {st} FAILED", flush=True)
            print(tb, flush=True); print("!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{MODEL_TAG}|{time.time()}", "stage": st,
                    "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
