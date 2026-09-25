"""Pilot A driver for Time-MoE (Shi et al. 2024), Maple728/TimeMoE-200M by default.

Runs in the fincast_v1 conda env (transformers 4.53, torch 2.5.0+cu124); the
env script may point HF_HOME at a scratch disk (scripts/env_timemoe.sh). Shares
generators and metrics with the other drivers through nulls.py /
probe_metrics.py; same JSONL schema and key format.

Time-MoE particulars (checked in smoke_timemoe.py):
  - Deterministic point forecaster: greedy autoregression over the model's
    own value output, no quantiles, no sampling, no MC term.
  - Continuous, context-normalised by the caller (mean/std of the context), no
    token grid: floor is norm_floor (~0), same schema field as TimesFM/FinCast.
  - Multi-horizon heads (1/8/32/64 steps): an H=16 forecast is decoded as 8+8
    from the 8-step head, an H=128 one as 64+64 from the 64-step head, so the
    first 16 steps of the two are NOT the same computation (smoke test reports
    the gap). Each stage therefore decodes at its own horizon, as for Moirai.
  - Takes values only. No frequency, timestamp or calendar argument exists, so
    the declared-frequency test is undefined for it, exactly as for Chronos;
    stage_freq records that and does nothing else.

Usage (inside env_timemoe.sh):  python pilot_a_timemoe.py --stage all
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, time, traceback
import numpy as np, torch
from transformers import AutoModelForCausalLM
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor

from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = os.environ.get("TIMEMOE_MODEL", "Maple728/TimeMoE-200M")
MODEL_TAG = "timemoe-" + MODEL_ID.split("-")[-1].lower()          # timemoe-200m / timemoe-50m
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

def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True); m.eval()
    log(f"loaded {MODEL_ID}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params, dtype {next(m.parameters()).dtype}")
    return m

def decode(model, x_norm, H):
    """Greedy multi-horizon decoding, a re-implementation of Time-MoE's TSGenerationMixin
    (its generate() relies on transformers-4.40 internals that 4.53 removed). Each call
    to the model with max_horizon_length=r uses the largest prediction head whose
    horizon <= r (heads 1/8/32/64) and returns that many steps at once from the last
    position; the steps are appended and the loop repeats until H values exist.
    No KV cache: every call re-encodes the whole sequence (512+H <= 4096)."""
    seq = x_norm.unsqueeze(-1)                      # [B, T, input_size=1]
    remaining = H
    with torch.no_grad():
        while remaining > 0:
            out = model(input_ids=seq, max_horizon_length=remaining, use_cache=False, return_dict=True)
            nxt = out.logits[:, -1, :].reshape(seq.shape[0], -1, seq.shape[2])   # [B, k, 1]
            seq = torch.cat([seq, nxt], dim=1); remaining -= nxt.shape[1]
    return seq[:, -H:, 0] if seq.shape[1] - x_norm.shape[1] == H else seq[:, x_norm.shape[1]:x_norm.shape[1] + H, 0]

def forecast_tm(model, ctx_np, H, batch=64):
    """Point forecast (n,H); chunked, OOM-adaptive. Normalise each context by its
    own mean/std as the model card prescribes, decode H new values, invert."""
    outs = []; i, bs = 0, batch
    while i < len(ctx_np):
        x = torch.tensor(np.asarray(ctx_np[i:i + bs], dtype=np.float32), device="cuda")
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
        try: y = decode(model, (x - mean) / std, H)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        outs.append((y * std + mean).float().cpu().numpy()); i += bs
    return np.concatenate(outs, 0)

def stage_ladder(args, model, done):
    n, T, H = args.n, CTX, args.horizon
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            y_hat = forecast_tm(model, y_ctx, H, args.batch); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung, "seed": seed,
                   "H": H, "n": n, "num_samples": 0, "floor_kind": "affine_normalisation_roundtrip",
                   **summarise(y_hat, y_ctx, y_true, info, rung, H, floor=fl, samples=None),
                   **{f"spec_{k}": v for k, v in spectra(y_hat, y_ctx).items()}}
            append(rec)
            log(f"{rung} s{seed}: dep h1={rec['departure_in_sigma'][0]:.3f} h{H}={rec['departure_in_sigma'][-1]:.3f} "
                f"skill_h1={rec['skill_vs_persistence'][0]:+.3f} up_h{H}={rec['frac_departure_up'][-1]:.2f}")

def stage_freq(args, model, done):
    """Time-MoE conditions on values only; record that the test is undefined, as for Chronos."""
    key = f"freq|{MODEL_TAG}|pretrained|not_applicable"
    if key in done: log(f"skip {key}"); return
    sig = list(inspect.signature(model.forward).parameters)
    append({"key": key, "stage": "freq", "model": MODEL_TAG, "init": "pretrained", "applicable": False,
            "forward_signature": sig,
            "reason": "Time-MoE conditions on values only; no frequency, timestamp or calendar argument, so the declared-frequency test is undefined for this model (same status as Chronos)."})
    log(f"freq: not applicable (forward args: {sig[:8]})")

def stage_direction(args, model, done):
    """H=128 on the shared direction contexts (seed 4000), per-series departures saved for the figure."""
    H, n = args.horizon_long, args.n_dir
    key = f"direction|{MODEL_TAG}|pretrained|H{H}|n{n}"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        y_hat = forecast_tm(model, ctx, H, args.batch); dep = y_hat - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{os.path.dirname(RESULTS)}/departures_timemoe_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        r = {"model": MODEL_TAG, "rung": rung, "H": H, "n": n,
             "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
             "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
             "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
             "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)},
             "spec_flatness_detrended": spectra(y_hat, ctx)["flatness_detrended"]}
        recs.append(r)
        log(f"direction {rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h={H} mean {r['mean_dep_sigma'][H]:+.2f} up {r['frac_up'][H]:.2f} corr(slope) {r['corr_slope64'][H]:+.2f} flat {r['spec_flatness_detrended']:.2f}")
    json.dump(recs, open(f"{os.path.dirname(RESULTS)}/diag_direction_timemoe.json", "w"), indent=1)
    append({"key": key, "stage": "direction", "model": MODEL_TAG, "init": "pretrained", "H": H, "n": n, "file": "diag_direction_timemoe.json"})

STAGES = {"ladder": stage_ladder, "freq": stage_freq, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-dir", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    model = load_model(); done = done_keys(); log(f"{len(done)} configs already done")
    failed = False
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        log(f"===== stage {st} =====")
        try: STAGES[st](args, model, done)
        except Exception:
            failed = True; tb = traceback.format_exc()
            print("!" * 70 + f"\nSTAGE {st} FAILED\n{tb}\n" + "!" * 70, flush=True)
            append({"key": f"ERROR|{st}|{MODEL_TAG}|{time.time()}", "stage": st, "error": True, "traceback": tb})
    log("DONE" if not failed else "DONE WITH FAILURES"); sys.exit(1 if failed else 0)
if __name__ == "__main__": main()
