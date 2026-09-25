"""Pilot A driver for Sundial-base-128M (Liu et al., ICML 2025; `thuml/sundial-base-128m`): a
decoder-only transformer over 16-point patches whose head is a flow-matching sampler
(TimeFlow), pretrained on the 1-trillion-point TimeBench corpus. Runs in the fincast_v1 env
(transformers 4.53, torch 2.5.0), like Time-MoE.

Shares generators and metrics with the other drivers through nulls.py / probe_metrics.py;
same JSONL schema and key format as the sampled models (Chronos, Moirai): the point forecast
is the mean of S trajectories and the Monte Carlo term is estimated from them and removed.

Particulars (smoke_sundial.py, 2026-09-18): 128.3M parameters, float32; the packaged
generate() fails under transformers 4.53 (its TSGenerationMixin is the same lineage as
Time-MoE's: `DynamicCache.get_max_length` is gone), so the forecast is one direct forward
call with the mixin's own normalisation (mean, population std + 1e-5) and num_samples=S:
the flow head emits 720 steps in one pass, so every horizon used here (16, 128) is a prefix
of one call and horizon-prefix equivalence is exact (rms 0). Sampling uses torch's global
RNG (50 Euler steps from Gaussian noise), seeded per call, so runs are reproducible
(max|diff| 0). Continuous input, no token grid: floor = norm_floor. Takes values only: no
frequency, timestamp or calendar argument, so the declared-frequency test is undefined for it.
Contexts of 512 to 4096 points run (the model card's lookback is 2880).

Usage (inside env_sundial.sh):  python pilot_a_sundial.py --stage all
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np, torch
from transformers import AutoModelForCausalLM
from nulls import RUNGS
from probe_metrics import make_data, summarise, spectra, norm_floor
from paths import RESULTS_DIR

RESULTS = os.environ.get("PILOT_A_OUT", os.path.join(RESULTS_DIR, "pilot_a.jsonl"))
MODEL_ID = os.environ.get("SUNDIAL_MODEL", "thuml/sundial-base-128m"); MODEL_TAG = "sundial"
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

def load_sundial():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to("cuda").eval()
    log(f"loaded {MODEL_ID}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params, dtype {next(m.parameters()).dtype}, "
        f"output_token_lens {m.config.output_token_lens}, sampling steps {m.config.num_sampling_steps}")
    return m

@torch.no_grad()
def forecast_sundial(model, ctx_np, H, S=100, batch=32, seed=0):
    """(n, S, H) sample trajectories in the data's units; one forward call per chunk, the
    mixin's normalisation, torch seeded per chunk (seed + chunk index); OOM-adaptive."""
    assert H <= max(model.config.output_token_lens), "one-pass forecast only up to the flow head's length"
    outs = []; i, bs, k = 0, batch, 0
    while i < len(ctx_np):
        x = torch.tensor(np.asarray(ctx_np[i:i + bs], dtype=np.float32), device="cuda")
        mean = x.mean(-1, keepdim=True); std = x.std(-1, keepdim=True, unbiased=False) + 1e-5
        try:
            torch.manual_seed(seed * 100003 + k)
            out = model(input_ids=(x - mean) / std, revin=False, num_samples=S, max_output_length=H, return_dict=True).logits
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); log(f"  OOM -> batch {bs}"); continue
        outs.append((out[:, :, :H] * std.unsqueeze(1) + mean.unsqueeze(1)).float().cpu().numpy()); i += bs; k += 1
    return np.concatenate(outs, 0)

def stage_ladder(args, model, done):
    n, T, H, S = args.n, CTX, args.horizon, args.num_samples
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for seed in range(args.seeds):
            key = f"ladder|{MODEL_TAG}|pretrained|{rung}|s{seed}|H{H}|n{n}|ns{S}"
            if key in done: log(f"skip {key}"); continue
            y_ctx, y_true, info = make_data(rung, n, T, H, seed=1000 + seed)
            s = forecast_sundial(model, y_ctx, H, S, args.batch, seed)
            y_hat = s.mean(1); fl = norm_floor(y_ctx, H)
            rec = {"key": key, "stage": "ladder", "model": MODEL_TAG, "init": "pretrained", "rung": rung, "seed": seed,
                   "H": H, "n": n, "num_samples": S, "point": "sample_mean", "floor_kind": "affine_normalisation_roundtrip",
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

def stage_direction(args, model, done):
    """Default setting at H=128 on the shared direction contexts (seed 4000)."""
    H, n, S = args.horizon_long, args.n_freq, args.num_samples
    key = f"direction|{MODEL_TAG}|pretrained|H{H}|n{n}|ns{S}"
    if key in done: log(f"skip {key}"); return
    recs = []
    for rung in ("N1", "N2", "N4"):
        y, info = RUNGS[rung](n, CTX, H, np.random.default_rng(4000)); ctx = y[:, :CTX]; sig = info["sigma"]
        s = forecast_sundial(model, ctx, H, S, args.batch, 0); dep = s.mean(1) - ctx[:, -1:]
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{os.path.dirname(RESULTS)}/departures_{MODEL_TAG}_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        r = {"model": MODEL_TAG, "rung": rung, "H": H, "n": n, "S": S,
             "mean_dep_sigma": {h + 1: float(dep[:, h].mean() / sig) for h in (0, 15, 63, H - 1)},
             "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean()) / sig) for h in (0, 15, 63, H - 1)},
             "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)},
             "corr_slope64": {h + 1: float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in (0, 15, 63, H - 1)}}
        recs.append(r)
        log(f"direction {rung}: h=16 mean {r['mean_dep_sigma'][16]:+.2f} up {r['frac_up'][16]:.2f} | h={H} mean {r['mean_dep_sigma'][H]:+.2f} up {r['frac_up'][H]:.2f} corr_slope {r['corr_slope64'][H]:+.2f}")
    json.dump(recs, open(f"{os.path.dirname(RESULTS)}/diag_direction_{MODEL_TAG}.json", "w"), indent=1)
    append({"key": key, "stage": "direction", "model": MODEL_TAG, "init": "pretrained", "H": H, "n": n, "num_samples": S, "file": f"diag_direction_{MODEL_TAG}.json"})
    append({"key": f"freq|{MODEL_TAG}|pretrained|N1|H{H}|n{n}", "stage": "freq", "model": MODEL_TAG, "init": "pretrained", "applicable": False,
            "note": "no frequency, timestamp or calendar input: the declared-frequency test is undefined for this model"})

STAGES = {"ladder": stage_ladder, "direction": stage_direction}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ladder", choices=list(STAGES) + ["all"])
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizon-long", type=int, default=128); ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    log(f"loading {MODEL_ID}"); model = load_sundial()
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
