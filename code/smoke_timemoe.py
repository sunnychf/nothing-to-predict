"""Smoke test for Time-MoE (Shi et al. 2024) before any Pilot A run.

Loads Maple728/TimeMoE-<size> through transformers' AutoModelForCausalLM with
trust_remote_code, on a few N1 random walks, and checks the things the driver
will rely on:
  1. inputs: the model wants sequences normalised by their own mean/std, and
     generate(max_new_tokens=H) returns the context followed by H new points,
     so the forecast is output[:, -H:] * std + mean
  2. deterministic: two calls give identical output
  3. horizon dependence: the 8-step and 64-step heads give different first-16
     forecasts, reported as an rms gap (informational; stages decode per horizon)
  4. no frequency / timestamp input exists in the forward signature
  5. finite output, sane departure from persistence in sigma units
Usage: python smoke_timemoe.py [--model Maple728/TimeMoE-200M] [--batch 8]
"""
import argparse, inspect, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM
from pilot_a_timemoe import decode
from nulls import RUNGS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Maple728/TimeMoE-200M")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""), flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", trust_remote_code=True)
model.eval()
print(f"loaded {args.model} in {time.time()-t0:.0f}s; params = {sum(p.numel() for p in model.parameters())/1e6:.0f}M; dtype = {next(model.parameters()).dtype}", flush=True)
sig = inspect.signature(model.forward)
print("forward signature:", list(sig.parameters)[:12], flush=True)
chk("no frequency/timestamp argument in forward", not any(k in sig.parameters for k in ("freq", "frequency", "timestamps", "time_features")))

def forecast(ctx_np, H):
    x = torch.tensor(np.asarray(ctx_np, dtype=np.float32), device="cuda")
    mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
    return (decode(model, (x - mean) / std, H) * std + mean).float().cpu().numpy()

y, info = RUNGS["N1"](args.batch, 512, 128, np.random.default_rng(4000)); ctx = y[:, :512]; s = info["sigma"]
t0 = time.time(); f16 = forecast(ctx, 16); t16 = time.time() - t0
chk("H=16 output shape (batch,16)", f16.shape == (args.batch, 16), f"{f16.shape}, {t16:.1f}s")
chk("finite", np.isfinite(f16).all())
f16b = forecast(ctx, 16)
chk("deterministic (two calls identical)", np.array_equal(f16, f16b), f"max|diff|={np.max(np.abs(f16-f16b)):.2e}")
t0 = time.time(); f128 = forecast(ctx, 128); t128 = time.time() - t0
chk("H=128 output shape (batch,128)", f128.shape == (args.batch, 128), f"{t128:.1f}s")
d = np.sqrt(np.mean((f128[:, :16] - f16) ** 2)) / s
print(f"   INFO  first 16 of H=128 (64-step head) vs H=16 (8-step head, twice): rms gap {d:.3f} sigma -> stages decode at their own horizon", flush=True)
dep16 = np.sqrt(np.mean((f16 - ctx[:, -1:]) ** 2, 0)) / s
print(f"   departure/sigma h=1 {dep16[0]:.3f}  h=16 {dep16[-1]:.3f}; signed mean h=128 {np.mean(f128[:, -1] - ctx[:, -1])/s:+.2f}σ up {np.mean(f128[:, -1] > ctx[:, -1]):.2f}", flush=True)
chk("departure not absurd (h=1 < 5 sigma)", dep16[0] < 5)
print("\n" + ("TIMEMOE SMOKE OK" if ok else "TIMEMOE SMOKE FAILED"), flush=True); sys.exit(0 if ok else 1)
