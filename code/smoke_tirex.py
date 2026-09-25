"""Smoke test for TiRex (NX-AI, 2025; xLSTM backbone), run before any TiRex experiment.

Checks, in this order: the package imports and the model loads on the GPU; the forecast API's
signature; output shapes and finiteness on (n, T) inputs; whether the returned point forecast is
the mean channel or the median (max |mean - q0.5|); horizon-prefix equivalence (H=16 against the
first 16 steps of H=128); the longest context the model accepts among 512 / 1024 / 2048 / 4096;
and the time per 128 windows at H=128.

Usage (inside env_tirex.sh):  python smoke_tirex.py
"""
import inspect, time
import numpy as np, torch
from nulls import RUNGS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

from tirex import load_model
log("loading NX-AI/TiRex")
model = load_model("NX-AI/TiRex")
log(f"loaded: {type(model).__name__}; forecast signature: {inspect.signature(model.forecast)}")
try:
    n_par = sum(p.numel() for p in model.parameters()); log(f"{n_par / 1e6:.1f}M parameters")
except Exception as e:
    log(f"parameter count unavailable: {e}")

y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0)); ctx = y[:, :512]; sig = info["sigma"]
x = torch.tensor(ctx, dtype=torch.float32)
t0 = time.time(); q, m = model.forecast(context=x, prediction_length=128); dt = time.time() - t0
q, m = np.asarray(q.float().cpu() if torch.is_tensor(q) else q), np.asarray(m.float().cpu() if torch.is_tensor(m) else m)
log(f"quantiles {q.shape}, mean {m.shape}, finite {np.isfinite(q).all() and np.isfinite(m).all()}, {dt:.1f}s for 8 windows")
if q.ndim == 3 and q.shape[-1] == 9:
    med = q[..., 4]
    log(f"point vs q0.5 max|diff| = {np.max(np.abs(m - med)) / sig:.3f} sigma; point vs quantile-average = {np.max(np.abs(m - q.mean(-1))) / sig:.3f} sigma")
elif q.ndim == 3 and q.shape[1] == 9:
    med = q[:, 4]
    log(f"(quantile axis is 1) point vs q0.5 max|diff| = {np.max(np.abs(m - med)) / sig:.3f} sigma")
q16, m16 = model.forecast(context=x, prediction_length=16)
m16 = np.asarray(m16.float().cpu() if torch.is_tensor(m16) else m16)
log(f"prefix equivalence: rms(H=16 - first 16 of H=128) = {np.sqrt(np.mean((m16 - m[:, :16]) ** 2)) / sig:.4f} sigma")
log(f"departure at h=128 on 8 random walks: mean {np.mean((m[:, -1] - ctx[:, -1]) / sig):+.2f} sigma")
for T in (1024, 2048, 4096):
    yy, ii = RUNGS["N1"](4, T, 16, np.random.default_rng(1))
    try:
        qq, mm = model.forecast(context=torch.tensor(yy[:, :T], dtype=torch.float32), prediction_length=16)
        mm = np.asarray(mm.float().cpu() if torch.is_tensor(mm) else mm)
        log(f"context {T}: ok, finite {np.isfinite(mm).all()}")
    except Exception as e:
        log(f"context {T}: FAILED ({type(e).__name__}: {str(e)[:120]})")
y2, _ = RUNGS["N1"](128, 512, 128, np.random.default_rng(2))
t0 = time.time(); model.forecast(context=torch.tensor(y2[:, :512], dtype=torch.float32), prediction_length=128); log(f"128 windows at H=128: {time.time() - t0:.1f}s")
log("SMOKE DONE")
