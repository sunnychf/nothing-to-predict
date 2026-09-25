"""Smoke test for TimesFM-2.5-200M (Google, 2025; timesfm 3.x API), run in the `tsfm25` env.

Checks: load and compile with the README's default ForecastConfig; the config's fields (it has
`force_flip_invariance`, a built-in symmetry averaging under value negation, and
`infer_is_positive`); output shapes and finiteness on 8 random walks at H=128; whether the point
output is the mean or the median of the quantile head; horizon-prefix equivalence; the
departure at h=128 with flip invariance ON (default) and OFF; contexts up to 2048; timing.

Usage (inside env_tsfm25.sh):  python smoke_timesfm25.py
"""
import inspect, time
import numpy as np, torch
from nulls import RUNGS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

import timesfm
log(f"timesfm {getattr(timesfm, '__version__', '?')}; ForecastConfig fields: {inspect.signature(timesfm.ForecastConfig)}")
log("loading google/timesfm-2.5-200m-pytorch")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
log(f"loaded {type(model).__name__}; forecast signature: {inspect.signature(model.forecast)}")

def compile_(flip, max_context=2048):
    model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=256, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=flip,
                                         infer_is_positive=True, fix_quantile_crossing=True))

y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0)); ctx = y[:, :512]; sig = info["sigma"]
compile_(True)
t0 = time.time(); p, q = model.forecast(horizon=128, inputs=[c.astype(np.float32) for c in ctx]); dt = time.time() - t0
p, q = np.asarray(p), np.asarray(q)
log(f"flip ON: point {p.shape}, quantiles {q.shape}, finite {np.isfinite(p).all() and np.isfinite(q).all()}, {dt:.1f}s for 8 windows")
if q.ndim == 3 and q.shape[-1] >= 10:
    log(f"  point vs q[...,0] (mean channel) max|diff| = {np.max(np.abs(p - q[..., 0])) / sig:.4f} sigma; point vs q[...,5] (median) = {np.max(np.abs(p - q[..., 5])) / sig:.4f} sigma")
p16, _ = model.forecast(horizon=16, inputs=[c.astype(np.float32) for c in ctx]); p16 = np.asarray(p16)
log(f"  prefix equivalence: rms(H=16 - first 16 of H=128) = {np.sqrt(np.mean((p16 - p[:, :16]) ** 2)) / sig:.4f} sigma")
log(f"  departure at h=128 on 8 random walks: mean {np.mean((p[:, -1] - ctx[:, -1]) / sig):+.2f} sigma")
compile_(False)
p2, _ = model.forecast(horizon=128, inputs=[c.astype(np.float32) for c in ctx]); p2 = np.asarray(p2)
log(f"flip OFF: departure at h=128: mean {np.mean((p2[:, -1] - ctx[:, -1]) / sig):+.2f} sigma; rms(on - off) = {np.sqrt(np.mean((p - p2) ** 2)) / sig:.3f} sigma")
neg = -ctx
p3, _ = model.forecast(horizon=128, inputs=[c.astype(np.float32) for c in neg]); p3 = np.asarray(p3)
log(f"flip OFF, negated inputs: departure {np.mean((p3[:, -1] - neg[:, -1]) / sig):+.2f} sigma (a flip-invariant model would give exactly the negative of the ON forecast)")
for T in (1024, 2048):
    yy, ii = RUNGS["N1"](4, T, 16, np.random.default_rng(1))
    try:
        pp, _ = model.forecast(horizon=16, inputs=[c.astype(np.float32) for c in yy[:, :T]]); log(f"context {T}: ok, finite {np.isfinite(np.asarray(pp)).all()}")
    except Exception as e:
        log(f"context {T}: FAILED ({type(e).__name__}: {str(e)[:120]})")
compile_(True)
y2, _ = RUNGS["N1"](128, 512, 128, np.random.default_rng(2))
t0 = time.time(); model.forecast(horizon=128, inputs=[c.astype(np.float32) for c in y2[:, :512]]); log(f"128 windows at H=128 (flip ON): {time.time() - t0:.1f}s")
log("SMOKE DONE")
