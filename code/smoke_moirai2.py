"""Smoke test for Moirai-2.0-R-small (Salesforce, 2025; decoder-only successor of Moirai-1.1),
run in the `moirai` env (uni2ts 2.0.0 has uni2ts.model.moirai2).

Checks: the module loads on the GPU; the quantile levels it returns; output shapes and
finiteness from a direct forward call with (n, T, 1) tensors; whether the 0.5 quantile is the
only point summary (no mean output); horizon-prefix equivalence (H=16 vs the first 16 of H=128);
contexts of 1024 / 2048 / 4096; time per 128 windows at H=128.

Usage (inside env_moirai.sh):  python smoke_moirai2.py
"""
import time
import numpy as np, torch
from nulls import RUNGS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
log("loading Salesforce/moirai-2.0-R-small")
module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
log(f"loaded; quantile_levels = {list(module.quantile_levels)}; params {sum(p.numel() for p in module.parameters()) / 1e6:.1f}M; patch_size {getattr(module, 'patch_size', '?')}")

@torch.no_grad()
def fc(ctx_np, H):
    T = ctx_np.shape[1]
    model = Moirai2Forecast(module=module, prediction_length=H, context_length=T, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to("cuda").eval()
    x = torch.tensor(ctx_np, dtype=torch.float32, device="cuda")[:, :, None]
    out = model(past_target=x, past_observed_target=torch.ones_like(x, dtype=torch.bool),
                past_is_pad=torch.zeros(x.shape[:2], dtype=torch.bool, device="cuda"))
    out = out.float().cpu().numpy()
    return out[..., 0] if out.ndim == 4 else out             # (n, num_quantiles, H)

y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0)); ctx = y[:, :512]; sig = info["sigma"]
t0 = time.time(); q = fc(ctx, 128); dt = time.time() - t0
log(f"forward output {q.shape}, finite {np.isfinite(q).all()}, {dt:.1f}s for 8 windows")
ql = list(module.quantile_levels); i50 = ql.index(0.5) if 0.5 in ql else None
log(f"median index {i50}; quantile-average minus median max|diff| = {np.max(np.abs(q.mean(1) - q[:, i50])) / sig:.3f} sigma")
q16 = fc(ctx, 16)
log(f"prefix equivalence: rms(H=16 - first 16 of H=128) = {np.sqrt(np.mean((q16[:, i50, :] - q[:, i50, :16]) ** 2)) / sig:.4f} sigma")
log(f"departure at h=128 on 8 random walks (median): mean {np.mean((q[:, i50, -1] - ctx[:, -1]) / sig):+.2f} sigma")
for T in (1024, 2048, 4096):
    yy, ii = RUNGS["N1"](4, T, 16, np.random.default_rng(1))
    try:
        qq = fc(yy[:, :T], 16); log(f"context {T}: ok, finite {np.isfinite(qq).all()}")
    except Exception as e:
        log(f"context {T}: FAILED ({type(e).__name__}: {str(e)[:120]})")
y2, _ = RUNGS["N1"](128, 512, 128, np.random.default_rng(2))
t0 = time.time(); fc(y2[:, :512], 128); log(f"128 windows at H=128: {time.time() - t0:.1f}s")
log("SMOKE DONE")
