"""FinCast smoke test. Verifies, before any GPU time is committed to the grid:
loading, forecast shape and finiteness, the quantile layout, the frequency
reader, and the assumption the driver rests on: that the first H steps of a
longer forecast equal a forecast made with horizon_len=H, so one model load
at the longest horizon serves every stage."""
import sys, os, numpy as np, torch
sys.path.insert(0, ".")
from pilot_a_fincast import load_fincast, forecast_fincast, norm_floor, log
from nulls import RUNGS

ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""))

from tools.inference_utils import freq_reader_inference as fr
chk("freq reader: H, D, W, M, Y map to 0,0,1,1,2",
    [fr("H"), fr("D"), fr("W"), fr("M"), fr("Y")] == [0, 0, 1, 1, 2],
    str([fr("H"), fr("D"), fr("W"), fr("M"), fr("Y")]))

y, info = RUNGS["N1"](8, 512, 256, np.random.default_rng(0))
ctx, truth = y[:, :512], y[:, 512:]

api16 = load_fincast(16)
m16, f16 = forecast_fincast(api16, ctx, 16, 0, batch=8)
chk("H=16: mean shape (8,16)", m16.shape == (8, 16), str(m16.shape))
chk("H=16: full shape (8,16,D) with D>=10", f16.ndim == 3 and f16.shape[:2] == (8, 16) and f16.shape[2] >= 10, str(f16.shape))
chk("H=16: finite", np.all(np.isfinite(m16)) and np.all(np.isfinite(f16)))
chk("full[...,0] is the mean output", np.allclose(f16[..., 0], m16, atol=1e-5),
    f"max|diff|={np.max(np.abs(f16[...,0]-m16)):.2e}")
q = f16[..., 1:]
chk("quantiles are monotone in q", np.all(np.diff(q, axis=-1) >= -1e-5),
    f"min step {np.min(np.diff(q, axis=-1)):.2e}")
dep16 = np.sqrt(np.mean((m16 - ctx[:, -1:]) ** 2)) / info["sigma"]
print(f"   H=16 departure rmse = {dep16:.4f} sigma ; norm floor = {np.sqrt(norm_floor(ctx,16).mean())/info['sigma']:.2e} sigma")
del api16; torch.cuda.empty_cache()

api256 = load_fincast(256)
m256, f256 = forecast_fincast(api256, ctx, 256, 0, batch=8)
chk("H=256: mean shape (8,256)", m256.shape == (8, 256), str(m256.shape))
chk("H=256: finite", np.all(np.isfinite(m256)))
d = np.max(np.abs(m256[:, :16] - m16))
chk("horizon-prefix equivalence: first 16 of H=256 == H=16 forecast", d < 1e-3 * info["sigma"],
    f"max|diff| = {d:.3e} (sigma={info['sigma']:.3g})")
dep256 = np.sqrt(np.mean((m256[:, -1] - ctx[:, -1]) ** 2)) / info["sigma"]
print(f"   H=256 departure at h=256 = {dep256:.3f} sigma")
print("\n" + ("FINCAST SMOKE OK" if ok else "FINCAST SMOKE FAILED"))
sys.exit(0 if ok else 1)
