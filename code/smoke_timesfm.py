"""TimesFM 2.0-500m smoke test (timesfm==1.3.0 API). Load, shapes, finiteness,
freq indicator, quantile layout, and prefix equivalence (decoder-only with
output patch 128: the first 16 of an H=128 forecast should equal an H=16 one).
Runs inside the timesfm env."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from nulls import RUNGS
import timesfm
ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""))
def load(H):
    return timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32, horizon_len=H,
                                                          num_layers=50, use_positional_embedding=False, context_len=2048),
                           checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0)); ctx = y[:, :512]
tfm = load(16)
pt, q = tfm.forecast(list(ctx), freq=[0] * 8)
pt, q = np.asarray(pt), np.asarray(q)
chk("H=16 point shape (8,16)", pt.shape == (8, 16), str(pt.shape))
chk("H=16 quantile shape (8,16,10)", q.shape == (8, 16, 10), str(q.shape))
chk("finite", np.all(np.isfinite(pt)) and np.all(np.isfinite(q)))
print(f"INFO  point vs q[...,5] (median) max|diff|={np.max(np.abs(q[...,5]-pt)):.2e}; vs q[...,0] (mean)={np.max(np.abs(q[...,0]-pt)):.2e}  (point_forecast_mode defaults to median; driver uses q[...,0])")
chk("quantiles monotone", np.all(np.diff(q[..., 1:], axis=-1) >= -1e-5))
dep = np.sqrt(np.mean((pt - ctx[:, -1:]) ** 2)) / info["sigma"]
print(f"   H=16 departure rmse = {dep:.4f} sigma")
for f in (0, 1, 2):
    p_f, _ = tfm.forecast(list(ctx), freq=[f] * 8); p_f = np.asarray(p_f)
    chk(f"freq={f} runs", p_f.shape == (8, 16) and np.all(np.isfinite(p_f)), f"dep={np.sqrt(np.mean((p_f-ctx[:,-1:])**2))/info['sigma']:.3f}σ")
tfm128 = load(128)
p128, _ = tfm128.forecast(list(ctx), freq=[0] * 8); p128 = np.asarray(p128)
chk("H=128 shape (8,128)", p128.shape == (8, 128))
d = np.max(np.abs(p128[:, :16] - pt))
chk("prefix equivalence: first 16 of H=128 == H=16", d < 1e-3 * info["sigma"], f"max|diff|={d:.3e}")
print(f"   H=128 departure at h=128 = {np.sqrt(np.mean((p128[:,-1]-ctx[:,-1])**2))/info['sigma']:.3f} sigma")
print("\n" + ("TIMESFM SMOKE OK" if ok else "TIMESFM SMOKE FAILED")); sys.exit(0 if ok else 1)
