"""Moirai smoke test. Load, forecast shape/finiteness, explicit patch sizes,
and the prefix-equivalence assumption. Run inside the `moirai` env."""
import sys, numpy as np, torch, pandas as pd
sys.path.insert(0, ".")
from nulls import RUNGS
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset
ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""))

MODEL = "Salesforce/moirai-1.1-R-small"
module = MoiraiModule.from_pretrained(MODEL)
chk("patch_sizes are the five documented", tuple(module.patch_sizes) == (8, 16, 32, 64, 128), str(module.patch_sizes))

def make(H, patch, S=20):
    return MoiraiForecast(module=module, prediction_length=H, context_length=512, patch_size=patch,
                          num_samples=S, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)

def run(model, ctx, bs=8):
    ds = ListDataset([{"start": pd.Period("2000-01-01", freq="D"), "target": c.astype(np.float32)} for c in ctx], freq="D")
    pred = model.create_predictor(batch_size=bs, device="cuda")
    fcs = list(pred.predict(ds))
    return np.stack([f.samples for f in fcs])          # (n, S, H)

y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0))
ctx = y[:, :512]
s16 = run(make(16, "auto"), ctx)
chk("auto, H=16: samples shape (8,20,16)", s16.shape == (8, 20, 16), str(s16.shape))
chk("finite", np.all(np.isfinite(s16)))
dep = np.sqrt(np.mean((s16.mean(1) - ctx[:, -1:]) ** 2)) / info["sigma"]
print(f"   auto H=16 departure rmse = {dep:.4f} sigma")
for ps in (8, 16, 32, 64, 128):
    s = run(make(16, ps), ctx)
    chk(f"patch_size={ps:>3} runs, shape ok", s.shape == (8, 20, 16) and np.all(np.isfinite(s)),
        f"dep={np.sqrt(np.mean((s.mean(1)-ctx[:,-1:])**2))/info['sigma']:.3f}σ")
# prefix equivalence at a fixed patch size (sampling makes it statistical, so compare means with S large)
torch.manual_seed(0); a = run(make(16, 32, S=200), ctx).mean(1)
torch.manual_seed(0); b = run(make(128, 32, S=200), ctx).mean(1)[:, :16]
d = np.sqrt(np.mean((a - b) ** 2)) / info["sigma"]
# Informational, not a gate: Moirai is a masked encoder and predicts all H at
# once, so the first patch of a longer forecast is NOT the same computation as
# a short forecast. The driver instantiates per horizon and never relies on this.
print(f"INFO  prefix (non-)equivalence at patch 32: first 16 of H=128 vs H=16, rms diff {d:.3f}σ "
      f"(sampling noise ~{np.sqrt(2/200)*dep:.3f}σ); masked encoder, horizon-dependent by design")
print("\n" + ("MOIRAI SMOKE OK" if ok else "MOIRAI SMOKE FAILED")); sys.exit(0 if ok else 1)
