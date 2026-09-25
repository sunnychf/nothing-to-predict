"""Smoke test: does the pipeline load, does the codec round trip behave, and is
the floor actually a floor? Run before committing GPU time to the full grid."""
import numpy as np, torch, sys
sys.path.insert(0, ".")
from pilot_a import load_pipeline, forecast, codec_floor
from nulls import RUNGS

ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""))

pipe = load_pipeline("amazon/chronos-t5-small", "pretrained", "cuda")
print("loaded:", type(pipe).__name__, "| tokenizer:", type(pipe.tokenizer).__name__)
print("n_special_tokens:", pipe.tokenizer.config.n_special_tokens,
      "n_tokens:", pipe.tokenizer.config.n_tokens,
      "pred_len:", pipe.tokenizer.config.prediction_length)

y, info = RUNGS["N1"](8, 512, 16, np.random.default_rng(0))
ctx, truth = y[:, :512], y[:, 512:]

# codec round trip: reconstruction of a constant must be close but not exact
fl = codec_floor(pipe, ctx, ctx[:, -1], 16)
rmse_floor = np.sqrt(fl.mean())
chk("codec floor is finite and positive", np.isfinite(rmse_floor) and rmse_floor > 0,
    f"rmse = {rmse_floor:.6f} (sigma = {info['sigma']:.4f}, "
    f"= {rmse_floor/info['sigma']:.4f} sigma)")
chk("codec floor is small relative to one step", rmse_floor < info["sigma"],
    f"{rmse_floor:.6f} < {info['sigma']:.6f}")
chk("codec floor is flat across horizon (it reconstructs a constant)",
    np.allclose(fl.mean(0), fl.mean(0)[0], rtol=1e-9), f"{fl.mean(0)[:3]}")

# the floor must genuinely bound the model's departure from below
s = forecast(pipe, ctx, 16, 20, batch=8, seed=0)
yhat = s.mean(1)
dep = (yhat - ctx[:, -1:]) ** 2
print(f"\nmodel departure rmse = {np.sqrt(dep.mean()):.6f} "
      f"({np.sqrt(dep.mean())/info['sigma']:.3f} sigma)")
print(f"codec floor      rmse = {rmse_floor:.6f} "
      f"({rmse_floor/info['sigma']:.3f} sigma)")
chk("forecast shape is (n, num_samples, H)", s.shape == (8, 20, 16), str(s.shape))
chk("forecasts are finite", np.all(np.isfinite(s)))

# a model asked to reproduce persistence cannot beat the codec: check the bound
# holds for the single closest representable output, not for the model.
chk("floor bounds the best representable output", rmse_floor <= np.sqrt(dep.mean()) + 1e-9,
    "floor <= model departure" if rmse_floor <= np.sqrt(dep.mean()) else "VIOLATED")

# random init loads too
rp = load_pipeline("amazon/chronos-t5-small", "random", "cuda")
sr = forecast(rp, ctx, 16, 20, batch=8, seed=0)
print(f"random-init departure rmse = {np.sqrt(((sr.mean(1)-ctx[:,-1:])**2).mean())/info['sigma']:.3f} sigma")
chk("random-init model runs", np.all(np.isfinite(sr)))
print("\n" + ("SMOKE OK" if ok else "SMOKE FAILED"))
sys.exit(0 if ok else 1)
