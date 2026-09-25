"""Smoke test for Sundial-base-128M (Liu et al., ICML 2025; `thuml/sundial-base-128m`), a
decoder-only generative forecaster with a flow-matching head, run in the fincast_v1 env
(transformers 4.53, torch 2.5.0; the model card pins transformers 4.40.1).

Checks, on 8 N1 random walks (seed 0): the model loads with trust_remote_code; whether the
packaged generate() still runs under transformers 4.53 (its TSGenerationMixin overrides
_greedy_search, which 4.53 no longer calls, the same lineage as Time-MoE's); a direct forward
call with the released normalisation (mean / population std + 1e-5, as in the mixin) and
num_samples=S returns (n, S, H) for H <= 720 in one pass, since the flow head emits 720 steps
at once; finiteness; that the samples are dispersed (a generative model, not a point one) and
reproducible under torch.manual_seed; horizon-prefix equivalence (H=16 vs the first 16 of
H=128 under the same seed); the departure at h=128 of the sample mean; contexts of 1024, 2048
and 2880 (the model card's lookback); time per 128 windows at H=128 with S=100.

Usage (inside env_sundial.sh):  python smoke_sundial.py
"""
import inspect, time
import numpy as np, torch
from transformers import AutoModelForCausalLM
from nulls import RUNGS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

MODEL_ID = "thuml/sundial-base-128m"
log(f"loading {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to("cuda").eval()
cfg = model.config
log(f"loaded {type(model).__name__}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, dtype {next(model.parameters()).dtype}; "
    f"input_token_len {cfg.input_token_len}, output_token_lens {cfg.output_token_lens}, num_sampling_steps {cfg.num_sampling_steps}, "
    f"layers {cfg.num_hidden_layers}, hidden {cfg.hidden_size}")
sig = inspect.signature(model.forward)
log(f"forward signature: {list(sig.parameters)}")
log(f"no frequency/timestamp argument: {not any(k in sig.parameters for k in ('freq', 'frequency', 'timestamps', 'time_features'))}")

@torch.no_grad()
def fc(ctx_np, H, S, seed=0):
    """(n, S, H) samples in the data's units: the mixin's normalisation, one forward call."""
    x = torch.tensor(np.asarray(ctx_np, dtype=np.float32), device="cuda")
    mean = x.mean(-1, keepdim=True); std = x.std(-1, keepdim=True, unbiased=False) + 1e-5
    torch.manual_seed(seed)
    out = model(input_ids=(x - mean) / std, revin=False, num_samples=S, max_output_length=H, return_dict=True).logits
    return (out[:, :, :H] * std.unsqueeze(1) + mean.unsqueeze(1)).float().cpu().numpy()

y, info = RUNGS["N1"](8, 512, 128, np.random.default_rng(0)); ctx = y[:, :512]; sig_ = info["sigma"]
# 1. packaged generate()
try:
    torch.manual_seed(0)
    g = model.generate(torch.tensor(ctx, dtype=torch.float32, device="cuda"), max_new_tokens=128, num_samples=20)
    g = np.asarray(g.float().cpu()); log(f"packaged generate(): ran, output {g.shape}, finite {np.isfinite(g).all()}")
    gen_ok = g.shape == (8, 20, 128) and np.isfinite(g).all()
except Exception as e:
    gen_ok = False; log(f"packaged generate(): FAILED under transformers 4.53 ({type(e).__name__}: {str(e)[:160]})")
# 2. direct forward
t0 = time.time(); s = fc(ctx, 128, 100); dt = time.time() - t0
log(f"direct forward: samples {s.shape}, finite {np.isfinite(s).all()}, {dt:.1f}s for 8 windows x 100 samples")
if gen_ok:
    torch.manual_seed(0); s20 = fc(ctx, 128, 20, seed=0)
    log(f"  generate() vs direct forward (S=20, same seed): max|diff| = {np.max(np.abs(g - s20)) / sig_:.2e} sigma")
disp = np.mean(s.std(1), 0)
log(f"  sample dispersion (std across samples / sigma): h=1 {disp[0] / sig_:.2f}, h=16 {disp[15] / sig_:.2f}, h=128 {disp[127] / sig_:.2f} (sqrt(h): 1, 4, 11.3)")
s2 = fc(ctx, 128, 100)
log(f"  reproducible under the seed: max|diff| = {np.max(np.abs(s - s2)):.2e}")
s16 = fc(ctx, 16, 100)
log(f"  prefix equivalence: rms(H=16 - first 16 of H=128) = {np.sqrt(np.mean((s16 - s[:, :, :16]) ** 2)) / sig_:.2e} sigma")
m = s.mean(1)
log(f"  departure at h=128 on 8 random walks (sample mean): mean {np.mean((m[:, -1] - ctx[:, -1]) / sig_):+.2f} sigma, up {np.mean(m[:, -1] > ctx[:, -1]):.2f}; "
    f"h=1 rms {np.sqrt(np.mean((m[:, 0] - ctx[:, -1]) ** 2)) / sig_:.3f} sigma")
med = np.median(s, 1)
log(f"  mean vs median of the samples at h=128: max|diff| = {np.max(np.abs(m[:, -1] - med[:, -1])) / sig_:.3f} sigma")
for T in (1024, 2048, 2880, 4096):
    yy, ii = RUNGS["N1"](4, T, 16, np.random.default_rng(1))
    try:
        qq = fc(yy[:, :T], 16, 20); log(f"context {T}: ok, finite {np.isfinite(qq).all()}")
    except Exception as e:
        log(f"context {T}: FAILED ({type(e).__name__}: {str(e)[:120]})")
y2, _ = RUNGS["N1"](128, 512, 128, np.random.default_rng(2))
t0 = time.time()
for i in range(0, 128, 32):
    fc(y2[i:i + 32, :512], 128, 100)
log(f"128 windows at H=128, S=100, batch 32: {time.time() - t0:.1f}s")
log("SMOKE DONE")
