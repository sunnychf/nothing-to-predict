"""Controlled architecture comparison: the same synthetic corpus (arch_corpus.py), the same blocks, the
same size, optimiser, steps and data stream; only the architecture differs.

    encoder  masked encoder (Moirai-style): the 32 context patches and 8 [MASK] tokens for the future,
             bidirectional attention, the mask positions decoded to the 8 future patches at once;
             loss on the future patches only.
    decoder  decoder-only (TimesFM / Time-MoE-style): causal attention over context + future patches,
             next-patch prediction at every position (teacher forcing); autoregressive at inference.

Both normalise the context by its own mean and standard deviation, embed 16-point patches, use 4
pre-LN transformer layers (d = 128, 4 heads, FFN 256) and a linear patch head, and are trained with MSE
in normalised units. After training the model is probed exactly as the released models are:
    direction  N1, seed 4000, 128 series, H = 128 (diag_direction's contexts): mean signed departure / sigma,
               fraction up, correlation with the recent slope, rms departure, and the even share from the
               mirrored contexts;
    ladder     N1 and N4, seeds 1000-1002, 512 series, H = 16: skill against persistence, N4 share of the
               exact optimum (nulls.oracle_forecast_exact) and of the linear one;
    corpus     held-out MSE on 4096 fresh corpus windows, so the two fits can be compared.

    python arch_train.py --arch encoder --corpus up --seed 0 [--steps 12000 --batch 256]
writes <out-dir>/arch_<arch>_<corpus>_s<seed>.json (and the checkpoint next to it).
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_corpus import make_corpus
from probe_metrics import make_data
from nulls import oracle_forecast, oracle_forecast_exact, n4_optima_skill

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["encoder", "decoder"])
ap.add_argument("--corpus", required=True, choices=["up", "sym"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--out-dir", default=os.environ.get("SPEC_OUT", "results"))
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
T, H, P = 512, 128, 16                                            # context, horizon, patch
NC, NF = T // P, H // P                                           # 32 context patches, 8 future patches
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(self, x, causal):
        h = self.ln1(x); L = x.shape[1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1) if causal else None
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class PatchModel(nn.Module):
    def __init__(self, arch, d, layers, heads=4):
        super().__init__()
        self.arch = arch
        self.embed = nn.Linear(P, d); self.pos = nn.Parameter(0.02 * torch.randn(NC + NF, d))
        self.mask_tok = nn.Parameter(0.02 * torch.randn(d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers)); self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, P)
    def run(self, tokens, causal):
        x = tokens + self.pos[: tokens.shape[1]]
        for b in self.blocks: x = b(x, causal)
        return self.head(self.ln(x))
    def forward_train(self, ctx_n, fut_n):
        """ctx_n (B, NC, P), fut_n (B, NF, P) in normalised units -> loss"""
        if self.arch == "encoder":
            toks = torch.cat([self.embed(ctx_n), self.mask_tok.expand(ctx_n.shape[0], NF, -1)], 1)
            pred = self.run(toks, causal=False)[:, NC:]
            return F.mse_loss(pred, fut_n)
        toks = self.embed(torch.cat([ctx_n, fut_n], 1))
        pred = self.run(toks, causal=True)[:, :-1]                     # position t predicts patch t+1
        return F.mse_loss(pred, torch.cat([ctx_n, fut_n], 1)[:, 1:])
    @torch.no_grad()
    def forecast(self, ctx):
        """ctx (B, T) raw -> (B, H) raw; instance normalisation by the context's mean and std."""
        mu, sd = ctx.mean(1, keepdim=True), ctx.std(1, keepdim=True).clamp_min(1e-6)
        c = ((ctx - mu) / sd).view(ctx.shape[0], NC, P)
        if self.arch == "encoder":
            toks = torch.cat([self.embed(c), self.mask_tok.expand(c.shape[0], NF, -1)], 1)
            out = self.run(toks, causal=False)[:, NC:]
        else:
            seq = c
            for _ in range(NF):
                nxt = self.run(self.embed(seq), causal=True)[:, -1:]
                seq = torch.cat([seq, nxt], 1)
            out = seq[:, NC:]
        return out.reshape(ctx.shape[0], H) * sd + mu


def batches(rng, corpus, n):
    x, _ = make_corpus(n, T + H, rng, corpus)
    return torch.tensor(x)

model = PatchModel(args.arch, args.d, args.layers).to(args.device)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = lambda step: min(1.0, step / 500) * (0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)) * 0.98 + 0.02)
rng = np.random.default_rng(10_000 + args.seed)                  # the corpus stream: identical for both architectures at a given seed
log(f"{args.arch} on corpus '{args.corpus}', seed {args.seed}: {n_params/1e6:.2f}M params, {args.steps} steps x batch {args.batch}")
t0 = time.time(); losses = []; chunk = None; ci = 0
for step in range(args.steps):
    if chunk is None or ci + args.batch > len(chunk):
        chunk = batches(rng, args.corpus, 64 * args.batch); ci = 0
    x = chunk[ci:ci + args.batch].to(args.device); ci += args.batch
    ctx, fut = x[:, :T], x[:, T:]
    mu, sd = ctx.mean(1, keepdim=True), ctx.std(1, keepdim=True).clamp_min(1e-6)
    ctx_n, fut_n = ((ctx - mu) / sd).view(-1, NC, P), ((fut - mu) / sd).view(-1, NF, P)
    for g in opt.param_groups: g["lr"] = args.lr * sched(step)
    loss = model.forward_train(ctx_n, fut_n); opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
    if step % 1000 == 0 or step == args.steps - 1: log(f"  step {step}: loss {np.mean(losses[-200:]):.4f} ({time.time()-t0:.0f}s)")
model.eval()

def fc(ctx_np, bs=256):
    out = []
    for i in range(0, len(ctx_np), bs):
        out.append(model.forecast(torch.tensor(ctx_np[i:i + bs], dtype=torch.float32, device=args.device)).cpu().numpy())
    return np.concatenate(out, 0)

res = {"arch": args.arch, "corpus": args.corpus, "seed": args.seed, "params": n_params, "steps": args.steps, "batch": args.batch, "d": args.d, "layers": args.layers,
       "train_loss_last1000": float(np.mean(losses[-1000:])), "seconds_train": round(time.time() - t0)}
# corpus held-out fit, normalised units, forecasting the future from the context as at inference
xh, kinds = make_corpus(4096, T + H, np.random.default_rng(99_000), args.corpus)
yh = fc(xh[:, :T]); mu, sd = xh[:, :T].mean(1, keepdims=True), xh[:, :T].std(1, keepdims=True)
res["corpus_heldout_mse_norm"] = float(np.mean(((yh - xh[:, T:]) / sd) ** 2))
res["corpus_heldout_mse_norm_by_kind"] = {k: float(np.mean(((yh[kinds == k] - xh[kinds == k, T:]) / sd[kinds == k]) ** 2)) for k in np.unique(kinds)}
res["corpus_heldout_persistence_mse_norm"] = float(np.mean(((xh[:, T - 1:T] - xh[:, T:]) / sd) ** 2))
# direction probe: N1, seed 4000, 128 series, H = 128, as diag_direction / diag_scale
rng_d = np.random.default_rng(4000)
from nulls import RUNGS
y, info = RUNGS["N1"](128, T, H, rng_d); ctx, truth = y[:, :T], y[:, T:]; sig = info["sigma"]
yhat = fc(ctx); dep = (yhat - ctx[:, -1:]) / sig
mir = 2 * ctx[:, :1] - ctx; dep_m = (fc(mir) - mir[:, -1:]) / sig
slope64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
even = 0.5 * (dep + dep_m)
res["direction"] = {str(h): {"mean_dep": float(dep[:, h - 1].mean()), "mean_dep_se": float(dep[:, h - 1].std(ddof=1) / np.sqrt(len(dep))),
                             "frac_up": float((dep[:, h - 1] > 0).mean()), "rms_dep": float(np.sqrt((dep[:, h - 1] ** 2).mean())),
                             "corr_slope64": float(np.corrcoef(dep[:, h - 1], slope64)[0, 1]),
                             "even_share": float((even[:, h - 1] ** 2).mean() / (dep[:, h - 1] ** 2).mean()),
                             "skill": float(1 - ((yhat[:, h - 1] - truth[:, h - 1]) ** 2).mean() / ((ctx[:, -1] - truth[:, h - 1]) ** 2).mean())} for h in (1, 16, 64, 128)}
# ladder probe: N1 and N4, seeds 1000-1002, 512 series, H = 16 (the forecast's first 16 steps of the 128)
res["ladder"] = {}
for rung in ("N1", "N4"):
    acc = {"skill_1": [], "skill_16": [], "mean_dep_16": [], "share_exact_1": [], "share_linear_1": []}
    for sd_ in (0, 1, 2):
        yc, yt, inf = make_data(rung, 512, T, 16, seed=1000 + sd_)
        yh_ = fc(yc)[:, :16]; last = yc[:, -1:]
        ep = ((yt - last) ** 2).mean(0); em = ((yt - yh_) ** 2).mean(0)
        acc["skill_1"].append(float(1 - em[0] / ep[0])); acc["skill_16"].append(float(1 - em[15] / ep[15])); acc["mean_dep_16"].append(float(((yh_ - last)[:, 15] / inf["sigma"]).mean()))
        if rung == "N4":
            eo = ((yt - oracle_forecast_exact("N4", yc, inf, 16)) ** 2).mean(0); el = ((yt - oracle_forecast("N4", yc, inf, 16)) ** 2).mean(0)
            acc["share_exact_1"].append(float((ep[0] - em[0]) / (ep[0] - eo[0]))); acc["share_linear_1"].append(float((ep[0] - em[0]) / (ep[0] - el[0])))
    res["ladder"][rung] = {k: float(np.mean(v)) for k, v in acc.items() if v}
os.makedirs(args.out_dir, exist_ok=True)
tag = f"arch_{args.arch}_{args.corpus}_s{args.seed}"
torch.save(model.state_dict(), os.path.join(args.out_dir, tag + ".pt"))
json.dump(res, open(os.path.join(args.out_dir, tag + ".json"), "w"), indent=1)
d = res["direction"]
log(f"{tag}: heldout mse {res['corpus_heldout_mse_norm']:.3f} (persistence {res['corpus_heldout_persistence_mse_norm']:.3f}) | N1 dir h=16 {d['16']['mean_dep']:+.2f}±{d['16']['mean_dep_se']:.2f} up {d['16']['frac_up']:.2f} | h=128 {d['128']['mean_dep']:+.2f}±{d['128']['mean_dep_se']:.2f} up {d['128']['frac_up']:.2f} even {d['128']['even_share']:.2f} corr {d['128']['corr_slope64']:+.2f} | ladder N1 skill16 {res['ladder']['N1']['skill_16']:+.3f} N4 share {res['ladder']['N4']['share_exact_1']:+.2f}")
