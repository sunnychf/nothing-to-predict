"""Experiment 2: fine-tune the released Chronos-Bolt-small on public Chronos training data, plainly or
with mirror augmentation, from the same initial weights on exactly the same windows (the only
difference between the arms is the mirror coin, drawn from its own generator).

  python ft_bolt.py --arm plain  --out runs/plain_s0
  python ft_bolt.py --arm mirror --out runs/mirror_s0
Data: ~/tsfm_rev/data/chronos_datasets/<subset>/*.parquet (autogluon/chronos_datasets from the Hugging Face Hub);
every list-valued numeric column is one series; the finance subset exchange_rate is excluded.
Windows: dataset uniform, series uniform, end point uniform; context up to 512 (left-padded with NaN),
target 64 (Bolt's native prediction length). Mirror: the whole window (context + target) is negated
about its first observed value, w -> 2 w_0 - w, with probability 1/2.
Loss: Bolt's own quantile loss (model(context, target).loss). AdamW, linear warm-up then cosine decay.
"""
import argparse, glob, json, os, time
import numpy as np, torch, pyarrow.parquet as pq
from chronos import BaseChronosPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["plain", "mirror"], required=True)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--max-per-dataset", type=int, default=20000)
ap.add_argument("--data", default=os.path.expanduser("~/tsfm_rev/data/chronos_datasets"))
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
CTX, PRED, MINCTX = 512, 64, 64
os.makedirs(args.out, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

pool, sizes = {}, {}
g_load = np.random.default_rng(12345)
for d in sorted(os.listdir(args.data)):
    if d in ("exchange_rate",):
        continue
    arrs = []
    for f in sorted(glob.glob(os.path.join(args.data, d, "*.parquet"))):
        t = pq.read_table(f)
        for c in t.schema.names:
            ty = str(t.schema.field(c).type)
            if c in ("id", "timestamp") or not ty.startswith("list<") or "timestamp" in ty:
                continue
            for v in t.column(c).to_pylist():
                a = np.array([np.nan if x is None else x for x in v], dtype=np.float64)
                a = a[np.isfinite(a)]
                if len(a) >= MINCTX + PRED and np.std(a) > 0:
                    arrs.append(a.astype(np.float32))
    if not arrs:
        continue
    if len(arrs) > args.max_per_dataset:
        idx = g_load.choice(len(arrs), size=args.max_per_dataset, replace=False); arrs = [arrs[i] for i in idx]
    pool[d] = arrs; sizes[d] = len(arrs)
names = sorted(pool)
log(f"{len(names)} datasets, {sum(sizes.values())} series: {sizes}")

g_win = np.random.default_rng(args.seed)                     # windows: identical across arms
g_flip = np.random.default_rng(args.seed + 1000)             # mirror coin: used only by the mirror arm


def batch():
    C = np.full((args.batch, CTX), np.nan, np.float32); Y = np.zeros((args.batch, PRED), np.float32); flips = 0
    b = 0
    while b < args.batch:
        a = pool[names[g_win.integers(len(names))]]; s = a[g_win.integers(len(a))]
        e = int(g_win.integers(MINCTX + PRED, len(s) + 1)); st = max(0, e - PRED - CTX)
        w = s[st:e].astype(np.float64)
        coin = g_flip.random() < 0.5
        if args.arm == "mirror" and coin:
            w = 2.0 * w[0] - w
        ctx, tgt = w[:-PRED], w[-PRED:]
        # reject windows the mean-absolute scaling cannot handle (flat or all-zero contexts, e.g. solar at
        # night); the test uses the window as it would be used under either arm, so both arms reject the same
        # windows whether or not they are mirrored
        ok = True
        for cand in (w, 2.0 * w[0] - w):
            cc, tt = cand[:-PRED], cand[-PRED:]; sc = np.mean(np.abs(cc))
            if not (np.std(cc) > 0 and sc > 0 and np.max(np.abs(tt)) / sc < 50.0):
                ok = False
        if not ok:
            continue
        if args.arm == "mirror" and coin:
            flips += 1
        C[b, CTX - len(ctx):] = ctx; Y[b] = tgt; b += 1
    return torch.tensor(C), torch.tensor(Y), flips


pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
model = pipe.model; model.train()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / args.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, k / args.steps))))
hist = []; t0 = time.time(); nflip = 0
for step in range(args.steps):
    C, Y, f = batch(); nflip += f
    loss = model(context=C, target=Y).loss
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    hist.append(float(loss))
    if step % 50 == 0 or step == args.steps - 1:
        log(f"step {step} loss {np.mean(hist[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e} flips {nflip} ({time.time() - t0:.0f}s)")
model.eval()
torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
json.dump({"args": vars(args), "datasets": sizes, "loss": hist, "flips": nflip, "seconds": time.time() - t0},
          open(os.path.join(args.out, "train.json"), "w"))
log("done")
