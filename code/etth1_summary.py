"""Metrics for the ETTh1 positive control (etth1_probe.py): skill against persistence at h = 1, 16, 64, 128,
pooled over windows in units of each context's one-step sigma, with bootstrap standard errors over
windows; the seasonal-naive benchmark (the value one day, 24 steps, before the target); the mirror-
corrected forecast and its even share, which show what the correction of Section 5 does to a
forecast that carries real structure.
    python etth1_summary.py --res results --tables paper/tables [--boot 2000]
Writes results/etth1_summary.json and paper/tables/etth1.tex.
"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); ap.add_argument("--tables", default="paper/tables")
ap.add_argument("--models", default="chronos,chronosbolt,chronos2,tirex,moirai,moirai2,timesfm,timesfm25,timemoe,sundial,fincast")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=19); args = ap.parse_args()
rng = np.random.default_rng(args.seed)
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
HS = (1, 16, 64, 128)
W = np.load(os.path.join(args.res, "etth1_windows.npz")); meta = json.load(open(os.path.join(args.res, "etth1_windows_meta.json")))
ctx, fut = W["ctx"], W["fut"]; n, T = ctx.shape; H = fut.shape[1]
sig = np.std(np.diff(ctx, axis=1), axis=1); last = ctx[:, -1:]
err_p = (fut - last) ** 2 / sig[:, None] ** 2
def boot(stat, idx):
    return float(np.std([stat(rng.choice(idx, size=len(idx), replace=True)) for _ in range(args.boot)]))
def summarise(yh, idx=None):
    idx = np.arange(n) if idx is None else idx
    em = (fut - yh) ** 2 / sig[:, None] ** 2; rec = {}
    for h in HS:
        j = h - 1; sk = lambda s: float(1 - em[s, j].mean() / err_p[s, j].mean())
        rec[str(h)] = {"skill": sk(idx), "skill_se": boot(sk, idx)}
    return rec
# seasonal naive: the value 24 steps before the target (one day), taken from the context or the already-observed future? Only the
# context is known at forecast time, so for step h use the value at position T + h - 24 * ceil(h / 24) of the window (always in the context).
win = np.concatenate([ctx, fut], axis=1); seas = np.stack([win[:, T + h - 1 - 24 * int(np.ceil(h / 24))] for h in range(1, H + 1)], axis=1)
summary = {"n_windows": int(n), "T": int(T), "H": int(H), "columns": sorted(set(m["column"] for m in meta)),
           "benchmarks": {"persistence": {str(h): {"skill": 0.0} for h in HS}, "seasonal_naive_24": summarise(seas)}, "models": {}}
print(f"{n} windows; seasonal naive skill: " + " / ".join(f"{summary['benchmarks']['seasonal_naive_24'][str(h)]['skill']:+.3f}" for h in HS))
for m in args.models.split(","):
    p = os.path.join(args.res, f"etth1_{m}.npz")
    if not os.path.exists(p): print("missing", p); continue
    r = np.load(p); yh, ym = r["yhat_raw"].astype(np.float64), r["yhat_mirror"].astype(np.float64)
    mir_last = (2 * ctx[:, :1] - ctx)[:, -1:]
    d_raw, d_mir = yh - last, ym - mir_last; d_even = 0.5 * (d_raw + d_mir); yc = yh - d_even
    S = {"raw": summarise(yh), "mirror": summarise(yc), "even_share": {str(h): float(np.mean((d_even[:, h - 1] / sig) ** 2) / np.mean((d_raw[:, h - 1] / sig) ** 2)) for h in HS}}
    summary["models"][m] = S
    print(f"{LABEL[m]:12s} skill " + " / ".join(f"{S['raw'][str(h)]['skill']:+.3f}" for h in HS) + " | mirror " + " / ".join(f"{S['mirror'][str(h)]['skill']:+.3f}" for h in HS) + f" | even share h=16 {S['even_share']['16']:.2f}")
M = summary["models"]
summary["counts"] = {"beat_persistence_all_h": [m for m in M if all(M[m]["raw"][str(h)]["skill"] > 0 for h in HS)],
                     "beat_seasonal_naive_h16": [m for m in M if M[m]["raw"]["16"]["skill"] > summary["benchmarks"]["seasonal_naive_24"]["16"]["skill"]],
                     "mirror_within_0.01_h16": [m for m in M if abs(M[m]["mirror"]["16"]["skill"] - M[m]["raw"]["16"]["skill"]) < 0.01]}
print("counts:", {k: len(v) for k, v in summary["counts"].items()})
os.makedirs(args.tables, exist_ok=True); json.dump(summary, open(os.path.join(args.res, "etth1_summary.json"), "w"), indent=1)
def pm(v, se): return f"${v:+.3f}\\pm{se:.3f}$"
L = [r"\begin{tabular}{l|cccc|cc}", r"\toprule", r" & \multicolumn{4}{c|}{skill against persistence, raw forecast} & \multicolumn{2}{c}{mirror-corrected} \\",
     r"Forecast & $h{=}1$ & $h{=}16$ & $h{=}64$ & $h{=}128$ & skill $h{=}16$ & even share \\", r"\midrule", r"persistence & $0$ & $0$ & $0$ & $0$ & & \\",
     "seasonal naive (one day back) & " + " & ".join(pm(summary["benchmarks"]["seasonal_naive_24"][str(h)]["skill"], summary["benchmarks"]["seasonal_naive_24"][str(h)]["skill_se"]) for h in HS) + r" & & \\", r"\midrule"]
for m in args.models.split(","):
    S = M.get(m)
    if not S: continue
    L.append(f"{LABEL[m]} & " + " & ".join(pm(S["raw"][str(h)]["skill"], S["raw"][str(h)]["skill_se"]) for h in HS) + f" & ${S['mirror']['16']['skill']:+.3f}$ & {S['even_share']['16']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "etth1.tex"), "w").write("\n".join(L) + "\n"); print("wrote etth1_summary.json, etth1.tex")
