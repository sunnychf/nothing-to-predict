"""Metrics for the real-data anchor, one code path for every model.

Reads results/real_windows.npz (+ _meta.json) and results/real_<model>.npz; writes
results/real_summary.json and paper/tables/real.tex. Nothing is typed by hand.

Per model, per input kind (sur = real-increment martingale surrogates, K per window,
pooled; raw = the real window) and per family (all / fx / eq), at h in {1, 16, 64, 128}:
  mean_dep      mean signed departure (yhat_h / y_T - 1) / sigma_r over windows, sigma_r the
                std of the context's simple returns (return units: the multiplicative
                surrogate keeps |returns|, so this is the unit in which it is exact)
  frac_up       fraction of windows with yhat_h > y_T
  rms_dep       sqrt(mean dep^2) with Var(samples)/S removed for sampled models
  skill         1 - MSE_model / MSE_persistence, pooled over windows with both errors in
                return units ((y - yhat) / y_T)^2, so that a window's weight in the pool does not
                depend on the scale of its level (IDR at 17000 against EURUSD at 1.1)
  realised      mean (y_{T+h} - y_T) / sigma of the data itself (the drift the models meet)
  corr_slope64  correlation of dep_h with the last 64-step slope of the context
Uncertainty: standard errors from a cluster bootstrap over calendar windows
(clusters = family x fut_end), because the equity portfolios share every calendar
window; for the surrogate the sign draws are independent per window, but the same
clustering is used so the two columns are comparable.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--tables", default="paper/tables")
ap.add_argument("--models", default="chronos,moirai,timesfm,timemoe,fincast,chronosbolt,chronos2,tirex,moirai2,timesfm25,sundial")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
HS = [1, 16, 64, 128]
LABEL = {"chronos": "Chronos-small", "moirai": "Moirai-small", "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "fincast": "FinCast",
         "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "sundial": "Sundial"}

z = np.load(os.path.join(args.res, "real_windows.npz"))
meta = json.load(open(os.path.join(args.res, "real_windows_meta.json")))
K = z["ctx_sur"].shape[1] if z["ctx_sur"].ndim == 3 else 1
def flat(a):
    """(n, K, ...) -> (n*K, ...); (n, ...) with K == 1 unchanged."""
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:]) if K > 1 else a
fam = np.array([m["family"] for m in meta]); fut_end = np.array([m["fut_end"] for m in meta])
post = np.array([m["fut_end"] >= "2025-01-01" for m in meta])
clusters = np.array([f"{a}|{b}" for a, b in zip(fam, fut_end)]); cl_ids = {c: i for i, c in enumerate(sorted(set(clusters)))}
cl = np.array([cl_ids[c] for c in clusters])
MASKS = {"all": np.ones(len(meta), bool), "fx": fam == "fx", "eq": fam == "eq", "post2025": post}
rng = np.random.default_rng(7)

def boot_se(values, mask, stat, cl=cl):
    """Cluster bootstrap SE of stat(values[mask]) with clusters resampled with replacement."""
    idx = np.where(mask)[0]; cls = np.unique(cl[idx]); out = []
    by = {c: idx[cl[idx] == c] for c in cls}
    for _ in range(args.boot):
        pick = rng.choice(cls, size=len(cls), replace=True)
        sel = np.concatenate([by[c] for c in pick]); out.append(stat(sel))
    return float(np.std(out))

summary = {"n_windows": int(len(meta)), "K": int(K), "n_clusters": int(len(cl_ids)), "n_post2025": int(post.sum()), "families": {f: int((fam == f).sum()) for f in ("fx", "eq")}, "models": {}}
for m in args.models.split(","):
    path = os.path.join(args.res, f"real_{m}.npz")
    if not os.path.exists(path):
        print("missing", path); continue
    r = np.load(path); summary["models"][m] = {}
    for kind in ("sur", "raw"):
        ctx, fut, sig = z[f"ctx_{kind}"], z[f"fut_{kind}"], z[f"rsigma_{kind}"]
        yh, mv = r[f"yhat_{kind}"].astype(np.float64), r[f"mcvar_{kind}"].astype(np.float64)
        if kind == "sur":
            ctx, fut, sig, yh, mv = flat(ctx), flat(fut), sig.reshape(-1), flat(yh), flat(mv)
            rep_ = K
        else:
            rep_ = 1
        cl_k = np.repeat(cl, rep_)
        last = ctx[:, -1:]
        dep = (yh / last - 1.0) / sig[:, None]; real = (fut / last - 1.0) / sig[:, None]
        mv = mv / last ** 2
        err_m = ((fut - yh) / last) ** 2; err_p = ((fut - last) / last) ** 2
        slope = (ctx[:, -1] / ctx[:, -65] - 1.0) / 64.0
        for fname, mask0 in MASKS.items():
            mask = np.repeat(mask0, rep_)
            rec = {}
            for h in HS:
                j = h - 1
                d = dep[:, j]
                mean_stat = lambda sel, j=j: float(np.mean(dep[sel, j]))
                skill_stat = lambda sel, j=j: float(1 - err_m[sel, j].mean() / err_p[sel, j].mean())
                rec[str(h)] = {
                    "mean_dep": float(d[mask].mean()), "mean_dep_se": boot_se(d, mask, mean_stat, cl_k),
                    "frac_up": float((d[mask] > 0).mean()),
                    "rms_dep": float(np.sqrt(max(np.mean(d[mask] ** 2) - np.mean(mv[mask, j] / sig[mask] ** 2), 0.0))),
                    "skill": skill_stat(np.where(mask)[0]), "skill_se": boot_se(d, mask, skill_stat, cl_k),
                    "realised": float(real[mask, j].mean()),
                    "corr_slope64": float(np.corrcoef(d[mask], slope[mask])[0, 1]),
                }
            summary["models"][m][f"{kind}|{fname}"] = rec
    a = summary["models"][m]
    print(f"{LABEL[m]:14s} sur/all h128 mean {a['sur|all']['128']['mean_dep']:+.2f}±{a['sur|all']['128']['mean_dep_se']:.2f} up {a['sur|all']['128']['frac_up']:.2f} skill16 {a['sur|all']['16']['skill']:+.3f} | "
          f"raw/all h128 mean {a['raw|all']['128']['mean_dep']:+.2f} up {a['raw|all']['128']['frac_up']:.2f} skill16 {a['raw|all']['16']['skill']:+.3f} skill128 {a['raw|all']['128']['skill']:+.3f} realised128 {a['raw|all']['128']['realised']:+.2f}")

json.dump(summary, open(os.path.join(args.res, "real_summary.json"), "w"), indent=1)

# ---------------------------------------------------------------- table body (language-neutral)
os.makedirs(args.tables, exist_ok=True)
def pm(v, se, d=2): return f"${v:+.{d}f}\\pm{se:.{d}f}$"
L = [r"\begin{tabular}{l|rrr|rrrr}", r"\toprule",
     r" & \multicolumn{3}{c|}{real-increment martingale} & \multicolumn{4}{c}{raw series} \\",
     r"Model & mean $h{=}128$ & up & skill $h{=}16$ & mean $h{=}128$ & up & skill $h{=}16$ & skill $h{=}128$ \\",
     r"\midrule"]
for m in args.models.split(","):
    if m not in summary["models"]: continue
    s_, w = summary["models"][m]["sur|all"], summary["models"][m]["raw|all"]
    L.append(f"{LABEL[m]} & {pm(s_['128']['mean_dep'], s_['128']['mean_dep_se'])} & {s_['128']['frac_up']:.2f} & {pm(s_['16']['skill'], s_['16']['skill_se'], 3)} & "
             f"{pm(w['128']['mean_dep'], w['128']['mean_dep_se'])} & {w['128']['frac_up']:.2f} & {pm(w['16']['skill'], w['16']['skill_se'], 3)} & {pm(w['128']['skill'], w['128']['skill_se'], 3)} \\\\")
L.append(r"\midrule")
any_m = next(iter(summary["models"].values()))
up_raw = float(np.mean(z["fut_raw"][:, -1] > z["ctx_raw"][:, -1])); up_sur = float(np.mean(z["fut_sur"][..., -1] > z["ctx_sur"][..., -1]))
L.append(f"the data itself & ${any_m['sur|all']['128']['realised']:+.2f}$ & {up_sur:.2f} & 0 & ${any_m['raw|all']['128']['realised']:+.2f}$ & {up_raw:.2f} & 0 & 0 \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "real.tex"), "w").write("\n".join(L) + "\n")
summary["realised"] = {"sur_h128": any_m["sur|all"]["128"]["realised"], "raw_h128": any_m["raw|all"]["128"]["realised"], "up_raw_h128": up_raw, "up_sur_h128": up_sur,
                       "raw_h16": any_m["raw|all"]["16"]["realised"]}
json.dump(summary, open(os.path.join(args.res, "real_summary.json"), "w"), indent=1)
# ---------------------------------------------------------------- appendix detail table: families and the post-cut-off split
D = [r"\begin{tabular}{ll|rr|rrr}", r"\toprule",
     r" & & \multicolumn{2}{c|}{real-increment martingale} & \multicolumn{3}{c}{raw series} \\",
     r"Model & windows & mean $h{=}128$ & skill $h{=}16$ & mean $h{=}128$ & skill $h{=}16$ & skill $h{=}128$ \\", r"\midrule"]
SPLIT = [("fx", f"FX ({int((fam == 'fx').sum())})"), ("eq", f"equity ({int((fam == 'eq').sum())})"), ("post2025", f"post-2025 ({int(post.sum())})")]
for m in args.models.split(","):
    if m not in summary["models"]: continue
    for i, (k, lab) in enumerate(SPLIT):
        s_, w = summary["models"][m][f"sur|{k}"], summary["models"][m][f"raw|{k}"]
        D.append(f"{LABEL[m] if i == 0 else ''} & {lab} & {pm(s_['128']['mean_dep'], s_['128']['mean_dep_se'])} & {pm(s_['16']['skill'], s_['16']['skill_se'], 3)} & "
                 f"{pm(w['128']['mean_dep'], w['128']['mean_dep_se'])} & {pm(w['16']['skill'], w['16']['skill_se'], 3)} & {pm(w['128']['skill'], w['128']['skill_se'], 3)} \\\\")
    D.append(r"\midrule")
D[-1] = r"\midrule"
rows_data = []
for k, lab in SPLIT:
    mask = {"fx": fam == "fx", "eq": fam == "eq", "post2025": post}[k]
    rows_data.append(f"{'the data' if not rows_data else ''} & {lab} & ${any_m['sur|' + k]['128']['realised']:+.2f}$ & 0 & ${any_m['raw|' + k]['128']['realised']:+.2f}$ & 0 & 0 \\\\")
D += rows_data + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "real_detail.tex"), "w").write("\n".join(D) + "\n")
print("wrote", os.path.join(args.res, "real_summary.json"), os.path.join(args.tables, "real.tex"), "and real_detail.tex")
