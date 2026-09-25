"""Experiment 1 analysis (pre-registered in experiments/PREREGISTRATION.md): skill and mirror-correction
changes on pre-2015 equity windows, split by the market's direction over each window's future.
Units and pooling as in the paper's daily anchor: return units relative to the last price, pooled
squared errors, h = 128 (and 16); cluster bootstrap over future-end dates (all 50 series share them).
Reads experiments/results/falling_windows.npz and <res>/falling_<model>[_mirror].npz; writes
experiments/results/falling_summary.json.
Run from finance/paper_revision:
  ~/miniconda3/envs/nature-figure/bin/python experiments/falling_summary.py --res experiments/results/falling
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--res", default=os.path.join("experiments", "results", "falling"))
ap.add_argument("--boot", type=int, default=2000); args = ap.parse_args()
W = np.load(os.path.join("experiments", "results", "falling_windows.npz")); meta = json.load(open(os.path.join("experiments", "results", "falling_windows_meta.json")))
ctx, fut = W["ctx_raw"], W["fut_raw"]; cs, fs = W["ctx_sur"], W["fut_sur"]
last = ctx[:, -1:]; fr = fut / last - 1.0; ls = cs[:, -1:]; frs = fs / ls - 1.0
ends = np.array([m["fut_end"] for m in meta]); mret = np.array([m["mkt_fut_ret"] for m in meta]); bear = np.array([m["in_bear"] for m in meta])
cl_ids = {d: i for i, d in enumerate(sorted(set(ends)))}; cl = np.array([cl_ids[d] for d in ends])
SPLITS = {"all": np.ones(len(ctx), bool), "down": mret < 0, "up": mret >= 0, "bear": bear, "not_bear": ~bear}
GROUP = {"upward": ["chronos", "chronosbolt", "chronos2", "tirex", "moirai"], "balanced": ["moirai2", "timesfm", "timesfm25", "timemoe", "sundial"], "other": ["fincast"]}
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
rng = np.random.default_rng(29)
def mirror_last(c):
    r = c[:, 1:] / c[:, :-1] - 1.0; return (c[:, :1] * np.cumprod(1.0 - r, axis=1))[:, -1:]
lm = mirror_last(ctx)


def boot_se(stat, idx):
    global rng
    gs = np.unique(cl[idx]); by = {g: idx[cl[idx] == g] for g in gs}; out = []
    for _ in range(args.boot):
        pick = rng.choice(gs, size=len(gs), replace=True); out.append(stat(np.concatenate([by[g] for g in pick])))
    return float(np.std(out))


out = {"n_windows": int(len(ctx)), "n_end_dates": int(len(cl_ids)), "splits": {k: int(v.sum()) for k, v in SPLITS.items()},
       "realised_mean_return_h128": {k: float(fr[v, 127].mean()) for k, v in SPLITS.items()}, "models": {}}
for m in LABEL:
    p = os.path.join(args.res, f"falling_{m}.npz"); pm = os.path.join(args.res, f"falling_{m}_mirror.npz")
    if not (os.path.exists(p) and os.path.exists(pm)):
        print("missing", m); continue
    rng = np.random.default_rng([29, list(LABEL).index(m)])   # per-model stream: SEs do not depend on which other models are present
    z = np.load(p); zm = np.load(pm)
    rel = z["yhat_raw"].astype(np.float64) / last - 1.0; rel_m = zm["yhat_mirror"].astype(np.float64) / lm - 1.0
    even = 0.5 * (rel + rel_m); rel_c = rel - even
    ys = z["yhat_sur"].astype(np.float64); ys = ys[:, 0] if ys.ndim == 3 else ys; rs = ys / ls - 1.0
    M = {}
    for sp, v in SPLITS.items():
        idx = np.where(v)[0]; R = {}
        for h in (16, 128):
            j = h - 1; P = lambda s: (fr[s, j] ** 2).mean()
            sk = lambda s, rr=rel: float(1 - ((fr[s, j] - rr[s, j]) ** 2).mean() / P(s))
            skc = lambda s: float(1 - ((fr[s, j] - rel_c[s, j]) ** 2).mean() / P(s))
            dsk = lambda s: skc(s) - sk(s)
            gain = lambda s: float((even[s, j] ** 2).mean() / P(s)); align = lambda s: float(-2 * (fr[s, j] * even[s, j]).mean() / P(s))
            cross = lambda s: float(2 * ((rel[s, j] - even[s, j]) * even[s, j]).mean() / P(s))
            sks = lambda s: float(1 - ((frs[s, j] - rs[s, j]) ** 2).mean() / (frs[s, j] ** 2).mean())
            R[str(h)] = {"skill": sk(idx), "skill_se": boot_se(sk, idx), "skill_mirror": skc(idx), "dskill": dsk(idx), "dskill_se": boot_se(dsk, idx),
                         "removed_energy": gain(idx), "alignment": align(idx), "cross": cross(idx),
                         "even_mean_annualised_pct": float(100 * (252 / 128) * even[idx, j].mean()) if h == 128 else None,
                         "odd_mean_annualised_pct": float(100 * (252 / 128) * (rel - even)[idx, j].mean()) if h == 128 else None,
                         "corr_even_future": float(np.corrcoef(even[idx, j], fr[idx, j])[0, 1]),
                         "frac_up": float((rel[idx, j] > 0).mean()), "null_skill": sks(idx), "null_frac_up": float((rs[idx, j] > 0).mean())}
        M[sp] = R
    out["models"][m] = M
    d, u = M["down"]["128"], M["up"]["128"]
    print(f"{LABEL[m]:12s} h=128 raw skill up {u['skill']:+.3f}±{u['skill_se']:.3f} down {d['skill']:+.3f}±{d['skill_se']:.3f} | mirror change up {u['dskill']:+.3f}±{u['dskill_se']:.3f} down {d['dskill']:+.3f}±{d['dskill_se']:.3f}"
          f" | align up {u['alignment']:+.3f} down {d['alignment']:+.3f} | even%/yr {M['all']['128']['even_mean_annualised_pct']:+.1f} | null skill {M['all']['128']['null_skill']:+.3f} up {M['all']['128']['null_frac_up']:.2f}")
# pre-registered checks
chk = {}
for m, M in out["models"].items():
    d, u = M["down"]["128"], M["up"]["128"]
    chk[m] = {"P1_skill_lower_down": d["skill"] < u["skill"], "P2_mirror_hurts_up_helps_down": (u["dskill"] < 0) and (d["dskill"] > 0),
              "gap": u["skill"] - d["skill"], "P4_null_loses_both": (M["down"]["128"]["null_skill"] < 0) and (M["up"]["128"]["null_skill"] < 0)}
gaps = {g: [chk[m]["gap"] for m in ms if m in chk] for g, ms in GROUP.items()}
out["preregistered"] = {"per_model": chk, "mean_gap": {g: (float(np.mean(v)) if v else None) for g, v in gaps.items()},
                        "P3_gap_larger_upward": (np.mean(gaps["upward"]) > np.mean(gaps["balanced"])) if gaps["upward"] and gaps["balanced"] else None}
# post hoc (not registered): what the contexts before falls looked like
lr_ctx = np.log(ctx[:, 1:] / ctx[:, :-1])
out["context_post_hoc"] = {sp: {"log_trend_512": float(lr_ctx[v].sum(1).mean()), "log_trend_last_60": float(lr_ctx[v][:, -60:].sum(1).mean()),
                                "daily_return_sd": float(lr_ctx[v].std(1).mean())} for sp, v in SPLITS.items()}
json.dump(out, open(os.path.join("experiments", "results", "falling_summary.json"), "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)


def num(v, d=3, signed=True):
    return (f"{v:+.{d}f}" if signed else f"{v:.{d}f}").replace("-", "\\textminus{}")


L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{\textbf{Raw skill}} & \multicolumn{2}{c}{\textbf{Mirror change}} & \multicolumn{2}{c}{\textbf{Even part (\%/yr)}} & \textbf{Null} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"\textbf{Model} & \textbf{Up} & \textbf{Down} & \textbf{Up} & \textbf{Down} & \textbf{Up} & \textbf{Down} & \textbf{skill} \\", r"\midrule"]
for g, ms in GROUP.items():
    rows = [m for m in ms if m in out["models"]]
    if not rows:
        continue
    for m in rows:
        u, d, a = out["models"][m]["up"]["128"], out["models"][m]["down"]["128"], out["models"][m]["all"]["128"]
        L.append(" & ".join([LABEL[m], num(u["skill"]) + f"\\,\\textpm\\,{u['skill_se']:.3f}", num(d["skill"]) + f"\\,\\textpm\\,{d['skill_se']:.3f}",
                             num(u["dskill"]) + f"\\,\\textpm\\,{u['dskill_se']:.3f}", num(d["dskill"]) + f"\\,\\textpm\\,{d['dskill_se']:.3f}",
                             num(u["even_mean_annualised_pct"], 1), num(d["even_mean_annualised_pct"], 1), num(a["null_skill"])]) + r" \\")
    if g != list(GROUP)[-1]:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join("tables", "falling.tex"), "w").write("\n".join(L) + "\n")
print("splits", out["splits"], "realised mean return h128", {k: round(v, 4) for k, v in out["realised_mean_return_h128"].items()})
print("pre-registered:", {m: (c["P1_skill_lower_down"], c["P2_mirror_hurts_up_helps_down"], round(c["gap"], 3), c["P4_null_loses_both"]) for m, c in chk.items()})
print("mean gap by group:", out["preregistered"]["mean_gap"], "P3:", out["preregistered"]["P3_gap_larger_upward"])
