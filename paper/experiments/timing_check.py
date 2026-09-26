"""Post hoc check (not registered), prompted by the falling-market test: do the models time the market?
For each end date, average the h = 128 forecast return over the windows ending then, and correlate these date means
with the realised date means. Three forecasts per model: on the raw context, on its multiplicative mirror (the
correction's second pass) and on its sign-randomised copy (same magnitudes, no direction), all scored against the
realised return of the raw window. A forecast that recalls the series times the market only from the raw context.
Windows: the falling-market windows (1996-2014, 50 equity series, 37 end dates) and the equities of the paper's daily
anchor (2017-2026, 18 end dates). Also records, for 2008-11-21, the forecasts for the portfolios that fell most, and the
timing of rules that read only the context (fitted drift, momentum over 20 to 511 days, each followed or reversed).
Writes experiments/results/timing_check.json and tables/timing.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/timing_check.py"""
import json, os
import numpy as np

MODELS = ["chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast"]
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
rng = np.random.default_rng(0)
SETS = {"falling": ("experiments/results/falling_windows.npz", "experiments/results/falling_windows_meta.json", "experiments/results/falling", "falling", None),
        "anchor_eq": ("../results/real_windows.npz", "../results/real_windows_meta.json", "../results", "real", "eq")}


def perm_p(x, y, B=20000):
    c = np.corrcoef(x, y)[0, 1]; yc = y - y.mean(); xc = x - x.mean()
    ys = np.array([rng.permutation(yc) for _ in range(B)])
    return float(np.mean(ys @ xc / (np.linalg.norm(xc) * np.linalg.norm(yc)) >= c))


out = {}
for tag, (wp, mp, res, prefix, fam) in SETS.items():
    W = np.load(wp); meta = json.load(open(mp))
    ctx, fut = W["ctx_raw"].astype(np.float64), W["fut_raw"].astype(np.float64); last = ctx[:, -1]
    cs = W["ctx_sur"].astype(np.float64); cs = cs if cs.ndim == 3 else cs[:, None]
    ends = np.array([m["fut_end"] for m in meta]); fams = np.array([m.get("family", "eq") for m in meta]); assets = np.array([m["asset"] for m in meta])
    sel = np.ones(len(ctx), bool) if fam is None else fams == fam
    Y = fut[:, -1] / last - 1.0; D = sorted(set(ends[sel])); ym = np.array([Y[sel & (ends == d)].mean() for d in D])
    r = ctx[:, 1:] / ctx[:, :-1] - 1.0; lm = ctx[:, 0] * np.cumprod(1.0 - r, axis=1)[:, -1]
    out[tag] = {"n_windows": int(sel.sum()), "n_dates": len(D), "models": {}, "context_rules": {}}
    rules = {"Fitted drift": (1.0 + r.mean(axis=1)) ** 128 - 1.0}
    for m_ in (20, 60, 128, 250, 511):
        rules[f"Momentum {m_}"] = ctx[:, -1] / ctx[:, -1 - m_] - 1.0
    for name, F in rules.items():
        out[tag]["context_rules"][name] = float(np.corrcoef([F[sel & (ends == d)].mean() for d in D], ym)[0, 1])
    kmax = max(out[tag]["context_rules"], key=lambda k: abs(out[tag]["context_rules"][k]))
    out[tag]["context_rule_max_abs"] = {"rule": kmax, "sign": "followed" if out[tag]["context_rules"][kmax] > 0 else "reversed",
                                        "abs_corr": abs(out[tag]["context_rules"][kmax])}
    for m in MODELS:
        p, pm = os.path.join(res, f"{prefix}_{m}.npz"), os.path.join(res, f"{prefix}_{m}_mirror.npz")
        if not (os.path.exists(p) and os.path.exists(pm)):
            continue
        z = np.load(p); F = z["yhat_raw"][:, -1] / last - 1.0
        ys = z["yhat_sur"]; ys = ys[:, 0] if ys.ndim == 3 else ys; Fs = ys[:, -1] / cs[:, 0, -1] - 1.0
        Fm = np.load(pm)["yhat_mirror"][:, -1] / lm - 1.0
        dm = lambda v: np.array([v[sel & (ends == d)].mean() for d in D])
        fr, fmr, fsr = dm(F), dm(Fm), dm(Fs)
        R = {"raw": float(np.corrcoef(fr, ym)[0, 1]), "raw_perm_p": perm_p(fr, ym), "raw_slope": float(np.polyfit(ym, fr, 1)[0]),
             "mirror": float(np.corrcoef(fmr, ym)[0, 1]), "sign_randomised": float(np.corrcoef(fsr, ym)[0, 1])}
        if tag == "falling":
            keep = np.array([d != "2008-11-21" for d in D])
            R["raw_without_2008_11"] = float(np.corrcoef(fr[keep], ym[keep])[0, 1])
            i = np.where(ends == "2008-11-21")[0]; worst = i[np.argsort(Y[i])[:5]]
            R["2008_11_worst5"] = [{"asset": str(assets[j]), "forecast": float(F[j]), "realised": float(Y[j])} for j in worst]
        out[tag]["models"][m] = R
json.dump(out, open(os.path.join("experiments", "results", "timing_check.json"), "w"), indent=1)


def num(v, d=2):
    return f"{v:+.{d}f}".replace("-", "\\textminus{}")


L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{\textbf{1996--2014, 37 end dates}} & \multicolumn{3}{c}{\textbf{2017--2026 equities, 18 end dates}} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"\textbf{Model} & \textbf{Raw} & \textbf{Mirror} & \textbf{Randomised} & \textbf{Raw} & \textbf{Mirror} & \textbf{Randomised} \\", r"\midrule"]
for m in MODELS:
    a = out["falling"]["models"].get(m); b = out["anchor_eq"]["models"].get(m)
    cells = [LABEL[m]]
    for R in (a, b):
        if R is None:
            cells += ["--"] * 3
        else:
            raw = num(R["raw"]) + ("\\textsuperscript{*}" if R["raw_perm_p"] < 0.01 else "")
            cells += [raw, num(R["mirror"]), num(R["sign_randomised"])]
    L.append(" & ".join(cells) + r" \\")
L.append(r"\addlinespace[3pt]")
L.append(" & ".join([r"Context rule, largest $|r|$", f"{out['falling']['context_rule_max_abs']['abs_corr']:.2f}", "--", "--",
                     f"{out['anchor_eq']['context_rule_max_abs']['abs_corr']:.2f}", "--", "--"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
import table_bold   # bold marks, by the rule stated in the table note
L = table_bold.apply("timing", L)
open(os.path.join("tables", "timing.tex"), "w").write("\n".join(L) + "\n")
for tag, o in out.items():
    print(tag, o["n_windows"], o["n_dates"], "context rules:", {k: round(v, 2) for k, v in o["context_rules"].items()}, "largest |r|:", o["context_rule_max_abs"])
    for m, R in o["models"].items():
        print(f"   {LABEL[m]:12s} raw {R['raw']:+.2f} (p {R['raw_perm_p']:.4f}, slope {R['raw_slope']:+.2f}) mirror {R['mirror']:+.2f} randomised {R['sign_randomised']:+.2f}"
              + (f" | without 2008-11 {R['raw_without_2008_11']:+.2f}" if "raw_without_2008_11" in R else ""))
