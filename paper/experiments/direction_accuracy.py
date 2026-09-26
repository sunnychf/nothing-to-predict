"""Part B of experiments/PREREGISTRATION_corpus_match.md: directional accuracy, the share of windows in which the forecast and
the realised h-step move from the last value have the same sign (ties count one half), from forecasts already made.
Daily anchor (results/real_windows.npz; 900 equity and 380 exchange-rate windows, four sign-randomised copies each):
  raw        the published forecasts of the raw windows (results/real_<model>.npz)
  mirror     the mirror-corrected forecast, raw minus the even part, as in code/remedy_summary.py (real_<model>_mirror.npz)
  randomised the forecasts of the four sign-randomised copies, each scored against its own future
Baselines: always up; the sign of the context's mean return (fitted drift). Standard errors: cluster bootstrap over
(family, end date), 2,000 draws. Also, without a prediction, the 1996-2014 windows (experiments/results/falling_*).
Writes experiments/results/direction_accuracy.json and tables/direction_accuracy.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/direction_accuracy.py"""
import json, os
import numpy as np

PUB, R = os.path.join("..", "results"), os.path.join("experiments", "results")
MODELS = ["chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast"]
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
UPWARD = ("chronos", "chronosbolt", "chronos2", "tirex", "moirai")        # the five models whose N1 forecasts lean upward
HS, B = (16, 128), 2000


def hit(f, y):
    """1 if same sign, 0 if opposite, 1/2 if either is zero."""
    return np.where((f == 0) | (y == 0), 0.5, (np.sign(f) == np.sign(y)).astype(float))


def boot(h, cl, seed):
    """mean of h and its cluster-bootstrap SE; h may be (n,) or (n, K) with the copies of a window in one cluster."""
    per = np.array([h[cl == c].mean() for c in np.unique(cl)]); w = np.array([np.sum(cl == c) for c in np.unique(cl)])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(per), size=(B, len(per)))
    bs = (per[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return {"da": float(h.mean()), "se": float(bs.std())}


def load_set(tag):
    if tag == "anchor":
        W = np.load(os.path.join(PUB, "real_windows.npz")); meta = json.load(open(os.path.join(PUB, "real_windows_meta.json")))
        fp = lambda m: os.path.join(PUB, f"real_{m}.npz"); fm = lambda m: os.path.join(PUB, f"real_{m}_mirror.npz")
    else:
        W = np.load(os.path.join(R, "falling_windows.npz")); meta = json.load(open(os.path.join(R, "falling_windows_meta.json")))
        fp = lambda m: os.path.join(R, "falling", f"falling_{m}.npz"); fm = lambda m: os.path.join(R, "falling", f"falling_{m}_mirror.npz")
    fam = np.array([m.get("family", "eq") for m in meta]); cl = np.array([f"{m.get('family', 'eq')}|{m['fut_end']}" for m in meta])
    return W, fam, cl, fp, fm


out = {"n_boot": B, "sets": {}}
for tag in ("anchor", "falling"):
    W, fam, cl, fp, fm = load_set(tag)
    ctx, fut = W["ctx_raw"].astype(np.float64), W["fut_raw"].astype(np.float64); last = ctx[:, -1]
    cs = W["ctx_sur"].astype(np.float64); fs = W["fut_sur"].astype(np.float64)
    cs, fs = (cs, fs) if cs.ndim == 3 else (cs[:, None], fs[:, None])
    rr = ctx[:, 1:] / ctx[:, :-1] - 1.0; last_m = ctx[:, 0] * np.cumprod(1.0 - rr, axis=1)[:, -1]
    drift = rr.mean(1); rs = cs[:, :, 1:] / cs[:, :, :-1] - 1.0; drift_s = rs.mean(2)
    splits = {"eq": fam == "eq", "fx": fam == "fx", "all": np.ones(len(fam), bool)} if tag == "anchor" else {"eq": np.ones(len(fam), bool)}
    S = out["sets"][tag] = {}
    for sname, sel in splits.items():
        if not sel.any():
            continue
        c = cl[sel]; T = S[sname] = {"n_windows": int(sel.sum()), "baselines": {}, "models": {}}
        for h in HS:
            y = fut[sel, h - 1] / last[sel] - 1.0; ys = fs[sel, :, h - 1] / cs[sel, :, -1] - 1.0
            T["baselines"][str(h)] = {"always_up": boot(hit(np.ones_like(y), y), c, 1), "fitted_drift": boot(hit(drift[sel], y), c, 2),
                                      "rising_share": float(np.mean(y > 0)),
                                      "always_up_randomised": boot(hit(np.ones_like(ys), ys), np.repeat(c[:, None], ys.shape[1], 1), 3)}
        for mi, m in enumerate(MODELS):
            if not (os.path.exists(fp(m)) and os.path.exists(fm(m))):
                continue
            z = np.load(fp(m)); yr = z["yhat_raw"].astype(np.float64); ysur = z["yhat_sur"].astype(np.float64)
            ysur = ysur if ysur.ndim == 3 else ysur[:, None]
            ymir = np.load(fm(m))["yhat_mirror"].astype(np.float64)
            M = T["models"][m] = {"label": LABEL[m]}
            for h in HS:
                y = fut[sel, h - 1] / last[sel] - 1.0; ys = fs[sel, :, h - 1] / cs[sel, :, -1] - 1.0
                d = yr[sel, h - 1] / last[sel] - 1.0; dm = ymir[sel, h - 1] / last_m[sel] - 1.0; odd = 0.5 * (d - dm)
                dsur = ysur[sel, :, h - 1] / cs[sel, :ysur.shape[1], -1] - 1.0
                M[str(h)] = {"raw": boot(hit(d, y), c, 10 + mi), "mirror": boot(hit(odd, y), c, 30 + mi),
                             "randomised": boot(hit(dsur, ys[:, :dsur.shape[1]]), np.repeat(c[:, None], dsur.shape[1], 1), 50 + mi),
                             "up_share_raw": float(np.mean(d > 0)), "up_share_mirror": float(np.mean(odd > 0))}
# the written predictions
A = out["sets"]["anchor"]["eq"]; up = A["baselines"]["128"]["always_up"]
out["predictions"] = {
    "D1_no_model_beats_always_up_by_2se": all(v["128"]["raw"]["da"] - up["da"] <= 2 * np.hypot(v["128"]["raw"]["se"], up["se"]) for v in A["models"].values()),
    "D1_detail": {m: round(v["128"]["raw"]["da"] - up["da"], 3) for m, v in A["models"].items()},
    "D2_randomised_within_2se_of_half": {m: abs(v["128"]["randomised"]["da"] - 0.5) <= 2 * v["128"]["randomised"]["se"] and abs(v["16"]["randomised"]["da"] - 0.5) <= 2 * v["16"]["randomised"]["se"]
                                         for m, v in out["sets"]["anchor"]["all"]["models"].items()},
    "D3_mirror_lowers_da_upward_models": {m: A["models"][m]["128"]["mirror"]["da"] < A["models"][m]["128"]["raw"]["da"] for m in UPWARD}}
out["predictions"]["D2_all"] = all(out["predictions"]["D2_randomised_within_2se_of_half"].values())
ups = [v["128"]["up_share_raw"] for v in A["models"].values()]; das = [v["128"]["raw"]["da"] for v in A["models"].values()]
out["anchor_eq_128_corr_upshare_da"] = float(np.corrcoef(ups, das)[0, 1])      # across the eleven models
out["predictions"]["D3_all"] = all(out["predictions"]["D3_mirror_lowers_da_upward_models"].values())
json.dump(out, open(os.path.join(R, "direction_accuracy.json"), "w"), indent=1)

for tag, S in out["sets"].items():
    for sname, T in S.items():
        b = T["baselines"]["128"]
        print(f"== {tag}/{sname} ({T['n_windows']} windows), h=128: rising {b['rising_share']:.3f}, always up {b['always_up']['da']:.3f}±{b['always_up']['se']:.3f}, "
              f"drift sign {b['fitted_drift']['da']:.3f}±{b['fitted_drift']['se']:.3f}; h=16 always up {T['baselines']['16']['always_up']['da']:.3f}")
        for m, M in T["models"].items():
            q = M["128"]
            print(f"   {M['label']:12s} raw {q['raw']['da']:.3f}±{q['raw']['se']:.3f} (up {q['up_share_raw']:.2f}) | mirror {q['mirror']['da']:.3f}±{q['mirror']['se']:.3f} "
                  f"(up {q['up_share_mirror']:.2f}) | randomised {q['randomised']['da']:.3f}±{q['randomised']['se']:.3f} | h=16 raw {M['16']['raw']['da']:.3f}")
print("predictions:", {k: v for k, v in out["predictions"].items() if not k.endswith("detail")})


def pc(o):
    return f"{100 * o['da']:.0f}\\,\\textpm\\,{100 * o['se']:.0f}"


A, F = out["sets"]["anchor"], out["sets"]["falling"]["eq"]
L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{\textbf{Daily anchor, equities, 2017--2026}} & \textbf{Randomised} & \textbf{1996--2014} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
     r"\textbf{Forecast} & \textbf{Raw} & \textbf{Upward} & \textbf{Mirror-corrected} & \textbf{All copies} & \textbf{Raw} \\", r"\midrule"]
for key, lab in (("always_up", "Always up"), ("fitted_drift", "Sign of fitted drift")):
    L.append(" & ".join([lab, pc(A["eq"]["baselines"]["128"][key]), "100" if key == "always_up" else "--", "--",
                         pc(A["all"]["baselines"]["128"]["always_up_randomised"]) if key == "always_up" else "--", pc(F["baselines"]["128"][key])]) + r" \\")
L.append(r"\midrule")
for m in MODELS:
    if m not in A["eq"]["models"]:
        continue
    q = A["eq"]["models"][m]["128"]
    L.append(" & ".join([LABEL[m], pc(q["raw"]), f"{100 * q['up_share_raw']:.0f}", pc(q["mirror"]), pc(A["all"]["models"][m]["128"]["randomised"]),
                         pc(F["models"][m]["128"]["raw"]) if m in F["models"] else "--"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
import table_bold   # bold marks, by the rule stated in the table note
L = table_bold.apply("direction_accuracy", L)
open(os.path.join("tables", "direction_accuracy.tex"), "w").write("\n".join(L) + "\n")
