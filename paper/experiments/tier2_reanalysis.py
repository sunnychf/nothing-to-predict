"""Tier-2 reanalyses of existing outputs (no new model runs). Reads ../results and ../paper_revision/figures
read-only and writes experiments/results/tier2_reanalysis.json.

  (a) direction uncertainty: exact binomial tests and Clopper-Pearson intervals for the fraction of
      forecasts above the last value on N1 (128 independent series, h = 16 and 128), Holm-adjusted over
      the eleven default models; cluster-bootstrap intervals for the same fraction on the daily
      sign-randomised windows (clusters = family x future-end date, copies stay with their window);
  (b) small-cap: paired skill differences model - benchmark (drift, AR(1), AR(5)) at h = 16 on the
      smallest decile, cluster bootstrap over future-end dates (same units as smallcap_summary.py);
  (c) Monte Carlo share of the mirror's gain for the sampled models: with independent sampling noise,
      raw risk carries E[v(x)]/S and the mirror-corrected risk (1/4)E[v(x)/S + v(xbar)/S], so the
      mirror removes E[v/S] - (1/4)E[v(x)/S + v(xbar)/S] of risk that is sampling noise and not
      structure; reported in skill units next to the observed skill change.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/tier2_reanalysis.py
"""
import json, math, os
import numpy as np

RES = os.path.join("..", "results"); OUT = os.path.join("experiments", "results")
os.makedirs(OUT, exist_ok=True)
BOOT = 2000
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1",
         "moirai2": "Moirai-2.0", "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
FF = {"chronos": "Chronos-small", "chronosbolt": "Chronos-Bolt-small", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-small",
      "moirai2": "Moirai-2.0", "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE-200M", "sundial": "Sundial", "fincast": "FinCast"}
MODELS = list(LABEL)
SAMPLED = {"chronos", "moirai", "sundial"}
out = {}


def binom_two_sided(k, n):
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(n + 1)]
    pk = [math.exp(v) for v in lp]; obs = pk[k]
    return min(1.0, sum(p for p in pk if p <= obs * (1 + 1e-12)))


def clopper_pearson(k, n, a=0.05):
    from math import comb
    def cdf(x, p): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) > 0: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: (1 - cdf(k - 1, p)) - a / 2 if False else a / 2 - (1 - cdf(k - 1, p)))
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) - a / 2)
    return lower, upper


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [0.0] * len(pvals); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - r) * pvals[i])); adj[i] = run
    return adj


def cluster_boot(stat, clusters, seed):
    g = np.random.default_rng(seed); gs = np.unique(clusters); by = {c: np.where(clusters == c)[0] for c in gs}; vals = []
    for _ in range(BOOT):
        pick = g.choice(gs, size=len(gs), replace=True); vals.append(stat(np.concatenate([by[c] for c in pick])))
    return np.array(vals)


# ---------------------------------------------------------------- (a) direction
ff = json.load(open(os.path.join("figures", "figures_facts.json")))
A = {"N1": {}}
for h in ("16", "128"):
    rows = {}
    for m in MODELS:
        d = ff["direction"][FF[m]]; n = int(d.get("n", 128)); frac = d[f"frac_up_h{h}"]; k = int(round(frac * n))
        assert abs(k / n - frac) < 1e-9, (m, h, frac)
        lo, hi = clopper_pearson(k, n)
        rows[m] = {"k": k, "n": n, "frac": k / n, "p_two_sided": binom_two_sided(k, n), "ci95": [lo, hi]}
    adj = holm([rows[m]["p_two_sided"] for m in MODELS])
    for m, a in zip(MODELS, adj): rows[m]["p_holm"] = a
    A["N1"][h] = rows
# daily sign-randomised windows: fraction of copies with a forecast above the copy's last value
W = np.load(os.path.join(RES, "real_windows.npz")); meta = json.load(open(os.path.join(RES, "real_windows_meta.json")))
fam = np.array([m["family"] for m in meta]); cl_ids = {c: i for i, c in enumerate(sorted({f"{m['family']}|{m['fut_end']}" for m in meta}))}
cl = np.array([cl_ids[f"{m['family']}|{m['fut_end']}"] for m in meta])
ctx_s = W["ctx_sur"]; n_w, K = ctx_s.shape[:2]
A["daily_sign_randomised"] = {}
for m in MODELS:
    z = np.load(os.path.join(RES, f"real_{m}.npz")); ys = z["yhat_sur"].astype(np.float64)[:, :K]
    up = (ys[:, :, 127] > ctx_s[:, :ys.shape[1], -1]).astype(float)          # (n, K) at h = 128
    per_w = up.mean(1)
    frac = float(per_w.mean()); bs = cluster_boot(lambda s: float(per_w[s].mean()), cl, 101)
    A["daily_sign_randomised"][m] = {"frac_h128": frac, "se": float(bs.std()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                     "check_figures_facts": ff["real"][FF[m]]["sur"]["frac_up_h128"]}
out["direction"] = A

# ---------------------------------------------------------------- (b) small-cap paired differences
SW = np.load(os.path.join(RES, "smallcap_windows.npz")); smeta = json.load(open(os.path.join(RES, "smallcap_windows_meta.json")))
ctx, fut, sig = SW["ctx_raw"], SW["fut_raw"], SW["rsigma_raw"]; H = fut.shape[1]
asset = np.array([m["asset"] for m in smeta]); dates = np.array([m["fut_end"] for m in smeta])
lo10 = np.where(asset == "ew:Lo10")[0]; dcl = {d: i for i, d in enumerate(sorted(set(dates)))}; scl = np.array([dcl[d] for d in dates])
last = ctx[:, -1:]; fut_rel = fut / last - 1.0


def ar_rel(c, p):
    r = c[:, 1:] / c[:, :-1] - 1.0; o = np.zeros((len(c), H))
    for i in range(len(c)):
        y = r[i, p:]; X = np.column_stack([np.ones(len(y))] + [r[i, p - kk:len(r[i]) - kk] for kk in range(1, p + 1)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]; hist = list(r[i, -p:]) if p else []
        for hh in range(H):
            nxt = b[0] + sum(b[kk] * hist[-kk] for kk in range(1, p + 1)); o[i, hh] = nxt; hist.append(nxt)
    return np.cumprod(1.0 + o, axis=1) - 1.0


j = 15                                                                                      # h = 16
ep = (fut_rel[:, j] / sig) ** 2
bench = {"drift": ar_rel(ctx, 0), "ar1": ar_rel(ctx, 1), "ar5": ar_rel(ctx, 5)}
eb = {k: ((fut_rel[:, j] - v[:, j]) / sig) ** 2 for k, v in bench.items()}
B = {"horizon": 16, "n_windows": int(len(lo10)), "benchmarks": {k: float(1 - eb[k][lo10].mean() / ep[lo10].mean()) for k in eb}, "models": {}}
for m in MODELS:
    yh = np.load(os.path.join(RES, f"smallcap_{m}.npz"))["yhat_raw"].astype(np.float64)
    em = ((fut_rel[:, j] - (yh[:, j] / last[:, 0] - 1.0)) / sig) ** 2
    rec = {"skill": float(1 - em[lo10].mean() / ep[lo10].mean())}
    for k in eb:
        diff = lambda s, k=k: float((eb[k][s].mean() - em[s].mean()) / ep[s].mean())       # skill(model) - skill(benchmark)
        idx_cl = scl[lo10]
        bs = cluster_boot(lambda s: diff(lo10[s]), idx_cl, 202 + len(k))
        rec[f"minus_{k}"] = {"diff": diff(lo10), "se": float(bs.std()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    B["models"][m] = rec
for k in ("ar1", "ar5"):
    d_b = lambda s, k=k: float((eb["drift"][s].mean() - eb[k][s].mean()) / ep[s].mean())
    bs = cluster_boot(lambda s: d_b(lo10[s]), scl[lo10], 303 + len(k))
    B[f"{k}_minus_drift"] = {"diff": d_b(lo10), "se": float(bs.std())}
B["count_below_ar5_gt_2se"] = [m for m in MODELS if B["models"][m]["minus_ar5"]["diff"] < -2 * B["models"][m]["minus_ar5"]["se"]]
B["count_above_drift_gt_2se"] = [m for m in MODELS if B["models"][m]["minus_drift"]["diff"] > 2 * B["models"][m]["minus_drift"]["se"]]
B["count_below_drift_gt_2se"] = [m for m in MODELS if B["models"][m]["minus_drift"]["diff"] < -2 * B["models"][m]["minus_drift"]["se"]]
B["count_above_ar1_gt_2se"] = [m for m in MODELS if B["models"][m]["minus_ar1"]["diff"] > 2 * B["models"][m]["minus_ar1"]["se"]]
B["count_above_ar5_gt_2se"] = [m for m in MODELS if B["models"][m]["minus_ar5"]["diff"] > 2 * B["models"][m]["minus_ar5"]["se"]]
out["smallcap_paired"] = B

# ---------------------------------------------------------------- (c) Monte Carlo share of the mirror's gain
C = {"ladder": {}, "daily": {}}
for m in sorted(SAMPLED):
    z = np.load(os.path.join(RES, f"remedy_{m}.npz")); info = json.load(open(os.path.join(RES, f"remedy_{m}.json")))
    C["ladder"][m] = {}
    for rung in info["rungs"]:
        fut_l, ctx_l = z[f"{rung}/fut"], z[f"{rung}/ctx"]; yh, mv = z[f"{rung}/yhat_raw"].astype(np.float64), z[f"{rung}/mcvar_raw"].astype(np.float64)
        n = len(yh); mi = (np.arange(n) + n // 2) % n; sel = np.arange(n) < n // 2
        lastl = ctx_l[:, -1:]; dep = yh - lastl; ym = yh - 0.5 * (dep + dep[mi])
        C["ladder"][m][rung] = {}
        for h in (16, 128):
            jj = h - 1; epl = ((fut_l - lastl) ** 2)[sel, jj].mean()
            obs = float((((fut_l - yh) ** 2)[sel, jj].mean() - ((fut_l - ym) ** 2)[sel, jj].mean()) / epl)
            mc = float((mv[sel, jj].mean() - 0.25 * (mv[sel, jj] + mv[mi][sel, jj]).mean()) / epl)
            C["ladder"][m][rung][str(h)] = {"observed_skill_gain": obs, "mc_part": mc, "structural_part": obs - mc,
                                            "mc_share": (mc / obs) if obs > 0 else None}
    # daily windows, return units, pooled squared errors as in remedy_summary.py (skill at h = 128)
    zr = np.load(os.path.join(RES, f"real_{m}.npz")); zm = np.load(os.path.join(RES, f"real_{m}_mirror.npz"))
    ctx_r, fut_r = W["ctx_raw"], W["fut_raw"]; last_r = ctx_r[:, -1:]; r_ctx = ctx_r[:, 1:] / ctx_r[:, :-1] - 1.0
    last_m = (ctx_r[:, :1] * np.cumprod(1.0 - r_ctx, axis=1))[:, -1:]
    yh, mv = zr["yhat_raw"].astype(np.float64), zr["mcvar_raw"].astype(np.float64)
    ym, mvm = zm["yhat_mirror"].astype(np.float64), zm["mcvar_mirror"].astype(np.float64)
    rel_raw = yh / last_r - 1.0; rel_mir = ym / last_m - 1.0; rel_c = rel_raw - 0.5 * (rel_raw + rel_mir); fr = fut_r / last_r - 1.0
    C["daily"][m] = {}
    for split, idx in (("all", np.arange(len(yh))), ("eq", np.where(fam == "eq")[0]), ("fx", np.where(fam == "fx")[0])):
        jj = 127; epr = (fr[idx, jj] ** 2).mean()
        obs = float(((fr[idx, jj] - rel_raw[idx, jj]) ** 2).mean() / epr - ((fr[idx, jj] - rel_c[idx, jj]) ** 2).mean() / epr)
        mc_raw = (mv[idx, jj] / last_r[idx, 0] ** 2).mean(); mc_c = 0.25 * (mv[idx, jj] / last_r[idx, 0] ** 2 + mvm[idx, jj] / last_m[idx, 0] ** 2).mean()
        mc = float((mc_raw - mc_c) / epr)
        C["daily"][m][f"raw|{split}"] = {"observed_skill_change": obs, "mc_part": mc, "structural_part": obs - mc}
    # sign-randomised copies
    zs = np.load(os.path.join(RES, f"real_{m}_mirror_sur.npz"))
    ys, ms = zr["yhat_sur"].astype(np.float64)[:, :K], zr["mcvar_sur"].astype(np.float64)[:, :K]
    yms, mms = zs["yhat_mirror_sur"].astype(np.float64)[:, :K], zs["mcvar_mirror_sur"].astype(np.float64)[:, :K]
    cs, fs = W["ctx_sur"][:, :K], W["fut_sur"][:, :K]; ls = cs[:, :, -1:]; rr = cs[:, :, 1:] / cs[:, :, :-1] - 1.0
    lms = (cs[:, :, :1] * np.cumprod(1.0 - rr, axis=2))[:, :, -1:]
    rs_ = ys / ls - 1.0; rms_ = yms / lms - 1.0; rsc = rs_ - 0.5 * (rs_ + rms_); frs = fs / ls - 1.0
    jj = 127; eps_ = (frs[:, :, jj] ** 2).mean()
    obs = float(((frs[:, :, jj] - rs_[:, :, jj]) ** 2).mean() / eps_ - ((frs[:, :, jj] - rsc[:, :, jj]) ** 2).mean() / eps_)
    mc_raw = (ms[:, :, jj] / ls[:, :, 0] ** 2).mean(); mc_c = 0.25 * (ms[:, :, jj] / ls[:, :, 0] ** 2 + mms[:, :, jj] / lms[:, :, 0] ** 2).mean()
    mc = float((mc_raw - mc_c) / eps_)
    C["daily"][m]["sign_randomised|all"] = {"observed_skill_change": obs, "mc_part": mc, "structural_part": obs - mc, "mc_share": mc / obs if obs > 0 else None}
out["mc_share"] = C
json.dump(out, open(os.path.join(OUT, "tier2_reanalysis.json"), "w"), indent=1)

# ---------------------------------------------------------------- print
print("(a) N1, h=128: k/n, p, Holm p, 95% CI")
for m in MODELS:
    r = A["N1"]["128"][m]; print(f"  {LABEL[m]:12s} {r['k']:3d}/{r['n']} p={r['p_two_sided']:.2g} holm={r['p_holm']:.2g} CI=[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"
                                  f" | daily null frac {A['daily_sign_randomised'][m]['frac_h128']:.3f} (ff {A['daily_sign_randomised'][m]['check_figures_facts']:.3f}) se {A['daily_sign_randomised'][m]['se']:.3f}")
print("(b) small-cap h=16, skill and paired differences (model - benchmark):", {k: round(v, 3) for k, v in B["benchmarks"].items()})
for m in MODELS:
    r = B["models"][m]; print(f"  {LABEL[m]:12s} skill {r['skill']:+.3f} | -drift {r['minus_drift']['diff']:+.3f}±{r['minus_drift']['se']:.3f} | -AR1 {r['minus_ar1']['diff']:+.3f}±{r['minus_ar1']['se']:.3f} | -AR5 {r['minus_ar5']['diff']:+.3f}±{r['minus_ar5']['se']:.3f}")
print("  AR1-drift", B["ar1_minus_drift"], "AR5-drift", B["ar5_minus_drift"], "below AR5 >2SE:", B["count_below_ar5_gt_2se"]); print("  above drift >2SE:", B["count_above_drift_gt_2se"], "below drift >2SE:", B["count_below_drift_gt_2se"], "above AR1 >2SE:", B["count_above_ar1_gt_2se"], "above AR5 >2SE:", B["count_above_ar5_gt_2se"])
print("(c) MC part of the mirror's skill gain")
for m in sorted(SAMPLED):
    for rung, d in C["ladder"][m].items():
        print(f"  {LABEL[m]:10s} {rung:6s} " + " | ".join(f"h={h}: obs {v['observed_skill_gain']:+.3f} mc {v['mc_part']:+.3f} struct {v['structural_part']:+.3f}" for h, v in d.items()))
    for k, v in C["daily"][m].items():
        print(f"  {LABEL[m]:10s} daily {k:22s} obs {v['observed_skill_change']:+.3f} mc {v['mc_part']:+.3f} struct {v['structural_part']:+.3f}")
