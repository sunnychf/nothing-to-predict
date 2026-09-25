"""Metrics for the small-cap anchor (smallcap_data.py windows, real_probe.py --prefix smallcap forecasts):
the models on a real daily series with a documented, exploitable autocorrelation (the equal-weighted
smallest size decile, ew:Lo10) and on a comparison series without one (the value-weighted largest
decile, vw:Hi10), the positive control of the ladder repeated on a market.

    python smallcap_summary.py --res results --tables paper/tables [--boot 2000]

Units: return units relative to the last price, divided by sigma_r (the std of the context's
returns), pooled over windows in those normalised units, so that the 1987, 2008 and 2020 windows
do not decide every ratio (the unnormalised pooled skill of the daily anchor is kept in the JSON
as skill_raw_units). Benchmarks fitted to the context alone and iterated over the horizon:
  drift    the context's mean return (intercept only);
  AR(1)    OLS on the returns with an intercept, the linear reading of the lag-1 autocorrelation;
  AR(5)    the same with five lags.
Per model and split: raw skill at h = 1 and 16, the share of AR(1)'s gain over persistence at h = 1,
the correlation across windows of the model's one-step departure with AR(1)'s (does the model move
the way the autocorrelation says?), the mirror-corrected versions, the even share, and the same on
the K sign-randomised copies, where the autocorrelation is destroyed and persistence is optimal.
Standard errors: cluster bootstrap over future-end dates (the two series share the calendar).
Writes results/smallcap_summary.json and paper/tables/smallcap.tex.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results"); ap.add_argument("--tables", default="paper/tables"); ap.add_argument("--windows", default=None)
ap.add_argument("--models", default="chronos,chronosbolt,chronos2,tirex,moirai,moirai2,timesfm,timesfm25,timemoe,sundial,fincast")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=17)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
HS = (1, 16)
WP = args.windows or os.path.join(args.res, "smallcap_windows.npz")
W = np.load(WP); meta = json.load(open(WP.replace(".npz", "_meta.json")))
ctx, fut, ctx_s, fut_s, sig, sig_s = W["ctx_raw"], W["fut_raw"], W["ctx_sur"], W["fut_sur"], W["rsigma_raw"], W["rsigma_sur"]
n, K, T = ctx_s.shape; H = fut.shape[1]
asset = np.array([m["asset"] for m in meta]); dates = np.array([m["fut_end"] for m in meta])
cl_ids = {d: i for i, d in enumerate(sorted(set(dates)))}; cl = np.array([cl_ids[d] for d in dates])
last = ctx[:, -1:]; fut_rel = fut / last - 1.0
SPLITS = {"lo10": asset == "ew:Lo10", "hi10": asset == "vw:Hi10", "all": np.ones(n, bool)}

def boot(stat, idx, clusters):
    gs = np.unique(clusters[idx]); by = {g: idx[clusters[idx] == g] for g in gs}; out = []
    for _ in range(args.boot):
        pick = rng.choice(gs, size=len(gs), replace=True); out.append(stat(np.concatenate([by[g] for g in pick])))
    return float(np.std(out))

def summarise(rel, err_m, err_p, sig_, clusters, idx, err_o=None, rel_o=None):
    """rel, err_* (n, H) in return units; sig_ (n,): everything pooled in sigma_r units."""
    rec = {}
    for h in HS:
        j = h - 1; ds = rel[:, j] / sig_; em, ep = err_m[:, j] / sig_ ** 2, err_p[:, j] / sig_ ** 2
        sk = lambda s: float(1 - em[s].mean() / ep[s].mean())
        r = {"mean_dep": float(ds[idx].mean()), "mean_dep_se": boot(lambda s: float(ds[s].mean()), idx, clusters), "frac_up": float((ds[idx] > 0).mean()),
             "skill": sk(idx), "skill_se": boot(sk, idx, clusters), "skill_raw_units": float(1 - err_m[idx, j].mean() / err_p[idx, j].mean())}
        if err_o is not None:
            eo = err_o[:, j] / sig_ ** 2
            sh = lambda s: float((ep[s].mean() - em[s].mean()) / (ep[s].mean() - eo[s].mean()))
            r["share"], r["share_se"] = sh(idx), boot(sh, idx, clusters)
        if rel_o is not None:
            do = rel_o[:, j] / sig_
            co = lambda s: float(np.corrcoef(ds[s], do[s])[0, 1])
            r["corr_with_ar1"], r["corr_with_ar1_se"] = co(idx), boot(co, idx, clusters)
        rec[str(h)] = r
    return rec

def ar_rel(c, p):
    """Iterated AR(p) (p=0: the drift) fitted by OLS on the context's returns; forecast in return units relative to the last price, (n, H)."""
    r = c[:, 1:] / c[:, :-1] - 1.0; out = np.zeros((len(c), H))
    for i in range(len(c)):
        y = r[i, p:]; X = np.column_stack([np.ones(len(y))] + [r[i, p - k:len(r[i]) - k] for k in range(1, p + 1)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]; hist = list(r[i, -p:]) if p else []
        for h in range(H):
            nxt = b[0] + sum(b[k] * hist[-k] for k in range(1, p + 1)); out[i, h] = nxt; hist.append(nxt)
    return np.cumprod(1.0 + out, axis=1) - 1.0

err_p = fut_rel ** 2
bench_rel = {"drift": ar_rel(ctx, 0), "ar1": ar_rel(ctx, 1), "ar5": ar_rel(ctx, 5)}
err_b = {k: (fut_rel - v) ** 2 for k, v in bench_rel.items()}
r_ctx = ctx[:, 1:] / ctx[:, :-1] - 1.0
rho1 = np.array([np.corrcoef(r[:-1], r[1:])[0, 1] for r in r_ctx])
summary = {"n_windows": int(n), "K": int(K), "H": int(H), "T": int(T), "splits": {k: int(v.sum()) for k, v in SPLITS.items()},
           "period": {"first_ctx_start": min(m["ctx_start"] for m in meta), "last_fut_end": max(m["fut_end"] for m in meta)},
           "rho1_ctx": {k: {"mean": float(rho1[v].mean()), "median": float(np.median(rho1[v])), "frac_above_0.1": float((rho1[v] > 0.1).mean())} for k, v in SPLITS.items()},
           "benchmarks": {k: {sp: summarise(bench_rel[k], err_b[k], err_p, sig, cl, np.where(v)[0], err_b["ar1"] if k != "ar1" else None) for sp, v in SPLITS.items()} for k in bench_rel},
           "models": {}}
for sp in SPLITS:
    b = summary["benchmarks"]
    print(f"{sp:5s} n={SPLITS[sp].sum()} rho1 mean {summary['rho1_ctx'][sp]['mean']:+.3f} | skill h=1: drift {b['drift'][sp]['1']['skill']:+.3f}, AR(1) {b['ar1'][sp]['1']['skill']:+.3f}±{b['ar1'][sp]['1']['skill_se']:.3f}, AR(5) {b['ar5'][sp]['1']['skill']:+.3f} | h=16: drift {b['drift'][sp]['16']['skill']:+.3f}, AR(1) {b['ar1'][sp]['16']['skill']:+.3f}")
# copies: the benchmarks' edge must vanish there
cs0, fs0 = ctx_s[:, 0], fut_s[:, 0]; last0 = cs0[:, -1:]; fr0 = fs0 / last0 - 1.0
ar1_0 = ar_rel(cs0, 1); summary["benchmarks"]["ar1_on_copies"] = {sp: summarise(ar1_0, (fr0 - ar1_0) ** 2, fr0 ** 2, sig_s[:, 0], cl, np.where(v)[0]) for sp, v in SPLITS.items()}
print("AR(1) on the first sign copy, skill h=1:", {sp: round(summary['benchmarks']['ar1_on_copies'][sp]['1']['skill'], 3) for sp in SPLITS})

for m in args.models.split(","):
    p = os.path.join(args.res, f"smallcap_{m}.npz")
    if not os.path.exists(p): print("missing", p); continue
    r = np.load(p); yh, ys = r["yhat_raw"].astype(np.float64), r["yhat_sur"].astype(np.float64)
    if ys.ndim == 2: ys = ys[:, None]
    Km = min(ys.shape[1], K); ys = ys[:, :Km]
    rel_raw = yh / last - 1.0; S = {}
    for sp, v in SPLITS.items():
        idx = np.where(v)[0]; S[f"raw|{sp}"] = summarise(rel_raw, (fut_rel - rel_raw) ** 2, err_p, sig, cl, idx, err_b["ar1"], bench_rel["ar1"])
    mp = os.path.join(args.res, f"smallcap_{m}_mirror.npz")
    if os.path.exists(mp):
        ym = np.load(mp)["yhat_mirror"].astype(np.float64); last_m = (ctx[:, :1] * np.cumprod(1.0 - r_ctx, axis=1))[:, -1:]
        rel_mir = ym / last_m - 1.0; d_even = 0.5 * (rel_raw + rel_mir); rel_c = rel_raw - d_even
        for sp, v in SPLITS.items():
            idx = np.where(v)[0]; S[f"mirror|{sp}"] = summarise(rel_c, (fut_rel - rel_c) ** 2, err_p, sig, cl, idx, err_b["ar1"], bench_rel["ar1"])
            S[f"even_share|{sp}"] = {str(h): float(np.mean((d_even[idx, h - 1] / sig[idx]) ** 2) / np.mean((rel_raw[idx, h - 1] / sig[idx]) ** 2)) for h in HS}
    last_s = ctx_s[:, :Km, -1:]; rel_s = ys / last_s - 1.0; fut_rel_s = fut_s[:, :Km] / last_s - 1.0
    flat = lambda a: a.reshape(n * Km, *a.shape[2:]); sig_f = sig_s[:, :Km].reshape(n * Km); cl_f = np.repeat(cl, Km)
    for sp, v in SPLITS.items():
        idx = np.where(np.repeat(v, Km))[0]; S[f"sur|{sp}"] = summarise(flat(rel_s), flat((fut_rel_s - rel_s) ** 2), flat(fut_rel_s ** 2), sig_f, cl_f, idx)
    msp = os.path.join(args.res, f"smallcap_{m}_mirror_sur.npz")
    if os.path.exists(msp):
        yms = np.load(msp)["yhat_mirror_sur"].astype(np.float64)[:, :Km]; rr_s = ctx_s[:, :Km, 1:] / ctx_s[:, :Km, :-1] - 1.0
        last_ms = (ctx_s[:, :Km, :1] * np.cumprod(1.0 - rr_s, axis=2))[:, :, -1:]; rel_ms = yms / last_ms - 1.0; rel_sc = rel_s - 0.5 * (rel_s + rel_ms)
        for sp, v in SPLITS.items():
            idx = np.where(np.repeat(v, Km))[0]; S[f"sur_mirror|{sp}"] = summarise(flat(rel_sc), flat((fut_rel_s - rel_sc) ** 2), flat(fut_rel_s ** 2), sig_f, cl_f, idx)
    summary["models"][m] = S
    a, c_ = S["raw|lo10"], S.get("mirror|lo10"); s_ = S["sur|lo10"]
    print(f"{LABEL[m]:12s} Lo10 raw h1 skill {a['1']['skill']:+.3f}±{a['1']['skill_se']:.3f} share {a['1']['share']:+.2f} corr {a['1']['corr_with_ar1']:+.2f} | h16 {a['16']['skill']:+.3f}"
          + (f" | mirror h1 {c_['1']['skill']:+.3f} share {c_['1']['share']:+.2f} corr {c_['1']['corr_with_ar1']:+.2f} even {S['even_share|lo10']['1']:.2f}" if c_ else "")
          + f" | copies h1 {s_['1']['skill']:+.3f} dep16 {s_['16']['mean_dep']:+.2f} up {s_['16']['frac_up']:.2f} | Hi10 raw h1 {S['raw|hi10']['1']['skill']:+.3f} corr {S['raw|hi10']['1']['corr_with_ar1']:+.2f}")

M = summary["models"]
summary["counts"] = {"lo10_beat_persistence_h16": [m for m in M if M[m]["raw|lo10"]["16"]["skill"] > 0],
                     "lo10_beat_persistence_h16_gt_1se": [m for m in M if M[m]["raw|lo10"]["16"]["skill"] > M[m]["raw|lo10"]["16"]["skill_se"]],
                     "lo10_above_ar1_h16": [m for m in M if M[m]["raw|lo10"]["16"]["skill"] > summary["benchmarks"]["ar1"]["lo10"]["16"]["skill"]],
                     "lo10_mirror_ge_raw_h16": [m for m in M if "mirror|lo10" in M[m] and M[m]["mirror|lo10"]["16"]["skill"] >= M[m]["raw|lo10"]["16"]["skill"]],
                     "lo10_mirror_beat_persistence_h16": [m for m in M if "mirror|lo10" in M[m] and M[m]["mirror|lo10"]["16"]["skill"] > 0],
                     "lo10_copies_beat_persistence_h16": [m for m in M if M[m]["sur|lo10"]["16"]["skill"] > 0],
                     "hi10_beat_persistence_h16": [m for m in M if M[m]["raw|hi10"]["16"]["skill"] > 0],
                     "lo10_beat_persistence_h1": [m for m in M if M[m]["raw|lo10"]["1"]["skill"] > 0],
                     "lo10_beat_persistence_h1_gt_1se": [m for m in M if M[m]["raw|lo10"]["1"]["skill"] > M[m]["raw|lo10"]["1"]["skill_se"]],
                     "lo10_corr_positive_gt_1se": [m for m in M if M[m]["raw|lo10"]["1"]["corr_with_ar1"] > M[m]["raw|lo10"]["1"]["corr_with_ar1_se"]],
                     "lo10_mirror_beat_persistence_h1": [m for m in M if "mirror|lo10" in M[m] and M[m]["mirror|lo10"]["1"]["skill"] > 0],
                     "lo10_mirror_helps_h1": [m for m in M if "mirror|lo10" in M[m] and M[m]["mirror|lo10"]["1"]["skill"] > M[m]["raw|lo10"]["1"]["skill"]],
                     "lo10_copies_beat_persistence_h1": [m for m in M if M[m]["sur|lo10"]["1"]["skill"] > 0],
                     "hi10_beat_persistence_h1": [m for m in M if M[m]["raw|hi10"]["1"]["skill"] > 0]}
print("counts:", {k: len(v) for k, v in summary["counts"].items()})
os.makedirs(args.tables, exist_ok=True)
json.dump(summary, open(os.path.join(args.res, "smallcap_summary.json"), "w"), indent=1)
def pm(v, se, d=3): return f"${v:+.{d}f}\\pm{se:.{d}f}$"
b = summary["benchmarks"]
# the table reads h = 16 (the first step is where several models carry an artefact, as on the daily anchor; h = 1 stays in the JSON)
L = [r"\begin{tabular}{l|cc|cc|c|c}", r"\toprule",
     r" & \multicolumn{2}{c|}{smallest decile, raw windows} & \multicolumn{2}{c|}{mirror-corrected} & copies & largest decile \\",
     r"Forecast, $h{=}16$ & skill & share of AR(1) & skill & even share & skill & skill \\", r"\midrule",
     r"persistence & $0$ & $0$ & & & $0$ & $0$ \\",
     f"drift (fit) & {pm(b['drift']['lo10']['16']['skill'], b['drift']['lo10']['16']['skill_se'])} & ${b['drift']['lo10']['16']['share']:+.2f}$ & & & & {pm(b['drift']['hi10']['16']['skill'], b['drift']['hi10']['16']['skill_se'])} \\\\",
     f"AR(1) (fit) & {pm(b['ar1']['lo10']['16']['skill'], b['ar1']['lo10']['16']['skill_se'])} & $1$ & & & {pm(b['ar1_on_copies']['lo10']['16']['skill'], b['ar1_on_copies']['lo10']['16']['skill_se'])} & {pm(b['ar1']['hi10']['16']['skill'], b['ar1']['hi10']['16']['skill_se'])} \\\\",
     f"AR(5) (fit) & {pm(b['ar5']['lo10']['16']['skill'], b['ar5']['lo10']['16']['skill_se'])} & ${b['ar5']['lo10']['16']['share']:+.2f}$ & & & & {pm(b['ar5']['hi10']['16']['skill'], b['ar5']['hi10']['16']['skill_se'])} \\\\", r"\midrule"]
for m in args.models.split(","):
    S = M.get(m)
    if not S: continue
    a, c_, s_ = S["raw|lo10"]["16"], S.get("mirror|lo10", {}).get("16"), S["sur|lo10"]["16"]
    L.append(f"{LABEL[m]} & {pm(a['skill'], a['skill_se'])} & ${a['share']:+.2f}$ & " + (f"${c_['skill']:+.3f}$ & {S['even_share|lo10']['16']:.2f}" if c_ else "-- & --")
             + f" & ${s_['skill']:+.3f}$ & ${S['raw|hi10']['16']['skill']:+.3f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "smallcap.tex"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(args.res, "smallcap_summary.json"), os.path.join(args.tables, "smallcap.tex"))
