"""Metrics for the intraday anchor (intraday_data.py windows, real_probe.py --prefix intraday
forecasts): the models on real transaction prices with a real bid-ask bounce, the positive
control of the ladder (rung N4) as it occurs in a market.

    python intraday_summary.py --res results --tables paper/tables [--boot 2000]

Everything is in return units relative to the last transaction price, as for the daily anchor:
D = yhat / y_T - 1 in units of sigma_r (std of the context's returns); skill = 1 - MSE / MSE of
persistence, pooled over windows; standard errors by a bootstrap over windows (the blocks of one
symbol are consecutive in time and are resampled independently, so the errors are read as
approximate). Two benchmarks that see only what the models see or less:
  MA(1) fit   the invertible MA(1) fitted to the context's returns by the lag-1 autocorrelation
              (rho_1 < 0 -> theta = (1 - sqrt(1 - 4 rho_1^2)) / (2 rho_1), else theta = 0), the
              innovations recovered by inverting the filter, forecast p_T + theta e_T at every
              horizon: rung N4's oracle with its parameter estimated from the window;
  mid proxy   the mean of the last ask-side and the last bid-side transaction prices in the
              context, an estimate of the quote midpoint from the trade directions, which the
              models never see; flat over the horizon.
Per model: raw skill, mirror-corrected skill (real_probe.py --mirror; the correction of
Section 5 in return units, as in remedy_summary.py), the share of the MA(1) benchmark's gain
captured (raw and corrected), the even share, and on the K sign-randomised copies (an exact
martingale with these increments) the mean departure and skill, raw and mirror-corrected.
Writes results/intraday_summary.json and paper/tables/intraday.tex.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--tables", default="paper/tables")
ap.add_argument("--windows", default=None)
ap.add_argument("--models", default="chronos,chronosbolt,chronos2,tirex,moirai,moirai2,timesfm,timesfm25,timemoe,sundial,fincast")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=13)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
HS = (1, 16)

W = np.load(args.windows or os.path.join(args.res, "intraday_windows.npz"))
meta = json.load(open((args.windows or os.path.join(args.res, "intraday_windows.npz")).replace(".npz", "_meta.json")))
ctx, fut, ctx_s, fut_s = W["ctx_raw"], W["fut_raw"], W["ctx_sur"], W["fut_sur"]
sig, sig_s = W["rsigma_raw"], W["rsigma_sur"]
q_ctx = W["q_ctx"].astype(np.int64)
n, K, T = ctx_s.shape; H = fut.shape[1]
last = ctx[:, -1:]; fut_rel = fut / last - 1.0; err_p = fut_rel ** 2
sym = np.array([w["asset"] for w in meta["windows"]])
groups = np.arange(n)                                                   # bootstrap over windows

def boot(stat, idx):
    out = []
    for _ in range(args.boot):
        out.append(stat(rng.choice(idx, size=len(idx), replace=True)))
    return float(np.std(out))

def summarise(rel, err_m, sig_, err_o=None, idx=None):
    idx = np.arange(len(rel)) if idx is None else idx
    rec = {}
    for h in HS:
        j = h - 1; ds = rel[:, j] / sig_
        sk = lambda s: float(1 - err_m[s, j].mean() / err_p_[s, j].mean())
        r = {"mean_dep": float(ds[idx].mean()), "mean_dep_se": boot(lambda s: float(ds[s].mean()), idx), "frac_up": float((ds[idx] > 0).mean()),
             "skill": sk(idx), "skill_se": boot(sk, idx)}
        if err_o is not None:
            sh = lambda s: float((err_p_[s, j].mean() - err_m[s, j].mean()) / (err_p_[s, j].mean() - err_o[s, j].mean()))
            r["share"], r["share_se"] = sh(idx), boot(sh, idx)
        rec[str(h)] = r
    return rec

# ---- benchmarks on the raw windows
r_ctx = ctx[:, 1:] / ctx[:, :-1] - 1.0
rho1 = np.array([np.corrcoef(r[:-1], r[1:])[0, 1] for r in r_ctx])
theta = np.where(rho1 < 0, (1.0 - np.sqrt(np.clip(1.0 - 4.0 * rho1 ** 2, 0.0, None))) / (2.0 * np.where(rho1 < 0, rho1, -1.0)), 0.0)
theta = np.clip(theta, -0.999, 0.0)
e = np.zeros_like(r_ctx)
for i in range(1, r_ctx.shape[1]):
    e[:, i] = r_ctx[:, i] - theta * e[:, i - 1]
ma1_rel = np.repeat((theta * e[:, -1])[:, None], H, axis=1)              # forecast return at every horizon
ask_last = np.array([c[np.where(qq > 0)[0][-1]] if (qq > 0).any() else c[-1] for c, qq in zip(ctx, q_ctx)])
bid_last = np.array([c[np.where(qq < 0)[0][-1]] if (qq < 0).any() else c[-1] for c, qq in zip(ctx, q_ctx)])
mid_rel = np.repeat((0.5 * (ask_last + bid_last) / last[:, 0] - 1.0)[:, None], H, axis=1)
err_p_ = err_p
err_ma1 = (fut_rel - ma1_rel) ** 2; err_mid = (fut_rel - mid_rel) ** 2
summary = {"n_windows": int(n), "K": int(K), "H": int(H), "T": int(T), "n_symbols": int(len(set(sym))),
           "windows_per_symbol": {s: int((sym == s).sum()) for s in sorted(set(sym))},
           "rho1_mean": float(rho1.mean()), "rho1_median": float(np.median(rho1)), "rho1_frac_negative": float((rho1 < 0).mean()),
           "rho1_frac_below_-0.1": float((rho1 < -0.1).mean()), "theta_mean": float(theta.mean()),
           "zero_return_share": float((r_ctx == 0).mean()),                                   # consecutive transactions at the same price
           "direction_flip_rate": float((q_ctx[:, 1:] != q_ctx[:, :-1]).mean()),               # 0.5 under i.i.d. trade directions (Roll)
           "ma1_fit_skill_h1_where_rho_below_-0.1": float(1 - np.mean(((fut_rel - ma1_rel) ** 2)[rho1 < -0.1, 0]) / np.mean(err_p[rho1 < -0.1, 0])) if (rho1 < -0.1).any() else None,
           "benchmarks": {"persistence": {str(h): {"skill": 0.0} for h in HS},
                          "ma1_fit": summarise(ma1_rel, err_ma1, sig),
                          "mid_proxy": summarise(mid_rel, err_mid, sig)},
           "models": {}}
print(f"{n} windows, {summary['n_symbols']} symbols, T {T}, H {H}, K {K}; lag-1 autocorrelation of returns: mean {rho1.mean():+.3f}, "
      f"{100 * (rho1 < 0).mean():.0f}% of windows negative; MA(1) fit skill h=1 {summary['benchmarks']['ma1_fit']['1']['skill']:+.3f}, "
      f"mid proxy {summary['benchmarks']['mid_proxy']['1']['skill']:+.3f}")

# ---- models
for m in args.models.split(","):
    p = os.path.join(args.res, f"intraday_{m}.npz")
    if not os.path.exists(p):
        print("missing", p); continue
    r = np.load(p); S = {}
    yh, ys = r["yhat_raw"].astype(np.float64), r["yhat_sur"].astype(np.float64)
    if ys.ndim == 2: ys = ys[:, None]
    Km = min(ys.shape[1], K); ys = ys[:, :Km]
    rel_raw = yh / last - 1.0
    S["raw"] = summarise(rel_raw, (fut_rel - rel_raw) ** 2, sig, err_ma1)
    # mirror-corrected (multiplicative mirror; the correction in return units, as remedy_summary.py)
    mp = os.path.join(args.res, f"intraday_{m}_mirror.npz")
    if os.path.exists(mp):
        ym = np.load(mp)["yhat_mirror"].astype(np.float64)
        last_m = (ctx[:, :1] * np.cumprod(1.0 - r_ctx, axis=1))[:, -1:]
        rel_mir = ym / last_m - 1.0
        d_even = 0.5 * (rel_raw + rel_mir); rel_c = rel_raw - d_even
        S["mirror"] = summarise(rel_c, (fut_rel - rel_c) ** 2, sig, err_ma1)
        S["even_share"] = {str(h): float(np.mean(d_even[:, h - 1] ** 2) / np.mean(rel_raw[:, h - 1] ** 2)) for h in HS}
    # the sign-randomised copies: an exact martingale with the real |increments|
    last_s = ctx_s[:, :Km, -1:]; rel_s = ys / last_s - 1.0
    fut_rel_s = fut_s[:, :Km] / last_s - 1.0
    flat = lambda a: a.reshape(n * Km, *a.shape[2:])
    err_p_ = flat(fut_rel_s ** 2); sig_f = sig_s[:, :Km].reshape(n * Km)
    S["sur"] = summarise(flat(rel_s), flat((fut_rel_s - rel_s) ** 2), sig_f)
    msp = os.path.join(args.res, f"intraday_{m}_mirror_sur.npz")
    if os.path.exists(msp):
        yms = np.load(msp)["yhat_mirror_sur"].astype(np.float64)[:, :Km]
        rr_s = ctx_s[:, :Km, 1:] / ctx_s[:, :Km, :-1] - 1.0
        last_ms = (ctx_s[:, :Km, :1] * np.cumprod(1.0 - rr_s, axis=2))[:, :, -1:]
        rel_ms = yms / last_ms - 1.0
        rel_sc = rel_s - 0.5 * (rel_s + rel_ms)
        S["sur_mirror"] = summarise(flat(rel_sc), flat((fut_rel_s - rel_sc) ** 2), sig_f)
    err_p_ = err_p
    summary["models"][m] = S
    print(f"{LABEL[m]:12s} raw skill h1 {S['raw']['1']['skill']:+.3f} (share {S['raw']['1']['share']:+.2f}) h16 {S['raw']['16']['skill']:+.3f}"
          + (f" | mirror h1 {S['mirror']['1']['skill']:+.3f} (share {S['mirror']['1']['share']:+.2f}) h16 {S['mirror']['16']['skill']:+.3f} even {S['even_share']['1']:.2f}" if "mirror" in S else "")
          + f" | copies dep16 {S['sur']['16']['mean_dep']:+.2f}±{S['sur']['16']['mean_dep_se']:.2f} up {S['sur']['16']['frac_up']:.2f} skill h1 {S['sur']['1']['skill']:+.3f} h16 {S['sur']['16']['skill']:+.3f}"
          + (f" -> mirror {S['sur_mirror']['1']['skill']:+.3f} / {S['sur_mirror']['16']['skill']:+.3f}" if "sur_mirror" in S else ""))

M = summary["models"]
summary["counts"] = {"beat_persistence_raw_h1": [m for m in M if M[m]["raw"]["1"]["skill"] > 0],
                     "mirror_helps_raw_h1": [m for m in M if "mirror" in M[m] and M[m]["mirror"]["1"]["skill"] > M[m]["raw"]["1"]["skill"]],
                     "mirror_helps_raw_h16": [m for m in M if "mirror" in M[m] and M[m]["mirror"]["16"]["skill"] > M[m]["raw"]["16"]["skill"]],
                     "mirror_helps_copies_h1": [m for m in M if "sur_mirror" in M[m] and M[m]["sur_mirror"]["1"]["skill"] > M[m]["sur"]["1"]["skill"]],
                     "mirror_helps_copies_h16": [m for m in M if "sur_mirror" in M[m] and M[m]["sur_mirror"]["16"]["skill"] > M[m]["sur"]["16"]["skill"]],
                     "copies_up_above_0.6": [m for m in M if M[m]["sur"]["16"]["frac_up"] > 0.6],
                     "copies_up_below_0.5": [m for m in M if M[m]["sur"]["16"]["frac_up"] < 0.5]}
print("counts:", {k: len(v) for k, v in summary["counts"].items()}, summary["counts"]["beat_persistence_raw_h1"], summary["counts"]["copies_up_above_0.6"])
os.makedirs(args.tables, exist_ok=True)
json.dump(summary, open(os.path.join(args.res, "intraday_summary.json"), "w"), indent=1)
# ---- table: transaction-price windows (raw -> mirror) and their sign-randomised copies
def pm(v, se, d=3): return f"${v:+.{d}f}\\pm{se:.{d}f}$"
L = [r"\begin{tabular}{l|cc|cc|cc|cc|cc}", r"\toprule",
     r" & \multicolumn{4}{c|}{transaction prices} & \multicolumn{6}{c}{sign-randomised copies (an exact martingale)} \\",
     r" & \multicolumn{2}{c|}{skill $h{=}1$} & \multicolumn{2}{c|}{skill $h{=}16$} & \multicolumn{2}{c|}{departure $h{=}16$} & \multicolumn{2}{c|}{skill $h{=}1$} & \multicolumn{2}{c}{skill $h{=}16$} \\",
     r"Model & raw & mirror & raw & mirror & mean$/\sigma_r$ & up & raw & mirror & raw & mirror \\", r"\midrule",
     f"persistence & $0$ & & $0$ & & & & & & & \\\\",
     f"MA(1) fitted to the context & {pm(summary['benchmarks']['ma1_fit']['1']['skill'], summary['benchmarks']['ma1_fit']['1']['skill_se'])} & & {pm(summary['benchmarks']['ma1_fit']['16']['skill'], summary['benchmarks']['ma1_fit']['16']['skill_se'])} & & & & & & & \\\\", r"\midrule"]
for m in args.models.split(","):
    S = summary["models"].get(m)
    if not S or m == "chronos": continue                     # Chronos-T5's row is the codec at this scale: stated in the text, not tabulated
    a, b, s_, sm = S["raw"], S.get("mirror"), S["sur"], S.get("sur_mirror")
    def sk(blk, h): return f"${blk[h]['skill']:+.3f}$" if blk else "--"
    L.append(f"{LABEL[m]} & {pm(a['1']['skill'], a['1']['skill_se'])} & {sk(b, '1')} & {sk(a, '16')} & {sk(b, '16')} & "
             f"{pm(s_['16']['mean_dep'], s_['16']['mean_dep_se'], 2)} & {s_['16']['frac_up']:.2f} & "
             f"{sk(s_, '1')} & {sk(sm, '1')} & {sk(s_, '16')} & {sk(sm, '16')} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "intraday.tex"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(args.res, "intraday_summary.json"), os.path.join(args.tables, "intraday.tex"))
