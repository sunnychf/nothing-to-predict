"""Metrics for the sign-averaged correction, one code path for every model.

Two symmetry-averaged corrections are evaluated (Section 5 of the paper):
  mirror     g_m(x) = g(x) - (D(x) + D(x_bar)) / 2, x_bar the context with every increment
             negated; exact for any process whose law is invariant under that flip (N1-N4);
             the ladder's antithetic layout supplies x_bar's forecast for free (row i + n/2)
  sign-avg   g_c(x) = g(x) - mean_k D(x~_k) over K random-sign copies; exact under conditional
             sign symmetry (N1-N3), with a finite-K noise term
Synthetic (results/remedy_<model>.npz from remedy_ladder.py), per rung and horizon:
  raw forecast   g(x):        mean signed departure / sigma, rms departure (MC term removed),
                              skill 1 - MSE/MSE_persistence, and on N4 the share of the exact
                              optimum's gain over persistence (nulls.oracle_forecast_exact; the
                              stored MA(1) oracle is the optimal linear forecast, not the optimum)
Test set: with the antithetic layout only the first half of the rows (independent draws with
their own futures) is evaluated; the mirror rows supply x_bar's forecast and nothing else, since
a test set containing both members of every pair would make the mirror's risk reduction the
identity E[(even part)^2] regardless of the data.
  corrected      g_c(x) = g(x) - dhat(x),  dhat_h(x) = mean_k [ g(x~_k)_h - x~_{k,T} ]
                              the same four quantities
  systematic share            mean(dhat^2) / mean(dep_raw^2), with and without the 1/K
                              noise term removed (Proposition: g_c beats g iff the share
                              exceeds 1/(K+1))
  loo                         corrected departure of each surrogate against the other K-1
                              (an estimator sanity check: zero in mean by exchangeability)
Real (results/real_<model>.npz with K surrogates per window), the same on the raw windows
in return units throughout: departure (g_h / y_T - 1) / sigma_r with sigma_r the std of the
context's simple returns, correction dhat = mean_k [ g(x~_k)_h / x~_{k,T} - 1 ] subtracted
from g_h / y_T - 1 (the surrogate is multiplicative, so return units are where it is exact
and where the leave-one-out mean is zero by construction), pooled and by family /
post-2025, plus the leave-one-out null on the surrogates. Skill is the ratio of pooled MSEs in
return units, so a window's weight does not depend on the scale of its level.
Standard errors: plain bootstrap over series (synthetic), cluster bootstrap over
family x future-end date (real). Writes results/remedy_summary.json and the LaTeX tables.
"""
import argparse, json, os
import numpy as np
from nulls import oracle_forecast, oracle_forecast_exact


def n4_exact_oracle(ctx, sigma, stored_linear, H):
    """The exact N4 optimum on stored contexts (spread 4 sigma, the generator's default); the stored
    linear optimum is recomputed and must match, which pins the parameters."""
    info = {"sigma": sigma, "spread": 4.0 * sigma}
    assert np.allclose(oracle_forecast("N4", ctx, info, H), stored_linear, atol=1e-9), "stored N4 oracle is not the MA(1) optimum at spread 4 sigma"
    return oracle_forecast_exact("N4", ctx, info, H)

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--real-dir", default=None, help="where real_<model>.npz live (default: --res)")
ap.add_argument("--windows", default=None, help="real windows file (default: <res>/real_windows.npz)")
ap.add_argument("--tables", default="paper/tables")
ap.add_argument("--models", default="chronos,moirai,timesfm,timemoe,fincast,chronosbolt,chronos2,tirex,moirai2,timesfm25,sundial")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
REAL = args.real_dir or args.res
WIN = args.windows or os.path.join(args.res, "real_windows.npz")
HS = [1, 16, 64, 128]
LABEL = {"chronos": "Chronos-small", "moirai": "Moirai-small", "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "fincast": "FinCast",
         "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "sundial": "Sundial"}
rng = np.random.default_rng(11)


def boot(stat, groups, idx_all):
    """Bootstrap SE of stat(idx) resampling groups (series or clusters) with replacement."""
    gs = np.unique(groups[idx_all]); by = {g: idx_all[groups[idx_all] == g] for g in gs}; out = []
    for _ in range(args.boot):
        pick = rng.choice(gs, size=len(gs), replace=True)
        out.append(stat(np.concatenate([by[g] for g in pick])))
    return float(np.std(out))


def boot_own(stat, groups, idx_all, seed):
    """As boot(), on a generator of its own, so that a statistic added later leaves every earlier
    bootstrap draw (and hence every published standard error) untouched."""
    g2 = np.random.default_rng(seed)
    gs = np.unique(groups[idx_all]); by = {g: idx_all[groups[idx_all] == g] for g in gs}; out = []
    for _ in range(args.boot):
        pick = g2.choice(gs, size=len(gs), replace=True)
        out.append(stat(np.concatenate([by[g] for g in pick])))
    return float(np.std(out))


def block(dep, err_m, err_p, mcv, sigma, mask, groups, err_o=None, noise=None):
    """Per-horizon summary of a forecast given its departures (n, H) and sigma (n,) in matching units.

    noise (n, H), when given, is the finite-K variance the sign-averaged estimate adds to the
    squared error (across-copy variance / K, in the units of err_m); removing it from the
    corrected MSE estimates the K -> infinity skill, exactly as the Monte Carlo term is removed
    from the departure elsewhere in the paper.
    """
    rec = {}
    idx_all = np.where(mask)[0]
    for h in HS:
        j = h - 1
        ds = dep[:, j] / sigma
        m = lambda sel: float(np.mean(ds[sel]))
        sk = lambda sel: float(1 - err_m[sel, j].mean() / err_p[sel, j].mean())
        r = {"mean_dep": m(idx_all), "mean_dep_se": boot(m, groups, idx_all),
             "frac_up": float((ds[idx_all] > 0).mean()),
             "rms_dep": float(np.sqrt(max(np.mean(ds[idx_all] ** 2) - np.mean(mcv[idx_all, j] / sigma[idx_all] ** 2), 0.0))),
             "skill": sk(idx_all), "skill_se": boot(sk, groups, idx_all)}
        if noise is not None:
            r["skill_Kinf"] = float(1 - (err_m[idx_all, j].mean() - noise[idx_all, j].mean()) / err_p[idx_all, j].mean())
        if err_o is not None:
            sh = lambda sel: float((err_p[sel, j].mean() - err_m[sel, j].mean()) / (err_p[sel, j].mean() - err_o[sel, j].mean()))
            r["oracle_share"], r["oracle_share_se"] = sh(idx_all), boot(sh, groups, idx_all)
        rec[str(h)] = r
    return rec


summary = {"models": {}, "K": None}
# ------------------------------------------------------------------ synthetic ladder
for mname in args.models.split(","):
    path = os.path.join(args.res, f"remedy_{mname}.npz")
    if not os.path.exists(path):
        print("missing", path); continue
    z = np.load(path); info = json.load(open(path.replace(".npz", ".json")))
    S = summary["models"].setdefault(mname, {})
    for ri, rung in enumerate(info["rungs"]):
        ctx, fut, sigma = z[f"{rung}/ctx"], z[f"{rung}/fut"], float(z[f"{rung}/sigma"])
        yh, mv, ys, ms, orc = (z[f"{rung}/yhat_raw"].astype(np.float64), z[f"{rung}/mcvar_raw"].astype(np.float64),
                               z[f"{rung}/yhat_sur"].astype(np.float64), z[f"{rung}/mcvar_sur"].astype(np.float64), z[f"{rung}/oracle"])
        n, K, T = ys.shape[0], ys.shape[1], ctx.shape[1]; summary["K"] = K
        if rung == "N4": orc = n4_exact_oracle(ctx, sigma, orc, fut.shape[1])                      # shares are against the exact optimum
        # regenerate the surrogates' last values exactly as remedy_ladder.py drew them
        g = np.random.default_rng(7000 + ri); eps = g.choice(np.array([-1.0, 1.0]), size=(n, K, T - 1))
        sur_last = ctx[:, 0] [:, None] + np.sum(eps * np.diff(ctx, axis=1)[:, None, :], axis=2)          # (n, K)
        last = ctx[:, -1:]
        dep_raw = yh - last
        dep_sur = ys - sur_last[:, :, None]                                                                # (n, K, H)
        dhat = dep_sur.mean(1)
        yc = yh - dhat
        mv_c = mv + ms.sum(1) / K ** 2
        sig = np.full(n, sigma); groups = np.arange(n)
        anti = bool(info["rungs"][rung].get("antithetic"))
        # Test set. With the antithetic layout the second half of the rows are the mirrors of the
        # first half, futures included; a test set that contains both members of every pair makes
        # the mirror's risk reduction the algebraic identity E[(even part)^2] on every pair, whatever
        # the data, and "improves on every rung" would follow from the layout and not from the
        # data. Every synthetic statistic is therefore taken on the first half only, n/2 independent
        # draws with their own futures; the mirror rows supply x_bar's forecast and enter nothing else.
        mask = (np.arange(n) < n // 2) if anti else np.ones(n, bool); sel = mask
        err_p = (fut - last) ** 2; err_o = (fut - orc) ** 2 if rung == "N4" else None
        v_k = dep_sur.var(1, ddof=1)                                                                       # (n, H) across-copy variance
        R = {"n": int(mask.sum()), "n_rows": n, "K": K, "sigma": sigma, "antithetic": anti,
             "raw": block(dep_raw, (fut - yh) ** 2, err_p, mv, sig, mask, groups, err_o),
             "corrected": block(yc - last, (fut - yc) ** 2, err_p, mv_c, sig, mask, groups, err_o, noise=v_k / K)}
        if rung == "N4":
            R["corrected"]["oracle_share_Kinf"] = {str(h): float((err_p[sel, h - 1].mean() - ((fut - yc) ** 2)[sel, h - 1].mean() + v_k[sel, h - 1].mean() / K) / (err_p[sel, h - 1].mean() - err_o[sel, h - 1].mean())) for h in HS}
        if anti:
            # mirror estimator: row i + n/2 is the mirror of row i (and vice versa)
            mi = (np.arange(n) + n // 2) % n
            d_even = 0.5 * (dep_raw + dep_raw[mi]); ym = yh - d_even
            R["mirror"] = block(ym - last, (fut - ym) ** 2, err_p, 0.25 * (mv + mv[mi]), sig, mask, groups, err_o)
            R["even_share"] = {str(h): float(np.mean(d_even[sel, h - 1] ** 2) / np.mean(dep_raw[sel, h - 1] ** 2)) for h in HS}
            # noise floor of the sign-blind part: the SE of the even part's mean over the pairs
            R["mirror_pair_se"] = {str(h): float(d_even[sel, h - 1].std() / np.sqrt(sel.sum()) / sigma) for h in HS}
            # the paired identity, for the record: on the full antithetic set the mirror's risk
            # reduction equals the mean squared even part exactly
            R["paired_identity_check"] = {str(h): {"risk_drop_full_set": float(((fut - yh) ** 2)[:, h - 1].mean() - ((fut - ym) ** 2)[:, h - 1].mean()),
                                                    "mean_even_sq": float((d_even[:, h - 1] ** 2).mean())} for h in HS}
        # systematic share of the imported term, and the finite-K threshold 1/(K+1)
        R["systematic_share"] = {str(h): {"naive": float(np.mean(dhat[sel, h - 1] ** 2) / np.mean(dep_raw[sel, h - 1] ** 2)),
                                          "debiased": float((np.mean(dhat[sel, h - 1] ** 2) - np.mean(v_k[sel, h - 1]) / K) / np.mean(dep_raw[sel, h - 1] ** 2)),
                                          "threshold": 1.0 / (K + 1)} for h in HS}
        # leave-one-out on the copies
        loo = np.stack([dep_sur[:, k] - (dep_sur.sum(1) - dep_sur[:, k]) / (K - 1) for k in range(K)], 1)  # (n, K, H)
        R["loo_mean_dep"] = {str(h): float(loo[sel, :, h - 1].mean() / sigma) for h in HS}
        R["loo_mean_dep_se"] = {str(h): float(loo[sel, :, h - 1].mean(1).std() / np.sqrt(sel.sum()) / sigma) for h in HS}
        R["prior_mean_dep"] = {str(h): float(dhat[sel, h - 1].mean() / sigma) for h in HS}
        if anti:
            # the mirror's gain in skill per cell, with a bootstrap SE of the paired difference (same
            # series, raw minus mirror error): the count of cells helped / hurt / unchanged in the paper
            e_raw, e_mir = ((fut - yh) ** 2)[sel], ((fut - ym) ** 2)[sel]; ep = err_p[sel]
            R["mirror_gain"] = {}
            for h in HS:
                d = (e_raw[:, h - 1] - e_mir[:, h - 1]) / ep[:, h - 1].mean()
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)]
                R["mirror_gain"][str(h)] = {"gain": float(d.mean()), "se": float(np.std(bs))}
        S[f"ladder|{rung}"] = R
        mr = R.get("mirror")
        print(f"{LABEL[mname]:14s} {rung:6s} h=16: dep {R['raw']['16']['mean_dep']:+.2f} | sign-avg {R['corrected']['16']['mean_dep']:+.2f} skill {R['raw']['16']['skill']:+.3f}->{R['corrected']['16']['skill']:+.3f} (K->inf {R['corrected']['16']['skill_Kinf']:+.3f})"
              + (f" | mirror skill ->{mr['16']['skill']:+.3f} even-share16 {R['even_share']['16']:.2f}" if mr else "")
              + (f" | N4 oracle share h=1 raw {R['raw']['1']['oracle_share']:+.2f} sign-avg {R['corrected']['1']['oracle_share']:+.2f}" + (f" mirror {mr['1']['oracle_share']:+.2f}" if mr else "") if rung == "N4" else ""))

# ------------------------------------------------------------------ real anchor
if os.path.exists(WIN):
    z = np.load(WIN); meta = json.load(open(WIN.replace(".npz", "_meta.json")))
    fam = np.array([m["family"] for m in meta]); post = np.array([m["fut_end"] >= "2025-01-01" for m in meta])
    cl_ids = {c: i for i, c in enumerate(sorted({f"{m['family']}|{m['fut_end']}" for m in meta}))}
    cl = np.array([cl_ids[f"{m['family']}|{m['fut_end']}"] for m in meta])
    ctx, fut, sig = z["ctx_raw"], z["fut_raw"], z["rsigma_raw"]
    ctx_s0, fut_s0, sig_s0 = z["ctx_sur"], z["fut_sur"], z["rsigma_sur"]
    if ctx_s0.ndim == 2:
        ctx_s0, fut_s0, sig_s0 = ctx_s0[:, None], fut_s0[:, None], sig_s0[:, None]
    n, K0 = ctx_s0.shape[:2]
    summary["real"] = {"n_windows": int(n), "K_windows_file": int(K0), "K_by_model": {}, "n_clusters": int(len(cl_ids)), "families": {f: int((fam == f).sum()) for f in ("fx", "eq")}, "n_post2025": int(post.sum())}
    SPLITS = (("all", np.ones(n, bool)), ("fx", fam == "fx"), ("eq", fam == "eq"), ("post2025", post))
    for mname in args.models.split(","):
        path = os.path.join(REAL, f"real_{mname}.npz")
        if not os.path.exists(path):
            print("missing", path); continue
        r = np.load(path); S = summary["models"].setdefault(mname, {})
        yh, mv = r["yhat_raw"].astype(np.float64), r["mcvar_raw"].astype(np.float64)
        ys, ms = r["yhat_sur"].astype(np.float64), r["mcvar_sur"].astype(np.float64)
        if ys.ndim == 2:
            ys, ms = ys[:, None], ms[:, None]
        # a model file may carry fewer copies than the windows file (K=4 against a K=16 file): the
        # first copies of both files are the same draws, so truncate per model
        K = min(ys.shape[1], K0); ctx_s, fut_s, sig_s = ctx_s0[:, :K], fut_s0[:, :K], sig_s0[:, :K]
        ys, ms = ys[:, :K], ms[:, :K]
        summary["real"]["K_by_model"][mname] = int(K)
        last = ctx[:, -1:]; last_s = ctx_s[:, :, -1:]
        rel_raw = yh / last - 1.0                                                                            # (n, H) departure in return units
        rel_sur = ys / last_s - 1.0                                                                          # (n, K, H)
        dhat_rel = rel_sur.mean(1)
        rel_c = rel_raw - dhat_rel; yc = last * (1.0 + rel_c)
        mv_rel, mv_rel_c = mv / last ** 2, mv / last ** 2 + (ms / last_s ** 2).sum(1) / K ** 2
        fut_rel = fut / last - 1.0
        err_p = fut_rel ** 2
        v_k = rel_sur.var(1, ddof=1)
        mpath = os.path.join(REAL, f"real_{mname}_mirror.npz")
        if os.path.exists(mpath):
            rm_ = np.load(mpath); ym_, mvm_ = rm_["yhat_mirror"].astype(np.float64), rm_["mcvar_mirror"].astype(np.float64)
            rr = ctx[:, 1:] / ctx[:, :-1] - 1.0
            last_m = (ctx[:, :1] * np.cumprod(1.0 - rr, axis=1))[:, -1:]                                   # the mirror's last value
            rel_mir = ym_ / last_m - 1.0
            d_even = 0.5 * (rel_raw + rel_mir); rel_m = rel_raw - d_even
            mv_rel_m = 0.25 * (mv / last ** 2 + mvm_ / last_m ** 2)
            mu_hat = rr.mean(1, keepdims=True) * np.arange(1, rel_raw.shape[1] + 1)[None, :]          # context mean return x h: the estimated drift
            gate = (d_even * (2.0 * mu_hat - d_even) < 0.0)                                           # (v): removing the even part pays iff the drift is below half of it
            rel_g = np.where(gate, rel_m, rel_raw); mv_rel_g = np.where(gate, mv_rel_m, mv_rel)
        for sname, mask in SPLITS:
            S[f"real|{sname}"] = {"raw": block(rel_raw, (fut_rel - rel_raw) ** 2, err_p, mv_rel, sig, mask, cl),
                                  "corrected": block(rel_c, (fut_rel - rel_c) ** 2, err_p, mv_rel_c, sig, mask, cl, noise=v_k / K),
                                  "prior_mean_dep": {str(h): float(np.mean(dhat_rel[mask, h - 1] / sig[mask])) for h in HS},
                                  "realised": {str(h): float(np.mean((fut[mask, h - 1] / last[mask, 0] - 1.0) / sig[mask])) for h in HS}}
            if os.path.exists(mpath):
                S[f"real|{sname}"]["mirror"] = block(rel_m, (fut_rel - rel_m) ** 2, err_p, mv_rel_m, sig, mask, cl)
                S[f"real|{sname}"]["gated"] = block(rel_g, (fut_rel - rel_g) ** 2, err_p, mv_rel_g, sig, mask, cl)
                S[f"real|{sname}"]["gate_rate"] = {str(h): float(gate[mask, h - 1].mean()) for h in HS}
                S[f"real|{sname}"]["even_share"] = {str(h): float(np.mean(d_even[mask, h - 1] ** 2) / np.mean(rel_raw[mask, h - 1] ** 2)) for h in HS}
                S[f"real|{sname}"]["mirror_mean_dep"] = {str(h): float(np.mean(rel_mir[mask, h - 1] / sig[mask])) for h in HS}
                # Proposition prop:sas (v), the exact identity with no assumption on the law, on the raw windows:
                #   Risk(mirror) - Risk(raw) = -E[Dbar^2] + 2 E[m Dbar] - 2 E[(D - Dbar) Dbar],
                # here in skill units (divided by the persistence MSE) and with the realised move y_{t+h}/y_t - 1 in
                # place of the conditional drift m, of which it is an unbiased proxy:
                #   dskill = gain + drift + cross,  gain = E[Dbar^2]/P,  drift = -2 E[(y_{t+h}/y_t-1) Dbar]/P,
                #   cross = +2 E[(D - Dbar) Dbar]/P.  The identity is exact on the sample (asserted below, before
                # the Monte Carlo variance correction that block() applies to the sampled models).
                dec = {}
                for h in HS:
                    d_, D_, y_ = d_even[mask, h - 1], rel_raw[mask, h - 1], fut_rel[mask, h - 1]
                    P = np.mean(y_ ** 2)
                    gain, drift, cross = float(np.mean(d_ ** 2) / P), float(-2.0 * np.mean(y_ * d_) / P), float(2.0 * np.mean((D_ - d_) * d_) / P)
                    dskill = float((np.mean((y_ - D_) ** 2) - np.mean((y_ - (D_ - d_)) ** 2)) / P)
                    assert abs(dskill - (gain + drift + cross)) < 1e-9, (mname, sname, h, dskill, gain + drift + cross)
                    # cluster-bootstrap standard error of the paired change (same calendar clusters as block()):
                    _ds = lambda sel: float((np.mean((y_[sel] - D_[sel]) ** 2) - np.mean((y_[sel] - (D_[sel] - d_[sel])) ** 2)) / np.mean(y_[sel] ** 2))
                    dskill_se = boot_own(_ds, cl[mask], np.arange(int(mask.sum())), seed=h * 1000 + len(sname))
                    ann = 252.0 / h
                    dec[str(h)] = {"gain": gain, "drift": drift, "cross": cross, "dskill": dskill, "dskill_se": dskill_se,
                                   "prior_mean_ret": float(np.mean(d_)),                     # mean even part, return units
                                   "prior_annualised": float(np.mean(d_) * ann),             # the sign-blind prior as a drift per year
                                   "breakeven_annualised": float(0.5 * np.mean(d_) * ann),   # (v): removing it helps iff the true drift is below half
                                   "realised_annualised": float(np.mean(y_) * ann)}          # the split's realised move per year over these windows
                S[f"real|{sname}"]["decomp"] = dec
        S["real|all"]["systematic_share"] = {str(h): {"naive": float(np.mean(dhat_rel[:, h - 1] ** 2) / np.mean(rel_raw[:, h - 1] ** 2)),
                                                      "debiased": float((np.mean(dhat_rel[:, h - 1] ** 2) - np.mean(v_k[:, h - 1]) / K) / np.mean(rel_raw[:, h - 1] ** 2)),
                                                      "threshold": 1.0 / (K + 1)} for h in HS}
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        # mirror of every surrogate copy, when that pass was run: the null test of the mirror correction on real increments
        mspath = os.path.join(REAL, f"real_{mname}_mirror_sur.npz")
        if os.path.exists(mspath):
            rms_ = np.load(mspath); yms_, mvms_ = rms_["yhat_mirror_sur"].astype(np.float64)[:, :K], rms_["mcvar_mirror_sur"].astype(np.float64)[:, :K]
            rr_s = ctx_s[:, :, 1:] / ctx_s[:, :, :-1] - 1.0
            last_ms = (ctx_s[:, :, :1] * np.cumprod(1.0 - rr_s, axis=2))[:, :, -1:]
            rel_ms = yms_ / last_ms - 1.0
            d_even_s = 0.5 * (rel_sur + rel_ms); rel_ms_c = rel_sur - d_even_s
            mv_ms_c = flat(0.25 * (ms / last_s ** 2 + mvms_ / last_ms ** 2))
        # leave-one-out null on the surrogates (each copy corrected by the other K-1), pooled over n x K
        if K >= 2:
            loo_rel = np.stack([rel_sur[:, k] - (rel_sur.sum(1) - rel_sur[:, k]) / (K - 1) for k in range(K)], 1)
            yc_s = last_s * (1.0 + loo_rel)
            fut_f, last_f = flat(fut_s), flat(last_s)
            sig_f, cl_f, fam_f, post_f = sig_s.reshape(-1), np.repeat(cl, K), np.repeat(fam, K), np.repeat(post, K)
            fut_rel_s = fut_f / last_f - 1.0
            err_p_s = fut_rel_s ** 2
            mv_s, mv_s_c = flat(ms / last_s ** 2), flat(ms / last_s ** 2) * (1 + 1.0 / (K - 1))
            for sname, mask in (("all", np.ones(n * K, bool)), ("fx", fam_f == "fx"), ("eq", fam_f == "eq"), ("post2025", post_f)):
                noise_s = flat(np.repeat(v_k[:, None, :], K, axis=1) / (K - 1))
                S[f"realsur|{sname}"] = {"raw": block(flat(rel_sur), (fut_rel_s - flat(rel_sur)) ** 2, err_p_s, mv_s, sig_f, mask, cl_f),
                                         "corrected": block(flat(loo_rel), (fut_rel_s - flat(loo_rel)) ** 2, err_p_s, mv_s_c, sig_f, mask, cl_f, noise=noise_s)}
                if os.path.exists(mspath):
                    S[f"realsur|{sname}"]["mirror"] = block(flat(rel_ms_c), (fut_rel_s - flat(rel_ms_c)) ** 2, err_p_s, mv_ms_c, sig_f, mask, cl_f)
                    S[f"realsur|{sname}"]["even_share"] = {str(h): float(np.mean(flat(d_even_s)[mask, h - 1] ** 2) / np.mean(flat(rel_sur)[mask, h - 1] ** 2)) for h in HS}
        a, b = S["real|all"], S.get("realsur|all")
        print(f"{LABEL[mname]:14s} real  h=128: dep {a['raw']['128']['mean_dep']:+.2f}->{a['corrected']['128']['mean_dep']:+.2f}  skill128 {a['raw']['128']['skill']:+.3f}->{a['corrected']['128']['skill']:+.3f}  skill16 {a['raw']['16']['skill']:+.3f}->{a['corrected']['16']['skill']:+.3f}"
              + (f" | mirror: dep128 {a['mirror']['128']['mean_dep']:+.2f} skill128 {a['mirror']['128']['skill']:+.3f} skill16 {a['mirror']['16']['skill']:+.3f} even-share128 {a['even_share']['128']:.2f}" if "mirror" in a else "")
              + (f" | sur mirror: skill16 {b['raw']['16']['skill']:+.3f}->{b['mirror']['16']['skill']:+.3f} skill128 {b['raw']['128']['skill']:+.3f}->{b['mirror']['128']['skill']:+.3f} even-share128 {b['even_share']['128']:.2f}" if b and "mirror" in b else "")
              + (f" | sur LOO dep128 {b['raw']['128']['mean_dep']:+.2f}->{b['corrected']['128']['mean_dep']:+.2f} skill16 {b['raw']['16']['skill']:+.3f}->{b['corrected']['16']['skill']:+.3f}" if b else ""))

# counts the text quotes: how many models the mirror correction helps on the raw windows, per split and horizon
if "real" in summary:
    cnt = {}
    for sname in ("all", "fx", "eq", "post2025"):
        for h in HS:
            ds_ = [summary["models"][m][f"real|{sname}"]["decomp"][str(h)] for m in summary["models"] if "decomp" in summary["models"][m].get(f"real|{sname}", {})]
            vals = [d["dskill"] for d in ds_]
            # "gt_2se": the change exceeds twice its cluster-bootstrap standard error (an approximate 95% interval excluding zero)
            cnt[f"{sname}|{h}"] = {"models": len(vals), "helped": int(sum(v > 0 for v in vals)), "hurt": int(sum(v < 0 for v in vals)),
                                   "helped_gt_2se": int(sum(d["dskill"] > 2 * d["dskill_se"] for d in ds_)), "hurt_gt_2se": int(sum(d["dskill"] < -2 * d["dskill_se"] for d in ds_))}
    summary["real"]["mirror_raw_counts"] = cnt
    print("mirror on raw windows, models helped/hurt:", {k: (v["helped"], v["hurt"]) for k, v in cnt.items() if k.endswith("|16") or k.endswith("|128")})

# ------------------------------------------------------------------ tables
os.makedirs(args.tables, exist_ok=True)
def pm(v, se, d=2): return f"${v:+.{d}f}\\pm{se:.{d}f}$"
def arrow(a, b, d=2): return f"${a:+.{d}f}\\to{b:+.{d}f}$"
ms_ = [m for m in args.models.split(",") if m in summary["models"]]

# ------------------------------------------------------------------ two classical baselines, same metrics
# Drift: the context's mean increment (returns on the anchor) extrapolated; AR(1): an AR(1) with intercept
# fitted by least squares to the context's increments (returns) and iterated. Both are fitted per window,
# see exactly what the models see, and are odd under the mirror (negating the increments negates the
# forecast), so the mirror correction leaves them unchanged up to the compounding of returns: they show
# where a forecaster that reads the context's drift lands on the same columns as the foundation models.
def _fit_ar1(d):                                             # d (n, T-1): per-row OLS of d_t on d_{t-1} with intercept
    x, y = d[:, :-1], d[:, 1:]
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    phi = ((x - xm) * (y - ym)).sum(1) / np.maximum(((x - xm) ** 2).sum(1), 1e-30)
    c = ym[:, 0] - phi * xm[:, 0]
    return c, phi
def baseline_forecast(kind, ctx, H, multiplicative):
    inc = ctx[:, 1:] / ctx[:, :-1] - 1.0 if multiplicative else np.diff(ctx, axis=1)
    if kind == "drift":
        step = np.repeat(inc.mean(1, keepdims=True), H, axis=1)
    else:
        c, phi = _fit_ar1(inc); step = np.empty((len(ctx), H)); prev = inc[:, -1]
        for h in range(H):
            prev = c + phi * prev; step[:, h] = prev
    last = ctx[:, -1:]
    return last * np.cumprod(1.0 + step, axis=1) if multiplicative else last + np.cumsum(step, axis=1)
BASELINES = {"drift": "Drift (fit)", "ar1": "AR(1) (fit)"}
summary["baselines"] = {}
_ref = next((m for m in args.models.split(",") if os.path.exists(os.path.join(args.res, f"remedy_{m}.npz"))), None)
for kind, lab in BASELINES.items():
    B = {}
    if _ref:
        zl = np.load(os.path.join(args.res, f"remedy_{_ref}.npz")); infl = json.load(open(os.path.join(args.res, f"remedy_{_ref}.json")))
        for rung in ("N1", "N4"):
            if rung not in infl["rungs"]: continue
            ctxl, futl, sigma_l = zl[f"{rung}/ctx"], zl[f"{rung}/fut"], float(zl[f"{rung}/sigma"]); H_l = futl.shape[1]; n_l = len(ctxl)
            yb = baseline_forecast(kind, ctxl, H_l, False); last_l = ctxl[:, -1:]
            anti_l = bool(infl["rungs"][rung].get("antithetic"))
            zeros = np.zeros_like(yb); sig_l = np.full(n_l, sigma_l); grp = np.arange(n_l); msk = (np.arange(n_l) < n_l // 2) if anti_l else np.ones(n_l, bool)   # unpaired half, as for the models
            err_p = (futl - last_l) ** 2; err_o = (futl - n4_exact_oracle(ctxl, sigma_l, zl[f"{rung}/oracle"], H_l)) ** 2 if rung == "N4" else None
            R = {"n": int(msk.sum()), "raw": block(yb - last_l, (futl - yb) ** 2, err_p, zeros, sig_l, msk, grp, err_o)}
            if anti_l:
                mi = (np.arange(n_l) + n_l // 2) % n_l; dep = yb - last_l
                d_even = 0.5 * (dep + dep[mi]); ym = yb - d_even
                R["mirror"] = block(ym - last_l, (futl - ym) ** 2, err_p, zeros, sig_l, msk, grp, err_o)
                R["even_share"] = {str(h): float(np.mean(d_even[msk, h - 1] ** 2) / max(np.mean(dep[msk, h - 1] ** 2), 1e-300)) for h in HS}
            B[f"ladder|{rung}"] = R
    if os.path.exists(WIN):
        zw = np.load(WIN); metaw = json.load(open(WIN.replace(".npz", "_meta.json")))
        clw_ids = {c: i for i, c in enumerate(sorted({f"{m['family']}|{m['fut_end']}" for m in metaw}))}
        clw = np.array([clw_ids[f"{m['family']}|{m['fut_end']}"] for m in metaw])
        ctxw, futw, sigw = zw["ctx_raw"], zw["fut_raw"], zw["rsigma_raw"]; ctxs, futs, sigs = zw["ctx_sur"], zw["fut_sur"], zw["rsigma_sur"]
        if ctxs.ndim == 2: ctxs, futs, sigs = ctxs[:, None], futs[:, None], sigs[:, None]
        nw, Kw, Tw = ctxs.shape; Hw = futw.shape[1]; lastw = ctxw[:, -1:]
        yb = baseline_forecast(kind, ctxw, Hw, True); rel_raw = yb / lastw - 1.0; fut_rel = futw / lastw - 1.0
        rr = ctxw[:, 1:] / ctxw[:, :-1] - 1.0; mirw = np.empty_like(ctxw); mirw[:, 0] = ctxw[:, 0]; mirw[:, 1:] = ctxw[:, :1] * np.cumprod(1.0 - rr, axis=1)
        ybm = baseline_forecast(kind, mirw, Hw, True); rel_mir = ybm / mirw[:, -1:] - 1.0
        d_even = 0.5 * (rel_raw + rel_mir); rel_c = rel_raw - d_even
        zeros = np.zeros_like(rel_raw); msk = np.ones(nw, bool)
        B["real|all"] = {"raw": block(rel_raw, (fut_rel - rel_raw) ** 2, fut_rel ** 2, zeros, sigw, msk, clw),
                         "mirror": block(rel_c, (fut_rel - rel_c) ** 2, fut_rel ** 2, zeros, sigw, msk, clw),
                         "even_share": {str(h): float(np.mean(d_even[:, h - 1] ** 2) / np.mean(rel_raw[:, h - 1] ** 2)) for h in HS}}
        flatc = ctxs.reshape(nw * Kw, Tw); flatf = futs.reshape(nw * Kw, Hw); lasts = flatc[:, -1:]
        ybs = baseline_forecast(kind, flatc, Hw, True); rel_s = ybs / lasts - 1.0; fut_rel_s = flatf / lasts - 1.0
        rrs = flatc[:, 1:] / flatc[:, :-1] - 1.0; mirs = np.empty_like(flatc); mirs[:, 0] = flatc[:, 0]; mirs[:, 1:] = flatc[:, :1] * np.cumprod(1.0 - rrs, axis=1)
        rel_ms = baseline_forecast(kind, mirs, Hw, True) / mirs[:, -1:] - 1.0; rel_sc = rel_s - 0.5 * (rel_s + rel_ms)
        sig_f = sigs.reshape(nw * Kw); cl_f = np.repeat(clw, Kw); mskf = np.ones(nw * Kw, bool); zerosf = np.zeros_like(rel_s)
        B["realsur|all"] = {"raw": block(rel_s, (fut_rel_s - rel_s) ** 2, fut_rel_s ** 2, zerosf, sig_f, mskf, cl_f),
                            "mirror": block(rel_sc, (fut_rel_s - rel_sc) ** 2, fut_rel_s ** 2, zerosf, sig_f, mskf, cl_f)}
    summary["baselines"][kind] = B
    print(f"baseline {lab:20s}: " + " | ".join(f"{k} skill16 {v['raw']['16']['skill']:+.3f}->{v['mirror']['16']['skill']:+.3f}" for k, v in B.items() if "mirror" in v and "16" in v["raw"])
          + (f" | N4 share {B['ladder|N4']['raw']['1']['oracle_share']:+.2f}->{B['ladder|N4']['mirror']['1']['oracle_share']:+.2f}" if "ladder|N4" in B and "mirror" in B["ladder|N4"] else ""))
# main table: raw / mirror-corrected / sign-averaged (K) for N1 skill at h=16, N4 oracle share at h=1, real raw windows skill at h=128
def tri(a, b, c, d=3):
    f = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}f}"
    return f"${f(a)}$ & ${f(b)}$ & ${f(c)}$"
L = [r"\begin{tabular}{l|ccc|ccc|ccc}", r"\toprule",
     r" & \multicolumn{3}{c|}{N1, skill $h{=}16$} & \multicolumn{3}{c|}{N4, oracle share $h{=}1$} & \multicolumn{3}{c}{real martingale, skill $h{=}128$} \\",
     r"Model & raw & mirror & sign-avg. & raw & mirror & sign-avg. & raw & mirror & sign-avg. \\", r"\midrule"]
for m in ms_:
    S = summary["models"][m]; n1, n4, ra = S.get("ladder|N1"), S.get("ladder|N4"), S.get("realsur|all")
    g = lambda blk, key, h, f: (blk[key][h][f] if blk and key in blk else None)
    cells = [LABEL[m],
             tri(g(n1, "raw", "16", "skill"), g(n1, "mirror", "16", "skill"), g(n1, "corrected", "16", "skill")),
             tri(g(n4, "raw", "1", "oracle_share"), g(n4, "mirror", "1", "oracle_share"), g(n4, "corrected", "1", "oracle_share"), 2),
             tri(g(ra, "raw", "128", "skill"), g(ra, "mirror", "128", "skill"), g(ra, "corrected", "128", "skill"))]
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "remedy.tex"), "w").write("\n".join(L) + "\n")
# detail tables, split so that each fits the text width: (A) ladder skills, (B) real windows
def row6(lab, blk, hs=("16", "128"), field="skill", d=3):
    cells = []
    for h in hs:
        for est in ("raw", "mirror", "corrected"):
            cells.append(pm(blk[est][h][field], blk[est][h][field + "_se"], d) if est in blk else "--")
    return lab + " & " + " & ".join(cells) + r" \\"
D = [r"\begin{tabular}{ll|rrr|rrr}", r"\toprule",
     r" & & \multicolumn{3}{c|}{skill $h{=}16$} & \multicolumn{3}{c}{skill $h{=}128$} \\",
     r"Model & rung & raw & mirror & sign-avg. & raw & mirror & sign-avg. \\", r"\midrule"]
for m in ms_:
    S = summary["models"][m]; first = True
    for key, lab in (("ladder|N1", "N1"), ("ladder|N2", "N2"), ("ladder|N3_nu5", "N3"), ("ladder|N4", "N4")):
        if key not in S: continue
        D.append(row6(f"{LABEL[m] if first else ''} & {lab}", S[key])); first = False
    D.append(r"\midrule")
D[-1] = r"\bottomrule"; D.append(r"\end{tabular}")
open(os.path.join(args.tables, "remedy_detail.tex"), "w").write("\n".join(D) + "\n")
E = [r"\begin{tabular}{ll|rrr|rrr}", r"\toprule",
     r" & & \multicolumn{3}{c|}{mean departure / $\sigma_r$, $h{=}128$} & \multicolumn{3}{c}{skill $h{=}128$} \\",
     r"Model & windows & raw & mirror & sign-avg. & raw & mirror & sign-avg. \\", r"\midrule"]
for m in ms_:
    S = summary["models"][m]; first = True
    for key, lab in (("realsur|all", "martingale"), ("real|all", "raw, all"), ("real|fx", "raw, FX"), ("real|eq", "raw, equity"), ("real|post2025", "raw, post-2025")):
        if key not in S: continue
        blk = S[key]
        cells = [pm(blk[est]["128"]["mean_dep"], blk[est]["128"]["mean_dep_se"]) if est in blk else "--" for est in ("raw", "mirror", "corrected")]
        cells += [pm(blk[est]["128"]["skill"], blk[est]["128"]["skill_se"], 3) if est in blk else "--" for est in ("raw", "mirror", "corrected")]
        E.append(f"{LABEL[m] if first else ''} & {lab} & " + " & ".join(cells) + r" \\"); first = False
    E.append(r"\midrule")
E[-1] = r"\bottomrule"; E.append(r"\end{tabular}")
open(os.path.join(args.tables, "remedy_detail_real.tex"), "w").write("\n".join(E) + "\n")
# main-text table: anchor + correction in one, raw -> mirror per cell (the sign-averaged variant and all splits are in the appendix tables)
SHORTLAB = {"Chronos-small": "Chronos-T5", "Moirai-small": "Moirai-1.1", "Time-MoE-200M": "Time-MoE"}   # width of the main table
G = [r"\begin{tabular}{l|rrr|r|rr}", r"\toprule",
     r" & \multicolumn{3}{c|}{real-increment martingale} & raw windows & \multicolumn{2}{c}{synthetic ladder} \\",
     r"Model & mean $h{=}128$ & skill $h{=}16$ & skill $h{=}128$ & skill $h{=}128$ & N1 skill $h{=}16$ & N4 share $h{=}1$ \\",
     r" & $/\sigma_r$ & raw$\to$mirror & raw$\to$mirror & raw$\to$mirror & raw$\to$mirror & raw$\to$mirror \\", r"\midrule"]
_rsj = os.path.join(args.res, "real_summary.json")          # the anchor's own bootstrap (seed 7) for the martingale columns, so the SEs match tab:real-detail
_rs = json.load(open(_rsj))["models"] if os.path.exists(_rsj) else {}
for m in ms_:
    S = summary["models"][m]; ms, rw, n1, n4 = S.get("realsur|all"), S.get("real|all"), S.get("ladder|N1"), S.get("ladder|N4")
    if not (ms and rw and n1 and n4 and "mirror" in ms and "mirror" in rw): continue
    _a = _rs.get(m, {}).get("sur|all", {}).get("128", ms["raw"]["128"])
    G.append(" & ".join([SHORTLAB.get(LABEL[m], LABEL[m]), f"${_a['mean_dep']:+.2f}$",
                         arrow(ms["raw"]["16"]["skill"], ms["mirror"]["16"]["skill"], 3), arrow(ms["raw"]["128"]["skill"], ms["mirror"]["128"]["skill"], 3),
                         arrow(rw["raw"]["128"]["skill"], rw["mirror"]["128"]["skill"], 3),
                         arrow(n1["raw"]["16"]["skill"], n1["mirror"]["16"]["skill"], 3), f"${n4['raw']['1']['oracle_share']:.2f}\\to{n4['mirror']['1']['oracle_share']:.2f}$"]) + r" \\")
for kind, lab in BASELINES.items():
    B = summary["baselines"].get(kind, {}); ms, rw, n1, n4 = B.get("realsur|all"), B.get("real|all"), B.get("ladder|N1"), B.get("ladder|N4")
    if not (ms and rw and n1 and n4 and "mirror" in n1): continue
    if kind == "drift": G.append(r"\midrule")
    G.append(" & ".join([lab, f"${ms['raw']['128']['mean_dep']:+.2f}$",
                         arrow(ms["raw"]["16"]["skill"], ms["mirror"]["16"]["skill"], 3), arrow(ms["raw"]["128"]["skill"], ms["mirror"]["128"]["skill"], 3),
                         arrow(rw["raw"]["128"]["skill"], rw["mirror"]["128"]["skill"], 3),
                         arrow(n1["raw"]["16"]["skill"], n1["mirror"]["16"]["skill"], 3), f"${n4['raw']['1']['oracle_share']:.2f}\\to{n4['mirror']['1']['oracle_share']:.2f}$"]) + r" \\")
G += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "main.tex"), "w").write("\n".join(G) + "\n")
# (v) decomposition of the mirror correction's skill change on the raw windows at h=128, FX and equity
F = [r"\begin{tabular}{l|rrrr|rr|rrrr|rr}", r"\toprule",
     r" & \multicolumn{6}{c|}{raw FX windows, $h{=}128$} & \multicolumn{6}{c}{raw equity windows, $h{=}128$} \\",
     r"Model & $\Delta$skill & gain & drift & cross & prior & break-even & $\Delta$skill & gain & drift & cross & prior & break-even \\", r"\midrule"]
for m in ms_:
    S = summary["models"][m]; cells = [LABEL[m]]
    for key in ("real|fx", "real|eq"):
        d = S.get(key, {}).get("decomp", {}).get("128")
        if d is None:
            cells += ["--"] * 6; continue
        cells += [f"${d['dskill']:+.3f}\\pm{d['dskill_se']:.3f}$", f"${d['gain']:+.3f}$", f"${d['drift']:+.3f}$", f"${d['cross']:+.3f}$",
                  f"${100 * d['prior_annualised']:+.1f}\\%$", f"${100 * d['breakeven_annualised']:+.1f}\\%$"]
    F.append(" & ".join(cells) + r" \\")
F += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "remedy_decomp.tex"), "w").write("\n".join(F) + "\n")
# the gated estimator (Algorithm alg:mirror with the drift estimated from the context) against raw and mirror
Q = [r"\begin{tabular}{l|rrr|rrr|rrr|r}", r"\toprule",
     r" & \multicolumn{3}{c|}{all windows} & \multicolumn{3}{c|}{FX} & \multicolumn{3}{c|}{equities} & gate \\",
     r"Model & raw & mirror & gated & raw & mirror & gated & raw & mirror & gated & applied \\", r"\midrule"]
gate_counts = {"all": {"gated_ge_raw": 0, "gated_ge_mirror": 0, "n": 0}, "fx": {"gated_ge_raw": 0, "gated_ge_mirror": 0, "n": 0}, "eq": {"gated_ge_raw": 0, "gated_ge_mirror": 0, "n": 0}}
for m in ms_:
    S = summary["models"][m]; cells = [LABEL[m]]
    for key in ("real|all", "real|fx", "real|eq"):
        b = S.get(key, {})
        if "gated" not in b:
            cells += ["--"] * 3; continue
        r_, m_, g_ = b["raw"]["128"]["skill"], b["mirror"]["128"]["skill"], b["gated"]["128"]["skill"]
        cells += [f"${r_:+.3f}$", f"${m_:+.3f}$", f"${g_:+.3f}$"]
        gc = gate_counts[key.split("|")[1]]; gc["n"] += 1; gc["gated_ge_raw"] += g_ >= r_ - 1e-12; gc["gated_ge_mirror"] += g_ >= m_ - 1e-12
    cells.append(f"{S['real|all']['gate_rate']['128']:.2f}" if "gate_rate" in S.get("real|all", {}) else "--")
    Q.append(" & ".join(cells) + r" \\")
Q += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(args.tables, "remedy_gate.tex"), "w").write("\n".join(Q) + "\n")
summary["real"]["gate_counts_h128"] = gate_counts
# ladder cell counts: does the mirror lower the risk on futures it has not seen? (unpaired half, every model, rung, horizon)
cnt = {"cells": 0, "better": 0, "worse": 0, "unchanged": 0, "better_gt_1se": 0, "worse_gt_1se": 0, "worse_gt_1se_cells": [],
       "N1_h16_better": [], "N1_h16_worse": [], "N1_h16_unchanged": [], "N4_h1_share_up": [], "N4_h1_share_down": [], "N4_h1_share_unchanged": []}
for mname, S in summary["models"].items():
    for key, R in S.items():
        if not key.startswith("ladder|") or "mirror_gain" not in R: continue
        for h, g in R["mirror_gain"].items():
            cnt["cells"] += 1
            if abs(g["gain"]) < 1e-6: cnt["unchanged"] += 1
            elif g["gain"] > 0:
                cnt["better"] += 1; cnt["better_gt_1se"] += int(g["gain"] > g["se"])
            else:
                cnt["worse"] += 1
                if -g["gain"] > g["se"]: cnt["worse_gt_1se"] += 1; cnt["worse_gt_1se_cells"].append([mname, key.split("|")[1], int(h), round(g["gain"], 4), round(g["se"], 4)])
    g16 = S["ladder|N1"]["mirror_gain"]["16"]["gain"] if "ladder|N1" in S and "mirror_gain" in S["ladder|N1"] else None
    if g16 is not None: cnt["N1_h16_better" if g16 > 1e-6 else "N1_h16_worse" if g16 < -1e-6 else "N1_h16_unchanged"].append(mname)
    if "ladder|N4" in S and "mirror" in S["ladder|N4"]:
        d4 = S["ladder|N4"]["mirror"]["1"]["oracle_share"] - S["ladder|N4"]["raw"]["1"]["oracle_share"]
        cnt["N4_h1_share_up" if d4 > 1e-6 else "N4_h1_share_down" if d4 < -1e-6 else "N4_h1_share_unchanged"].append(mname)
summary["ladder_counts"] = cnt
print("ladder cells (unpaired half): " + ", ".join(f"{k} {cnt[k]}" for k in ("cells", "better", "worse", "unchanged", "better_gt_1se", "worse_gt_1se"))
      + f" | N1 h16 better {len(cnt['N1_h16_better'])}, worse {cnt['N1_h16_worse']}, unchanged {cnt['N1_h16_unchanged']} | N4 share up {len(cnt['N4_h1_share_up'])}, down {cnt['N4_h1_share_down']}, unchanged {cnt['N4_h1_share_unchanged']}")
print("gated estimator, h=128:", gate_counts)
json.dump(summary, open(os.path.join(args.res, "remedy_summary.json"), "w"), indent=1)
print("wrote", os.path.join(args.res, "remedy_summary.json"), "remedy.tex, remedy_detail.tex, remedy_detail_real.tex, remedy_decomp.tex, remedy_gate.tex, main.tex")
