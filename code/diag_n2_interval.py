"""Remark 4.1 (rem:n2): on N2 the mean is unpredictable but the variance is not.
For a probabilistic model the fair test is whether its predictive interval
width tracks the conditional volatility, or is a function of horizon alone.

Chronos returns sampled trajectories, so the interval is read off the sample
quantiles. N2's generator exposes the true conditional volatility path, so the
comparison is against the quantity the model would need to have recovered.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from pilot_a import load_pipeline, forecast, log
from nulls import RUNGS

from paths import RESULTS_DIR

OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-t5-small"
    n, T, H, ns = 512, 512, 16, 100
    pipe = load_pipeline(model, "pretrained", "cuda")
    recs = []
    for seed in (5001, 5002):
        y, info = RUNGS["N2"](n, T, H, np.random.default_rng(seed))
        ctx, truth = y[:, :T], y[:, T:]
        cv = info["cond_vol"][:, T:]                      # true sigma_t over the horizon
        cv_last = info["cond_vol"][:, T - 1]              # last in-context sigma
        s = forecast(pipe, ctx, H, ns, batch=8, seed=0)
        lo, hi = np.quantile(s, [0.1, 0.9], axis=1)       # (n, H) each
        width = hi - lo                                   # 80% interval
        med = np.median(s, axis=1)
        cover = np.mean((truth >= lo) & (truth <= hi), axis=0)   # per-h coverage
        # Correlation of width with true conditional vol, per horizon, across series
        r_true = [float(np.corrcoef(width[:, h], cv[:, h])[0, 1]) for h in range(H)]
        # and with the last in-context vol, which is all the model could estimate
        r_last = [float(np.corrcoef(width[:, h], cv_last)[0, 1]) for h in range(H)]
        # Same test against the realised in-context vol proxy (abs returns, last 20)
        rv20 = np.sqrt(np.mean(np.diff(ctx[:, -21:], axis=1) ** 2, axis=1))
        r_rv = [float(np.corrcoef(width[:, h], rv20)[0, 1]) for h in range(H)]
        # How much of width variation across series is explained vs horizon alone
        w_h = width.mean(0)                                # horizon-only model
        resid = width - w_h[None, :]
        frac_series = float(np.var(resid) / np.var(width))
        rec = {"seed": seed, "n": n, "H": H,
               "coverage_80": cover.tolist(),
               "mean_width_over_sigma": (w_h / info["sigma"]).tolist(),
               "corr_width_true_condvol": r_true,
               "corr_width_last_condvol": r_last,
               "corr_width_realised_vol20": r_rv,
               "frac_width_var_not_explained_by_horizon": frac_series,
               "condvol_spread_max_over_min": float(cv_last.max() / cv_last.min()),
               "median_departure_sigma": float(np.sqrt(np.mean((med - ctx[:, -1:]) ** 2)) / info["sigma"])}
        recs.append(rec)
        log(f"seed {seed}: cov80 h1={cover[0]:.3f} h16={cover[-1]:.3f}  "
            f"corr(width, true sigma_t) h1={r_true[0]:+.3f} h16={r_true[-1]:+.3f}  "
            f"corr(width, last sigma) h1={r_last[0]:+.3f}  "
            f"width var not from horizon={frac_series:.3f}")
    fn = os.path.join(OUT, "diag_n2_interval.json")
    json.dump(recs, open(fn, "w"), indent=1)
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
