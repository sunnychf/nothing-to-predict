"""Is the mean squared departure driven by a few outlier series?

The mean of a squared quantity is fragile to outliers, and two seeds of the same
N1 configuration disagreed by 3.6x at h=1 while their horizon-averaged intervals
overlapped. If a handful of contexts produce wild forecasts and carry the
headline, that has to be said plainly rather than averaged away. This measures
the concentration directly and reports robust alternatives to the mean.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from pilot_a import load_pipeline, forecast, codec_floor, log
from nulls import RUNGS

from paths import RESULTS_DIR

OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-t5-small"
    n, T, H, ns = 512, 512, 16, 100
    pipe = load_pipeline(model, "pretrained", "cuda")
    recs = []
    for seed in (2001, 2002, 2003, 2004):
        y, info = RUNGS["N1"](n, T, H, np.random.default_rng(seed))
        ctx, truth = y[:, :512], y[:, 512:]
        s = forecast(pipe, ctx, H, ns, batch=16, seed=0)
        yhat = s.mean(1)
        sig = info["sigma"]
        d2 = ((yhat - ctx[:, -1:]) ** 2)[:, 0] / sig ** 2   # per-series, h=1
        order = np.sort(d2)[::-1]
        tot = d2.sum()
        rec = {
            "seed": seed, "n": n,
            "mean_sq": float(d2.mean()),
            "rms": float(np.sqrt(d2.mean())),
            "median_sq": float(np.median(d2)),
            "rms_of_median": float(np.sqrt(np.median(d2))),
            "q25": float(np.sqrt(np.quantile(d2, .25))),
            "q75": float(np.sqrt(np.quantile(d2, .75))),
            "q95": float(np.sqrt(np.quantile(d2, .95))),
            "max": float(np.sqrt(d2.max())),
            "share_top1pct": float(order[:max(1, n // 100)].sum() / tot),
            "share_top5pct": float(order[:max(1, n // 20)].sum() / tot),
            "trimmed_rms_95": float(np.sqrt(d2[d2 <= np.quantile(d2, .95)].mean())),
            "codec_floor_rms": float(np.sqrt(codec_floor(pipe, ctx, ctx[:, -1], H).mean()) / sig),
        }
        recs.append(rec)
        log(f"seed {seed}: rms={rec['rms']:.4f}  median={rec['rms_of_median']:.4f}  "
            f"trimmed95={rec['trimmed_rms_95']:.4f}  top1%share={rec['share_top1pct']:.3f}  "
            f"max={rec['max']:.2f}")
    with open(os.path.join(OUT, "diag_dispersion.json"), "w") as f:
        json.dump(recs, f, indent=1)
    log(f"wrote {OUT}/diag_dispersion.json")

if __name__ == "__main__":
    main()
