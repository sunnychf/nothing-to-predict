"""Deeper spectral analysis of the departure, kept separate from pilot_a.py so
it can be run without disturbing a job in flight.

Motivation. The first pass reported spectral flatness ~0.09 (not white) but a
median peak period equal to the whole forecast window, and 82 percent of the
power inside a band nominally labelled 'period 52'. That label is an artefact of
binning: at H=128 a +-1 bin window around period 52 spans bins 1-3, i.e. periods
128, 64 and 43, so it measures the lowest frequencies rather than any rhythm.

The distinction matters because Section 4.1 rests on it. A departure that is a
slow DRIFT is structured but is not an imported seasonality; a departure with
mass at 7 or 24 is. This script separates the two explicitly and saves the raw
departure paths so the claim can be re-examined without another GPU run.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from pilot_a import load_pipeline, forecast, codec_floor, log
from nulls import RUNGS

from paths import RESULTS_DIR

OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)


def decompose(dep, detrend_order=2):
    """Split departure power into drift, seasonal and high-frequency parts.

    dep: (n_series, H). Returns the mean periodogram and the shares. Bands are
    defined on the frequency grid itself, so nothing is attributed to a period
    the window cannot resolve.

    Two defences against the failure test_decompose.py caught. First, each
    series is detrended with a low-order polynomial before the periodogram, so
    a smooth drift does not spill power into every low bin. Second, the local
    continuum a seasonal band is judged against never includes the drift bins
    (period > H/4): a band that sits next to the drift must not be compared to
    the drift. The raw drift share is still reported from the undetrended
    periodogram, since that is the quantity that says how much of the
    departure is a slow wander.
    """
    n, H = dep.shape
    t = np.arange(H, dtype=float)
    x0 = dep - dep.mean(-1, keepdims=True)
    # drift share from the raw periodogram
    P0 = np.abs(np.fft.rfft(x0, axis=-1)) ** 2
    P0 = P0[:, 1:]
    freqs = np.fft.rfftfreq(H)[1:]
    periods = 1.0 / freqs
    Pm0 = P0.mean(0)
    drift = periods > H / 4
    # detrended periodogram for the band test
    V = np.vander(t, detrend_order + 1)            # polynomial basis
    coef, *_ = np.linalg.lstsq(V, dep.T, rcond=None)
    x = dep - (V @ coef).T
    x = x - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    P = P[:, 1:]
    Pm = P.mean(0)
    out = {
        "psd_raw": Pm0.tolist(),
        "psd_detrended": Pm.tolist(),
        "periods": periods.tolist(),
        "share_drift_periodGtH4": float(Pm0[drift].sum() / Pm0.sum()),
        "share_rest": float(Pm0[~drift].sum() / Pm0.sum()),
        "peak_period_of_mean_psd": float(periods[int(np.argmax(Pm0))]),
        "peak_period_detrended": float(periods[int(np.argmax(Pm))]),
        "detrend_order": detrend_order,
    }
    for p in (5, 7, 12, 24, 30):
        if p > H / 3:
            out[f"band{p}"] = None
            continue
        k = int(np.argmin(np.abs(periods - p)))
        lo, hi = max(0, k - 1), min(len(Pm), k + 2)
        band = np.zeros(len(Pm), bool); band[lo:hi] = True
        peak = Pm[lo:hi].mean()
        # continuum: nearby bins, excluding the band AND the drift bins; widen
        # on the high-frequency side if the low side has nothing usable
        w = 6
        nb = np.zeros(len(Pm), bool)
        nb[max(0, k - w):min(len(Pm), k + w + 1)] = True
        nb &= ~band & ~drift
        if nb.sum() < 4:
            nb[k + 2:min(len(Pm), k + 2 * w + 1)] = True
            nb &= ~band & ~drift
        cont = Pm[nb].mean() if nb.sum() else np.nan
        out[f"band{p}"] = {
            "period_of_bin": float(periods[k]),
            "share_of_total_detrended": float(Pm[lo:hi].sum() / Pm.sum()),
            "excess_over_continuum": float(peak / cont) if cont and np.isfinite(cont) else None,
            "n_continuum_bins": int(nb.sum()),
        }
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-t5-small"
    H = int(os.environ.get("SPEC_H", 128))
    n = int(os.environ.get("SPEC_N", 128))
    ns = int(os.environ.get("SPEC_NS", 100))
    os.makedirs(OUT, exist_ok=True)
    pipe = load_pipeline(model, "pretrained", "cuda")
    recs = []
    for rung in ["N1", "N2", "N4"]:
        rng = np.random.default_rng(4000)
        y, info = RUNGS[rung](n, 512, H, rng)
        ctx, truth = y[:, :512], y[:, 512:]
        s = forecast(pipe, ctx, H, ns, batch=16, seed=0)
        yhat = s.mean(1)
        dep = yhat - ctx[:, -1:]
        fl = codec_floor(pipe, ctx, ctx[:, -1], H)
        np.savez_compressed(os.path.join(OUT, f"departures_{rung}_H{H}.npz"),
                            dep=dep.astype(np.float32),
                            yhat=yhat.astype(np.float32),
                            last=ctx[:, -1].astype(np.float32),
                            truth=truth.astype(np.float32),
                            sigma=info["sigma"])
        d = decompose(dep)
        d.update({"rung": rung, "H": H, "n": n, "num_samples": ns, "model": model,
                  "sigma": float(info["sigma"]),
                  "dep_profile_mean": dep.mean(0).tolist(),
                  "dep_profile_absmean": np.abs(dep).mean(0).tolist(),
                  "codec_floor_rmse_sigma": float(np.sqrt(fl.mean()) / info["sigma"])})
        recs.append(d)
        log(f"{rung}: drift share {d['share_drift_periodGtH4']:.3f}  "
            f"peak period {d['peak_period_of_mean_psd']:.1f}  "
            f"band24 excess {(d['band24'] or {}).get('excess_over_continuum')}  "
            f"band7 excess {(d['band7'] or {}).get('excess_over_continuum')}")
    fn = os.path.join(OUT, f"spectral_deep_H{H}.json")
    with open(fn, "w") as f:
        json.dump(recs, f, indent=1, default=float)
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
