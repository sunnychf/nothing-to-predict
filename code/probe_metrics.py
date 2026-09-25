"""Pure-numpy metrics shared by every model driver.

Kept free of torch, chronos and any model library so that a driver living in a
different conda environment (FinCast pins its own torch build) can import the
same measurement code and produce rows the report reads identically.
"""
from __future__ import annotations

import numpy as np

from nulls import (RUNGS, LEVEL0, SIGMA_REL, oracle_forecast, departure_metrics,
                   spectral_flatness, band_power_fraction, peak_period,
                   bootstrap_ci)


# ------------------------------------------------------------------- helpers

def make_data(rung, n, T, H, seed, sigma_rel=SIGMA_REL, level0=LEVEL0):
    rng = np.random.default_rng(seed)
    y, info = RUNGS[rung](n, T, H, rng, sigma_rel=sigma_rel, level0=level0)
    return y[:, :T], y[:, T:], info


def summarise(y_hat, y_ctx, y_true, info, rung, H, floor=None, samples=None):
    """Per-horizon decomposition, plus the scalars the paper reports.

    When the sampled trajectories are passed we also subtract the Monte Carlo
    component exactly rather than by extrapolation. The point forecast is the
    mean of S trajectories, so its sampling variance is Var(samples)/S, and that
    variance enters the measured squared departure additively. Estimating it
    inside the same run gives a per-rung, per-horizon correction for free,
    where the 1/S extrapolation needs a whole sweep and only covers the rung it
    was swept on.
    """
    m = departure_metrics(y_hat, y_ctx, y_true)
    sigma = info["sigma"]
    last = y_ctx[:, -1:]
    per_series_dep = np.mean((y_hat - last) ** 2, axis=1)
    dep_signed = y_hat - last
    # Direction of the departure. Under a symmetric null a model with no
    # directional prior departs up and down equally; a mean that grows with h
    # and a fraction-up far from one half is a drift the model brings with it.
    slope64 = (y_ctx[:, -1] - y_ctx[:, -65]) / 64.0 if y_ctx.shape[1] >= 65 else np.zeros(len(y_ctx))
    out = {
        "departure_mean_signed_sigma": (dep_signed.mean(0) / sigma).tolist(),
        "frac_departure_up": (dep_signed > 0).mean(0).tolist(),
        "corr_departure_slope64": [float(np.corrcoef(dep_signed[:, h], slope64)[0, 1])
                                   if np.std(slope64) > 0 else float("nan")
                                   for h in range(dep_signed.shape[1])],
        "irreducible": m["irreducible"].tolist(),
        "imported": m["imported"].tolist(),
        "risk_model": m["risk_model"].tolist(),
        "risk_persistence": m["risk_persistence"].tolist(),
        "cross": m["cross"].tolist(),
        # normalised forms
        "imported_over_irreducible": (m["imported"] / m["irreducible"]).tolist(),
        # the interpretable one: departure in units of a single step's sigma
        "departure_in_sigma": (np.sqrt(m["imported"]) / sigma).tolist(),
        "skill_vs_persistence": (1 - m["risk_model"] / m["risk_persistence"]).tolist(),
        "dep_ci95": bootstrap_ci(per_series_dep),
        "sigma": float(sigma),
    }
    if samples is not None:
        S = samples.shape[1]
        mc = np.mean(np.var(samples, axis=1, ddof=1), axis=0) / S
        out["mc_component"] = mc.tolist()
        corrected = np.maximum(m["imported"] - mc, 0.0)
        out["imported_mc_corrected"] = corrected.tolist()
        out["departure_in_sigma_mc_corrected"] = (np.sqrt(corrected) / sigma).tolist()
    if floor is not None:
        fl = np.mean(floor, axis=0)
        out["codec_floor"] = fl.tolist()
        out["imported_net_of_floor"] = np.maximum(m["imported"] - fl, 0.0).tolist()
        out["floor_share"] = (fl / np.maximum(m["imported"], 1e-300)).tolist()
    # N4 carries an oracle, so we can ask whether the model finds real structure
    if rung == "N4":
        orc = oracle_forecast("N4", y_ctx, info, H)
        mse_orc = np.mean((y_true - orc) ** 2, axis=0)
        out["risk_oracle"] = mse_orc.tolist()
        out["oracle_skill_vs_persistence"] = (
            1 - mse_orc / m["risk_persistence"]).tolist()
    return out


def spectra(y_hat, y_ctx):
    """Structure of the departure path: patterned prior, or just noise."""
    dep = y_hat - y_ctx[:, -1:]
    periods = [5, 7, 12, 24, 30, 52]
    return {
        "flatness_raw": float(np.mean(spectral_flatness(dep, detrend=False))),
        "flatness_detrended": float(np.mean(spectral_flatness(dep, detrend=True))),
        "peak_period": float(np.median(peak_period(dep))),
        "band_power": {k: float(np.mean(v)) for k, v in
                       band_power_fraction(dep, periods).items()},
        "flatness_ci95": bootstrap_ci(np.asarray(spectral_flatness(dep, detrend=True))),
    }


def norm_floor(y_ctx, H):
    """Affine-normalisation round trip of y_T: the continuous-model analogue of
    the codec floor, for models that scale by context mean/std and have no
    token grid (FinCast, Moirai). Float32 precision, so effectively zero; stored
    under codec_floor so every driver's rows share one schema."""
    x = np.asarray(y_ctx, dtype=np.float32)
    mu = x.mean(1, keepdims=True); sd = x.std(1, keepdims=True) + 1e-8
    last = x[:, -1:]
    rt = ((last - mu) / sd) * sd + mu
    return np.repeat((rt - last) ** 2, H, axis=1).astype(np.float64)
