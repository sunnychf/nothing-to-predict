"""Why the calendar-shift check pairs the forecast change with the realised target level (simulation, no model involved).
Random walks (log levels): 50 series with a common factor (correlation 0.6), 37 end dates 128 days apart, context 512,
future 128, context end moved by k in (-40, -20, 0, +20, +40); daily drift 0, 0.0005, or a drift of 0.0015 that switches
sign every 300 days. Forecasters that only read the context: trend-following and trend-opposing (forecast change = +/- the
move over the last m days) and the fitted drift of the whole context; and remembering forecasters (weight w on the
realised move, alone or with a trend-opposing term). Within-date correlation of cell means, as in shift_check.py:
  returns   forecast and realised returns from each shifted context's end;
  levels    forecast and realised levels at each shift's target date, relative to the unshifted context's end;
  change    the forecast change from each shifted context's end against the realised level at the target date,
            relative to the unshifted context's end (the statistic used in the paper).
Returns are biased by construction: a later shift's context contains part of the earlier shifts' futures, so removing a
date's mean over its shifts ties the realised deviations to the recent move in each context. Levels share the drift of
the context ends and of the targets. The change version pairs a quantity computed from the context with target
deviations that use only moves at least 48 days after the latest context end, and is unbiased while the drift is
constant; when the drift switches sign, readers of the context can also score (real structure), which is why the paper
reports drift and momentum forecasts on the same windows.
Writes experiments/results/shift_bias_sim.json.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/shift_bias_sim.py"""
import json, os
import numpy as np

rng = np.random.default_rng(1)
SH, H, CTX = (-40, -20, 0, 20, 40), 128, 512


def run(kind, m=20, n_dates=37, n_series=50, rho=0.6, reps=200, w=1.0, drift="none"):
    res = {"returns": [], "levels": [], "change": []}
    for _ in range(reps):
        T = CTX + H + 128 * n_dates + 200
        f = rng.standard_normal(T)
        mu = {"none": np.zeros(T), "constant": np.full(T, 0.0005), "switching": 0.0015 * np.where(np.sin(np.arange(T) / 300 * np.pi) >= 0, 1.0, -1.0)}[drift]
        c = np.cumsum(mu[None] + 0.01 * (np.sqrt(rho) * f[None] + np.sqrt(1 - rho) * rng.standard_normal((n_series, T))), axis=1)   # log levels
        acc = {k: ([], []) for k in res}
        for d in range(n_dates):
            e0 = 100 + CTX + 128 * d
            V = {k: ([], []) for k in res}
            for k in SH:
                e = e0 + k; trend = c[:, e] - c[:, e - m]; real = c[:, e + H] - c[:, e]
                fc = {"trend_following": trend, "trend_opposing": -trend, "fitted_drift": (c[:, e] - c[:, e - CTX + 1]) / (CTX - 1) * H,
                      "remembering": w * real, "remembering_trend_opposing": w * real - 0.5 * trend}[kind]
                V["returns"][0].append(fc.mean()); V["returns"][1].append(real.mean())
                V["levels"][0].append((c[:, e] + fc - c[:, e0]).mean()); V["levels"][1].append((c[:, e + H] - c[:, e0]).mean())
                V["change"][0].append(fc.mean()); V["change"][1].append((c[:, e + H] - c[:, e0]).mean())
            for key in res:
                F, Y = np.array(V[key][0]), np.array(V[key][1]); acc[key][0].extend(F - F.mean()); acc[key][1].extend(Y - Y.mean())
        for key in res:
            res[key].append(np.corrcoef(*acc[key])[0, 1])
    return {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items()} | {"reps": reps}


out = {}
for drift in ("none", "constant", "switching"):
    for kind, m in (("trend_following", 20), ("trend_following", 128), ("trend_opposing", 20), ("trend_opposing", 128), ("fitted_drift", 20)):
        out[f"{drift}|{kind}|m={m}" if kind != "fitted_drift" else f"{drift}|{kind}"] = run(kind, m, drift=drift)
    for kind in ("remembering", "remembering_trend_opposing"):
        for w in (1.0, 0.5, 0.25):
            out[f"{drift}|{kind}|w={w}"] = run(kind, 20, w=w, reps=100, drift=drift)
json.dump(out, open(os.path.join("experiments", "results", "shift_bias_sim.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:44s} " + " | ".join(f"{s} {v[s][0]:+.3f} (sd {v[s][1]:.3f})" for s in ("returns", "levels", "change")))
