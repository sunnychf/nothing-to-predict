"""Self-test of the null ladder. Run before any model touches it.

The probe's entire logic rests on N1-N3 being martingales and N4 not being one.
If a generator is wrong the measured 'departure' is meaningless, so these checks
gate the experiment in the same way verify_theory.py gates the paper.
"""
from __future__ import annotations

import sys

import numpy as np

from nulls import (RUNGS, LEVEL0, SIGMA_REL, n4_bounce, oracle_forecast,
                   theta_population, spectral_flatness, departure_metrics)

ok = True


def chk(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + label + (("   " + detail) if detail else ""))


rng = np.random.default_rng(0)
N, T, H = 20_000, 512, 16

# ------------------------------------------------ N1-N3 are martingales
print("=== N1-N3 satisfy the null (mean increment zero, no return autocorrelation)")
for name in ["N1", "N2", "N3_nu3", "N3_nu5"]:
    y, info = RUNGS[name](N, T, H, np.random.default_rng(1))
    r = np.diff(y, axis=1)
    mu, se = r.mean(), r.std() / np.sqrt(r.size)
    chk(f"{name}: mean increment is zero", abs(mu) < 4 * se, f"{mu:+.3e} +- {4*se:.3e}")
    # First-order autocorrelation of returns; a martingale gives 0.
    # The naive iid standard error 1/sqrt(n) is NOT valid here: under conditional
    # heteroskedasticity (N2) or heavy tails (N3) the asymptotic variance of the
    # sample autocorrelation involves fourth moments and is strictly larger. Using
    # the iid bound would flag a correct generator as broken, so we use the
    # heteroskedasticity-robust standard error throughout.
    a, b = r[:, :-1].ravel(), r[:, 1:].ravel()
    rho1 = float(np.mean(a * b) / np.mean(r ** 2))
    se_rob = float(np.sqrt(np.mean((a * b) ** 2)) / np.mean(r ** 2) / np.sqrt(len(a)))
    chk(f"{name}: returns are serially uncorrelated", abs(rho1) < 4 * se_rob,
        f"rho1 = {rho1:+.5f}, robust 4*se = {4*se_rob:.5f} "
        f"(iid 4*se would be {4/np.sqrt(len(a)):.5f})")
    # persistence is optimal: it must beat any fixed drift
    last = y[:, T - 1:T]
    mse_pers = np.mean((y[:, T:T + 1] - last) ** 2)
    drifts = [np.mean((y[:, T:T + 1] - (last + d * info["sigma"])) ** 2)
              for d in (-0.5, -0.1, 0.1, 0.5)]
    chk(f"{name}: persistence beats every drifted forecast", all(d >= mse_pers for d in drifts),
        f"pers {mse_pers:.5f} vs best drift {min(drifts):.5f}")

# ------------------------------------------------ N2 variance IS predictable
print("\n=== N2 has an unpredictable mean wrapped around a predictable variance")
y, info = RUNGS["N2"](N, T, H, np.random.default_rng(2))
r = np.diff(y, axis=1)
cv = info["cond_vol"][:, 1:]
# squared returns are autocorrelated even though returns are not
a, b = (r[:, :-1] ** 2).ravel(), (r[:, 1:] ** 2).ravel()
rho = float(np.corrcoef(a, b)[0, 1])
chk("N2: squared returns are autocorrelated", rho > 0.05, f"corr = {rho:+.4f}")
# the conditional vol genuinely forecasts realised size
rho2 = float(np.corrcoef(cv[:, :-1].ravel(), np.abs(r[:, 1:]).ravel())[0, 1])
chk("N2: conditional vol forecasts realised |r|", rho2 > 0.1, f"corr = {rho2:+.4f}")
chk("N2: vol varies by a wide factor across the panel",
    cv.max() / cv.min() > 5, f"max/min = {cv.max()/cv.min():.1f}")

# ------------------------------------------------ N4 breaks the null
print("\n=== N4 violates the null by construction (Roll 1984) and has a known optimum")
y, info = n4_bounce(N, T, H, np.random.default_rng(3))
s, sigma_m = info["spread"], info["sigma"]
r = np.diff(y, axis=1)
acov1 = float(np.mean((r[:, 1:] - r.mean()) * (r[:, :-1] - r.mean())))
chk("N4: first-order autocovariance == -s^2/4",
    abs(acov1 + s ** 2 / 4) < 6 * r.std() ** 2 / np.sqrt(r.size),
    f"measured {acov1:+.5f}, predicted {-s**2/4:+.5f}")
chk("N4: autocovariance is negative", acov1 < 0)
beta = float(np.polyfit(r[:, :-1].ravel(), r[:, 1:].ravel(), 1)[0])
chk("N4: observed price is NOT a martingale", abs(beta) > 10 / np.sqrt(r[:, 1:].size),
    f"AR(1) = {beta:+.4f}")

theta = theta_population(sigma_m, s)
chk("N4: population MA(1) root is invertible", abs(theta) < 1, f"theta = {theta:+.4f}")
g0, g1 = sigma_m ** 2 + s ** 2 / 2, -s ** 2 / 4
chk("N4: theta satisfies theta/(1+theta^2) == gamma_1/gamma_0",
    abs(theta / (1 + theta ** 2) - g1 / g0) < 1e-12,
    f"{theta/(1+theta**2):+.6f} vs {g1/g0:+.6f}")

y_ctx, y_true = y[:, :T], y[:, T:]
orc = oracle_forecast("N4", y_ctx, info, H)
pers = np.repeat(y_ctx[:, -1:], H, axis=1)
mse_orc = float(np.mean((y_true[:, 0] - orc[:, 0]) ** 2))
mse_pers = float(np.mean((y_true[:, 0] - pers[:, 0]) ** 2))
chk("N4: the oracle beats persistence at h=1", mse_orc < mse_pers,
    f"oracle {mse_orc:.5f} vs persistence {mse_pers:.5f} "
    f"({100*(mse_pers-mse_orc)/mse_pers:.2f}% better)")
# and on N1 the same oracle machinery must NOT beat persistence
y1, i1 = RUNGS["N1"](N, T, H, np.random.default_rng(4))
o1 = oracle_forecast("N1", y1[:, :T], i1, H)
chk("N1: the oracle IS persistence (no free lunch on the null)",
    np.allclose(o1, np.repeat(y1[:, T - 1:T], H, axis=1)))

# ------------------------------------------------ Prop 3.2 on ladder data
print("\n=== Proposition 3.2 holds on ladder data with an arbitrary forecast")
y, info = RUNGS["N1"](N, T, H, np.random.default_rng(5))
y_ctx, y_true = y[:, :T], y[:, T:]
fake = y_ctx[:, -1:] + 0.3 * info["sigma"] * np.sin(
    2 * np.pi * np.arange(H) / 7.0)[None, :]      # a 'prior' with a weekly rhythm
m = departure_metrics(fake, y_ctx, y_true)
lhs, rhs = m["risk_model"], m["irreducible"] + m["imported"]
chk("risk == irreducible + imported, per horizon",
    np.all(np.abs(lhs - rhs) / lhs < 0.02), f"max rel err {np.max(np.abs(lhs-rhs)/lhs):.2e}")
chk("cross term vanishes",
    np.all(np.abs(m["cross"]) < 6 * info["sigma"] ** 2 * np.sqrt(H) / np.sqrt(N)),
    f"max |cross| = {np.max(np.abs(m['cross'])):.3e}")

# ------------------------------------------------ flatness discriminates
print("\n=== Spectral flatness separates a rhythm from noise (Section 4.1)")
g = np.random.default_rng(6)
white = g.standard_normal((256, 128))
per = np.sin(2 * np.pi * np.arange(128) / 24.0)[None, :] + 0.1 * g.standard_normal((256, 128))
trend = np.arange(128)[None, :] * 0.01 + 0.1 * g.standard_normal((256, 128))
fw, fp = float(np.mean(spectral_flatness(white))), float(np.mean(spectral_flatness(per)))
chk("flatness ~ 1 for white departures", fw > 0.4, f"{fw:.3f}")
chk("flatness << 1 for a periodic departure", fp < 0.1, f"{fp:.4f}")
chk("periodic and white are cleanly separated", fw > 10 * fp, f"{fw:.3f} vs {fp:.4f}")
ftr_raw = float(np.mean(spectral_flatness(trend, detrend=False)))
ftr_det = float(np.mean(spectral_flatness(trend, detrend=True)))
# The property that matters is absolute, not a ratio: undetrended, a pure drift
# looks structured (low flatness) and could be mistaken for a corpus rhythm;
# detrended, it must read as white. Asserting a ratio instead would be an
# arbitrary threshold on two quantities we can bound directly.
chk("a pure trend looks structured before detrending", ftr_raw < 0.2, f"raw {ftr_raw:.4f}")
chk("and reads as white after detrending, so it cannot pose as a rhythm",
    ftr_det > 0.4, f"detrended {ftr_det:.3f}")
chk("a real rhythm survives detrending (the test is not just erasing structure)",
    float(np.mean(spectral_flatness(per, detrend=True))) < 0.1,
    f"periodic detrended {float(np.mean(spectral_flatness(per, detrend=True))):.4f}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
