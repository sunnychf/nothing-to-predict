"""Numerical checks of every analytical claim in the paper. Pure numpy.

Run: python verify_theory.py

Each check corresponds to one statement in the text; a failure means the paper is
wrong. Written in the same spirit as the equivalent script of an earlier project,
where this exercise caught two real errors before they reached a reviewer.
"""
from __future__ import annotations

import sys

import numpy as np

rng = np.random.default_rng(0)
ok = True


def chk(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + label + (("   " + detail) if detail else ""))


N = 400_000

# ---------------------------------------------------------------- Prop 3.2
# Risk decomposes into an irreducible term and the mean squared departure from
# persistence, with NO cross term, under the martingale null.
print("=== Proposition 3.2 (departure is the prior)")
y_t = rng.standard_normal(N) * 3.0                     # F_t-measurable state
step = rng.standard_normal(N)                          # martingale increment
y_next = y_t + step
# an arbitrary F_t-measurable forecast: any function of y_t
y_hat = 0.8 * y_t + 0.5 + 0.1 * np.sin(y_t)
lhs = np.mean((y_next - y_hat) ** 2)
rhs = np.mean((y_next - y_t) ** 2) + np.mean((y_hat - y_t) ** 2)
chk("R(yhat) == irreducible + departure", abs(lhs - rhs) / lhs < 3 / np.sqrt(N),
    f"{lhs:.6f} vs {rhs:.6f}")
cross = np.mean((y_next - y_t) * (y_hat - y_t))
chk("cross term vanishes", abs(cross) < 5 / np.sqrt(N), f"cross = {cross:+.2e}")

# ---------------------------------------------------------------- Prop 5.1
print("\n=== Proposition 5.1 (no harm under the null)")
for pi_desc, pi in [("constant 0.3", np.full(N, 0.3)),
                    ("F_t-measurable", 1 / (1 + np.exp(-y_t)))]:
    y_pi = (1 - pi) * y_t + pi * y_hat
    lhs = np.mean((y_next - y_pi) ** 2)
    rhs = np.mean((y_next - y_t) ** 2) + np.mean(pi ** 2 * (y_hat - y_t) ** 2)
    chk(f"R(yhat^pi) == R(y_t) + E[pi^2 (yhat-y_t)^2]  [{pi_desc}]",
        abs(lhs - rhs) / lhs < 3 / np.sqrt(N), f"{lhs:.6f} vs {rhs:.6f}")
# monotonicity in pi, and the two endpoints
risks = [np.mean((y_next - ((1 - p) * y_t + p * y_hat)) ** 2) for p in np.linspace(0, 1, 11)]
chk("risk is non-decreasing in pi", all(b >= a - 1e-9 for a, b in zip(risks, risks[1:])),
    f"pi=0: {risks[0]:.6f} -> pi=1: {risks[-1]:.6f}")
chk("pi=0 recovers persistence risk",
    abs(risks[0] - np.mean((y_next - y_t) ** 2)) < 1e-12)
chk("pi=1 recovers the model's risk",
    abs(risks[-1] - np.mean((y_next - y_hat) ** 2)) < 1e-12)

# ------------------------------------------------- Section 4.1 normalisation
# Claim: under a random walk the irreducible term grows LINEARLY in h, so a
# departure that is flat in h makes the normalised ratio decay for arithmetic
# reasons. Verify the linear growth so the caveat in the text is correct.
print("\n=== Section 4.1 (the normalisation is h-dependent by arithmetic)")
T = 4000
paths = np.cumsum(rng.standard_normal((20_000, T)), axis=1)
hs = np.array([1, 2, 4, 8, 16, 32])
irr = np.array([np.mean((paths[:, h:] - paths[:, :-h]) ** 2) for h in hs])
slope = np.polyfit(hs, irr, 1)
chk("irreducible term is linear in h", np.corrcoef(hs, irr)[0, 1] > 0.9999,
    f"slope {slope[0]:.4f} (unit variance => 1.0), intercept {slope[1]:+.4f}")

# ------------------------------------------------------------- N4 rung, Roll
# The bid-ask bounce model. Two things to establish, because the paper asserts
# both: the induced first-order autocovariance is -s^2/4 (Roll 1984), and the
# OBSERVED price is NOT a martingale, so N4 violates Definition 3.1 by design.
print("\n=== N4 rung (bid-ask bounce): Roll's result, and the null violation")
s = 0.4                                                  # spread
m = np.cumsum(rng.standard_normal(N)) * 0.1              # efficient price
q = rng.choice([-1.0, 1.0], size=N)                      # trade direction
p = m + (s / 2) * q                                      # observed price
r = np.diff(p)
acov1 = np.mean((r[1:] - r.mean()) * (r[:-1] - r.mean()))
chk("first-order autocovariance == -s^2/4 (Roll 1984)",
    abs(acov1 + s ** 2 / 4) < 6 * np.std(r) ** 2 / np.sqrt(N),
    f"measured {acov1:+.5f}, predicted {-s**2/4:+.5f}")
chk("autocovariance is NEGATIVE (paper's N4 description)", acov1 < 0)
# the observed price is not a martingale: E[p_{t+1} - p_t | p_t, p_{t-1}] != 0
# regress the next observed return on the previous one; a martingale gives 0.
beta = np.polyfit(r[:-1], r[1:], 1)[0]
chk("observed price is NOT a martingale, so N4 violates Definition 3.1",
    abs(beta) > 10 / np.sqrt(N), f"AR(1) coefficient of observed returns = {beta:+.4f}")

# closed-form optimal forecast for N4. Observed returns are MA(1):
#   r_t = e_t + theta * e_{t-1},  with theta and var(e) set by s and sigma_m.
# The optimal one-step forecast of r_{t+1} is theta * e_t, recovered by
# inverting the MA. Check that this beats persistence (which forecasts 0).
gamma0, gamma1 = np.var(r), acov1
# solve gamma1/gamma0 = theta/(1+theta^2) for the invertible root |theta|<1
rho1 = gamma1 / gamma0
theta = (1 - np.sqrt(max(0.0, 1 - 4 * rho1 ** 2))) / (2 * rho1)
chk("MA(1) representation is invertible", abs(theta) < 1, f"theta = {theta:+.4f}")
e = np.zeros_like(r)
for i in range(1, len(r)):                               # invert the MA filter
    e[i] = r[i] - theta * e[i - 1]
pred = theta * e[:-1]                                    # optimal forecast of r_{t+1}
mse_opt = np.mean((r[1:] - pred) ** 2)
mse_zero = np.mean(r[1:] ** 2)                           # persistence on the price
chk("closed-form MA(1) forecast beats persistence on N4",
    mse_opt < mse_zero, f"optimal {mse_opt:.6f} vs persistence {mse_zero:.6f} "
                        f"({100*(mse_zero-mse_opt)/mse_zero:.2f}% better)")

# The MA(1) forecast is the optimal LINEAR forecast of the price. The conditional mean itself is
# p_T - (s/2) E[q_T | p_{1:T}], the trade-direction posterior of a two-state chain (nulls.oracle_forecast_exact);
# q is two-valued, so the exact optimum beats the linear one, its posterior is calibrated against the
# generator's hidden directions, and its departure is odd under the mirror like the linear one's.
print("\n=== N4 rung: the exact conditional mean against the optimal linear forecast")
from nulls import n4_bounce as _n4b, oracle_forecast as _of, oracle_forecast_exact as _ofe, q_posterior_up as _qp
ne, Te = 20000, 128
ye, info_e = _n4b(ne, Te, 1, rng); ctx_e, fut_e = ye[:, :Te], ye[:, Te]
lin_e, ex_e = _of("N4", ctx_e, info_e, 1)[:, 0], _ofe("N4", ctx_e, info_e, 1)[:, 0]
mse_p, mse_l, mse_e = np.mean((fut_e - ctx_e[:, -1]) ** 2), np.mean((fut_e - lin_e) ** 2), np.mean((fut_e - ex_e) ** 2)
chk("exact conditional mean beats the optimal linear forecast on N4", mse_e < mse_l,
    f"skill vs persistence: linear {1 - mse_l / mse_p:+.4f}, exact {1 - mse_e / mse_p:+.4f}")
p_up = _qp(ctx_e, info_e["sigma"], info_e["spread"]); q_T = info_e["q"][:, Te - 1]
cal_gap = max(abs(p_up[b].mean() - (q_T[b] > 0).mean()) for b in [np.clip((p_up * 5).astype(int), 0, 4) == k for k in range(5)] if b.sum() > 100)
chk("trade-direction posterior is calibrated (bins of P(q_T=+1) vs realised frequency)", cal_gap < 0.03, f"largest gap {cal_gap:.3f}")
mir_e = 2 * ctx_e[:, :1] - ctx_e; ex_m = _ofe("N4", mir_e, info_e, 1)[:, 0]
chk("exact optimum's departure is odd under the mirror", np.allclose(ex_e - ctx_e[:, -1], -(ex_m - mir_e[:, -1]), atol=1e-9),
    f"max |m(x) + m(xbar)| = {np.max(np.abs((ex_e - ctx_e[:, -1]) + (ex_m - mir_e[:, -1]))):.2e}")
resid = fut_e - ex_e
chk("exact optimum's residual is uncorrelated with the linear forecast's departure (no linear improvement left)",
    abs(np.corrcoef(resid, lin_e - ctx_e[:, -1])[0, 1]) < 3 / np.sqrt(ne), f"corr = {np.corrcoef(resid, lin_e - ctx_e[:, -1])[0, 1]:+.4f}")

# ------------------------------------------------- gate statistics, Section 5
print("\n=== Section 5 gate statistics behave as claimed under the null")
# variance ratio equals 1 for a random walk
rw = np.cumsum(rng.standard_normal(N))
d1 = np.diff(rw)
for k in (2, 4, 8):
    dk = rw[k:] - rw[:-k]
    vr = (np.var(dk) / k) / np.var(d1)
    chk(f"variance ratio ~ 1 under the null (k={k})", abs(vr - 1) < 0.05, f"VR = {vr:.4f}")
# spectral flatness of white innovations is ~1; of a trending series, far below
def flatness(x):
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    P = P[1:]                                            # drop DC
    return np.exp(np.mean(np.log(P + 1e-300))) / np.mean(P)
chk("spectral flatness ~ 1 for white increments", flatness(d1[:8192]) > 0.5,
    f"{flatness(d1[:8192]):.3f}")
season = np.sin(2 * np.pi * np.arange(8192) / 24) + 0.1 * rng.standard_normal(8192)
chk("spectral flatness << 1 for a periodic series", flatness(season) < 0.1,
    f"{flatness(season):.4f}")

# ---------------------------------------------------------------- Prop 5.2 (sign-averaged correction)
# Under conditional sign symmetry the sign-averaged departure is the conditional mean
# departure, so subtracting it (i) zeroes the mean departure, (ii) lowers the risk by
# E[Dbar^2], (iii) with K draws adds E[V]/K, which under CSS is (E[D^2]-E[Dbar^2])/K, and
# (iv) the correction is sign-blind, so a sign-reading component of the forecast survives.
print("\n=== Section 5, the sign-averaged correction")
n, T, K, M = 5000, 32, 4, 1000
steps = rng.standard_normal((n, T))                    # N1 context increments: CSS holds
ctx = 100.0 + np.cumsum(steps, axis=1)
fut = ctx[:, -1] + rng.standard_normal(n)              # martingale future, h = 1
def model(c):
    """A stand-in forecaster: a sign-blind prior that scales with the magnitude path, plus a
    sign-reading component (it follows the last increment), plus its own noise."""
    d = np.diff(c, axis=1)
    return c[:, -1] + 0.8 * np.mean(np.abs(d), axis=1) + 0.5 * d[:, -1] + 0.1 * rng.standard_normal(len(c))
def copies(c, k):
    d = np.diff(c, axis=1); eps = rng.choice(np.array([-1.0, 1.0]), size=(len(c), k, d.shape[1]))
    out = np.empty((len(c), k, c.shape[1])); out[:, :, 0] = c[:, :1]
    out[:, :, 1:] = c[:, None, :1] + np.cumsum(eps * d[:, None, :], axis=2); return out
g = model(ctx); D = g - ctx[:, -1]
cp = copies(ctx, M)                                     # many draws: the K -> infinity limit
Dcp = model(cp.reshape(-1, T)).reshape(n, M) - cp[:, :, -1]
Dbar, V = np.mean(Dcp, axis=1), np.var(Dcp, axis=1, ddof=1)
gc = g - Dbar
chk("(i) mean departure of the corrected forecast ~ 0 under CSS", abs(np.mean(gc - ctx[:, -1])) < 4 * np.std(gc - ctx[:, -1]) / np.sqrt(n),
    f"raw {np.mean(D):+.4f}, corrected {np.mean(gc - ctx[:, -1]):+.4f}")
risk_g, risk_c = np.mean((fut - g) ** 2), np.mean((fut - gc) ** 2)
chk("(ii) Risk(g_c) == Risk(g) - E[Dbar^2]", abs(risk_g - risk_c - np.mean(Dbar ** 2)) < 5 * np.std((fut - g) ** 2 - (fut - gc) ** 2) / np.sqrt(n),
    f"{risk_g - risk_c:.4f} vs E[Dbar^2] = {np.mean(Dbar ** 2):.4f}")
cpK = copies(ctx, K); DK = np.mean(model(cpK.reshape(-1, T)).reshape(n, K) - cpK[:, :, -1], axis=1)
risk_K = np.mean((fut - (g - DK)) ** 2)
chk(f"(iii) Risk(g_c,K) == Risk(g_c) + E[V]/K  (K={K})", abs(risk_K - risk_c - np.mean(V) / K) < 5 * np.std((fut - (g - DK)) ** 2 - (fut - gc) ** 2) / np.sqrt(n),
    f"{risk_K - risk_c:.4f} vs E[V]/K = {np.mean(V) / K:.4f}")
chk("(iii) under CSS E[V] == E[D^2] - E[Dbar^2]", abs(np.mean(V) - (np.mean(D ** 2) - np.mean(Dbar ** 2))) < 0.05 * np.mean(D ** 2),
    f"{np.mean(V):.4f} vs {np.mean(D ** 2) - np.mean(Dbar ** 2):.4f}")
# (iv) the correction depends on the magnitudes only: it equals the model's sign-blind term
# 0.8 * mean|d| up to Monte Carlo noise, is the same for the sign-flipped context, and leaves
# the sign-reading term 0.5 * d_T in the corrected forecast
truth = 0.8 * np.mean(np.abs(np.diff(ctx, axis=1)), axis=1)
flipped = np.concatenate([ctx[:, :1], ctx[:, :1] - np.cumsum(np.diff(ctx, axis=1), axis=1)], axis=1)
cpF = copies(flipped, M); DbarF = np.mean(model(cpF.reshape(-1, T)).reshape(n, M) - cpF[:, :, -1], axis=1)
mc_se = np.sqrt(np.mean(V) / M)
chk("(iv) Dbar equals the sign-blind term of the model (up to MC noise)", np.corrcoef(Dbar, truth)[0, 1] > 0.95 and abs(np.mean(Dbar - truth)) < 4 * mc_se / np.sqrt(n),
    f"corr {np.corrcoef(Dbar, truth)[0, 1]:.4f}, mean diff {np.mean(Dbar - truth):+.4f} (MC se {mc_se:.4f})")
chk("(iv) Dbar is unchanged when every sign of the context is flipped", abs(np.mean(Dbar - DbarF)) < 4 * np.sqrt(2) * mc_se / np.sqrt(n) and np.std(Dbar - DbarF) < 3 * np.sqrt(2) * mc_se,
    f"mean diff {np.mean(Dbar - DbarF):+.4f}, std {np.std(Dbar - DbarF):.4f} vs MC {np.sqrt(2) * mc_se:.4f}")
chk("(iv) the sign-reading term survives the correction", np.corrcoef(gc - ctx[:, -1], np.diff(ctx, axis=1)[:, -1])[0, 1] > 0.9,
    f"corr(g_c - y_t, d_T) = {np.corrcoef(gc - ctx[:, -1], np.diff(ctx, axis=1)[:, -1])[0, 1]:.3f}")

# ---------------------------------------------------------------- the mirror group on a non-martingale
# The mirror instance needs only invariance under negating every increment: on the N4 bounce
# series (not a martingale, not conditionally sign-symmetric) the corrected risk still drops by
# E[(AD)^2], the optimal departure theta*e_T is odd and untouched, and the correction applied to
# the optimal forecaster changes nothing.
print("\n=== Section 5, the mirror group on the N4 bounce series")
from nulls import n4_bounce, oracle_forecast
nb, Tb = 20000, 64
yb, info_b = n4_bounce(nb, Tb, 1, rng)
ctx_b, fut_b = yb[:, :Tb], yb[:, Tb]
mir_b = 2 * ctx_b[:, :1] - ctx_b
orc = oracle_forecast("N4", ctx_b, info_b, 1)[:, 0]; orc_m = oracle_forecast("N4", mir_b, info_b, 1)[:, 0]
chk("optimal departure is odd under the mirror", np.allclose(orc - ctx_b[:, -1], -(orc_m - mir_b[:, -1])),
    f"max |m(x) + m(xbar)| = {np.max(np.abs((orc - ctx_b[:, -1]) + (orc_m - mir_b[:, -1]))):.2e}")
def model_b(c):
    """a forecaster with a sign-blind prior, a partial read of the bounce, and noise"""
    d = np.diff(c, axis=1)
    return c[:, -1] + 0.6 * np.mean(np.abs(d), axis=1) - 0.3 * d[:, -1] + 0.2 * rng.standard_normal(len(c))
gb, gbm = model_b(ctx_b), model_b(mir_b)
Db, Dbm = gb - ctx_b[:, -1], gbm - mir_b[:, -1]
AD = 0.5 * (Db + Dbm); gc = gb - AD
risk_g, risk_c = np.mean((fut_b - gb) ** 2), np.mean((fut_b - gc) ** 2)
chk("(ii) mirror: Risk(g_c) == Risk(g) - E[(AD)^2] on N4 (no martingale needed)",
    abs(risk_g - risk_c - np.mean(AD ** 2)) < 5 * np.std((fut_b - gb) ** 2 - (fut_b - gc) ** 2) / np.sqrt(nb),
    f"{risk_g - risk_c:.4f} vs E[(AD)^2] = {np.mean(AD ** 2):.4f}")
chk("(i) mirror-corrected departure has mean ~ 0 on N4", abs(np.mean(gc - ctx_b[:, -1])) < 4 * np.std(gc - ctx_b[:, -1]) / np.sqrt(nb),
    f"raw {np.mean(Db):+.4f}, corrected {np.mean(gc - ctx_b[:, -1]):+.4f}")
chk("(iv) mirror correction leaves the optimal forecaster unchanged", np.allclose(0.5 * ((orc - ctx_b[:, -1]) + (orc_m - mir_b[:, -1])), 0.0),
    "A m == 0 exactly")
share_gain = (np.mean((fut_b - ctx_b[:, -1]) ** 2) - risk_c) / (np.mean((fut_b - ctx_b[:, -1]) ** 2) - np.mean((fut_b - orc) ** 2))
share_raw = (np.mean((fut_b - ctx_b[:, -1]) ** 2) - risk_g) / (np.mean((fut_b - ctx_b[:, -1]) ** 2) - np.mean((fut_b - orc) ** 2))
chk("mirror correction raises the share of the oracle's gain on N4", share_gain > share_raw, f"{share_raw:+.3f} -> {share_gain:+.3f}")

# ---------------------------------------------------------------- Prop 5.2 (v): no symmetry at all
# With no assumption on the law, Risk(g_c) - Risk(g) = -E[AD^2] + 2 E[m AD] - 2 E[(D - AD) AD], m the
# conditional drift. It is an algebraic identity in the realised future (y_{t+h} - y_t in place of m),
# so it must hold to rounding on any sample; the two extra terms are what a drift and an asymmetric
# law add, and for a sign-blind prior D == c the change is c (2 E[m] - c): removing the prior helps iff
# the drift it stands for is below half of it, or of the opposite sign.
print("\n=== Section 5, Proposition (v): the identity with no assumption on the law")
nv, Tv, hv = 200000, 64, 16
mu, sd = 0.05, 1.0
sym = rng.standard_normal((nv, Tv + hv)) * sd                           # symmetric: the assumptions of (ii) hold
inc = rng.standard_normal((nv, Tv + hv)) * sd + mu                      # a drifted Gaussian walk: not invariant, m = mu h
skw = (rng.exponential(1.0, (nv, Tv + hv)) - 1.0) * sd                  # zero-mean right-skewed walk: not invariant, m = 0
def model_v(x):                                                         # even part a*|d_T|, odd part b*d_T, constant prior k
    d = np.diff(x, axis=1); return x[:, -1] + (0.4 * np.abs(d[:, -1]) + 0.5 * d[:, -1] + 0.3) * hv / 8
for name, e, drift_term, cross_term in (("symmetric Gaussian", sym, False, False), ("drifted Gaussian", inc, True, True), ("zero-mean skewed", skw, False, True)):
    y = np.cumsum(e, axis=1); c = y[:, :Tv]; fut = y[:, Tv + hv - 1]; last = c[:, -1]
    mir = 2 * c[:, :1] - c
    g, gm = model_v(c), model_v(mir); D, Dm = g - last, gm - mir[:, -1]; AD = 0.5 * (D + Dm); gc = g - AD
    lhs = np.mean((fut - gc) ** 2) - np.mean((fut - g) ** 2)
    t_drift, t_cross = 2 * np.mean((fut - last) * AD), -2 * np.mean((D - AD) * AD)
    rhs = -np.mean(AD ** 2) + t_drift + t_cross
    chk(f"(v) identity exact on the sample ({name})", abs(lhs - rhs) < 1e-8 * max(1.0, abs(lhs)), f"lhs {lhs:+.5f} rhs {rhs:+.5f}")
    se_d, se_c = 2 * np.std((fut - last) * AD) / np.sqrt(nv), 2 * np.std((D - AD) * AD) / np.sqrt(nv)
    chk(f"(v) drift term {'nonzero' if drift_term else '~ zero'} and cross term {'nonzero' if cross_term else '~ zero'} ({name})",
        ((abs(t_drift) > 5 * se_d) == drift_term) and ((abs(t_cross) > 5 * se_c) == cross_term),
        f"2E[m AD] = {t_drift:+.4f} (se {se_d:.4f}), -2E[(D-AD)AD] = {t_cross:+.4f} (se {se_c:.4f})")
    if name.startswith("symmetric"):
        chk("(v) reduces to (ii) under the assumptions", abs(lhs + np.mean(AD ** 2)) < 5 * np.std((fut - gc) ** 2 - (fut - g) ** 2) / np.sqrt(nv),
            f"risk change {lhs:+.4f} vs -E[AD^2] = {-np.mean(AD ** 2):+.4f}")
# the corollary for a sign-blind prior on a drifted walk: risk change c (2 mu h - c), sign flip at c = 2 mu h
y = np.cumsum(inc, axis=1); c = y[:, :Tv]; fut = y[:, Tv + hv - 1]; last = c[:, -1]; mir = 2 * c[:, :1] - c
for cc in (0.5 * mu * hv, 1.5 * mu * hv, 2.5 * mu * hv):
    g = last + cc; gc = last                                            # the mirror removes a constant prior entirely
    lhs = np.mean((fut - gc) ** 2) - np.mean((fut - g) ** 2); pred = cc * (2 * mu * hv - cc)
    chk(f"(v) sign-blind prior c = {cc / (mu * hv):.1f} x drift: risk change {'>' if pred > 0 else '<'} 0 as predicted",
        abs(lhs - pred) < 5 * np.std((fut - gc) ** 2 - (fut - g) ** 2) / np.sqrt(nv) and (lhs > 0) == (pred > 0),
        f"measured {lhs:+.4f}, c(2 mu h - c) = {pred:+.4f}")
# a forecaster that reads the drift from the data is odd, so the mirror leaves it alone even though the law has a drift
g = last + np.mean(np.diff(c, axis=1), axis=1) * hv; gm = mir[:, -1] + np.mean(np.diff(mir, axis=1), axis=1) * hv
chk("(iv)/(v) a forecaster that reads the drift from the context is odd: the mirror removes nothing",
    np.allclose(0.5 * ((g - last) + (gm - mir[:, -1])), 0.0, atol=1e-9), "A D == 0")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
