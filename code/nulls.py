"""The null ladder of Section 4.1, plus the metrics the probe reports.

Pure numpy, no model dependency, so it can be tested on its own. Every rung
returns a *level* series (the thing handed to a forecaster) together with the
theoretically optimal forecast of that level, which is what makes the departure
measurable rather than merely observable.

N1-N3 satisfy the martingale null exactly: the optimal forecast is persistence.
N4 violates it by construction and its optimum is the inverted MA(1) filter.
Without N4 the probe cannot tell correct abstention from inertness.
"""
from __future__ import annotations

import numpy as np

LEVEL0 = 100.0          # starting price level, fixed across rungs
SIGMA_REL = 0.01        # per-step return std as a fraction of the level


# --------------------------------------------------------------- generators
# Each returns (y, info): y has shape (n_series, T + H), info carries whatever
# the oracle forecast needs. Context is y[:, :T]; truth is y[:, T:].

def n1_random_walk(n, T, H, rng, sigma_rel=SIGMA_REL, level0=LEVEL0, sigma=None):
    """N1. The bare null: Gaussian random walk on the level.

    The step size is sigma_rel * |level0| so that 'one percent of the level'
    means the same thing at every level. Taking |level0| matters: the earlier
    form sigma_rel * level0 gave a NEGATIVE sigma at level0 = -100, which
    flipped every 'in sigma units' statistic downstream, and gave sigma = 0 at
    level0 = 0. Pass sigma explicitly to decouple the step from the level.
    """
    sigma = sigma_rel * abs(level0) if sigma is None else float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma} (level0={level0})")
    steps = rng.standard_normal((n, T + H)) * sigma
    y = level0 + np.cumsum(steps, axis=1)
    return y, {"sigma": sigma}


def n2_garch(n, T, H, rng, sigma_rel=SIGMA_REL, level0=LEVEL0,
             alpha=0.09, beta=0.90):
    """N2. Unpredictable mean, highly predictable variance: GARCH(1,1).

    Also returns the conditional volatility path, so the interval-width test of
    Remark~\\ref{rem:n2} can ask whether predictive width tracks sigma_t.
    """
    sigma_uncond = sigma_rel * abs(level0)
    omega = sigma_uncond ** 2 * (1.0 - alpha - beta)
    L = T + H
    z = rng.standard_normal((n, L))
    r = np.zeros((n, L))
    s2 = np.full(n, sigma_uncond ** 2)
    sig = np.zeros((n, L))
    for t in range(L):
        sig[:, t] = np.sqrt(s2)
        r[:, t] = sig[:, t] * z[:, t]
        s2 = omega + alpha * r[:, t] ** 2 + beta * s2
    y = level0 + np.cumsum(r, axis=1)
    return y, {"sigma": sigma_uncond, "cond_vol": sig}


def n3_student_t(n, T, H, rng, nu=5, sigma_rel=SIGMA_REL, level0=LEVEL0):
    """N3. Heavy tails, mean untouched. Innovations standardised to unit variance
    so that the point-forecast problem is identical to N1 and only tails move."""
    sigma = sigma_rel * abs(level0)
    z = rng.standard_t(nu, size=(n, T + H)) / np.sqrt(nu / (nu - 2.0))
    y = level0 + np.cumsum(z * sigma, axis=1)
    return y, {"sigma": sigma, "nu": nu}


def n4_bounce(n, T, H, rng, sigma_rel=SIGMA_REL, level0=LEVEL0, spread_mult=4.0):
    """N4. Positive control: bid-ask bounce. The observed price is NOT a
    martingale, so persistence is not optimal and a model that always abstains
    is exposed as inert rather than honest.

    p_t = m_t + (s/2) q_t with m an efficient random walk and q = +-1 i.i.d.
    Observed returns are MA(1) with gamma_1 = -s^2/4 (Roll 1984).

    spread_mult sets s relative to one step of the efficient price, and it is the
    parameter that decides whether this rung can do its job. At spread_mult=1 the
    population optimum beats persistence by only 2.9 percent in MSE, which is far
    below the quantisation noise of the models under test, so a model could fail
    to capture it for reasons that have nothing to do with skill and the control
    would prove nothing. spread_mult=4 gives an edge of about 27 percent, which is
    the configuration verify_theory.py checks (s=0.4 against sigma_m=0.1) and the
    figure the paper already quotes, so we match it.
    """
    sigma_m = sigma_rel * abs(level0)
    s = spread_mult * sigma_m          # spread comparable to one step of m
    m = level0 + np.cumsum(rng.standard_normal((n, T + H)) * sigma_m, axis=1)
    q = rng.choice(np.array([-1.0, 1.0]), size=(n, T + H))
    y = m + (s / 2.0) * q
    return y, {"sigma": sigma_m, "spread": s, "q": q, "m": m}


RUNGS = {
    "N1": n1_random_walk,
    "N2": n2_garch,
    "N3_nu3": lambda n, T, H, rng, **kw: n3_student_t(n, T, H, rng, nu=3, **kw),
    "N3_nu5": lambda n, T, H, rng, **kw: n3_student_t(n, T, H, rng, nu=5, **kw),
    "N4": n4_bounce,
}


# ------------------------------------------------------------------ oracles

def theta_population(sigma_m, s):
    """Invertible MA(1) root for the bounce model, from population moments.

    gamma_0 = sigma_m^2 + s^2/2,  gamma_1 = -s^2/4,
    and theta solves theta/(1+theta^2) = gamma_1/gamma_0 with |theta| < 1.
    """
    g0 = sigma_m ** 2 + s ** 2 / 2.0
    g1 = -s ** 2 / 4.0
    rho1 = g1 / g0
    if abs(rho1) < 1e-15:
        return 0.0
    return (1.0 - np.sqrt(max(0.0, 1.0 - 4.0 * rho1 ** 2))) / (2.0 * rho1)


def oracle_forecast(rung, y_ctx, info, H):
    """Theoretically optimal forecast of the level, shape (n, H).

    Persistence on N1-N3. On N4 the MA(1) filter is inverted on the observed
    return series to recover the innovations, and the one-step optimum is
    p_T + theta * e_T; beyond one step the bounce contributes nothing further,
    so the optimum stays flat at that value.
    """
    last = y_ctx[:, -1:]
    if rung != "N4":
        return np.repeat(last, H, axis=1)
    theta = theta_population(info["sigma"], info["spread"])
    r = np.diff(y_ctx, axis=1)
    e = np.zeros_like(r)
    for i in range(1, r.shape[1]):          # invert the MA filter
        e[:, i] = r[:, i] - theta * e[:, i - 1]
    step1 = last + (theta * e[:, -1:])
    return np.repeat(step1, H, axis=1)


def q_posterior_up(y_ctx, sigma_m, s):
    """P(q_T = +1 | p_{1:T}) for each series: the forward algorithm of the two-state chain
    q_t in {-1, +1} (i.i.d., prior 1/2) with emission p_t - p_{t-1} ~ N((s/2)(q_t - q_{t-1}), sigma_m^2);
    the prior on q_1 is uninformative (the level a model sees is not tied to the generator's origin)."""
    r = np.diff(y_ctx, axis=1)
    states = np.array([-1.0, 1.0])
    shift = (s / 2.0) * (states[None, :] - states[:, None])           # [q_prev, q_now]
    log_a = np.full((r.shape[0], 2), np.log(0.5))
    for i in range(r.shape[1]):
        ll = -0.5 * ((r[:, i][:, None, None] - shift[None, :, :]) / sigma_m) ** 2
        log_a = np.logaddexp.reduce(log_a[:, :, None] + ll + np.log(0.5), axis=1)
        log_a -= log_a.max(axis=1, keepdims=True)
    a = np.exp(log_a)
    return a[:, 1] / a.sum(axis=1)


def oracle_forecast_exact(rung, y_ctx, info, H):
    """The conditional mean E[p_{T+h} | p_{1:T}], shape (n, H): persistence on N1-N3; on N4
    p_T - (s/2) E[q_T | p_{1:T}] at every h >= 1 (m is a martingale, future directions have mean
    zero), the trade-direction posterior from q_posterior_up. oracle_forecast() is the optimal
    *linear* forecast of the same rung, which this one dominates because q is two-valued."""
    last = y_ctx[:, -1:]
    if rung != "N4":
        return np.repeat(last, H, axis=1)
    eq = 2.0 * q_posterior_up(y_ctx, info["sigma"], info["spread"]) - 1.0
    return np.repeat(last - (info["spread"] / 2.0) * eq[:, None], H, axis=1)


def n4_optima_skill(n, T, H, seed, sigma_rel=SIGMA_REL, level0=LEVEL0):
    """Regenerate an N4 draw exactly as the drivers do (RUNGS['N4'] with default_rng(seed)) and return
    the per-horizon skill against persistence of the linear optimum and of the exact conditional mean,
    so that every table can quote shares of the exact optimum without the drivers having stored it."""
    rng = np.random.default_rng(seed)
    y, info = RUNGS["N4"](n, T, H, rng, sigma_rel=sigma_rel, level0=level0)
    y_ctx, y_true = y[:, :T], y[:, T:]
    last = np.repeat(y_ctx[:, -1:], H, axis=1)
    rp = np.mean((y_true - last) ** 2, axis=0)
    rl = np.mean((y_true - oracle_forecast("N4", y_ctx, info, H)) ** 2, axis=0)
    rb = np.mean((y_true - oracle_forecast_exact("N4", y_ctx, info, H)) ** 2, axis=0)
    return {"linear": (1 - rl / rp).tolist(), "exact": (1 - rb / rp).tolist(), "mse_persistence": rp.tolist(), "mse_linear": rl.tolist(), "mse_exact": rb.tolist()}


# ------------------------------------------------------------------ metrics

def departure_metrics(y_hat, y_ctx, y_true):
    """The decomposition of Proposition 3.2, measured per horizon.

    Returns the irreducible term, the imported term (squared departure from
    persistence), the model's own risk, and the empirical cross term, which the
    proposition says must vanish under the null.
    """
    last = y_ctx[:, -1:]
    dep = y_hat - last                       # what the model adds
    inc = y_true - last                      # what actually happened
    return {
        "irreducible": np.mean(inc ** 2, axis=0),
        "imported": np.mean(dep ** 2, axis=0),
        "risk_model": np.mean((y_true - y_hat) ** 2, axis=0),
        "risk_persistence": np.mean(inc ** 2, axis=0),
        "cross": np.mean(inc * dep, axis=0),
    }


def spectral_flatness(x, detrend=True, axis=-1):
    """Geometric over arithmetic mean of the periodogram, DC dropped.

    Near 1 means white, which per Section 4.1 REFUTES the imported-prior reading
    and leaves only the weaker claim that the model adds noise. Detrending is
    reported alongside the raw value because a smooth drift also depresses
    flatness without being periodic structure, and conflating the two would let
    a trend masquerade as a corpus rhythm.
    """
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    if detrend:
        n = x.shape[-1]
        t = np.arange(n)
        tc = t - t.mean()
        denom = np.sum(tc ** 2)
        slope = (x * tc).sum(-1, keepdims=True) / denom
        x = x - slope * tc - x.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    P = P[..., 1:]
    if P.shape[-1] == 0:
        return np.nan
    gm = np.exp(np.mean(np.log(P + 1e-300), axis=-1))
    am = np.mean(P, axis=-1)
    return gm / np.maximum(am, 1e-300)


def band_power_fraction(x, periods, axis=-1):
    """Fraction of periodogram power sitting within one bin of each named period.

    The declared-frequency test needs this: mass at period 24 when the model is
    told 'hourly' and at 7 when told 'daily' is what locates a prior rather than
    merely detecting a departure.
    """
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    P = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    P = P[..., 1:]
    freqs = np.fft.rfftfreq(n)[1:]
    total = np.sum(P, axis=-1)
    out = {}
    for p in periods:
        f0 = 1.0 / p
        if f0 > freqs[-1] or f0 < freqs[0]:
            out[f"p{p}"] = np.nan
            continue
        k = int(np.argmin(np.abs(freqs - f0)))
        lo, hi = max(0, k - 1), min(P.shape[-1], k + 2)
        out[f"p{p}"] = np.sum(P[..., lo:hi], axis=-1) / np.maximum(total, 1e-300)
    return out


def peak_period(x, axis=-1):
    """Period carrying the most power, DC excluded."""
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    P = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    P = P[..., 1:]
    freqs = np.fft.rfftfreq(n)[1:]
    k = np.argmax(P, axis=-1)
    return 1.0 / freqs[k]


def bootstrap_ci(vals, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI over series, so every reported number carries a spread."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = stat(vals[idx], axis=1)
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))
