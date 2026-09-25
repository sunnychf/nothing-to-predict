"""A synthetic 'general' pretraining corpus for the controlled architecture comparison (arch_train.py).

The paper attributes the models' drift on structureless input to what their corpora look like: series
that are autocorrelated, seasonal and trending, with growth more common than decline. This generator
produces such series with the mixture weights below, in units of a unit innovation, so that the same
corpus (same seed stream) can be fed to two architectures and only the architecture differs. Every
window has `T + H` points; the level is irrelevant because every model normalises its context.

    kinds (weights): seasonal AR (0.45): AR(1) with phi ~ U(0.5, 0.98) plus daily and weekly sinusoids;
                     trending (0.25): the same plus a linear trend beta * t;
                     integrated (0.20): a random walk with drift, plus seasonality;
                     bounce (0.10): the N4 bid-ask model p = m + (s/2) q with s ~ U(1, 4), so that the
                                    positive control's structure is learnable in principle.
    corpus 'up':  beta ~ N(+0.02, 0.03) per step and the random walks' drift ~ N(+0.02, 0.03): growth is the norm.
    corpus 'sym': the same with zero means, so no direction is preferred.
"""
import numpy as np

WEIGHTS = {"seasonal": 0.45, "trending": 0.25, "integrated": 0.20, "bounce": 0.10}


def _ar1(n, L, rng, phi):
    phi = np.asarray(phi).reshape(n)
    e = rng.standard_normal((n, L)); x = np.zeros((n, L)); x[:, 0] = e[:, 0] / np.sqrt(1 - phi ** 2)
    for t in range(1, L): x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def _season(n, L, rng):
    t = np.arange(L)[None, :]
    a1, a2 = rng.uniform(0, 3, (n, 1)), rng.uniform(0, 3, (n, 1)); p1, p2 = rng.uniform(0, 2 * np.pi, (n, 1)), rng.uniform(0, 2 * np.pi, (n, 1))
    return a1 * np.sin(2 * np.pi * t / 24 + p1) + a2 * np.sin(2 * np.pi * t / 168 + p2)


def make_corpus(n, L, rng, corpus="up"):
    """(n, L) float32 windows and their kinds."""
    mu = 0.02 if corpus == "up" else 0.0
    kinds = rng.choice(list(WEIGHTS), size=n, p=list(WEIGHTS.values()))
    out = np.zeros((n, L)); t = np.arange(L)[None, :]
    for kind in WEIGHTS:
        idx = np.where(kinds == kind)[0]; m = len(idx)
        if m == 0: continue
        if kind == "seasonal":
            out[idx] = _ar1(m, L, rng, rng.uniform(0.5, 0.98, (m, 1))) + _season(m, L, rng)
        elif kind == "trending":
            beta = rng.normal(mu, 0.03, (m, 1))
            out[idx] = _ar1(m, L, rng, rng.uniform(0.5, 0.98, (m, 1))) + _season(m, L, rng) + beta * t
        elif kind == "integrated":
            drift = rng.normal(mu, 0.03, (m, 1))
            out[idx] = np.cumsum(rng.standard_normal((m, L)) + drift, axis=1) + _season(m, L, rng)
        else:                                                    # bounce: N4 with a random spread
            s = rng.uniform(1.0, 4.0, (m, 1)); q = rng.choice([-1.0, 1.0], size=(m, L))
            out[idx] = np.cumsum(rng.standard_normal((m, L)), axis=1) + 0.5 * s * q
    return out.astype(np.float32), kinds


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for c in ("up", "sym"):
        x, k = make_corpus(20000, 640, rng, c)
        d = x[:, -1] - x[:, 511]
        print(c, {kind: int((k == kind).sum()) for kind in WEIGHTS}, "mean 128-step move", round(float(d.mean()), 2), "std", round(float(d.std()), 2), "frac up", round(float((d > 0).mean()), 3))
