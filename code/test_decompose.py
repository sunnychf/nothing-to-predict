"""Does decompose() actually separate a drift from a rhythm? Test it on signals
whose answer we know, before trusting it on model output."""
import sys, numpy as np
sys.path.insert(0, ".")
from spectral_deep import decompose

ok = True
def chk(l, c, d=""):
    global ok; ok &= bool(c); print(("PASS  " if c else "FAIL  ") + l + ("   " + d if d else ""))

H, n = 128, 128
rng = np.random.default_rng(0)
t = np.arange(H)

# 1. pure linear drift -> should be almost all drift, no seasonal excess
drift = (0.01 * t)[None, :] + 0.02 * rng.standard_normal((n, H))
d = decompose(drift)
chk("pure drift: drift share is dominant", d["share_drift_periodGtH4"] > 0.8,
    f"{d['share_drift_periodGtH4']:.3f}")
chk("pure drift: no period-24 excess over continuum",
    d["band24"]["excess_over_continuum"] < 2.0,
    f"{d['band24']['excess_over_continuum']:.2f}x")

# 2. pure period-24 rhythm -> seasonal excess must be large, drift share small
seas = np.sin(2*np.pi*t/24.0)[None, :] + 0.2*rng.standard_normal((n, H))
d = decompose(seas)
chk("period-24 rhythm: large excess at band24",
    d["band24"]["excess_over_continuum"] > 10,
    f"{d['band24']['excess_over_continuum']:.1f}x")
chk("period-24 rhythm: drift share is small", d["share_drift_periodGtH4"] < 0.3,
    f"{d['share_drift_periodGtH4']:.3f}")
chk("period-24 rhythm: peak period is ~24", abs(d["peak_period_of_mean_psd"]-24) < 4,
    f"{d['peak_period_of_mean_psd']:.1f}")

# 3. drift PLUS a weak rhythm -> must still surface the rhythm, which is the
#    case that matters: a real prior sitting on a steep low-frequency slope.
both = (0.01*t)[None, :] + 0.15*np.sin(2*np.pi*t/24.0)[None, :] + 0.02*rng.standard_normal((n, H))
d = decompose(both)
chk("drift+rhythm: drift still dominates total power",
    d["share_drift_periodGtH4"] > 0.5, f"{d['share_drift_periodGtH4']:.3f}")
chk("drift+rhythm: band24 excess STILL detected (this is why we use the",
    d["band24"]["excess_over_continuum"] > 3,
    f"local continuum, not total power) {d['band24']['excess_over_continuum']:.1f}x")

# 4. white noise -> no excess anywhere
d = decompose(rng.standard_normal((n, H)))
chk("white: no band excess", max(d[f"band{p}"]["excess_over_continuum"] for p in (7,24)) < 2.0,
    f"max {max(d[f'band{p}']['excess_over_continuum'] for p in (7,24)):.2f}x")
print("\n" + ("DECOMPOSE OK" if ok else "DECOMPOSE FAILED"))
sys.exit(0 if ok else 1)
