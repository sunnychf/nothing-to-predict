"""Is the departure trend extrapolation? Regenerate the exact contexts (seed 4000,
same generator) and correlate each series' departure with its recent in-context
slope. Also test the alternative: a fixed directional bias (always up / down)."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nulls import RUNGS
RES = sys.argv[1] if len(sys.argv) > 1 else "results"          # directory holding departures_*.npz
for H in (128, 256):
    try:
        z = np.load(f"{RES}/departures_N1_H{H}.npz")
    except FileNotFoundError:
        continue
    dep, last, sigma = z["dep"], z["last"], float(z["sigma"])
    n = dep.shape[0]
    y, info = RUNGS["N1"](n, 512, H, np.random.default_rng(4000))
    ctx = y[:, :512]
    assert np.allclose(ctx[:, -1], last, atol=1e-4), "context regeneration mismatch"
    print(f"=== Chronos N1, H={H}, n={n} ===")
    print(f"{'h':>5} {'corr(dep, slope20)':>19} {'corr(dep, slope64)':>19} {'corr(dep, ctx drift)':>21} {'mean dep (sigma)':>17} {'frac dep>0':>11}")
    for k_desc, k in (("slope20", 20), ("slope64", 64)):
        pass
    s20 = (ctx[:, -1] - ctx[:, -21]) / 20.0          # recent slope, per step
    s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
    drift = (ctx[:, -1] - ctx[:, 0]) / 511.0          # whole-context drift
    for h in [0, 3, 15, 63, H - 1]:
        d = dep[:, h]
        c20 = np.corrcoef(d, s20)[0, 1]; c64 = np.corrcoef(d, s64)[0, 1]; cd = np.corrcoef(d, drift)[0, 1]
        print(f"{h+1:>5} {c20:>19.3f} {c64:>19.3f} {cd:>21.3f} {d.mean()/sigma:>17.3f} {np.mean(d>0):>11.3f}")
    # how much of the departure variance does a linear extrapolation of slope64 explain?
    for h in [15, H - 1]:
        d = dep[:, h]; x = s64 * (h + 1)                 # what pure extrapolation of slope64 would give
        beta = np.polyfit(x, d, 1)[0]
        r2 = np.corrcoef(d, x)[0, 1] ** 2
        print(f"  h={h+1}: dep ≈ {beta:.2f} × (slope64 × h); R² = {r2:.3f}   (beta=1 would be full extrapolation)")
    print()
