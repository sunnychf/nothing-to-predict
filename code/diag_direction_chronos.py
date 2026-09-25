import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nulls import RUNGS
RES = sys.argv[1] if len(sys.argv) > 1 else "results"          # directory holding departures_*.npz
print(f"{'rung':>4} {'H':>4} {'h':>5} {'mean dep(σ)':>12} {'RMS dep(σ)':>11} {'bias²/RMS²':>11} {'frac up':>8} {'binom p (H0: 50%)':>18}")
from math import comb
def binom_p(k, n):
    # two-sided exact test of frac = 0.5
    p_one = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(p_one, 1 - p_one + comb(n, k) / 2 ** n))
for rung in ("N1", "N2", "N4"):
    for H in (128, 256):
        try: z = np.load(f"{RES}/departures_{rung}_H{H}.npz")
        except FileNotFoundError: continue
        dep, sigma = z["dep"], float(z["sigma"])
        n = dep.shape[0]
        for h in [0, 15, H - 1]:
            d = dep[:, h]
            m, r = d.mean() / sigma, np.sqrt((d ** 2).mean()) / sigma
            k = int((d > 0).sum())
            print(f"{rung:>4} {H:>4} {h+1:>5} {m:>+12.3f} {r:>11.3f} {(m/r)**2:>11.2f} {k/n:>8.3f} {binom_p(k, n):>18.2e}")
