"""N4's exact conditional mean against its optimal linear forecast (nulls.oracle_forecast_exact vs
nulls.oracle_forecast). The bounce model p_t = m_t + (s/2) q_t has MA(1) observed returns, so the
MA(1) forecast p_T + theta e_T is the optimal linear forecast of p_{T+h}; the conditional mean is
E[p_{T+h} | p_{1:T}] = p_T - (s/2) E[q_T | p_{1:T}] with the trade-direction posterior from a
two-state forward filter, and it dominates the linear one because q is two-valued.

    python n4_bayes.py [results]     recomputes both optima on the ladder's own N4 draws (seeds 1000-1002,
                                     512 series, context 512, horizon 16) and on the scale sweep's draw
                                     (seed 4000, 128 series, horizon 64), checks the posterior's calibration
                                     against the generator's hidden directions, and writes results/n4_optima.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nulls import RUNGS, n4_optima_skill, q_posterior_up, oracle_forecast, oracle_forecast_exact

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
out = {"ladder": {"n": 512, "T": 512, "H": 16, "seeds": [1000, 1001, 1002]}, "scale": {"n": 128, "T": 512, "H": 64, "seed": 4000}}
rows = [n4_optima_skill(512, 512, 16, sd) for sd in out["ladder"]["seeds"]]
out["ladder"]["per_seed"] = rows
out["ladder"]["skill_linear_h1"] = float(np.mean([r["linear"][0] for r in rows])); out["ladder"]["skill_exact_h1"] = float(np.mean([r["exact"][0] for r in rows]))
out["ladder"]["skill_linear_h16"] = float(np.mean([r["linear"][15] for r in rows])); out["ladder"]["skill_exact_h16"] = float(np.mean([r["exact"][15] for r in rows]))
out["ladder"]["exact_over_linear_h1"] = out["ladder"]["skill_exact_h1"] / out["ladder"]["skill_linear_h1"]
sc = n4_optima_skill(128, 512, 64, 4000); out["scale"].update({"skill_linear_h1": sc["linear"][0], "skill_exact_h1": sc["exact"][0]})
# calibration of the posterior against the hidden directions, and the two one-step departures compared
rng = np.random.default_rng(1000); y, info = RUNGS["N4"](512, 512, 16, rng); ctx = y[:, :512]
p_up = q_posterior_up(ctx, info["sigma"], info["spread"]); q_T = info["q"][:, 511]
bins = np.clip((p_up * 5).astype(int), 0, 4)
out["calibration_seed1000"] = {str(b): {"n": int((bins == b).sum()), "mean_posterior": float(p_up[bins == b].mean()) if (bins == b).any() else None,
                                        "frac_up": float((q_T[bins == b] > 0).mean()) if (bins == b).any() else None} for b in range(5)}
lin = oracle_forecast("N4", ctx, info, 1)[:, 0] - ctx[:, -1]; ex = oracle_forecast_exact("N4", ctx, info, 1)[:, 0] - ctx[:, -1]
out["corr_linear_exact_step_seed1000"] = float(np.corrcoef(lin, ex)[0, 1])
os.makedirs(out_dir, exist_ok=True); json.dump(out, open(os.path.join(out_dir, "n4_optima.json"), "w"), indent=1)
L = out["ladder"]
print(f"ladder N4, h=1 skill vs persistence: linear {L['skill_linear_h1']:+.4f}, exact {L['skill_exact_h1']:+.4f} (ratio {L['exact_over_linear_h1']:.3f}); h=16: {L['skill_linear_h16']:+.4f} / {L['skill_exact_h16']:+.4f}")
print(f"scale draw (seed 4000, n=128): linear {out['scale']['skill_linear_h1']:+.4f}, exact {out['scale']['skill_exact_h1']:+.4f}")
print("posterior calibration (bins of P(q_T=+1)):", {b: (v['n'], round(v['mean_posterior'], 2) if v['mean_posterior'] is not None else None, round(v['frac_up'], 2) if v['frac_up'] is not None else None) for b, v in out['calibration_seed1000'].items()})
