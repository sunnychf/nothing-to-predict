"""Does the imported prior shrink, grow, or hold as the model gets bigger?

Five Chronos-T5 sizes on identical N1 (direction, departure, skill) and N4
(positive control) contexts, H=64, n=128, S=100. Same seed as the direction
runs so numbers line up with §4.10. Resumable per (model, rung).
"""
import json, os, sys, time, numpy as np, torch
from pilot_a import load_pipeline, forecast, codec_floor, log
from nulls import RUNGS, oracle_forecast
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
J = f"{OUT}/scale_chronos.jsonl"
H, n, S, T = 64, 128, 100, 512
SIZES = [("amazon/chronos-t5-tiny", "8M"), ("amazon/chronos-t5-mini", "20M"),
         ("amazon/chronos-t5-small", "46M"), ("amazon/chronos-t5-base", "200M"),
         ("amazon/chronos-t5-large", "710M")]
done = set()
if os.path.exists(J):
    for l in open(J):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
for model, size in SIZES:
    keys = {r: f"{model}|{r}|H{H}|n{n}|S{S}" for r in ("N1", "N4")}
    if all(k in done for k in keys.values()):
        log(f"skip {model}"); continue
    log(f"=== {model} ({size}) ===")
    try:
        pipe = load_pipeline(model, "pretrained", "cuda")
    except Exception as e:
        log(f"!!! load failed for {model}: {e}"); continue
    for rung in ("N1", "N4"):
        if keys[rung] in done: continue
        y, info = RUNGS[rung](n, T, H, np.random.default_rng(4000))
        ctx, truth = y[:, :T], y[:, T:]
        sig = info["sigma"]
        t0 = time.time()
        s = forecast(pipe, ctx, H, S, batch=8, seed=0)
        yhat = s.mean(1); dep = yhat - ctx[:, -1:]
        mc = np.var(s, axis=1, ddof=1).mean(0) / S
        imp = (dep ** 2).mean(0)
        risk_m = ((truth - yhat) ** 2).mean(0); risk_p = ((truth - ctx[:, -1:]) ** 2).mean(0)
        fl = codec_floor(pipe, ctx, ctx[:, -1], H).mean(0)
        s64 = (ctx[:, -1] - ctx[:, -65]) / 64.0
        np.savez_compressed(f"{OUT}/departures_scale_{size}_{rung}_H{H}.npz", dep=dep.astype(np.float32), sigma=sig)
        rec = {"key": keys[rung], "model": model, "size": size, "rung": rung, "H": H, "n": n, "S": S,
               "dep_sigma": (np.sqrt(imp) / sig).tolist(),
               "dep_mc_sigma": (np.sqrt(np.maximum(imp - mc, 0)) / sig).tolist(),
               "grid_sigma": (np.sqrt(fl) / sig).tolist(),
               "skill": (1 - risk_m / risk_p).tolist(),
               "mean_dep_sigma": (dep.mean(0) / sig).tolist(),
               "frac_up": (dep > 0).mean(0).tolist(),
               "corr_slope64": [float(np.corrcoef(dep[:, h], s64)[0, 1]) for h in range(H)],
               "seconds": time.time() - t0}
        if rung == "N4":
            orc = oracle_forecast("N4", ctx, info, H)
            rec["oracle_skill"] = (1 - ((truth - orc) ** 2).mean(0) / risk_p).tolist()
        with open(J, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(keys[rung])
        if rung == "N1":
            log(f"  N1: dep-MC h1 {rec['dep_mc_sigma'][0]:.3f} h64 {rec['dep_mc_sigma'][-1]:.3f} | "
                f"skill h1 {rec['skill'][0]:+.3f} | mean h64 {rec['mean_dep_sigma'][-1]:+.2f}σ up {rec['frac_up'][-1]:.2f} "
                f"corr(slope) {rec['corr_slope64'][-1]:+.2f} | {rec['seconds']:.0f}s")
        else:
            log(f"  N4: skill h1 {rec['skill'][0]:+.3f} / oracle {rec['oracle_skill'][0]:+.3f} "
                f"= {100*rec['skill'][0]/rec['oracle_skill'][0]:.0f}% | {rec['seconds']:.0f}s")
    del pipe; torch.cuda.empty_cache()
log("=== SCALE SWEEP COMPLETE ===")
