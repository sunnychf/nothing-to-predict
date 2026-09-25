"""Experiment 4 summary: the corpus intervention of Section 4 with larger models (code/arch_train.py with --d 256
--layers 6, three seeds per design and corpus). Reads experiments/results/arch_scale/arch_<arch>_<corpus>_s<seed>.json,
aggregates exactly as code/arch_summary.py does (mean and between-seed standard deviation), and writes
experiments/results/arch_scale/arch_scale_summary.json and tables/arch_scale.tex (same layout as tables/arch.tex).
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/arch_scale_summary.py"""
import glob, json, os
import numpy as np

RES = os.path.join("experiments", "results", "arch_scale")
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, "arch_*_s*.json"))) if "summary" not in os.path.basename(p)]
assert runs, "no arch_*.json in " + RES


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


S = {}
for corpus in ("up", "sym"):
    for arch in ("encoder", "decoder"):
        rs = [r for r in runs if r["arch"] == arch and r["corpus"] == corpus]
        if not rs:
            continue
        S[f"{arch}|{corpus}"] = {
            "n_seeds": len(rs), "seeds": [r["seed"] for r in rs], "params": rs[0]["params"], "steps": rs[0]["steps"], "d": rs[0]["d"], "layers": rs[0]["layers"],
            "heldout_mse": ms([r["corpus_heldout_mse_norm"] for r in rs]), "heldout_persistence_mse": ms([r["corpus_heldout_persistence_mse_norm"] for r in rs]),
            "dir": {h: {f: ms([r["direction"][h][f] for r in rs]) for f in ("mean_dep", "frac_up", "even_share", "corr_slope64", "rms_dep", "skill")} for h in ("1", "16", "64", "128")},
            "dir_mean_dep_se_within": {h: ms([r["direction"][h]["mean_dep_se"] for r in rs])[0] for h in ("1", "16", "64", "128")},
            "per_seed_dir128": [(r["seed"], r["direction"]["128"]["mean_dep"], r["direction"]["128"]["mean_dep_se"], r["direction"]["128"]["frac_up"]) for r in rs],
            "ladder": {rung: {f: ms([r["ladder"][rung][f] for r in rs]) for f in rs[0]["ladder"][rung]} for rung in ("N1", "N4")}}
# registered prediction (experiments/PREREGISTRATION.md, Experiment 4): the growth corpus produces an upward N1 bias in both designs
S["preregistered_growth_upward_both_designs"] = all(
    S[f"{a}|up"]["dir"]["128"]["mean_dep"][0] > 0 and S[f"{a}|up"]["dir"]["128"]["frac_up"][0] > 0.5 for a in ("encoder", "decoder") if f"{a}|up" in S)
json.dump(S, open(os.path.join(RES, "arch_scale_summary.json"), "w"), indent=1)


def num(v, d, signed=True):
    return (f"{v:+.{d}f}" if signed else f"{v:.{d}f}").replace("-", "\\textminus{}")


W = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
     r" &  & \multicolumn{2}{c}{\textbf{Held-out MSE}} & \multicolumn{3}{c}{\textbf{N1, $h$ = 128}} & \textbf{N1 skill} & \textbf{N4 share} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
     r"\textbf{Model design} & \textbf{Corpus} & \textbf{Model} & \textbf{Persistence} & \textbf{Mean / $\sigma$} & \textbf{Upward} & \textbf{$\rho_{\rm even}$} & $h$ = 16 & $h$ = 1 \\",
     r"\midrule"]
NAME = {"encoder": "Masked encoder", "decoder": "Decoder-only"}; CORP = {"up": "Growth", "sym": "Symmetric"}
for arch in ("encoder", "decoder"):
    for corpus in ("up", "sym"):
        k = f"{arch}|{corpus}"
        if k not in S:
            continue
        r = S[k]; d = r["dir"]["128"]
        mse = r["heldout_mse"][0]; pers = r["heldout_persistence_mse"][0]
        W.append(" & ".join([NAME[arch] if corpus == "up" else "", CORP[corpus],
                             (r"\textbf{%.3f}" % mse) if mse < pers else f"{mse:.3f}", f"{pers:.3f}",
                             num(d["mean_dep"][0], 2) + f"\\,\\textpm\\,{d['mean_dep'][1]:.2f}", num(d["frac_up"][0], 2, False),
                             num(d["even_share"][0], 2, False), num(r["ladder"]["N1"]["skill_16"][0], 3), num(r["ladder"]["N4"]["share_exact_1"][0], 2, False)]) + r" \\")
    if arch == "encoder":
        W.append(r"\addlinespace[3pt]")
W += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join("tables", "arch_scale.tex"), "w").write("\n".join(W) + "\n")
for k, r in S.items():
    if not isinstance(r, dict):
        continue
    d = r["dir"]["128"]
    print(f"{k:14s} seeds {r['seeds']} params {r['params']} | heldout {r['heldout_mse'][0]:.3f} (pers {r['heldout_persistence_mse'][0]:.3f}) | N1 h=128 dep {d['mean_dep'][0]:+.2f}±{d['mean_dep'][1]:.2f} up {d['frac_up'][0]:.2f} even {d['even_share'][0]:.2f} | N1 skill16 {r['ladder']['N1']['skill_16'][0]:+.3f} | N4 share {r['ladder']['N4']['share_exact_1'][0]:+.2f}")
    print("   per seed h=128 (seed, mean, se, up):", [tuple(round(x, 3) if isinstance(x, float) else x for x in t) for t in r["per_seed_dir128"]])
print("pre-registered growth -> upward in both designs:", S["preregistered_growth_upward_both_designs"])
