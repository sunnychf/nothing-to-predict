"""Aggregate the controlled architecture comparison (arch_train.py): results/arch_<arch>_<corpus>_s<seed>.json
-> results/arch_summary.json and paper/tables/arch.tex (mean +- sd over seeds).
Usage: python arch_summary.py --res results --tables paper/tables
"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); ap.add_argument("--tables", default="paper/tables"); args = ap.parse_args()
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.res, "arch_*_s*.json")))]
assert runs, "no arch_*.json"
ARCH = {"encoder": "masked encoder", "decoder": "decoder-only"}; CORP = {"up": "growth corpus", "sym": "symmetric corpus"}
S = {}
def ms(v): v = np.asarray(v, float); return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
for corpus in ("up", "sym"):
    for arch in ("encoder", "decoder"):
        rs = [r for r in runs if r["arch"] == arch and r["corpus"] == corpus]
        if not rs: continue
        S[f"{arch}|{corpus}"] = {"n_seeds": len(rs), "params": rs[0]["params"], "steps": rs[0]["steps"],
            "heldout_mse": ms([r["corpus_heldout_mse_norm"] for r in rs]), "heldout_persistence_mse": ms([r["corpus_heldout_persistence_mse_norm"] for r in rs]),
            "heldout_by_kind": {k: ms([r["corpus_heldout_mse_norm_by_kind"][k] for r in rs]) for k in rs[0]["corpus_heldout_mse_norm_by_kind"]},
            "dir": {h: {f: ms([r["direction"][h][f] for r in rs]) for f in ("mean_dep", "frac_up", "even_share", "corr_slope64", "rms_dep", "skill")} for h in ("1", "16", "64", "128")},
            "dir_mean_dep_se_within": {h: ms([r["direction"][h]["mean_dep_se"] for r in rs])[0] for h in ("1", "16", "64", "128")},
            "per_seed_dir128": [(r["seed"], r["direction"]["128"]["mean_dep"], r["direction"]["128"]["frac_up"]) for r in rs],
            "ladder": {rung: {f: ms([r["ladder"][rung][f] for r in rs]) for f in rs[0]["ladder"][rung]} for rung in ("N1", "N4")}}
json.dump(S, open(os.path.join(args.res, "arch_summary.json"), "w"), indent=1)
# LaTeX body
W = [r"\begin{tabular}{ll|cc|ccc|cc}", r"\toprule",
     r" & & \multicolumn{2}{c|}{corpus fit (MSE)} & \multicolumn{3}{c|}{N1, $h{=}128$} & N1 skill & N4 share \\",
     r"Corpus & Architecture & model & persist. & mean dep.\ $/\sigma$ & frac.\ up & even share & $h{=}16$ & $h{=}1$ \\", r"\midrule"]
for corpus in ("up", "sym"):
    for arch in ("encoder", "decoder"):
        k = f"{arch}|{corpus}"
        if k not in S: continue
        r = S[k]; d = r["dir"]["128"]
        W.append(f"{CORP[corpus]} & {ARCH[arch]} & {r['heldout_mse'][0]:.3f} & {r['heldout_persistence_mse'][0]:.3f} & ${d['mean_dep'][0]:+.2f}\\pm{d['mean_dep'][1]:.2f}$ & {d['frac_up'][0]:.2f} & {d['even_share'][0]:.2f} & ${r['ladder']['N1']['skill_16'][0]:+.3f}$ & {r['ladder']['N4']['share_exact_1'][0]:.2f} \\\\")
    if corpus == "up": W.append(r"\midrule")
W += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(args.tables, exist_ok=True); open(os.path.join(args.tables, "arch.tex"), "w").write("\n".join(W) + "\n")
for k, r in S.items():
    d = r["dir"]["128"]
    print(f"{k:14s} seeds {r['n_seeds']} | heldout {r['heldout_mse'][0]:.3f} (pers {r['heldout_persistence_mse'][0]:.3f}) | N1 h=128 dep {d['mean_dep'][0]:+.2f}±{d['mean_dep'][1]:.2f} (within-seed se {r['dir_mean_dep_se_within']['128']:.2f}) up {d['frac_up'][0]:.2f} even {d['even_share'][0]:.2f} corr {d['corr_slope64'][0]:+.2f} | N1 skill16 {r['ladder']['N1']['skill_16'][0]:+.3f} | N4 share {r['ladder']['N4']['share_exact_1'][0]:+.2f} (linear {r['ladder']['N4']['share_linear_1'][0]:+.2f})")
    print("   per seed h=128:", r["per_seed_dir128"])
