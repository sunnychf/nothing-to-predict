"""Experiment 2 summary: released Chronos-Bolt-small vs fine-tuned plainly vs fine-tuned with mirror augmentation
(three seeds each), from experiments/results/finetune/eval_bolt_*.json. Writes finetune_summary.json and tables/finetune.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/finetune_summary.py"""
import json, os
import numpy as np

D = os.path.join("experiments", "results", "finetune")


def load(t):
    return json.load(open(os.path.join(D, f"eval_bolt_{t}.json")))


# (row label, getter, decimals, signed)
ROWS = [("N1 upward fraction, $h{=}128$", lambda R: R["N1_direction"]["128"]["frac_up"], 2, False),
        ("N1 mean departure ($\\sigma$), $h{=}128$", lambda R: R["N1_direction"]["128"]["mean_dep_sigma"], 2, True),
        ("N1 skill, $h{=}16$", lambda R: R["ladder_N1"]["skill_h16"], 3, True),
        ("N4 share of optimum, $h{=}1$", lambda R: R["ladder_N4"]["oracle_share_h1"], 3, False),
        ("ETTh1 skill, $h{=}16$", lambda R: R["etth1"]["16"], 3, True),
        ("Raw equities, skill, $h{=}128$", lambda R: R["daily_raw"]["eq"]["128"]["skill"], 3, True),
        ("Raw equities, mirror-corrected", lambda R: R["daily_raw"]["eq"]["128"]["skill_mirror"], 3, True),
        ("Raw equities, even part (\\%/yr)", lambda R: R["daily_raw"]["eq"]["128"]["even_mean_annualised"], 1, True),
        ("Sign-randomised skill, $h{=}128$", lambda R: R["daily_sign_randomised"]["128"]["skill"], 3, True),
        ("Sign-randomised upward fraction", lambda R: R["daily_sign_randomised"]["128"]["frac_up"], 2, False)]
ARMS = {"released": ["released"], "plain": [f"plain_s{s}" for s in range(3)], "mirror": [f"mirror_s{s}" for s in range(3)]}
out = {}
for arm, tags in ARMS.items():
    Rs = [load(t) for t in tags]
    out[arm] = {}
    for name, f, _, _ in ROWS:
        v = [float(f(R)) for R in Rs]
        out[arm][name] = {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
# pre-registered checks (experiments/PREREGISTRATION.md, Experiment 2), per seed pairing and on the means
up = "N1 upward fraction, $h{=}128$"; et = "ETTh1 skill, $h{=}16$"; eq = "Raw equities, skill, $h{=}128$"
pv = {k: out["plain"][k]["values"] for k in (up, et, eq)}; mv = {k: out["mirror"][k]["values"] for k in (up, et, eq)}
rel_up = out["released"][up]["values"][0]
out["preregistered"] = {
    "N1_up_closer_to_half_every_seed": all(abs(m - 0.5) < min(abs(p - 0.5) for p in pv[up] + [rel_up]) for m in mv[up]),
    "ETTh1_within_0.02_every_seed": all(abs(m - p) <= 0.02 for m, p in zip(mv[et], pv[et])),
    "raw_equity_lower_every_seed": all(m < p for m, p in zip(mv[eq], pv[eq])),
}
json.dump(out, open(os.path.join(D, "finetune_summary.json"), "w"), indent=1)


def num(v, d, signed):
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return s.replace("-", "\\textminus{}")


L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     r"\textbf{Probe} & \textbf{Released} & \textbf{Fine-tuned} & \textbf{Mirror-augmented} \\", r"\midrule"]
for name, _, d, signed in ROWS:
    cells = [num(out["released"][name]["mean"], d, signed)]
    for arm in ("plain", "mirror"):
        v = out[arm][name]
        cells.append(num(v["mean"], d, signed) + f"\\,\\textpm\\,{v['sd']:.{d}f}")
    L.append(name + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join("tables", "finetune.tex"), "w").write("\n".join(L) + "\n")
for arm in ARMS:
    print(arm, {k: [round(x, 3) for x in v["values"]] for k, v in out[arm].items()})
print("pre-registered:", out["preregistered"])
