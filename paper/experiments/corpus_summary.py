"""Experiment 3 summary (descriptive; no directional prediction was registered): trend statistics of three public
pretraining corpora at h = 128, from experiments/results/corpus/corpus_trend_v2.json (corpus_trend.py, seed 0).
Per subset: share of moving windows whose 128-step change is positive, median change and change clipped to
+-10 context sigmas. Per corpus: median over subsets, series-weighted pool, the same pool without the M4 subsets,
and the share of subsets leaning up (> 0.55) or down (< 0.45). Writes experiments/results/corpus/corpus_summary.json
and tables/corpus_trend.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/corpus_summary.py"""
import json, os
import numpy as np

D = os.path.join("experiments", "results", "corpus")
src = json.load(open(os.path.join(D, "corpus_trend_v2.json")))
NAME = {"chronos": "Chronos datasets", "lotsa": "LOTSA (Moirai)", "time300b": "Time-300B (Time-MoE)"}
H = "128"
out = {}
for c, v in src.items():
    rows = [(s, x["series"], x["stats"][H]) for s, x in v["subsets"].items() if x["stats"][H] and x["stats"][H]["up_share_of_moves"] is not None]
    w = np.array([n for _, n, _ in rows], float); us = np.array([st["up_share_of_moves"] for _, _, st in rows])
    cm = np.array([st["mean_z_clip10"] for _, _, st in rows]); md = np.array([st["median_z"] for _, _, st in rows])
    m4 = np.array(["m4_" in s for s, _, _ in rows])
    out[c] = {"subsets": int(len(rows)), "series": int(w.sum()), "windows": int(sum(st["n"] for _, _, st in rows)),
              "rising_median_subset": float(np.median(us)), "rising_pooled": float((us * w).sum() / w.sum()),
              "rising_pooled_without_m4": float((us[~m4] * w[~m4]).sum() / w[~m4].sum()),
              "subsets_leaning_up": float(np.mean(us > 0.55)), "subsets_leaning_down": float(np.mean(us < 0.45)),
              "median_change_median_subset": float(np.median(md)),
              "clipped_mean_median_subset": float(np.median(cm)), "clipped_mean_pooled": float((cm * w).sum() / w.sum()),
              "clipped_mean_pooled_without_m4": float((cm[~m4] * w[~m4]).sum() / w[~m4].sum()),
              "m4_subsets": [s for s, _, _ in rows if "m4_" in s], "m4_rising": {s: st["up_share_of_moves"] for s, _, st in rows if "m4_" in s},
              "m4_share_of_pooled_clipped_mean": float((cm[m4] * w[m4]).sum() / (cm * w).sum())}
json.dump(out, open(os.path.join(D, "corpus_summary.json"), "w"), indent=1)


def num(v, d, signed=False):
    return (f"{v:+.{d}f}" if signed else f"{v:.{d}f}").replace("-", "\\textminus{}")


L = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
     r" & & & \multicolumn{3}{c}{\textbf{Share of moves that rise}} & \multicolumn{2}{c}{\textbf{Subsets leaning}} & \multicolumn{2}{c}{\textbf{Change ($\sigma$)}} \\",
     r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
     r"\textbf{Corpus} & \textbf{Subsets} & \textbf{Series} & \textbf{Median} & \textbf{Pooled} & \textbf{No M4} & \textbf{Up} & \textbf{Down} & \textbf{Median} & \textbf{Clipped mean} \\",
     r"\midrule"]
for c in ("chronos", "lotsa", "time300b"):
    o = out[c]
    L.append(" & ".join([NAME[c], str(o["subsets"]), f"{o['series']:,}".replace(",", "\\,"), num(o["rising_median_subset"], 2), num(o["rising_pooled"], 2),
                         num(o["rising_pooled_without_m4"], 2), num(o["subsets_leaning_up"], 2), num(o["subsets_leaning_down"], 2),
                         num(o["median_change_median_subset"], 2, True), num(o["clipped_mean_pooled"], 2, True) + " (" + num(o["clipped_mean_pooled_without_m4"], 2, True) + ")"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join("tables", "corpus_trend.tex"), "w").write("\n".join(L) + "\n")
for c, o in out.items():
    print(c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in o.items() if k not in ("m4_subsets",)})
