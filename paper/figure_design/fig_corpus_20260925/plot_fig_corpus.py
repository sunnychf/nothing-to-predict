"""Section 4 figure: the controlled corpus experiment. A masked encoder and a decoder-only model, trained from scratch
on a growth corpus (trends and drifts with mean +0.02 per step) or on a symmetric one (mean zero), three seeds each,
at two sizes (0.54M and 3.18M parameters).

Reads only existing outputs (nothing is typed by hand):
  ../results/arch_{encoder,decoder}_{up,sym}_s{0,1,2}.json                   0.54M runs (code/arch_train.py)
  experiments/results/arch_scale/arch_{encoder,decoder}_{up,sym}_s{0,1,2}.json 3.18M runs (--d 256 --layers 6)
  ../results/arch_summary.json, experiments/results/arch_scale/arch_scale_summary.json  seed means, checked against
(a) mean departure on the 128 N1 random walks at h = 128 (sigma units), (b) fraction of those forecasts above the last
value, (c) skill against the last value on held-out corpus windows, 1 - MSE(model) / MSE(last value), the paper's
definition of skill. Writes figures/fig_corpus.{pdf,svg}, figures/png/fig_corpus.png, figures/fig_corpus.alignment.json
and figure_design/fig_corpus_20260925/fig_corpus_facts.json. Run from paper_revision/ with the nature-figure env and the
skill scripts on PYTHONPATH."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_common"))
import subcaptions as sc   # '(a) short name' under each panel (authors' style, 2026-09-24)

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = "figures"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
# design colours follow the paper's families: Moirai-1.1's teal for the masked encoder, TimesFM-2.0's grey for decoder-only
DES = [("encoder", "masked encoder", "#42949E"), ("decoder", "decoder-only", "#4D4D4D")]
CORP = [("up", "growth"), ("sym", "symmetric")]
SIZES = [("0.54M", os.path.join("..", "results"), os.path.join("..", "results", "arch_summary.json"), True),
         ("3.18M", os.path.join("experiments", "results", "arch_scale"),
          os.path.join("experiments", "results", "arch_scale", "arch_scale_summary.json"), False)]

runs, facts = {}, {"runs": {}, "means": {}}
for size, d, summ, filled in SIZES:
    S = json.load(open(summ))
    for arch, _, _ in DES:
        for corp, _ in CORP:
            vals = []
            for seed in (0, 1, 2):
                r = json.load(open(os.path.join(d, f"arch_{arch}_{corp}_s{seed}.json")))
                dep, up = r["direction"]["128"]["mean_dep"], r["direction"]["128"]["frac_up"]
                sk = 1.0 - r["corpus_heldout_mse_norm"] / r["corpus_heldout_persistence_mse_norm"]
                vals.append((dep, up, sk))
                facts["runs"][f"{size}|{arch}|{corp}|s{seed}"] = {"mean_dep_h128": dep, "frac_up_h128": up, "heldout_skill": sk}
            v = np.array(vals); runs[(size, arch, corp)] = v
            m = S[f"{arch}|{corp}"]
            assert abs(v[:, 0].mean() - m["dir"]["128"]["mean_dep"][0]) < 1e-9, (size, arch, corp, "mean departure")
            assert abs(v[:, 1].mean() - m["dir"]["128"]["frac_up"][0]) < 1e-9, (size, arch, corp, "upward fraction")
            assert abs(np.mean([json.load(open(os.path.join(d, f"arch_{arch}_{corp}_s{s}.json")))["corpus_heldout_mse_norm"]
                                for s in (0, 1, 2)]) - m["heldout_mse"][0]) < 1e-9, (size, arch, corp, "held-out MSE")
            facts["means"][f"{size}|{arch}|{corp}"] = {"mean_dep_h128": float(v[:, 0].mean()), "frac_up_h128": float(v[:, 1].mean()),
                                                       "heldout_skill": float(v[:, 2].mean())}

W_IN, H_IN = 5.5, 1.6
fig, ax = plt.subplots(1, 3, figsize=(W_IN, H_IN), layout="constrained")
XD = {"encoder": -0.2, "decoder": +0.2}; XS = {"0.54M": -0.065, "3.18M": +0.065}; JIT = (-0.018, 0.0, 0.018)
for k, (a, ref, ylab) in enumerate(zip(ax, (0.0, 0.5, 0.0), ("mean departure (σ)", "fraction above last value", "held-out skill"))):
    a.axhline(ref, color="0.55", ls=":", lw=0.8, zorder=0)
    for i, (corp, _) in enumerate(CORP):
        for arch, _, col in DES:
            for size, _, _, filled in SIZES:
                v = runs[(size, arch, corp)][:, k]; x0 = i + XD[arch] + XS[size]
                a.plot([x0 - 0.045, x0 + 0.045], [v.mean()] * 2, color=col, lw=1.2, solid_capstyle="butt", zorder=2)
                a.plot(x0 + np.array(JIT), v, "o", ms=3.0, color=col, mfc=col if filled else "white", mew=0.8, zorder=3)
    a.set_xticks([0, 1]); a.set_xticklabels([lab for _, lab in CORP]); a.set_xlim(-0.55, 1.55)
    a.tick_params(axis="x", length=0); a.set_ylabel(ylab)
ax[1].set_ylim(0.4, 1.0); ax[2].set_ylim(0.0, 0.7)
leg = fig.legend(handles=[Line2D([], [], color=col, lw=2.0, label=lab) for _, lab, col in DES]   # designs: colour; sizes: fill
                 + [Line2D([], [], ls="", marker="o", ms=3.0, color="0.3", label="0.54M parameters"),
                    Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec="0.3", mew=0.8, label="3.18M parameters")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.2, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.2)

fig.canvas.draw(); fig.set_layout_engine("none")
px = fig.dpi / 72.0; Wpt = W_IN * 72.0
PANEL_H = 60.0
for _ in range(2):
    dec = sc.decorations_pt(fig, ax)
    left = dec[0][0] + 5.0; gutter = max(d[0] for d in dec[1:]) + 8.0; right = 6.0
    w = (Wpt - left - right - 2 * gutter) / 3
    bottom = sc.band_pt(fig, ax) + sc.MARGIN
    top = (fig.get_figheight() * 72.0 - leg.get_window_extent(fig.canvas.get_renderer()).y0 / px) + 4.0 + max(d[2] for d in dec)
    Hpt = top + PANEL_H + bottom
    fig.set_size_inches(W_IN, Hpt / 72.0)
    for k, a in enumerate(ax):
        a.set_position([(left + k * (w + gutter)) / Wpt, bottom / Hpt, w / Wpt, PANEL_H / Hpt])
facts["panel_labels"] = sc.add(fig, [[(ax[0], "(a) Mean departure at $h$ = 128"), (ax[1], "(b) Fraction above the last value"),
                                      (ax[2], "(c) Skill on held-out windows")]])
base = os.path.join(FIG, "fig_corpus")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_corpus.png"), dpi=300)
facts["axes"] = {"b_ylim": list(ax[1].get_ylim()), "c_ylim": list(ax[2].get_ylim())}
facts["figure_height_in"] = fig.get_figheight()
json.dump(facts, open(os.path.join(HERE, "fig_corpus_facts.json"), "w"), indent=1)
print("wrote", base + ".pdf", round(fig.get_figheight(), 3), "in |", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in facts["means"].items()})
