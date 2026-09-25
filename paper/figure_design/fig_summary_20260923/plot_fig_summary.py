"""Figure 1 of the revised paper: the directional bias under the null and its cost on raw equities.

Reads only existing outputs (nothing is typed by hand):
  figures/figures_facts.json      written by code/make_paper_figures.py from results/
                                  (direction[...]['frac_up_h128'] on N1, real[...]['sur']['frac_up_h128'])
  ../results/remedy_summary.json  written by code/remedy_summary.py
                                  (models[m]['real|eq'][raw|mirror]['128'][skill, skill_se])
Writes figures/fig_summary.{pdf,svg}, figures/png/fig_summary.png, figures/fig_summary.alignment.json
and figure_design/fig_summary_20260923/fig_summary_facts.json (every drawn number).
Run from paper_revision/ with the nature-figure env and the skill scripts on PYTHONPATH.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.lines import Line2D
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = "figures"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
# same colours and display names as code/make_paper_figures.py
COL = {"Chronos-small": "#0F4D92", "Chronos-Bolt-small": "#4F86C6", "Chronos-2": "#9DC3E6", "Moirai-small": "#42949E",
       "TimesFM-2.0": "#4D4D4D", "Time-MoE-200M": "#8F8F8F", "FinCast": "#B64342", "TiRex": "#D98C21",
       "Moirai-2.0": "#8FC7CC", "TimesFM-2.5": "#6E6E6E", "Sundial": "#7A5C9E"}
DISPLAY = {"Chronos-small": "Chronos-T5", "Chronos-Bolt-small": "Chronos-Bolt", "Moirai-small": "Moirai-1.1", "Time-MoE-200M": "Time-MoE"}
KEY = {"Chronos-small": "chronos", "Chronos-Bolt-small": "chronosbolt", "Chronos-2": "chronos2", "TiRex": "tirex",
       "Moirai-small": "moirai", "Moirai-2.0": "moirai2", "TimesFM-2.0": "timesfm", "TimesFM-2.5": "timesfm25",
       "Time-MoE-200M": "timemoe", "Sundial": "sundial", "FinCast": "fincast"}
disp = lambda n: DISPLAY.get(n, n)

ff = json.load(open(os.path.join(FIG, "figures_facts.json")))
rs = json.load(open(os.path.join("..", "results", "remedy_summary.json")))
names = sorted(KEY, key=lambda n: ff["direction"][n]["frac_up_h128"])   # bottom-to-top: most upward at the top
up_null = {n: ff["direction"][n]["frac_up_h128"] for n in names}
up_sur = {n: ff["real"][n]["sur"]["frac_up_h128"] for n in names}
T2 = json.load(open(os.path.join("experiments", "results", "tier2_reanalysis.json")))["direction"]
ci_null = {n: T2["N1"]["128"][KEY[n]]["ci95"] for n in names}
ci_sur = {n: T2["daily_sign_randomised"][KEY[n]]["ci95"] for n in names}
eq = {n: {w: (rs["models"][KEY[n]]["real|eq"][w]["128"]["skill"], rs["models"][KEY[n]]["real|eq"][w]["128"]["skill_se"])
          for w in ("raw", "mirror")} for n in names}

W_IN, H_IN = 5.5, 2.45
fig, ax = plt.subplots(1, 2, figsize=(W_IN, H_IN), sharey=True, layout="constrained")
y = np.arange(len(names))

a = ax[0]
a.axvline(0.5, color="0.55", ls=":", lw=0.8, zorder=0)
for i, n in enumerate(names):
    a.plot([up_sur[n], up_null[n]], [i, i], color="0.8", lw=0.8, zorder=1)
    lo, hi = ci_null[n]
    a.errorbar(up_null[n], i + 0.14, xerr=[[up_null[n] - lo], [hi - up_null[n]]], fmt="o", ms=4.2, color=COL[n],
               elinewidth=0.7, capsize=0, zorder=3)
    lo, hi = ci_sur[n]
    a.errorbar(up_sur[n], i - 0.14, xerr=[[up_sur[n] - lo], [hi - up_sur[n]]], fmt="o", ms=4.2, mfc="white", mec=COL[n],
               mew=1.0, ecolor=COL[n], elinewidth=0.7, capsize=0, zorder=3)
a.set_xlim(0.28, 0.95); a.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
a.set_yticks(y); a.set_yticklabels([disp(n) for n in names])
a.set_xlabel("fraction of forecasts above the last value")
a.set_title("no predictable mean, $h$ = 128", loc="left")
a.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.2, color="0.3", label="random walks (N1)"),
                  Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec="0.3", label="daily, signs randomised")],
         loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)

b = ax[1]
LO, HI = -0.34, 0.25
b.axvline(0.0, color="0.55", ls=":", lw=0.8, zorder=0)
clipped = {}
for i, n in enumerate(names):
    r, rse = eq[n]["raw"]; m, mse = eq[n]["mirror"]
    if r < LO or m < LO:
        clipped[disp(n)] = {"raw": round(r, 3), "mirror": round(m, 3)}
        b.annotate("", xy=(LO, i), xytext=(LO + 0.045, i),
                   arrowprops=dict(arrowstyle="-|>", color=COL[n], lw=0.9, mutation_scale=6))
        b.text(LO + 0.055, i, f"{r:+.2f} \u2192 {m:+.2f}".replace("-", "\u2212"), fontsize=6, va="center", ha="left", color="0.25")
        continue
    b.plot([r, m], [i, i], color="0.8", lw=0.8, zorder=1)
    b.errorbar(r, i + 0.14, xerr=rse, fmt="o", ms=4.2, color=COL[n], elinewidth=0.7, capsize=0, zorder=3)
    b.errorbar(m, i - 0.14, xerr=mse, fmt="o", ms=4.2, mfc="white", mec=COL[n], mew=1.0,
               ecolor=COL[n], elinewidth=0.7, capsize=0, zorder=3)
b.set_xlim(LO, HI); b.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
b.set_xticklabels(["−0.3", "−0.2", "−0.1", "0", "+0.1", "+0.2"])
b.set_xlabel("skill against the last value")
b.set_title("raw equity windows, $h$ = 128", loc="left")
b.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.2, color="0.3", label="as released"),
                  Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec="0.3", label="mirror-corrected")],
         loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
for a_ in ax:
    a_.tick_params(axis="y", length=0)

fig.canvas.draw(); fig.set_layout_engine("none")
rend = fig.canvas.get_renderer(); px = fig.dpi / 72.0; Wpt = W_IN * 72.0
dec0 = (ax[0].get_window_extent(rend).x0 - ax[0].get_tightbbox(rend).x0) / px
dec1 = (ax[1].get_window_extent(rend).x0 - ax[1].get_tightbbox(rend).x0) / px
left, gutter, right = dec0 + 5.0, max(dec1, 8.0) + 14.0, 6.0
w = (Wpt - left - right - gutter) / 2
pos = [p.get_position() for p in ax]; h0 = min(p.height for p in pos); y0 = max(p.y0 for p in pos)
for i, a_ in enumerate(ax):
    a_.set_position([(left + i * (w + gutter)) / Wpt, y0, w / Wpt, h0])
for a_, lab in zip(ax, "ab"):
    a_.text(0, 1, lab, transform=a_.transAxes + ScaledTranslation(-4 / 72, 3 / 72, fig.dpi_scale_trans),
            fontsize=8, fontweight="bold", ha="right", va="bottom")

base = os.path.join(FIG, "fig_summary")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_summary.png"), dpi=300)
json.dump({"order_bottom_to_top": [disp(n) for n in names],
           "a_frac_up_h128_N1": {disp(n): up_null[n] for n in names},
           "a_frac_up_h128_daily_signs_randomised": {disp(n): up_sur[n] for n in names},
           "a_ci95_N1": {disp(n): ci_null[n] for n in names}, "a_ci95_daily": {disp(n): ci_sur[n] for n in names},
           "b_equity_skill_h128": {disp(n): {"raw": eq[n]["raw"], "mirror": eq[n]["mirror"]} for n in names},
           "b_clipped_below_axis": clipped, "b_axis": [LO, HI]},
          open(os.path.join(HERE, "fig_summary_facts.json"), "w"), indent=1)
print("wrote", base + ".pdf", "clipped:", clipped)
