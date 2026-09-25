"""Appendix figure for the calendar-shift check (Appendix F.5): how the check is built (a, schematic of the five context
ends of one end date, drawn to scale in trading days) and why it pairs the forecast change with the realised target level
(b, simulation). Panel b reads only experiments/results/shift_bias_sim.json (experiments/shift_bias_sim.py, random walks
without drift): mean and standard deviation over replications of the within-date correlation, for the return version
(open) and for the statistic used (filled).
Writes figures/fig_shiftcheck.{pdf,svg}, figures/png/fig_shiftcheck.png, figures/fig_shiftcheck.alignment.json and
figure_design/fig_shiftcheck_20260924/fig_shiftcheck_facts.json.
Run from paper_revision/ with the nature-figure env and the skill scripts on PYTHONPATH."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_common"))
import subcaptions as sc   # '(a) short name' under each panel (authors' style, 2026-09-24)
from matplotlib.lines import Line2D
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = "figures"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
BLUE, LIGHT, INK, GREY, BAND = "#0F4D92", "#9DC3E6", "#262626", "#8F8F8F", "#EDEDED"
SH, H = (-40, -20, 0, 20, 40), 128
sim = json.load(open(os.path.join("experiments", "results", "shift_bias_sim.json")))
ROWS = [("follows last 20 days", "none|trend_following|m=20"), ("opposes last 20 days", "none|trend_opposing|m=20"),
        ("follows last 128 days", "none|trend_following|m=128"), ("opposes last 128 days", "none|trend_opposing|m=128"),
        ("fitted drift", "none|fitted_drift"), ("recalls the path", "none|remembering|w=1.0"),
        ("recalls a quarter,\nopposes 20 days", "none|remembering_trend_opposing|w=0.25")]
facts = {"rows": {lab: {"returns": sim[k]["returns"], "change": sim[k]["change"], "reps": sim[k]["reps"]} for lab, k in ROWS},
         "gap_days": 88 - 40}

W_IN, H_IN = 5.5, 2.1
fig, ax = plt.subplots(1, 2, figsize=(W_IN, H_IN), layout="constrained")
# a: five context ends of one end date
a = ax[0]
LEFT = -150
a.axvspan(40, 88, color=BAND, lw=0, zorder=0)
a.text(64, 5.35, "48-day\ngap", ha="center", va="top", fontsize=6, color="0.35")
for r, k in enumerate(SH):
    yv = r + 0.5
    a.plot([LEFT, k], [yv, yv], color=INK, lw=2.2, solid_capstyle="butt")
    a.plot([k, k + H], [yv, yv], color=LIGHT, lw=2.2, solid_capstyle="butt")
    a.plot([k + H], [yv], "o", ms=3.4, color=BLUE, zorder=3)
a.set_yticks([r + 0.5 for r in range(len(SH))]); a.set_yticklabels([f"k = {k:+d}".replace("+0", "0").replace("-", "−") for k in SH])
a.set_ylim(-0.1, 5.4); a.set_xlim(LEFT, 185); a.set_xticks([-128, -40, 0, 40, 88, 168])
a.set_xticklabels(["−128", "−40", "0", "40", "88", "168"])
a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
a.set_xlabel("trading day, 0 = unshifted context end")
a.legend(handles=[Line2D([], [], color=INK, lw=2.2, label="context (512 days)"), Line2D([], [], color=LIGHT, lw=2.2, label="forecast horizon"),
                  Line2D([], [], ls="", marker="o", ms=3.4, color=BLUE, label="target date")],
         loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=6, handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
         borderaxespad=0.0)
# b: simulation
b = ax[1]
b.axvline(0, color=GREY, lw=0.7, ls=":", zorder=0)
for r, (lab, k) in enumerate(ROWS):
    yv = len(ROWS) - 1 - r
    for key, mfc in (("returns", "white"), ("change", BLUE)):
        m, sd = sim[k][key]
        b.plot([m - sd, m + sd], [yv, yv], color=BLUE if key == "change" else GREY, lw=0.8, zorder=1)
        b.plot([m], [yv], "o", ms=3.4, mfc=mfc, mec=BLUE if key == "change" else GREY, mew=0.9, zorder=2)
b.axhline(1.5, color=GREY, lw=0.5, zorder=0)
b.set_yticks(range(len(ROWS))); b.set_yticklabels([lab for lab, _ in ROWS][::-1], fontsize=6)
b.set_xlim(-0.7, 1.1); b.set_xticks([-0.5, 0, 0.5, 1.0]); b.set_xticklabels(["−0.5", "0", "0.5", "1.0"])
b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
b.set_xlabel("within-date correlation on random walks")
b.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=GREY, mew=0.9, label="returns"),
                  Line2D([], [], ls="", marker="o", ms=3.4, color=BLUE, label="statistic used")],
         loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)

fig.canvas.draw(); fig.set_layout_engine("none")
px = fig.dpi / 72.0; Wpt = W_IN * 72.0
PANEL_H = 91.2                              # plot-area height of the previous version (appendix: the figure may grow)
for _ in range(2):
    dec = sc.decorations_pt(fig, ax)        # 'above' includes each panel's legend
    right = 8.0; gutter = dec[1][0] + 10.0; left = dec[0][0] + 4.0
    w = (Wpt - left - right - gutter) / 2
    bottom = sc.band_pt(fig, ax) + sc.MARGIN
    top = max(d[2] for d in dec) + 3.0
    Hpt = top + PANEL_H + bottom
    fig.set_size_inches(W_IN, Hpt / 72.0)
    for i, x in enumerate(ax):
        x.set_position([(left + i * (w + gutter)) / Wpt, bottom / Hpt, w / Wpt, PANEL_H / Hpt])
facts["panel_labels"] = sc.add(fig, [[(ax[0], "(a) Five context ends of one date"), (ax[1], "(b) The two statistics on random walks")]])
base = os.path.join(FIG, "fig_shiftcheck")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_shiftcheck.png"), dpi=300)
json.dump(facts, open(os.path.join(HERE, "fig_shiftcheck_facts.json"), "w"), indent=1)
print("wrote", base)
