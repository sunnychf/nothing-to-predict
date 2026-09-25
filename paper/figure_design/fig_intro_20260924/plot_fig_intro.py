"""Figure 1 of the English paper: the question on one random walk (a), how often the models lean upward (b) and what
the lean does to skill on a rising market (c). Panel a is the paper's original Figure 1 (code/make_paper_figures.py,
fig:teaser) redrawn relative to the last value; panels b and c are the previous Figure 1
(figure_design/fig_summary_20260923/plot_fig_summary.py) unchanged in data and encoding. Model colours are shared.

Reads only existing outputs (nothing is typed by hand):
  ../code/nulls.py                          the N1 generator; the 128 direction contexts are RUNGS["N1"](128, 512, 128,
                                            default_rng(4000)) and panel a shows context 115 (chosen once for the
                                            original figure: flat recent slope, last value near the level)
  ../results/departures_*_N1_H128.npz       every model's 128-step departures on those contexts
  figures/figures_facts.json                the original figure's departures at h = 128 (checked against), and the
                                            upward fractions of panel b
  experiments/results/tier2_reanalysis.json 95 percent intervals of panel b
  ../results/remedy_summary.json            raw and mirror-corrected equity skill of panel c
Writes figures/fig_intro.{pdf,svg}, figures/png/fig_intro.png, figures/fig_intro.alignment.json and
figure_design/fig_intro_20260924/fig_intro_facts.json. Run from paper_revision/ with the nature-figure env and the
skill scripts on PYTHONPATH."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_common"))
import subcaptions as sc   # '(a) short name' under each panel (authors' style, 2026-09-24)
from matplotlib.lines import Line2D
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None
sys.path.insert(0, os.path.join("..", "code"))
from nulls import RUNGS

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = "figures"; RES = os.path.join("..", "results")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
# colours, line styles and display names of code/make_paper_figures.py
COL = {"Chronos-small": "#0F4D92", "Chronos-Bolt-small": "#4F86C6", "Chronos-2": "#9DC3E6", "Moirai-small": "#42949E",
       "TimesFM-2.0": "#4D4D4D", "Time-MoE-200M": "#8F8F8F", "FinCast": "#B64342", "TiRex": "#D98C21",
       "Moirai-2.0": "#8FC7CC", "TimesFM-2.5": "#6E6E6E", "Sundial": "#7A5C9E"}
LS = {"Chronos-small": "-", "Chronos-Bolt-small": (0, (5, 1.5)), "Chronos-2": (0, (1, 1)), "Moirai-small": "--", "TimesFM-2.0": "-.",
      "Time-MoE-200M": (0, (3, 1, 1, 1, 1, 1)), "FinCast": ":", "TiRex": (0, (4, 1, 1, 1)), "Moirai-2.0": (0, (2, 1)),
      "TimesFM-2.5": (0, (6, 2, 1, 2)), "Sundial": (0, (7, 1.5, 1.5, 1.5, 1.5, 1.5))}
DISPLAY = {"Chronos-small": "Chronos-T5", "Chronos-Bolt-small": "Chronos-Bolt", "Moirai-small": "Moirai-1.1", "Time-MoE-200M": "Time-MoE"}
KEY = {"Chronos-small": "chronos", "Chronos-Bolt-small": "chronosbolt", "Chronos-2": "chronos2", "TiRex": "tirex",
       "Moirai-small": "moirai", "Moirai-2.0": "moirai2", "TimesFM-2.0": "timesfm", "TimesFM-2.5": "timesfm25",
       "Time-MoE-200M": "timemoe", "Sundial": "sundial", "FinCast": "fincast"}
DEP = {"Chronos-small": "departures_ctrl_pos_H128.npz", "Chronos-Bolt-small": "departures_bolt_N1_H128.npz",
       "Chronos-2": "departures_chronos2_N1_H128.npz", "TiRex": "departures_tirex_N1_H128.npz",
       "Moirai-small": "departures_moirai_N1_H128.npz", "Moirai-2.0": "departures_moirai2_N1_H128.npz",
       "TimesFM-2.0": "departures_timesfm_N1_H128.npz", "TimesFM-2.5": "departures_timesfm25_N1_H128.npz",
       "Time-MoE-200M": "departures_timemoe_N1_H128.npz", "Sundial": "departures_sundial_N1_H128.npz",
       "FinCast": "departures_fincast_N1_H128.npz"}
GROUPS = [("not decoder-only", ["Chronos-small", "Chronos-Bolt-small", "Chronos-2", "TiRex", "Moirai-small"]),
          ("decoder-only", ["Moirai-2.0", "TimesFM-2.0", "TimesFM-2.5", "Time-MoE-200M", "Sundial", "FinCast"])]
disp = lambda n: DISPLAY.get(n, n)
NEUTRAL = "#272727"

# ---------------------------------------------------------------- panel a: the original Figure 1, relative to the last value
y_all, info = RUNGS["N1"](128, 512, 128, np.random.default_rng(4000)); ctx = y_all[:, :512]; sig = float(info["sigma"])
I, TAIL = 115, 96
ff = json.load(open(os.path.join(FIG, "figures_facts.json")))
assert ff["teaser"]["series_index"] == I and abs(ff["teaser"]["last_value"] - ctx[I, -1]) < 1e-9, "not the original figure's context"
dep = {}
for n, f in DEP.items():
    z = np.load(os.path.join(RES, f)); d = z["dep"].astype(np.float64) / float(z["sigma"])
    dep[n] = d[I]
    assert abs(dep[n][-1] - ff["teaser"][n]["departure_h128"]) < 1e-9, f"{n}: departure differs from the original figure"
hist = (ctx[I, -TAIL:] - ctx[I, -1]) / sig; t_hist = np.arange(-TAIL + 1, 1); h = np.arange(1, 129)

# ---------------------------------------------------------------- panels b, c: the previous Figure 1, same data
rs = json.load(open(os.path.join(RES, "remedy_summary.json")))
names = sorted(KEY, key=lambda n: ff["direction"][n]["frac_up_h128"])
up_null = {n: ff["direction"][n]["frac_up_h128"] for n in names}
up_sur = {n: ff["real"][n]["sur"]["frac_up_h128"] for n in names}
T2 = json.load(open(os.path.join("experiments", "results", "tier2_reanalysis.json")))["direction"]
ci_null = {n: T2["N1"]["128"][KEY[n]]["ci95"] for n in names}
ci_sur = {n: T2["daily_sign_randomised"][KEY[n]]["ci95"] for n in names}
eq = {n: {w: (rs["models"][KEY[n]]["real|eq"][w]["128"]["skill"], rs["models"][KEY[n]]["real|eq"][w]["128"]["skill_se"])
          for w in ("raw", "mirror")} for n in names}

HA, HB = 66.0, 100.0          # plot-area heights of a and of b, c (points); legends of b and c sit inside their panels
W_IN, H_IN = 5.5, 3.44        # provisional height; the final one follows from the decorations and the panel labels
fig = plt.figure(figsize=(W_IN, H_IN))
gs = GridSpec(2, 2, figure=fig)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
# a
a.fill_between(h, -np.sqrt(h), np.sqrt(h), color=NEUTRAL, alpha=0.08, lw=0)
a.plot([0, 128], [0, 0], color=NEUTRAL, lw=0.9, ls=(0, (1, 1)))
a.plot(t_hist, hist, color=NEUTRAL, lw=0.9)
a.axvline(0, color=NEUTRAL, lw=0.5, alpha=0.5)
for _, members in GROUPS:
    for n in members:
        a.plot(np.r_[0, h], np.r_[0, dep[n]], color=COL[n], ls=LS[n], lw=0.9)
a.set_xlim(-TAIL, 130); a.set_xticks([-96, -64, -32, 0, 32, 64, 96, 128])
a.set_xlabel("step (0 = last observation)"); a.set_ylabel(r"level $-$ last value ($\sigma$)")
# b
y = np.arange(len(names))
b.axvline(0.5, color="0.55", ls=":", lw=0.8, zorder=0)
for i, n in enumerate(names):
    b.plot([up_sur[n], up_null[n]], [i, i], color="0.8", lw=0.8, zorder=1)
    lo, hi = ci_null[n]
    b.errorbar(up_null[n], i + 0.14, xerr=[[up_null[n] - lo], [hi - up_null[n]]], fmt="o", ms=4.0, color=COL[n], elinewidth=0.7, capsize=0, zorder=3)
    lo, hi = ci_sur[n]
    b.errorbar(up_sur[n], i - 0.14, xerr=[[up_sur[n] - lo], [hi - up_sur[n]]], fmt="o", ms=4.0, mfc="white", mec=COL[n],
               mew=1.0, ecolor=COL[n], elinewidth=0.7, capsize=0, zorder=3)
b.set_xlim(0.28, 0.97); b.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
b.set_yticks(y); b.set_yticklabels([disp(n) for n in names])
b.set_xlabel("fraction of forecasts above the last value")
b.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.0, color="0.3", label="random walks (N1)"),
                  Line2D([], [], ls="", marker="o", ms=4.0, mfc="white", mec="0.3", label="daily, signs randomised")],
         loc="lower right", ncol=1, handlelength=0.8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)
# c
LO, HI = -0.34, 0.25
c.axvline(0.0, color="0.55", ls=":", lw=0.8, zorder=0)
clipped = {}; cut_bars = {}
for i, n in enumerate(names):
    r, rse = eq[n]["raw"]; m, mse = eq[n]["mirror"]
    if r < LO or m < LO:
        clipped[disp(n)] = {"raw": round(r, 3), "mirror": round(m, 3)}
        c.annotate("", xy=(LO, i), xytext=(LO + 0.045, i), arrowprops=dict(arrowstyle="-|>", color=COL[n], lw=0.9, mutation_scale=6))
        c.text(LO + 0.055, i, f"{r:+.2f} → {m:+.2f}".replace("-", "−"), fontsize=6, va="center", ha="left", color="0.25")
        continue
    c.plot([r, m], [i, i], color="0.8", lw=0.8, zorder=1)
    # bars that leave the axis end at its edge (as clipping would show them), so no hidden path runs into panel b
    xe = {k: [[min(se, v - LO)], [min(se, HI - v)]] for k, (v, se) in (("raw", (r, rse)), ("mirror", (m, mse)))}
    for k, (v, se) in (("raw", (r, rse)), ("mirror", (m, mse))):
        if v - se < LO or v + se > HI:
            cut_bars.setdefault(disp(n), {})[k] = [round(v - se, 3), round(v + se, 3)]
    c.errorbar(r, i + 0.14, xerr=xe["raw"], fmt="o", ms=4.0, color=COL[n], elinewidth=0.7, capsize=0, zorder=3)
    c.errorbar(m, i - 0.14, xerr=xe["mirror"], fmt="o", ms=4.0, mfc="white", mec=COL[n], mew=1.0, ecolor=COL[n], elinewidth=0.7, capsize=0, zorder=3)
c.set_xlim(LO, HI); c.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
c.set_xticklabels(["−0.3", "−0.2", "−0.1", "0", "+0.1", "+0.2"])
c.set_yticks(y); c.set_yticklabels([])
c.set_xlabel("skill against the last value")
c.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.0, color="0.3", label="as released"),
                  Line2D([], [], ls="", marker="o", ms=4.0, mfc="white", mec="0.3", label="mirror-corrected")],
         loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=1, handlelength=0.8, handletextpad=0.4, labelspacing=0.3,
         borderaxespad=0.0)   # below TiRex's row, left of every other model's interval
for x in (b, c):
    x.tick_params(axis="y", length=0)
    x.set_ylim(-0.6, len(names) - 0.4)

# ---------------------------------------------------------------- layout: a spans the width minus its legend; b and c share a row
fig.canvas.draw()
px = fig.dpi / 72.0; Wpt = W_IN * 72.0
LEG = 118.0                                              # room for panel a's legend at its right
for _ in range(2):                                       # the second pass settles tick labels that move with the plot area
    dec_b = sc.decorations_pt(fig, [b])[0][0]
    left, right, gutter = dec_b + 5.0, 6.0, 14.0
    w = (Wpt - left - right - gutter) / 2
    bottom = sc.band_pt(fig, [b, c]) + sc.MARGIN                                         # b, c: ticks, axis labels, (b) (c)
    gap = sc.band_pt(fig, [a]) + max(d[2] for d in sc.decorations_pt(fig, [b, c])) + 4.0   # a: ticks, axis label, (a)
    top = max(sc.decorations_pt(fig, [a])[0][2] + 3.0, 6.0)                              # a's legend starts 4 pt above it
    Hpt = bottom + HB + gap + HA + top
    fig.set_size_inches(W_IN, Hpt / 72.0)
    b.set_position([left / Wpt, bottom / Hpt, w / Wpt, HB / Hpt])
    c.set_position([(left + w + gutter) / Wpt, bottom / Hpt, w / Wpt, HB / Hpt])
    a.set_position([left / Wpt, (bottom + HB + gap) / Hpt, (Wpt - left - right - LEG) / Wpt, HA / Hpt])
ya = (bottom + HB + gap + HA) / Hpt
for k, (title, members) in enumerate(GROUPS):
    fig.legend(handles=[Line2D([], [], color=COL[n], ls=LS[n], lw=0.9, label=disp(n)) for n in members],
               loc="upper left", bbox_to_anchor=((Wpt - right - LEG + 8 + k * 58) / Wpt, ya + 4 / Hpt), ncol=1, fontsize=6,
               title=title, title_fontsize=6, alignment="left", handlelength=1.8, handletextpad=0.35, labelspacing=0.2,
               borderaxespad=0.0, borderpad=0.0)
panel_labels = sc.add(fig, [[(a, "(a) One zero-drift random walk")],
                             [(b, "(b) No predictable mean, $h$ = 128"), (c, "(c) Raw equity windows, $h$ = 128")]])
base = os.path.join(FIG, "fig_intro")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
                                       panel_ids={a: "a", b: "b", c: "c"},
                                       exemptions=[{"panels": ["a"], "checks": ["column", "panel-width"],
                                                    "reason": "panel a spans the row above b and c but stops short of c's right edge to leave room for the legend of its eleven forecasts; its left edge is aligned with b"}])
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_intro.png"), dpi=300)
json.dump({"a_context_index": I, "a_context_length": int(ctx.shape[1]), "a_horizon": int(h[-1]), "a_tail_steps": TAIL, "a_departure_h128_sigma": {disp(n): float(dep[n][-1]) for n in DEP},
           "a_band": "+-1 sigma sqrt(h) of the future", "order_bottom_to_top": [disp(n) for n in names],
           "b_frac_up_h128_N1": {disp(n): up_null[n] for n in names}, "b_frac_up_h128_daily_signs_randomised": {disp(n): up_sur[n] for n in names},
           "b_ci95_N1": {disp(n): ci_null[n] for n in names}, "b_ci95_daily": {disp(n): ci_sur[n] for n in names},
           "c_equity_skill_h128": {disp(n): {"raw": eq[n]["raw"], "mirror": eq[n]["mirror"]} for n in names},
           "c_clipped_below_axis": clipped, "c_se_bars_cut_at_axis_edge": cut_bars, "c_axis": [LO, HI], "panel_labels": panel_labels},
          open(os.path.join(HERE, "fig_intro_facts.json"), "w"), indent=1)
print("wrote", base + ".pdf", "clipped:", clipped, "bars cut at the axis edge:", cut_bars, "| a departures h=128:", {disp(n): round(float(dep[n][-1]), 2) for n in DEP})
