"""Appendix figure for the falling-market test: market timing from the raw and the mirrored contexts.

Reads only existing outputs (nothing is typed by hand):
  experiments/results/falling_windows.npz, falling_windows_meta.json   (experiments/falling_windows.py)
  experiments/results/falling/falling_<model>[_mirror].npz            (code/real_probe.py on those windows)
For each of the 37 end dates: mean realised 128-day return of the 50 windows ending then (x) against the mean
forecast return from the raw contexts (filled) and from their multiplicative mirrors (open), in percent.
Writes figures/fig_timing.{pdf,svg}, figures/png/fig_timing.png, figures/fig_timing.alignment.json and
figure_design/fig_timing_20260924/fig_timing_facts.json (every drawn number).
Run from paper_revision/ with the nature-figure env and the skill scripts on PYTHONPATH.
"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = "figures"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
# colours of code/make_paper_figures.py
MODELS = [("chronosbolt", "Chronos-Bolt", "#4F86C6"), ("tirex", "TiRex", "#D98C21"),
          ("moirai2", "Moirai-2.0", "#42949E"), ("moirai", "Moirai-1.1", "#0F4D92")]

R = os.path.join("experiments", "results")
W = np.load(os.path.join(R, "falling_windows.npz")); meta = json.load(open(os.path.join(R, "falling_windows_meta.json")))
ctx, fut = W["ctx_raw"].astype(np.float64), W["fut_raw"].astype(np.float64); last = ctx[:, -1]
ends = np.array([m["fut_end"] for m in meta]); D = sorted(set(ends))
r = ctx[:, 1:] / ctx[:, :-1] - 1.0; lm = ctx[:, 0] * np.cumprod(1.0 - r, axis=1)[:, -1]
Y = fut[:, -1] / last - 1.0
dm = lambda v: np.array([100 * v[ends == d].mean() for d in D])
x = dm(Y)
facts = {"end_dates": D, "x_realised_pct": x.tolist(), "models": {}}
pts = {}
for key, name, col in MODELS:
    F = np.load(os.path.join(R, "falling", f"falling_{key}.npz"))["yhat_raw"][:, -1] / last - 1.0
    Fm = np.load(os.path.join(R, "falling", f"falling_{key}_mirror.npz"))["yhat_mirror"][:, -1] / lm - 1.0
    yr, ym = dm(F), dm(Fm)
    pts[key] = (yr, ym)
    facts["models"][name] = {"raw_pct": yr.tolist(), "mirror_pct": ym.tolist(),
                             "corr_raw": float(np.corrcoef(yr, x)[0, 1]), "corr_mirror": float(np.corrcoef(ym, x)[0, 1])}

W_IN, H_IN = 5.5, 1.95
fig, ax = plt.subplots(1, 4, figsize=(W_IN, H_IN), sharex=True, sharey=True, layout="constrained")
lo = min(x.min(), min(min(a.min(), b.min()) for a, b in pts.values())); hi = max(x.max(), max(max(a.max(), b.max()) for a, b in pts.values()))
LIM = (np.floor(lo / 10) * 10 - 2, np.ceil(hi / 10) * 10 + 2)
for a, (key, name, col) in zip(ax, MODELS):
    yr, ym = pts[key]
    a.plot(LIM, LIM, color="0.6", ls=":", lw=0.8, zorder=0)
    # no zero lines: the y = x diagonal is the reference, and the corner labels stay clear of every stroke
    a.plot(x, ym, "o", ms=3.2, mfc="white", mec=col, mew=0.8, zorder=2)
    a.plot(x, yr, "o", ms=3.2, color=col, zorder=3)
    f = facts["models"][name]
    a.text(0.04, 0.97, f"raw r = {f['corr_raw']:+.2f}\nmirror r = {f['corr_mirror']:+.2f}".replace("-", "−"),
           transform=a.transAxes, ha="left", va="top", fontsize=6, color="0.25")
    a.set_xlim(*LIM); a.set_ylim(*LIM)
ax[0].set_ylabel("mean forecast (%)")
for a in ax:
    a.set_xlabel("realised (%)")
leg = fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.2, color="0.3", label="from the raw contexts"),
                          Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec="0.3", label="from their mirrors")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)

fig.canvas.draw(); fig.set_layout_engine("none")
px = fig.dpi / 72.0; Wpt = W_IN * 72.0
PANEL_H, MAX_H = 82.4, H_IN * 72.0          # plot-area height of the previous version; the figure does not grow
for _ in range(2):
    dec = sc.decorations_pt(fig, ax)
    left, gutter, right = dec[0][0] + 5.0, 12.0, 6.0
    w = (Wpt - left - right - 3 * gutter) / 4
    bottom = sc.band_pt(fig, ax) + sc.MARGIN
    top = (fig.get_figheight() * 72.0 - leg.get_window_extent(fig.canvas.get_renderer()).y0 / px) + 4.0 + max(d[2] for d in dec)
    Hpt = min(top + PANEL_H + bottom, MAX_H); ph = Hpt - top - bottom
    fig.set_size_inches(W_IN, Hpt / 72.0)
    for i, a in enumerate(ax):
        a.set_position([(left + i * (w + gutter)) / Wpt, bottom / Hpt, w / Wpt, ph / Hpt])
facts["panel_labels"] = sc.add(fig, [[(a, f"({l}) {name}") for a, l, (_, name, _) in zip(ax, "abcd", MODELS)]])
base = os.path.join(FIG, "fig_timing")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_timing.png"), dpi=300)
facts["axis_limits_pct"] = list(LIM)
json.dump(facts, open(os.path.join(HERE, "fig_timing_facts.json"), "w"), indent=1)
print("wrote", base + ".pdf", {n: (round(v["corr_raw"], 2), round(v["corr_mirror"], 2)) for n, v in facts["models"].items()})
