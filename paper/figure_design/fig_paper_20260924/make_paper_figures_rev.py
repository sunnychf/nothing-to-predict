"""Paper figures, drawn from results/ only. Nothing here is typed by hand.

REVISION COPY (paper_revision, 2026-09-24): code/make_paper_figures.py with (1) paths for running from
paper_revision/ (results and nulls.py from the read-only ../results and ../code), (2) the panel labels moved under
the panels as "(a) short name" (figure_design/fig_common/subcaptions.py) in place of the top-left letters and the panel
titles, shared legends moved above the panels, each figure keeping its previous plot-area height, (3) fig_teaser not
redrawn (only the Chinese draft uses it) and figures_facts.json not overwritten: the recomputed facts go to
paper_figures_facts.json (beside this script) and must equal figures/figures_facts.json (checked at the end). Data and drawn values are those
of the original script (checked pixel for pixel before the layout change).
Run from paper_revision/:  PYTHONPATH=<nature-figure scripts> python figure_design/fig_paper_20260924/make_paper_figures_rev.py --out figures

Run locally in the `nature-figure` conda env (matplotlib 3.11, numpy, PyMuPDF):
    PYTHONPATH=<directory holding audit_panel_alignment.py> \\
    ~/miniconda3/envs/nature-figure/bin/python code/make_paper_figures.py --res results --out paper/figures
Outputs paper/figures/fig_*.pdf (+ .svg, png/ previews, *.alignment.json) and
paper/figures/figures_facts.json (every number that is drawn or annotated, so
the caption text can be checked against it).

Figure contract (publication-figure workflow):
  fig_direction   claim: on identical zero-drift random walks the sign of the
                  departure is model-specific; Chronos and Moirai drift up,
                  TimesFM and Time-MoE do not drift, FinCast drifts down.
                  a = mean signed departure over the horizon (hero, sign and
                  size); b = fraction of forecasts above the last value (sign
                  consistency across series, independent of size); c = spread
                  of the departure at h=128 across the 128 series (median,
                  interquartile range, 5th-95th percentile), showing that the
                  sign is a shift of the whole distribution and not a tail.
  fig_declaration claim: changing only the declared sampling frequency changes
                  the drift; one panel per model that takes a declaration.
  fig_scale       claim: across five Chronos sizes null skill stays negative and
                  the upward bias grows while N4 skill improves.
  fig_level       claim: the upward drift persists across price levels; only
                  the constant line at +1000 is left alone.
Palette: one blue family for the two encoder-family models that drift up, one
neutral family for the two decoder-only general-corpus models with no drift,
one red accent for the finance-native model that drifts down; every series is
also encoded by line style, so the figure survives greyscale. Ordered
declarations use a truncated viridis ramp (ordered, colour-blind safe).
Uncertainty: panels aggregate across series, not seeds; the definition is stated
in the caption (128 series; c: median, IQR, 5th-95th percentile). No series is
excluded anywhere.

Per-series departure files (dep in raw price units, sigma stored alongside;
all five models on the SAME seed-4000 N1 contexts, H=128):
    departures_ctrl_pos_H128.npz       Chronos-small, level +100
    departures_moirai_N1_H128.npz      Moirai-small, patch auto
    departures_timesfm_N1_H128.npz     TimesFM-2.0, freq 0
    departures_timemoe_N1_H128.npz     Time-MoE-200M (no declaration), if present
    departures_fincast_N1_H128.npz     FinCast, freq 0
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_common"))
import subcaptions as sc

try:                                     # render-time alignment gate (skill script on PYTHONPATH)
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:                      # pragma: no cover
    require_matplotlib_panel_alignment = None

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join("..", "results"))
ap.add_argument("--out", default="figures")
ap.add_argument("--facts", default=os.path.join("figures", "figures_facts.json"))
args = ap.parse_args()
RES, OUT = args.res, args.out
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, "png"), exist_ok=True)

# ---- mandatory font / editable-text settings, then the publication defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none",   # keeps text as <text> nodes, not paths
                     "pdf.fonttype": 42,       # editable TrueType text in PDF
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
# palette: blue family (drift up), neutral family (no drift), red accent (drifts down)
COL = {"Chronos-small": "#0F4D92", "Chronos-Bolt-small": "#4F86C6", "Chronos-2": "#9DC3E6", "Moirai-small": "#42949E",
       "TimesFM-2.0": "#4D4D4D", "Time-MoE-200M": "#8F8F8F", "FinCast": "#B64342", "TiRex": "#D98C21",
       "Moirai-2.0": "#8FC7CC", "TimesFM-2.5": "#6E6E6E", "Sundial": "#7A5C9E"}      # Chronos family: three blues; Moirai family: two teals; TimesFM family: two greys; TiRex (xLSTM) orange; Sundial purple
# Display names in legends, ticks and titles follow the paper text (M31, 09-23); the keys above stay as they were,
# because figures_facts.json and the caption checks are keyed by them.
DISPLAY = {"Chronos-small": "Chronos-T5", "Chronos-Bolt-small": "Chronos-Bolt", "Moirai-small": "Moirai-1.1", "Time-MoE-200M": "Time-MoE"}
def disp(n): return DISPLAY.get(n, n)
LS = {"Chronos-small": "-", "Chronos-Bolt-small": (0, (5, 1.5)), "Chronos-2": (0, (1, 1)), "Moirai-small": "--", "TimesFM-2.0": "-.",
      "Time-MoE-200M": (0, (3, 1, 1, 1, 1, 1)), "FinCast": ":", "TiRex": (0, (4, 1, 1, 1)), "Moirai-2.0": (0, (2, 1)), "TimesFM-2.5": (0, (6, 2, 1, 2)), "Sundial": (0, (7, 1.5, 1.5, 1.5, 1.5, 1.5))}
NEUTRAL = "#272727"
facts = {}


SUBCAPS = {}


def top_legend(fig, handles, labels, ncols=(6, 4), **kw):
    """Shared legend above the panels: the most columns that fit the figure width with 4 pt to spare."""
    W = fig.get_figwidth() * 72.0
    for nc in ncols:
        lg = fig.legend(handles, labels, loc="upper center", ncol=nc, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2, **kw)
        fig.canvas.draw()
        if lg.get_window_extent(fig.canvas.get_renderer()).width / (fig.dpi / 72.0) <= W - 8.0 or nc == ncols[-1]:
            return lg
        lg.remove()


def layout_row(fig, axes, names, legend=None, panel_h=None, max_h_pt=None, pad_pt=5.0, right_pad_pt=6.0, gap_pt=4.0):
    """One row of panels with identical plot-area widths, heights and gutters (every gutter is the widest decoration a
    non-first panel carries, as before), the shared legend (if any) above them and the '(a) name' labels under them.
    The plot areas keep the height panel_h of the previous figure; the figure height follows, capped at max_h_pt."""
    for a in axes:
        a.set_title("")
    fig.canvas.draw(); fig.set_layout_engine("none")
    W = fig.get_figwidth() * 72.0; px = fig.dpi / 72.0
    for _ in range(2):                       # tick labels can move with the plot area; the second pass settles them
        dec = sc.decorations_pt(fig, axes)
        left = dec[0][0] + pad_pt
        gutter = (max(d[0] for d in dec[1:]) + pad_pt) if len(axes) > 1 else 0.0
        n = len(axes); w = (W - left - right_pad_pt - (n - 1) * gutter) / n
        bottom = sc.band_pt(fig, axes) + sc.MARGIN
        above = max(d[2] for d in dec)
        Hc = fig.get_figheight() * 72.0
        if legend is not None:
            fig.canvas.draw()
            top = (Hc - legend.get_window_extent(fig.canvas.get_renderer()).y0 / px) + gap_pt + above
        else:
            top = above + 3.0
        ph = panel_h if panel_h is not None else Hc - top - bottom
        H = top + ph + bottom
        if max_h_pt is not None and H > max_h_pt:
            H = max_h_pt; ph = H - top - bottom
        fig.set_size_inches(W / 72.0, H / 72.0)
        for i, a in enumerate(axes):
            a.set_position([(left + i * (w + gutter)) / W, bottom / H, w / W, ph / H])
    return sc.add(fig, [list(zip(axes, names))])


def finish(fig, name, axes):
    """Alignment gate (blocking), then PDF for the paper, SVG with editable text, PNG preview."""
    base = os.path.join(OUT, name)
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5, strict=True)
    else:
        print(f"WARNING {name}: audit_panel_alignment not importable; alignment gate NOT run", file=sys.stderr)
    fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
    fig.savefig(os.path.join(OUT, "png", name + ".png"), dpi=300)   # preview only; the paper embeds the PDF
    plt.close(fig)


def load_dep(fname):
    z = np.load(os.path.join(RES, fname))
    return z["dep"].astype(np.float64) / float(z["sigma"])


def rows(path):
    out = []
    for l in open(path):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def zero_line(ax, y=0.0, axis="h"):
    (ax.axhline if axis == "h" else ax.axvline)(y, color=NEUTRAL, lw=0.6, ls=":", zorder=0)


# ------------------------------------------------------------------ fig:direction
DIR_FILES = [("Chronos-small", "departures_ctrl_pos_H128.npz"),
             ("Chronos-Bolt-small", "departures_bolt_N1_H128.npz"),
             ("Chronos-2", "departures_chronos2_N1_H128.npz"),
             ("TiRex", "departures_tirex_N1_H128.npz"),
             ("Moirai-small", "departures_moirai_N1_H128.npz"),
             ("Moirai-2.0", "departures_moirai2_N1_H128.npz"),
             ("TimesFM-2.0", "departures_timesfm_N1_H128.npz"),
             ("TimesFM-2.5", "departures_timesfm25_N1_H128.npz"),
             ("Time-MoE-200M", "departures_timemoe_N1_H128.npz"),
             ("Sundial", "departures_sundial_N1_H128.npz"),
             ("FinCast", "departures_fincast_N1_H128.npz")]
DIR_FILES = [(n, f) for n, f in DIR_FILES if os.path.exists(os.path.join(RES, f))]
# cross-check against the diagnostics the tables were built from
ref = {"Chronos-small": json.load(open(os.path.join(RES, "diag_direction_control.json")))["pos"]["mean"]["128"],
       "Moirai-small": next(r for r in json.load(open(os.path.join(RES, "diag_direction_moirai.json"))) if r["rung"] == "N1")["mean_dep_sigma"]["128"],
       "TimesFM-2.0": next(r for r in json.load(open(os.path.join(RES, "diag_direction_timesfm.json"))) if r["rung"] == "N1")["mean_dep_sigma"]["128"],
       "FinCast": next(r for r in json.load(open(os.path.join(RES, "diag_direction_fincast.json"))) if r["rung"] == "N1" and r["H"] == 128)["mean_dep_sigma"]["128"]}
if os.path.exists(os.path.join(RES, "diag_direction_timemoe.json")):
    ref["Time-MoE-200M"] = next(r for r in json.load(open(os.path.join(RES, "diag_direction_timemoe.json"))) if r["rung"] == "N1")["mean_dep_sigma"]["128"]
for _short, _lab in (("bolt", "Chronos-Bolt-small"), ("chronos2", "Chronos-2"), ("tirex", "TiRex"), ("moirai2", "Moirai-2.0"), ("timesfm25", "TimesFM-2.5"), ("sundial", "Sundial")):
    if os.path.exists(os.path.join(RES, f"diag_direction_{_short}.json")):
        ref[_lab] = next(r for r in json.load(open(os.path.join(RES, f"diag_direction_{_short}.json"))) if r["rung"] == "N1")["mean_dep_sigma"]["128"]

fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.62 if len(DIR_FILES) <= 8 else (1.85 if len(DIR_FILES) <= 10 else 2.1)), constrained_layout=True)   # taller for ten or eleven rows in panel c
h = np.arange(1, 129)
facts["direction"] = {}
names = [n for n, _ in DIR_FILES]
for name, f in DIR_FILES:
    d = load_dep(f)
    m = d.mean(0); up = (d > 0).mean(0)
    assert abs(m[-1] - ref[name]) < 0.02, (name, m[-1], ref[name])
    ax[0].plot(h, m, color=COL[name], ls=LS[name], label=disp(name))
    ax[1].plot(h, up, color=COL[name], ls=LS[name])
    q = np.quantile(d[:, -1], [0.05, 0.25, 0.5, 0.75, 0.95])
    y = len(names) - names.index(name)
    ax[2].plot([q[0], q[4]], [y, y], color=COL[name], lw=0.8, solid_capstyle="butt")
    ax[2].plot([q[1], q[3]], [y, y], color=COL[name], lw=2.6, solid_capstyle="butt")
    ax[2].plot(q[2], y, marker="o", ms=3.2, color="white", mec=COL[name], mew=0.9, zorder=3)
    facts["direction"][name] = {"mean_h16": float(m[15]), "mean_h128": float(m[-1]), "frac_up_h16": float(up[15]),
                                "frac_up_h128": float(up[-1]), "n": int(d.shape[0]),
                                "h128_quantiles_5_25_50_75_95": [float(v) for v in q]}
zero_line(ax[0]); zero_line(ax[1], 0.5); zero_line(ax[2], 0.0, "v")
ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("mean departure / σ")
ax[1].set_xlabel("horizon h"); ax[1].set_ylabel("fraction forecast up"); ax[1].set_ylim(0.3, 1.0)
ax[2].set_xlabel("departure at h = 128 / σ")
SHORT = {"Chronos-small": "Chronos-T5", "Chronos-Bolt-small": "Chronos-Bolt", "Chronos-2": "Chronos-2", "TiRex": "TiRex", "Moirai-small": "Moirai-1.1", "Moirai-2.0": "Moirai-2.0", "TimesFM-2.0": "TimesFM-2.0", "TimesFM-2.5": "TimesFM-2.5", "Time-MoE-200M": "Time-MoE", "Sundial": "Sundial", "FinCast": "FinCast"}
ax[2].set_yticks(range(len(names), 0, -1)); ax[2].set_yticklabels([SHORT[n] for n in names], fontsize=6 if len(names) > 10 else None); ax[2].set_ylim(0.4, len(names) + 0.6)
ax[2].tick_params(axis="y", length=0)
for a in ax[:2]:
    a.set_xticks([1, 32, 64, 96, 128])
leg = top_legend(fig, *ax[0].get_legend_handles_labels(), handlelength=2.2, columnspacing=1.2)
SUBCAPS["fig_direction"] = layout_row(fig, ax, ["(a) Signed mean departure", "(b) Fraction above the last value",
                                                "(c) Spread at $h$ = 128"], legend=leg, panel_h=74.4, max_h_pt=151.2)
finish(fig, "fig_direction", ax)

# ------------------------------------------------------------------ fig:declaration
recs = rows(os.path.join(RES, "pilot_a.jsonl"))
freq = [r for r in recs if r.get("stage") == "freq" and r.get("applicable") and "departure_mean_signed_sigma" in r]
moirai = sorted((r for r in freq if r["model"] == "moirai-1.1-R-small"), key=lambda r: r["freq"])
timesfm = sorted((r for r in freq if r["model"] == "timesfm-2.0-500m"), key=lambda r: r["freq"])
fincast = {k: load_dep(f"departures_fincast_freq{k}_H128.npz") for k in (0, 1, 2)}
ffreq = {r["freq"]: r for r in json.load(open(os.path.join(RES, "diag_freq_direction_fincast.json")))}

fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.42), constrained_layout=True, sharey=True)
facts["declaration"] = {"Moirai-small": {}, "TimesFM-2.0": {}, "FinCast": {}}
cm = plt.get_cmap("viridis")
ramp5 = [cm(0.08 + 0.2 * i) for i in range(5)]; ramp3 = [cm(0.08), cm(0.48), cm(0.88)]
band = {8: "yearly/quarterly", 16: "weekly/daily", 32: "daily/hourly", 64: "hourly/minute", 128: "minute/second"}
for i, r in enumerate(moirai):
    m = np.asarray(r["departure_mean_signed_sigma"])
    ax[0].plot(h, m, color=ramp5[i], label=f"patch {r['freq']} ({band[r['freq']]})")
    facts["declaration"]["Moirai-small"][r["freq"]] = {"mean_h128": float(m[-1]), "frac_up_h128": float(r["frac_departure_up"][-1])}
tlab = {0: "declared ≤ daily", 1: "declared weekly/monthly", 2: "declared quarterly/yearly"}
for i, r in enumerate(timesfm):
    m = np.asarray(r["departure_mean_signed_sigma"])
    ax[1].plot(h, m, color=ramp3[i], label=tlab[r["freq"]])
    facts["declaration"]["TimesFM-2.0"][r["freq"]] = {"mean_h128": float(m[-1]), "frac_up_h128": float(r["frac_departure_up"][-1])}
flab = {0: "declared ≤ daily", 1: "declared weekly/monthly", 2: "declared yearly"}
for i, k in enumerate((0, 1, 2)):
    m = fincast[k].mean(0)
    assert abs(m[-1] - ffreq[k]["mean_dep_sigma"]["128"]) < 0.02, (k, m[-1])
    ax[2].plot(h, m, color=ramp3[i], label=flab[k])
    facts["declaration"]["FinCast"][k] = {"mean_h128": float(m[-1]), "frac_up_h128": float((fincast[k][:, -1] > 0).mean())}
for a in ax:
    zero_line(a); a.set_xlabel("horizon h"); a.set_xticks([1, 32, 64, 96, 128])
ax[1].legend(loc="upper left", handlelength=1.6); ax[2].legend(loc="upper left", handlelength=1.6)
ax[0].set_ylim(-3.4, 10.8); ax[0].set_yticks([0, 5])
ax[0].text(2, 10.1, "patch 8, 16, 32 (yearly to daily): drift up", fontsize=6, color=NEUTRAL, va="top")
ax[0].text(2, -1.8, "patch 64, 128 (hourly to second): none", fontsize=6, color=NEUTRAL, va="top")
ax[0].set_ylabel("mean departure / σ")
SUBCAPS["fig_declaration"] = layout_row(fig, ax, ["(a) Moirai-1.1, patch size", "(b) TimesFM-2.0, frequency",
                                                  "(c) FinCast, frequency"], panel_h=61.8)
finish(fig, "fig_declaration", ax)

# ------------------------------------------------------------------ fig:scale
scl = rows(os.path.join(RES, "scale_chronos.jsonl"))
by = {(r["size"], r["rung"]): r for r in scl}
sizes = ["8M", "20M", "46M", "200M", "710M"]; params = np.array([8, 20, 46, 200, 710])
fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.55), constrained_layout=True)
sk1 = [by[(s, "N1")]["skill"][0] for s in sizes]; sk64 = [by[(s, "N1")]["skill"][-1] for s in sizes]
up16 = [by[(s, "N1")]["frac_up"][15] for s in sizes]; up64 = [by[(s, "N1")]["frac_up"][-1] for s in sizes]
sys.path.insert(0, os.path.join("..", "code"))                   # nulls.py of the read-only code/
from nulls import n4_optima_skill                                  # the exact N4 optimum on the sweep's own draw (rng(4000), n=128, H=64);
_sc_opt = n4_optima_skill(128, 512, 64, 4000)                       # the stored oracle_skill is the optimal *linear* forecast's
assert abs(_sc_opt["linear"][0] - by[("46M", "N4")]["oracle_skill"][0]) < 1e-9
n4 = [100 * by[(s, "N4")]["skill"][0] / _sc_opt["exact"][0] for s in sizes]
c = COL["Chronos-small"]
ax[0].plot(params, sk1, "o-", color=c, ms=3.2, label="h = 1"); ax[0].plot(params, sk64, "s--", color=c, ms=3.0, label="h = 64")
zero_line(ax[0]); ax[0].set_ylabel("N1 skill vs persistence"); ax[0].set_ylim(-0.24, 0.09)
ax[1].plot(params, up16, "o-", color=c, ms=3.2, label="h = 16"); ax[1].plot(params, up64, "s--", color=c, ms=3.0, label="h = 64")
zero_line(ax[1], 0.5); ax[1].set_ylabel("N1 fraction forecast up"); ax[1].set_ylim(0.4, 1.0)
ax[2].plot(params, n4, "o-", color=c, ms=3.2); zero_line(ax[2], 100)
ax[2].set_ylabel("N4 skill, % of optimum")
for a in ax:
    a.set_xscale("log"); a.set_xticks(params); a.set_xticklabels(sizes, rotation=35, ha="right", rotation_mode="anchor")
    a.set_xlabel("Chronos-T5 parameters"); a.minorticks_off()
ax[0].legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.8); ax[1].legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.8)
facts["scale"] = {s: {"skill_h1": sk1[i], "skill_h64": sk64[i], "up16": up16[i], "up64": up64[i], "n4_pct": n4[i],
                      "n4_pct_linear": 100 * by[(s, "N4")]["skill"][0] / by[(s, "N4")]["oracle_skill"][0]} for i, s in enumerate(sizes)}
facts["scale_n4_optimum_skill_h1"] = {"exact": _sc_opt["exact"][0], "linear": _sc_opt["linear"][0]}
SUBCAPS["fig_scale"] = layout_row(fig, ax, ["(a) Skill on the null", "(b) Upward bias", "(c) Share of the N4 optimum"],
                                  panel_h=62.8)
finish(fig, "fig_scale", ax)

# ------------------------------------------------------------------ fig:level
lv = sorted(rows(os.path.join(RES, "diag_direction_level.jsonl")), key=lambda r: r["level"])
labels = [f"{int(r['level']):+d}" if r["level"] != 0 else "0" for r in lv]; x = np.arange(len(lv))
fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.55), constrained_layout=True)
for hh, mk, ls in ((16, "o", "-"), (64, "s", "--"), (128, "^", ":")):
    ax[0].plot(x, [r["mean_dep_sigma"][str(hh)] for r in lv], marker=mk, ls=ls, ms=3.2, color=c, label=f"h = {hh}")
zero_line(ax[0]); ax[0].set_ylabel("mean departure / σ"); ax[0].legend(loc="upper right")
ax[1].plot(x, [r["frac_up"]["16"] for r in lv], marker="o", ls="-", ms=3.2, color=c, label="h = 16")
ax[1].plot(x, [r["frac_up"]["128"] for r in lv], marker="^", ls=":", ms=3.2, color=c, label="h = 128")
zero_line(ax[1], 0.5); ax[1].set_ylabel("fraction forecast up"); ax[1].set_ylim(0.2, 1.0); ax[1].legend(loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("price level of the same increments")
facts["level"] = {labels[i]: {"mean_h128": r["mean_dep_sigma"]["128"], "frac_up_h128": r["frac_up"]["128"],
                              "grid_step_sigma": r["grid_step_sigma"]} for i, r in enumerate(lv)}
SUBCAPS["fig_level"] = layout_row(fig, ax, ["(a) Signed mean departure", "(b) Fraction above the last value"], panel_h=71.1)
finish(fig, "fig_level", ax)

# ------------------------------------------------------------------ fig:real (real-data anchor)
REAL = [("Chronos-small", "chronos"), ("Chronos-Bolt-small", "chronosbolt"), ("Chronos-2", "chronos2"), ("TiRex", "tirex"), ("Moirai-small", "moirai"), ("Moirai-2.0", "moirai2"), ("TimesFM-2.0", "timesfm"), ("TimesFM-2.5", "timesfm25"), ("Time-MoE-200M", "timemoe"), ("Sundial", "sundial"), ("FinCast", "fincast")]
REAL = [(n, k) for n, k in REAL if os.path.exists(os.path.join(RES, f"real_{k}.npz"))]
if REAL and os.path.exists(os.path.join(RES, "real_windows.npz")):
    zw = np.load(os.path.join(RES, "real_windows.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85), constrained_layout=True)
    hh = np.arange(1, zw["fut_raw"].shape[1] + 1)
    facts["real"] = {}; offscale = {}
    # departures in return units, (yhat / y_T - 1) / sigma_r with sigma_r the std of the context's
    # simple returns: the unit in which the multiplicative surrogate is exact (real_summary.py uses
    # the same); K surrogates per window are pooled
    def pool(a):
        return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:]) if a.ndim == 3 else a
    for kind, a in (("sur", ax[0]), ("raw", ax[1])):
        ctx, fut, sig = zw[f"ctx_{kind}"], zw[f"fut_{kind}"], zw[f"rsigma_{kind}"]
        if kind == "sur":
            ctx, fut, sig = pool(ctx), pool(fut), sig.reshape(-1)
        last = ctx[:, -1:]
        for name, k in REAL:
            yh = np.load(os.path.join(RES, f"real_{k}.npz"))[f"yhat_{kind}"].astype(np.float64)
            if kind == "sur":
                yh = pool(yh)
            dep = (yh / last - 1.0) / sig[:, None]
            a.plot(hh, dep.mean(0), color=COL[name], ls=LS[name], label=disp(name))
            if kind == "raw":
                err_m = (((fut - yh) / last) ** 2).mean(0); err_p = (((fut - last) / last) ** 2).mean(0); sk = 1 - err_m / err_p   # return units, scale-free pooling
                if sk[15:].min() < -0.85: offscale[name] = (float(sk[15:].min()), float(sk[15:].max()))   # drawn as text, not as a clipped path
                else: ax[2].plot(hh, sk, color=COL[name], ls=LS[name])
            facts["real"].setdefault(name, {})[kind] = {"mean_h16": float(dep[:, 15].mean()), "mean_h128": float(dep[:, -1].mean()),
                                                       "frac_up_h128": float((dep[:, -1] > 0).mean())}
        real = ((fut / last - 1.0) / sig[:, None]).mean(0)
        if kind == "raw":
            a.plot(hh, real, color=NEUTRAL, lw=1.4, label="the data's own move")
            facts["real"]["realised"] = {"mean_h16": float(real[15]), "mean_h128": float(real[-1])}
        zero_line(a)
    zero_line(ax[2])
    # shorter than "mean departure / σ (returns)" so that the label fits the plot height (the caption says "mean")
    ax[0].set_ylabel("departure (σ of returns)"); ax[1].set_ylabel("departure (σ of returns)"); ax[2].set_ylabel("skill vs persistence")
    for a, lab in zip(ax, "abc"):
        a.set_xlabel("horizon h"); a.set_xticks([1, 32, 64, 96, 128])
    ax[2].set_ylim(-0.85, 0.35)
    for j, (name, (lo, hi)) in enumerate(offscale.items()):                 # stacked at the top, where no curve runs
        ax[2].text(2, 0.30 - 0.11 * j, f"{SHORT[name]} below " + f"{hi:+.1f}".replace("-", "−") + ", not drawn", fontsize=6, color=COL[name], va="top")
        facts["real"][name]["raw"]["skill_h16_to_128_range"] = [lo, hi]
    hnd, lab = ax[1].get_legend_handles_labels()
    leg = top_legend(fig, hnd, lab, handlelength=1.8, columnspacing=1.0)
    SUBCAPS["fig_real"] = layout_row(fig, ax, ["(a) Real-increment martingale", "(b) Raw series", "(c) Raw series, skill"],
                                     legend=leg, panel_h=63.4, max_h_pt=133.2)
    finish(fig, "fig_real", ax)

# ------------------------------------------------------------------ fig:remedy (symmetry-averaged correction)
if os.path.exists(os.path.join(RES, "remedy_summary.json")):
    RS = json.load(open(os.path.join(RES, "remedy_summary.json")))
    NAMES = [(n, k) for n, k in [("Chronos-small", "chronos"), ("Chronos-Bolt-small", "chronosbolt"), ("Chronos-2", "chronos2"), ("TiRex", "tirex"), ("Moirai-small", "moirai"), ("Moirai-2.0", "moirai2"), ("TimesFM-2.0", "timesfm"), ("TimesFM-2.5", "timesfm25"), ("Time-MoE-200M", "timemoe"), ("Sundial", "sundial"), ("FinCast", "fincast")] if k in RS["models"]]
    K = RS.get("K") or 4
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.85), constrained_layout=True)
    xs = np.arange(len(NAMES)); facts["remedy"] = {}
    # three estimators per model: raw (open circle), mirror-corrected (filled circle), sign-averaged K (triangle)
    def triple(a, i, vals, ses, name, key):
        good = [v for v in vals if v is not None and np.isfinite(v)]
        a.plot([i - 0.22, i, i + 0.22][:len(vals)], vals, color=COL[name], lw=0.7, zorder=1)
        for xo, v, se, mk, mfc in zip((i - 0.22, i, i + 0.22), vals, ses, ("o", "o", "^"), ("white", COL[name], COL[name])):
            if v is None or not np.isfinite(v): continue
            a.errorbar(xo, v, yerr=se if se else None, fmt=mk, ms=3.0, mfc=mfc, color=COL[name], elinewidth=0.5, capsize=1.2, zorder=2)
        facts["remedy"].setdefault(name, {})[key] = {"raw": vals[0], "mirror": vals[1], "sign_avg": vals[2]}
    def get(blk, est, h, f):
        return (blk[est][h][f], blk[est][h].get(f + "_se", 0.0)) if blk and est in blk else (np.nan, 0.0)
    off_c = []
    for i, (name, k) in enumerate(NAMES):
        n1, n4, ra = RS["models"][k].get("ladder|N1"), RS["models"][k].get("ladder|N4"), RS["models"][k].get("realsur|all")
        v = [get(n1, e, "16", "skill") for e in ("raw", "mirror", "corrected")]
        triple(ax[0], i, [x[0] for x in v], [x[1] for x in v], name, "N1_skill_h16")
        v = [get(n4, e, "1", "oracle_share") for e in ("raw", "mirror", "corrected")]
        triple(ax[1], i, [x[0] for x in v], [x[1] for x in v], name, "N4_oracle_share_h1")
        v = [get(ra, e, "128", "skill") for e in ("raw", "mirror", "corrected")]
        vals = [x[0] for x in v]
        if np.nanmin(vals) < -0.7:
            off_c.append((name, vals)); facts["remedy"].setdefault(name, {})["realsur_skill_h128"] = {"raw": vals[0], "mirror": vals[1], "sign_avg": vals[2]}
        else:
            triple(ax[2], i, vals, [x[1] for x in v], name, "realsur_skill_h128")
    for a_ in ax: zero_line(a_)
    ax[0].set_ylabel("skill vs persistence, N1, h = 16"); ax[1].set_ylabel("share of oracle gain, N4, h = 1"); ax[2].set_ylabel("skill vs persistence, h = 128")
    ax[2].set_ylim(-0.85, 0.35)                                             # Sundial's error bars reach -0.81; nothing drawn is clipped
    for j, (name, vals) in enumerate(off_c):
        ax[2].text(0.04, 0.96 - 0.1 * j, f"{SHORT[name]} below " + f"{np.nanmax(vals):+.1f}".replace("-", "−"), transform=ax[2].transAxes, fontsize=6, color=COL[name], va="top")   # top-left: empty above the markers
    for a_, lab in zip(ax, "abc"):                                            # staggered rows so that neighbouring labels never touch
        _rows = 2 if len(NAMES) <= 5 else (3 if len(NAMES) <= 8 else 4)     # staggered rows so that neighbouring labels never touch
        a_.set_xticks(xs); a_.set_xticklabels(["\n" * (i % _rows) + SHORT[n] for i, (n, _) in enumerate(NAMES)], fontsize=6.5 if len(NAMES) <= 8 else 6)
    leg = top_legend(fig, [Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=NEUTRAL),
                           Line2D([], [], ls="", marker="o", ms=3.0, color=NEUTRAL),
                           Line2D([], [], ls="", marker="^", ms=3.0, color=NEUTRAL)],
                     ["raw forecast", "mirror correction", "sign-averaged correction"], ncols=(3,), handletextpad=0.3,
                     columnspacing=1.6)
    SUBCAPS["fig_remedy"] = layout_row(fig, ax, ["(a) Null (N1)", "(b) Positive control (N4)", "(c) Real-increment martingale"],
                                       legend=leg, panel_h=83.6)
    finish(fig, "fig_remedy", ax)

# ------------------------------------------------------------------ fig:teaser (Figure 1): one martingale, seven forecasts
# One of the 128 shared direction contexts (seed 4000, N1, level +100, sigma 1) with the last 96 steps
# shown, the correct forecast (flat at the last value) and every model's 128-step forecast, read
# straight from the per-series departure files the direction table was built from.
if False:   # fig_teaser: Chinese draft only, not redrawn (English Figure 1a is figure_design/fig_intro_20260924)
    from nulls import RUNGS as _RUNGS
    _y, _info = _RUNGS["N1"](128, 512, 128, np.random.default_rng(4000)); _ctx = _y[:, :512]
    _i = 115                                                           # flat recent slope (-0.002/step), last value near the level; chosen once and fixed
    _last = _ctx[_i, -1]; _tail = 96
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.1), constrained_layout=True)               # taller when the legend needs three rows
    ax.plot(np.arange(-_tail + 1, 1), _ctx[_i, -_tail:], color=NEUTRAL, lw=0.9, label="martingale context")
    ax.plot([0, 128], [_last, _last], color=NEUTRAL, lw=0.9, ls=(0, (1, 1)), label="optimal forecast (last value)")
    sd = np.sqrt(np.arange(1, 129)) * _info["sigma"]
    ax.fill_between(np.arange(1, 129), _last - sd, _last + sd, color=NEUTRAL, alpha=0.08, lw=0, label="±1σ√h of the future")
    facts["teaser"] = {"series_index": _i, "last_value": float(_last), "tail_steps": _tail}
    for name, f in DIR_FILES:
        d = load_dep(f)
        ax.plot(np.arange(1, 129), _last + d[_i], color=COL[name], ls=LS[name], lw=0.9, label=name)
        facts["teaser"][name] = {"departure_h128": float(d[_i, -1])}
    ax.axvline(0, color=NEUTRAL, lw=0.5, alpha=0.5)
    ax.set_xlabel("step (0 = last observation)"); ax.set_ylabel("level")
    hnd, lab = ax.get_legend_handles_labels()
    byl = dict(zip(lab, hnd))
    # three legend blocks: the reference lines, then the models split as in Section 6.3 (M31, 09-23)
    _grp = [(None, ["martingale context", "optimal forecast (last value)", "±1σ√h of the future"], 1, 0.005),
            ("not decoder-only", ["Chronos-small", "Chronos-Bolt-small", "Chronos-2", "TiRex", "Moirai-small"], 2, 0.33),
            ("decoder-only", ["Moirai-2.0", "TimesFM-2.0", "TimesFM-2.5", "Time-MoE-200M", "Sundial", "FinCast"], 2, 0.655)]
    for title, keys, nc, x0 in _grp:
        keys = [k for k in keys if k in byl]
        lg = fig.legend([byl[k] for k in keys], [disp(k) for k in keys], loc="lower left", ncol=nc, fontsize=6, frameon=False,
                        handlelength=2.0, columnspacing=0.9, bbox_to_anchor=(x0, -0.01), title=title, title_fontsize=6, alignment="left")
    _bot = 0.30
    fig.get_layout_engine().set(rect=(0, _bot, 1, 1 - _bot))
    finish_single(fig, "fig_teaser", ax) if "finish_single" in globals() else None
    if "finish_single" not in globals():
        base = os.path.join(OUT, "fig_teaser")
        fig.savefig(base + ".pdf"); fig.savefig(base + ".svg"); fig.savefig(os.path.join(OUT, "png", "fig_teaser.png"), dpi=300); plt.close(fig)
        json.dump({"status": "PASS", "note": "single panel: no comparable panels to align"}, open(base + ".alignment.json", "w"), indent=1)

json.dump(facts, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures_facts.json"), "w"), indent=1)
_old = json.load(open(args.facts))
def _same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a))
    return a == b
_bad = [k for k in facts if not _same(json.loads(json.dumps(facts[k])), _old.get(k))]
assert not _bad, f"drawn values differ from {args.facts}: {_bad}"
json.dump(SUBCAPS, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "subcaptions_qa.json"), "w"), indent=1)
print("facts identical to", args.facts, "| panel labels:", {k: [r["text"] for r in v] for k, v in SUBCAPS.items()})
print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith((".pdf", ".json"))))
