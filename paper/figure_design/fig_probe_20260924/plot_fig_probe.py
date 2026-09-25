"""Section 2 figure: the probe on one context (Chronos-T5), a worked example of the mirror decomposition and of the
sign-randomised copy. Reads only existing outputs (nothing is typed by hand):
  ../results/remedy_chronos.npz   correction sample on N1: 128 zero-drift random walks (rows 0-127) and their mirrors
                                  about the first value (rows 128-255), context 512, forecast 128 (code/remedy_ladder.py)
  ../results/real_windows.npz, real_windows_meta.json, real_chronos.npz   daily anchor: raw windows, sign-randomised
                                  copies (K = 4) and Chronos-T5's forecasts on both (code/real_data.py, real_probe.py)
Examples are chosen by rule, not by eye:
  a, b  the context whose even part and absolute odd part at h = 128 are closest to their medians over the 128 contexts
        (standardised squared distance);
  c     the market-portfolio window whose first copy's h = 128 departure (in sigma_r) is closest to the median over all
        equity copies.
Writes figures/fig_probe.{pdf,svg}, figures/png/fig_probe.png, figures/fig_probe.alignment.json and
figure_design/fig_probe_20260924/fig_probe_facts.json (every number used in the caption).
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

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = "figures"; RES = os.path.join("..", "results")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
                     "lines.linewidth": 1.1, "axes.titlepad": 5.0})
BLUE, LIGHT, INK, GREY, BAND = "#0F4D92", "#9DC3E6", "#262626", "#8F8F8F", "#E6E6E6"   # Chronos-T5 colour of the paper

# ---------------------------------------------------------------- a, b: one random walk and its mirror
z = np.load(os.path.join(RES, "remedy_chronos.npz")); c, y, s = z["N1/ctx"], z["N1/yhat_raw"], float(z["N1/sigma"]); n = 128
assert np.allclose(c[n:] - c[n:, -1:], -(c[:n] - c[:n, -1:])), "rows 128-255 are not the mirrors of rows 0-127"
D = (y - c[:, -1:]) / s                                   # departures over h = 1..128, in units of the step sigma
Dx, Dm = D[:n], D[n:]
even, odd = (Dx + Dm) / 2, (Dx - Dm) / 2
E, O = even[:, -1], odd[:, -1]
score = ((E - np.median(E)) / E.std()) ** 2 + ((np.abs(O) - np.median(np.abs(O))) / np.abs(O).std()) ** 2
i = int(np.argmin(score))
TAIL = 96
hist = (c[i, -TAIL:] - c[i, -1]) / s; t_hist = np.arange(-TAIL + 1, 1)
h = np.arange(1, 129)
# ---------------------------------------------------------------- c: a real window and its sign-randomised copy
W = np.load(os.path.join(RES, "real_windows.npz")); meta = json.load(open(os.path.join(RES, "real_windows_meta.json")))
P = np.load(os.path.join(RES, "real_chronos.npz"))
eq = np.array([m["family"] == "eq" for m in meta]); mkt = np.where([m["asset"] == "eq:Mkt" for m in meta])[0]
dsur = (P["yhat_sur"][:, :, -1] / W["ctx_sur"][:, :, -1] - 1.0) / W["rsigma_sur"]
med = float(np.median(dsur[eq]))
j = int(mkt[np.argmin(np.abs(dsur[mkt, 0] - med))])
rs_raw, rs_sur = float(W["rsigma_raw"][j]), float(W["rsigma_sur"][j, 0])
TAILC = 128; t_ctx = np.arange(-TAILC + 1, 1)
raw_ctx = (W["ctx_raw"][j, -TAILC:] / W["ctx_raw"][j, -1] - 1.0) / rs_raw
raw_fut = (W["fut_raw"][j] / W["ctx_raw"][j, -1] - 1.0) / rs_raw
raw_fc = (P["yhat_raw"][j] / W["ctx_raw"][j, -1] - 1.0) / rs_raw
sur_ctx = (W["ctx_sur"][j, 0, -TAILC:] / W["ctx_sur"][j, 0, -1] - 1.0) / rs_sur
sur_fc = (P["yhat_sur"][j, 0] / W["ctx_sur"][j, 0, -1] - 1.0) / rs_sur

facts = {"a_b": {"context_index": i, "tail_steps": TAIL, "last96_move_sigma": float(hist[-1] - hist[0]),
                 "departure_h128": float(Dx[i, -1]), "departure_mirror_h128": float(Dm[i, -1]),
                 "even_h128": float(E[i]), "odd_h128": float(O[i]), "median_even_h128": float(np.median(E)),
                 "median_abs_odd_h128": float(np.median(np.abs(O))), "mean_even_h128": float(E.mean()),
                 "iqr_even_h128": np.percentile(E, [25, 75]).tolist(), "iqr_odd_h128": np.percentile(O, [25, 75]).tolist(),
                 "n_contexts": n, "context_length": int(c.shape[1])},
         "c": {"window_index": j, "asset": meta[j]["asset"], "ctx_start": meta[j]["ctx_start"], "ctx_end": meta[j]["ctx_end"],
               "fut_end": meta[j]["fut_end"], "raw_departure_h128_sigma_r": float(raw_fc[-1]),
               "copy_departure_h128_sigma_r": float(sur_fc[-1]), "raw_realised_h128_sigma_r": float(raw_fut[-1]),
               "median_copy_departure_equities": med, "n_equity_copies": int(dsur[eq].size)}}

W_IN, H_IN = 5.5, 1.95
fig, ax = plt.subplots(1, 3, figsize=(W_IN, H_IN), layout="constrained")
# a
a = ax[0]
a.axhline(0, color=GREY, lw=0.7, ls=":", zorder=0)
a.plot(t_hist, hist, color=INK, lw=0.9)
a.plot(t_hist, -hist, color=GREY, lw=0.9)
a.plot(np.r_[0, h], np.r_[0, Dx[i]], color=BLUE, lw=1.3)
a.plot(np.r_[0, h], np.r_[0, Dm[i]], color=BLUE, lw=1.3, ls=(0, (3, 1.5)))
a.text(126, Dx[i, -1] + 0.9, f"{Dx[i, -1]:+.1f}", color=BLUE, fontsize=6, ha="right", va="bottom")
a.text(126, Dm[i, -1] - 0.9, f"{Dm[i, -1]:+.1f}", color=BLUE, fontsize=6, ha="right", va="top")
a.set_xlim(-TAIL, 130); a.set_xticks([-96, 0, 64, 128])
a.set_xlabel("step (0 = last value)"); a.set_ylabel(r"level $-$ last value ($\sigma$)")
# b
b = ax[1]
b.axhline(0, color=GREY, lw=0.7, ls=":", zorder=0)
b.fill_between(h, *np.percentile(odd, [25, 75], axis=0), color=BAND, lw=0, zorder=0)
b.fill_between(h, *np.percentile(even, [25, 75], axis=0), color=LIGHT, alpha=0.55, lw=0, zorder=1)
b.plot(h, odd[i], color=GREY, lw=1.1, ls=(0, (3, 1.5)))
b.plot(h, even[i], color=BLUE, lw=1.4)
b.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, label="even part: removed"),
                  Line2D([], [], color=GREY, lw=1.1, ls=(0, (3, 1.5)), label="odd part: kept")],
         loc="upper left", bbox_to_anchor=(-0.02, 1.03), fontsize=6, handlelength=1.8, handletextpad=0.4, labelspacing=0.25)
b.set_ylim(-1.8, 5.6)
b.set_xlim(0, 130); b.set_xticks([1, 64, 128])
b.set_xlabel("horizon h"); b.set_ylabel(r"part of the departure ($\sigma$)")
# c
cc = ax[2]
cc.axhline(0, color=GREY, lw=0.7, ls=":", zorder=0)
cc.plot(t_ctx, raw_ctx, color=INK, lw=0.9)
cc.plot(t_ctx, sur_ctx, color=GREY, lw=0.9)
cc.plot(np.r_[0, h], np.r_[0, raw_fc], color=BLUE, lw=1.3)
cc.plot(np.r_[0, h], np.r_[0, sur_fc], color=BLUE, lw=1.3, ls=(0, (3, 1.5)))
cc.plot([128], [raw_fut[-1]], "o", ms=3.2, color=INK, zorder=4, clip_on=False)   # at the axis edge: drawn whole
cc.text(119, raw_fut[-1], f"realised {raw_fut[-1]:+.1f}", color=INK, fontsize=6, ha="right", va="center")
cc.text(126, raw_fc[-1] + 0.9, f"{raw_fc[-1]:+.1f}", color=BLUE, fontsize=6, ha="right", va="bottom")
cc.text(126, sur_fc[-1] + 0.9, f"{sur_fc[-1]:+.1f}", color=BLUE, fontsize=6, ha="right", va="bottom")
cc.set_xlim(-TAILC, 130); cc.set_xticks([-128, 0, 64, 128])
cc.set_xlabel("trading day (0 = last value)"); cc.set_ylabel("change (σ of returns)")   # short enough for the plot height
facts["axes"] = {"a_ylim": list(map(float, a.get_ylim())), "b_ylim": list(map(float, b.get_ylim())), "c_ylim": list(map(float, cc.get_ylim()))}

leg = fig.legend(handles=[Line2D([], [], color=INK, lw=0.9, label="original context"),
                          Line2D([], [], color=GREY, lw=0.9, label="mirror (a) or sign-randomised copy (c)"),
                          Line2D([], [], color=BLUE, lw=1.3, label="forecast after the original"),
                          Line2D([], [], color=BLUE, lw=1.3, ls=(0, (3, 1.5)), label="forecast after the mirror or copy")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=2.0, handletextpad=0.4,
                 columnspacing=1.6, labelspacing=0.25, borderaxespad=0.2)
fig.canvas.draw(); fig.set_layout_engine("none")
px = fig.dpi / 72.0; Wpt = W_IN * 72.0
PANEL_H, MAX_H = 82.4, H_IN * 72.0          # plot-area height of the previous version; the figure does not grow
for _ in range(2):                          # the second pass settles tick labels that move with the plot area
    dec = sc.decorations_pt(fig, ax)
    left = max(d[0] for d in dec) + 5.0; gutter = max(d[0] for d in dec) + 8.0; right = 6.0
    w = (Wpt - left - right - 2 * gutter) / 3
    bottom = sc.band_pt(fig, ax) + sc.MARGIN
    top = (fig.get_figheight() * 72.0 - leg.get_window_extent(fig.canvas.get_renderer()).y0 / px) + 4.0 + max(d[2] for d in dec)
    Hpt = min(top + PANEL_H + bottom, MAX_H); ph = Hpt - top - bottom
    fig.set_size_inches(W_IN, Hpt / 72.0)
    for k, x in enumerate(ax):
        x.set_position([(left + k * (w + gutter)) / Wpt, bottom / Hpt, w / Wpt, ph / Hpt])
facts["panel_labels"] = sc.add(fig, [[(ax[0], "(a) A random walk and its mirror"), (ax[1], "(b) Even and odd parts"),
                                      (ax[2], "(c) A real window and its copy")]])
base = os.path.join(FIG, "fig_probe")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig, json_out=base + ".alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
else:
    print("WARNING: alignment gate NOT run", file=sys.stderr)
fig.savefig(base + ".pdf"); fig.savefig(base + ".svg")
os.makedirs(os.path.join(FIG, "png"), exist_ok=True); fig.savefig(os.path.join(FIG, "png", "fig_probe.png"), dpi=300)
json.dump(facts, open(os.path.join(HERE, "fig_probe_facts.json"), "w"), indent=1)
print(json.dumps(facts, indent=1)[:1500])
