"""Analysis-framework figure (version 5: all text Arial Bold; layout of version 4, the authors' preferred panel style): one column per question of Sections 2-6,
grey panels for theory and instruments, blue panels for experiments (Section 2 has no experiment, so both its panels
are grey). CHECKS = True adds a strip with the checks of Section 7 (version 3 had it; the authors' figure has none).
Methods only, no results; every curve is an idealised sketch, not data (no dates, no values).

Each column is one axes whose units are points (x right, y up). All text is measured and asserted to lie inside its
column; the PDF is then checked with the nature-figure text, collision and alignment audits.
Writes figures/fig_framework.{pdf,svg}, figures/png/fig_framework.png. Run from paper_revision/ with the nature-figure env.
Earlier versions are in versions/: _v1_text (text boxes), _v2_columns (sketch columns), _v3_panels (this style with the
checks strip), _v4_regular (this layout in regular weight). FW_SLACK=1 prints the tightest texts per column."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'Arial'                  # Arial only, no fallback (authors' request, 2026-09-26)
plt.rcParams['font.weight'] = 'bold'                   # all text bold (authors' request, 2026-09-26)
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

FIG = "figures"
W = 396.0                                  # \linewidth, pt
GAP = 5.0
WID = [74.0, 74.0, 78.0, 80.0, 70.0]       # column widths, sum = W - 4 GAP
assert abs(sum(WID) + 4 * GAP - W) < 1e-9
HEAD, TOPP, BOTP, PG = 19.0, 64.0, 80.0, 4.0
CH = BOTP + PG + TOPP + 2.0 + HEAD         # column height
CHECKS = False
STRIP, SG = (21.0, 5.0) if CHECKS else (0.0, 0.0)
H = CH + SG + STRIP
FS, FH = 6.1, 7.0                          # body and header sizes
DPI = 720.0
INK, GREY, TAG = "#222222", "#555555", "#8C8C8C"
BLUE, RED = "#1F5A9E", "#C0392B"           # experiment boxes and the odd part; the even part (bias)
PGREY, PBLUE = "#EFEFEF", "#E6EDF6"        # panel fills
LW = 0.8
DOT = (0, (1.2, 1.2))

fig = plt.figure(figsize=(W / 72, H / 72), dpi=DPI)
TEXTS = []


def column(i):
    x0 = sum(WID[:i]) + i * GAP
    ax = fig.add_axes([x0 / W, (STRIP + SG) / H, WID[i] / W, CH / H])
    ax.set_xlim(0, WID[i]); ax.set_ylim(0, CH); ax.set_axis_off()
    ax._x0, ax._w = x0, WID[i]
    return ax


def T(ax, x, y, s, size=FS, **kw):
    kw.setdefault("color", INK); kw.setdefault("va", "center"); kw.setdefault("fontweight", "bold")
    t = ax.text(x, y, s, fontsize=size, zorder=5, **kw)
    TEXTS.append((t, ax))
    return t


def panel(ax, y0, h, fill, tag, note=None):
    ax.add_patch(FancyBboxPatch((0.3, y0), ax._w - 0.6, h, boxstyle="round,pad=0,rounding_size=3",
                                fc=fill, ec="none", zorder=0))
    T(ax, 4, y0 + h - 6, tag, FS, color=TAG, style="italic")
    if note:
        T(ax, ax._w - 4, y0 + h - 6, note, FS, color=TAG, ha="right")


def rbox(ax, x, y, w, h, ec=GREY, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc="white", ec=ec, lw=lw, zorder=2))


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=3,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, mutation_scale=5, shrinkA=0, shrinkB=0))


def walk(n, seed, drift=0.0):
    return np.cumsum(np.random.default_rng(seed).normal(drift, 1.0, n))


def unit(v):
    return (v - v.min()) / (np.ptp(v) + 1e-9)


def path(ax, x, y, w, h, seed, color=INK, lw=LW, end_at=None, flip=False, drift=0.0):
    """Idealised random-walk sketch in [x, x+w] x [y, y+h]; flip gives its mirror about the last value."""
    v = walk(30, seed, drift)
    if flip:
        v = 2 * v[-1] - v
    xs = np.linspace(x, x + w, len(v)); ys = y + h * unit(v)
    if end_at is not None:
        ys = ys - ys[-1] + end_at
    ax.plot(xs, ys, color=color, lw=lw, zorder=3, solid_capstyle="round")
    return xs[-1], ys[-1]


def header(ax, l1, l2):
    T(ax, ax._w / 2, CH - 5, l1, FH, fontweight="bold", ha="center")
    T(ax, ax._w / 2, CH - 13.5, l2, FH, fontweight="bold", ha="center")


AX = [column(i) for i in range(5)]
TOP0 = BOTP + PG                          # bottom edge of the top panel

# ---------------- Section 2 ----------------
ax = AX[0]; w = ax._w
header(ax, "What is optimal", "on a martingale?")
panel(ax, TOP0, TOPP, PGREY, "theory", "Prop. 1")
ex, ey = path(ax, 5, TOP0 + 31, 30, 13, 11, end_at=TOP0 + 38)
ax.plot([ex, ex], [TOP0 + 28, TOP0 + 49], color="#B8B8B8", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
ax.plot([ex, w - 5], [ey, ey], color=INK, lw=0.9, ls=DOT, zorder=3)
T(ax, ex + 3, ey + 7, "last value", FS)
T(ax, ex + 3, ey - 6.5, "is optimal", FS)
rbox(ax, 4, TOP0 + 4, w - 8, 18)
T(ax, w / 2, TOP0 + 16.5, "excess risk = mean", FS, ha="center")
T(ax, w / 2, TOP0 + 9.5, "squared departure", FS, ha="center")
panel(ax, 0, BOTP, PGREY, "instrument: mirror")
T(ax, 4, 60, "x", FS); path(ax, 10, 54, 24, 12, 4)
T(ax, w / 2 + 1, 60, "−x", FS, color=GREY); path(ax, w / 2 + 10, 54, 24, 12, 4, flip=True, color="#8A8A8A")
arrow(ax, w / 2, 52, w / 2, 46.5)
rbox(ax, 4, 26, w - 8, 20, ec=RED)
T(ax, 7, 39.5, "even  ½(D(x) + D(−x))", FS, color=RED, fontweight="bold")
T(ax, 7, 31.5, "same for x and −x", FS)
rbox(ax, 4, 3.5, w - 8, 20, ec=BLUE)
T(ax, 7, 17, "odd  ½(D(x) − D(−x))", FS, color=BLUE, fontweight="bold")
T(ax, 7, 9, "flips sign with x", FS)

# ---------------- Section 3 ----------------
ax = AX[1]; w = ax._w
header(ax, "What do models", "forecast on the null?")
panel(ax, TOP0, TOPP, PGREY, "null ladder")
for k, (s, ec) in enumerate((("N1  random walk", GREY), ("N2  GARCH", GREY), ("N3  heavy tails", GREY),
                             ("N4  bid–ask bounce", BLUE))):
    yk = TOP0 + 47 - k * 10
    rbox(ax, 4, yk - 4.2, w - 8, 8.6, ec=ec, lw=0.7)
    T(ax, 7, yk, s, FS, color=BLUE if ec == BLUE else INK)
T(ax, w / 2, TOP0 + 6, "optimum known on each", FS, ha="center", color=GREY)
panel(ax, 0, BOTP, PBLUE, "experiment")
ex, ey = path(ax, 5, 36, 26, 16, 5, end_at=46)
ax.plot([ex, ex + 15], [ey, ey], color=INK, lw=0.9, ls=DOT, zorder=3)
for e in (12, 7, 2.5, -2.5, -7, -12):
    ax.plot([ex, ex + 15], [ey, ey + e], color=BLUE, lw=0.6, zorder=2)
T(ax, ex + 17, ey + 11, "above", FS); T(ax, ex + 17, ey - 11, "below", FS)
T(ax, w / 2, 20, "11 frozen models,", FS, ha="center")
T(ax, w / 2, 12, "horizons 1 to 128", FS, ha="center")

# ---------------- Section 4 ----------------
ax = AX[2]; w = ax._w
header(ax, "Where does the", "direction come from?")
panel(ax, TOP0, TOPP, PGREY, "controls")
for k, s in enumerate(["level, sign", "frequency", "model size", "context"]):
    yk = TOP0 + 47 - k * 10
    rbox(ax, 4, yk - 4.2, 36, 8.6, lw=0.7)
    T(ax, 6.5, yk, s, FS)
    arrow(ax, 40.5, yk, 44.5, TOP0 + 32)
rbox(ax, 45, TOP0 + 23, w - 49, 18)
T(ax, (45 + w - 4) / 2, TOP0 + 35.5, "released", FS, ha="center")
T(ax, (45 + w - 4) / 2, TOP0 + 28.5, "model", FS, ha="center")
T(ax, w / 2, TOP0 + 6, "one input at a time", FS, ha="center", color=GREY)
panel(ax, 0, BOTP, PBLUE, "experiment: corpus")
for k, (s, lines) in enumerate((("growth", ((0.22, 0.0), (0.16, -1.2), (0.10, 1.0))),
                                 ("symmetric", ((0.2, 0.0), (-0.2, 0.0), (0.0, 0.0))))):
    yk = 52 - k * 22
    rbox(ax, 4, yk - 8.5, 36, 17, lw=0.7)
    xs = np.linspace(7, 37, 24)
    for j, (sl, off) in enumerate(lines):   # idealised series: rising (growth) or rising and falling equally
        ax.plot(xs, yk + 3.5 + off + sl * (xs - 22) + 0.5 * np.sin(1.3 * xs + 2 * j),
                color=[INK, "#8A8A8A", "#B8B8B8"][j], lw=0.6, zorder=3)
    T(ax, 6.5, yk - 5, s, FS)
    arrow(ax, 40.5, yk, 45, yk)
    rbox(ax, 45.5, yk - 8.5, w - 49.5, 17, ec=BLUE)
    T(ax, (45.5 + w - 4) / 2, yk + 3.2, "encoder", FS, ha="center", color=BLUE)
    T(ax, (45.5 + w - 4) / 2, yk - 3.8, "decoder", FS, ha="center", color=BLUE)
T(ax, w / 2, 7, "only the corpus differs", FS, ha="center")

# ---------------- Section 5 ----------------
ax = AX[3]; w = ax._w
header(ax, "Is skill on prices", "more than the bias?")
panel(ax, TOP0, TOPP, PGREY, "instrument")
yc = TOP0 + 38            # a window and its sign-randomised copy: same magnitudes, random signs
r = np.random.default_rng(7); inc = r.normal(0.3, 1.0, 30)
for v, c in ((np.cumsum(inc), INK), (np.cumsum(np.abs(inc) * r.choice([-1, 1], 30)), BLUE)):
    ax.plot(np.linspace(5, 30, 30), yc - 7 + 14 * unit(v), color=c, lw=0.6, zorder=3)
T(ax, 33, yc + 3.5, "randomise", FS); T(ax, 33, yc - 3.5, "return signs", FS)
yc = TOP0 + 13            # even part: a common tilt; odd part: the response to the history
ax.plot([5, 30], [yc - 7, yc + 7], color=RED, lw=0.9, zorder=3)
ax.plot(np.linspace(5, 30, 30), yc + 9 * (unit(walk(30, 3)) - 0.5), color=BLUE, lw=0.6, zorder=3)
T(ax, 33, yc + 3.5, "split into even", FS); T(ax, 33, yc - 3.5, "and odd parts", FS)
panel(ax, 0, BOTP, PBLUE, "experiment")
yc = 50                   # a context, then a future in which the market fell (shaded)
ax.add_patch(Rectangle((21, yc - 9), 9, 18, fc="#D3DCE8", ec="none", zorder=2))
v = unit(walk(30, 9, 0.2))
ax.plot(np.linspace(5, 21, 30), yc - 6 + 12 * v, color=INK, lw=0.6, zorder=3)
ax.plot([21, 30], [yc - 6 + 12 * v[-1], yc - 8], color=INK, lw=0.6, zorder=3)
T(ax, 33, yc + 3.5, "falling markets,", FS); T(ax, 33, yc - 3.5, "predicted first", FS)
yc = 20                   # a forecast set against a trend rule read from the same context
v = unit(walk(30, 13))
ax.plot(np.linspace(5, 22, 30), yc - 7 + 13 * v, color=INK, lw=0.6, zorder=3)
ax.plot([22, 30], [yc - 7 + 13 * v[-1], yc - 1 + 13 * v[-1]], color="#8A8A8A", lw=0.7, ls=(0, (2, 1.2)), zorder=3)
T(ax, 33, yc + 3.5, "timing and", FS); T(ax, 33, yc - 3.5, "calendar shift", FS)

# ---------------- Section 6 ----------------
ax = AX[4]; w = ax._w
header(ax, "Can the bias be", "removed safely?")
panel(ax, TOP0, TOPP, PGREY, "theory", "Prop. 2")
rbox(ax, 4, TOP0 + 28, w - 8, 19)
T(ax, w / 2, TOP0 + 41, "gain guaranteed", FS, ha="center")
T(ax, w / 2, TOP0 + 34, "under symmetry", FS, ha="center")
T(ax, w / 2, TOP0 + 18, "otherwise an exact", FS, ha="center", color=GREY)
T(ax, w / 2, TOP0 + 11, "split of the risk", FS, ha="center", color=GREY)
panel(ax, 0, BOTP, PBLUE, "experiment")
rbox(ax, 4, 41, w - 8, 19, ec=BLUE)
T(ax, w / 2, 54, "keep the odd part", FS, ha="center", color=BLUE, fontweight="bold")
T(ax, w / 2, 47, "at inference", FS, ha="center")
path(ax, 4, 14, 16, 12, 8); path(ax, 4, 14, 16, 12, 8, flip=True, color="#8A8A8A")
arrow(ax, 21.5, 20, 26, 20)
rbox(ax, 26.5, 10.5, w - 30.5, 19, ec=BLUE)
T(ax, (26.5 + w - 4) / 2, 23.5, "fine-tune on", FS, ha="center")
T(ax, (26.5 + w - 4) / 2, 16.5, "x and −x", FS, ha="center")

# ---------------- Section 7 strip (optional) ----------------
if CHECKS:
  bx = fig.add_axes([0, 0, 1, STRIP / H]); bx.set_xlim(0, W); bx.set_ylim(0, STRIP); bx.set_axis_off()
  bx._x0, bx._w = 0.0, W
  bx.add_patch(FancyBboxPatch((0.3, 0.3), W - 0.6, STRIP - 0.6, boxstyle="round,pad=0,rounding_size=3",
                              fc=PGREY, ec="none", zorder=0))
  T(bx, 5, STRIP / 2 + 3.5, "checks for", FS, color=TAG, style="italic")
  T(bx, 5, STRIP / 2 - 3.5, "financial series", FS, color=TAG, style="italic")
  items = [("compare with the last value, a", "fitted drift and an always-up forecast"),
           ("score on sign-randomised", "copies of the windows"),
           ("report the skill left after", "the mirror correction")]
  for k, ((a, b2), x) in enumerate(zip(items, [66, 206, 306])):
      T(bx, x, STRIP / 2 + 3.5, a, FS); T(bx, x, STRIP / 2 - 3.5, b2, FS)
      if k:
          bx.plot([x - 6, x - 6], [4, STRIP - 4], color="#C8C8C8", lw=0.5)

# ---------------- fit check: every text inside its column (1 pt margin) ----------------
fig.canvas.draw()
R = fig.canvas.get_renderer()
bad = []
for t, a in TEXTS:
    e = t.get_window_extent(R)
    x0, x1 = e.x0 * 72 / DPI, e.x1 * 72 / DPI
    if not (a._x0 + 1.0 <= x0 and x1 <= a._x0 + a._w - 1.0):
        bad.append((t.get_text(), round(x0 - a._x0, 1), round(x1 - a._x0, 1), a._w))
if os.environ.get("FW_SLACK"):   # tightest texts per column (pt to the column edge), for layout tuning
    sl = []
    for t, a_ in TEXTS:
        e = t.get_window_extent(R)
        sl.append((round(min(e.x0 * 72 / DPI - a_._x0, a_._x0 + a_._w - e.x1 * 72 / DPI), 1), round(a_._x0), t.get_text()))
    for row in sorted(sl)[:12]:
        print("slack", row)
assert not bad, bad
# render-time alignment gate (nature-figure, when its scripts are on PYTHONPATH): shared tops, bottoms and gutters;
# widths differ by design
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None
IDS = ["s2", "s3", "s4", "s5", "s6"]
if require_matplotlib_panel_alignment is not None:
  require_matplotlib_panel_alignment(
      fig, json_out=os.path.join(FIG, "fig_framework.alignment.json"), axes=AX, panel_ids=IDS, row_groups=[IDS],
      exemptions=[{"panels": IDS, "checks": ["panel-width"],
                   "reason": "column widths follow each section's content; tops, bottoms and gutters are shared"}])
os.makedirs(os.path.join(FIG, "png"), exist_ok=True)
fig.savefig(os.path.join(FIG, "fig_framework.pdf"))
fig.savefig(os.path.join(FIG, "fig_framework.svg"))
fig.savefig(os.path.join(FIG, "png", "fig_framework.png"), dpi=300)
print(f"fig_framework: {W:.0f} x {H:.0f} pt ({W / 72:.2f} x {H / 72:.2f} in)")
