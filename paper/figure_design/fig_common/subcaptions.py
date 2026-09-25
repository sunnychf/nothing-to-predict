"""Panel labels under the panels (the authors' style of 2026-09-24): '(a) short name', centred on each panel's plot area,
with all labels of one row on one baseline just below the row's lowest tick label or axis label. They replace the bold
top-left letters and the panel titles. Shared legends sit above the panels, so only the labels run under them.

Used by every figure script under figure_design/; import with
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig_common"))
    import subcaptions as sc
"""

FS = 7.5      # label size in points (tick labels are 6.5, axis labels 7)
PAD = 3.0     # gap between a row's lowest decoration and the top of its labels, points
MARGIN = 2.0  # space kept below the lowest label of a figure, points


def _px(fig):
    return fig.dpi / 72.0


def decorations_pt(fig, axes):
    """(left, below, above) extent of each axes' decorations (tick labels, axis labels, texts) outside its plot area,
    in points."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); px = _px(fig)
    out = []
    for a in axes:
        tb, pb = a.get_tightbbox(r), a.get_window_extent(r)
        out.append(((pb.x0 - tb.x0) / px, (pb.y0 - tb.y0) / px, (tb.y1 - pb.y1) / px))
    return out


def label_height_pt(fig, fontsize=FS):
    """Height of one label line (parentheses, ascenders, descenders, an italic h), in points."""
    t = fig.text(0, 0, "(a) Hg$h$", fontsize=fontsize)
    fig.canvas.draw()
    h = t.get_window_extent(fig.canvas.get_renderer()).height / _px(fig)
    t.remove()
    return h


def band_pt(fig, axes, fontsize=FS):
    """Space to keep under a row of plot areas for its tick labels, axis labels and panel labels, in points."""
    below = max(d[1] for d in decorations_pt(fig, axes))
    return below + PAD + label_height_pt(fig, fontsize)


def legend_height_pt(fig, legend):
    fig.canvas.draw()
    return legend.get_window_extent(fig.canvas.get_renderer()).height / _px(fig)


def add(fig, rows, fontsize=FS, pad=PAD):
    """rows: [[(axes, '(a) name'), ...], ...], one list per row of panels. Draws the labels and checks them: one
    baseline per row, each centred on its plot area within 0.5 pt, inside the figure, no two labels of a row touching.
    Returns one record per label (text, centre offset, top and bottom in points from the figure's bottom edge)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); px = _px(fig); W, H = fig.bbox.width, fig.bbox.height
    placed = []
    for row in rows:
        low = min(a.get_tightbbox(r).y0 for a, _ in row)
        for a, s in row:
            pb = a.get_window_extent(r)
            t = fig.text((pb.x0 + pb.x1) / 2 / W, (low - pad * px) / H, s, ha="center", va="top", fontsize=fontsize)
            placed.append((len(placed), a, t))
    fig.canvas.draw()
    recs = []
    for _, a, t in placed:
        pb, tb = a.get_window_extent(r), t.get_window_extent(r)
        recs.append({"text": t.get_text(), "centre_offset_pt": round(((tb.x0 + tb.x1) - (pb.x0 + pb.x1)) / 2 / px, 3),
                     "top_pt": round(tb.y1 / px, 3), "bottom_pt": round(tb.y0 / px, 3),
                     "x0_pt": round(tb.x0 / px, 3), "x1_pt": round(tb.x1 / px, 3)})
    k = 0
    for row in rows:
        rr = recs[k:k + len(row)]; k += len(row)
        assert max(x["top_pt"] for x in rr) - min(x["top_pt"] for x in rr) < 0.05, ("labels of a row not on one line", rr)
        for x in rr:
            assert abs(x["centre_offset_pt"]) < 0.5, ("label not centred on its panel", x)
            assert x["bottom_pt"] >= 0.5 and x["x0_pt"] >= 0.0 and x["x1_pt"] <= W / px, ("label outside the figure", x)
        for x, y in zip(sorted(rr, key=lambda v: v["x0_pt"]), sorted(rr, key=lambda v: v["x0_pt"])[1:]):
            assert x["x1_pt"] + 4.0 <= y["x0_pt"], ("neighbouring labels closer than 4 pt", x, y)
    return recs
