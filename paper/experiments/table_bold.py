"""Bold marks for the paper's tables, applied to the LaTeX rows before a table is written.

Each rule is the one stated in the table's note. Values are the displayed point estimates; displayed ties are all
bold; a +/- term is never bold. falling_summary.py, direction_accuracy.py, finetune_summary.py, timing_check.py and
shift_check.py call apply() on their rows. Run as a script from finance/paper_revision to re-apply every rule to
tables/*.tex in place, which also covers tables/coverage.tex and tables/remedy_decomp.tex:
  ~/miniconda3/envs/nature-figure/bin/python experiments/table_bold.py
"""
import os, re, sys

NUM = re.compile(r"(?:\\textminus\{\}|[+-])?\d+(?:\.\d+)?")
STAR = r"\textsuperscript{*}"


def _rows(lines):
    """Indices and cells of the data rows after the first \\midrule (panel titles skipped)."""
    out, seen = [], False
    for i, line in enumerate(lines):
        if line.startswith(r"\midrule"):
            seen = True
            continue
        s = line.rstrip()
        if not seen or " & " not in s or not s.endswith("\\\\") or s.startswith(r"\multicolumn"):
            continue
        out.append((i, [c.strip() for c in s[:-2].split(" & ")]))
    return out


def _strip(c):
    """The cell without bold around its point estimate."""
    if c.startswith(r"\textbf{"):
        depth = 0
        for k in range(len(r"\textbf"), len(c)):
            depth += {"{": 1, "}": -1}.get(c[k], 0)
            if depth == 0:
                return c[len(r"\textbf{"):k] + c[k + 1:]
    return c


def _value(c):
    m = NUM.match(_strip(c))
    return float(m.group().replace(r"\textminus{}", "-")) if m else None


def _bold(c):
    c = _strip(c)
    m = NUM.match(c)
    return r"\textbf{" + m.group() + "}" + c[m.end():]


def _mark(cells, picks, cols):
    """Bold exactly the (row, col) pairs in picks among the ruled columns cols."""
    for r, row in enumerate(cells):
        for c in cols:
            if c < len(row) and _value(row[c]) is not None:
                row[c] = _bold(row[c]) if (r, c) in picks else _strip(row[c])


def _best(cells, rows, col, key):
    vals = [(r, key(_value(cells[r][col]))) for r in rows if _value(cells[r][col]) is not None]
    if not vals:
        return set()
    top = max(v for _, v in vals)
    return {(r, col) for r, v in vals if abs(v - top) < 1e-12}


def _is_rule(row):
    return row[0].startswith("Context rule")


def rule_falling(cells):
    """Table 17: the highest raw skill in each regime and the highest null skill."""
    rows = range(len(cells)); cols = [1, 2, 7]
    return set().union(*(_best(cells, rows, c, lambda v: v) for c in cols)), cols


def rule_direction_accuracy(cells):
    """Table 20: the highest accuracy in each accuracy column, baselines included (the upward column is not ranked)."""
    rows = range(len(cells)); cols = [1, 3, 4, 5]
    return set().union(*(_best(cells, rows, c, lambda v: v) for c in cols)), cols


def rule_finetune(cells):
    """Table 27: the highest entry in each skill or share row; direction rows are not ranked."""
    picks = set()
    for r, row in enumerate(cells):
        if any(w in row[0] for w in ("skill", "share", "mirror-corrected")):
            picks |= {(r, c) for c in [1, 2, 3] if (r, c) in _best_row(cells, r, [1, 2, 3])}
    return picks, [1, 2, 3]


def _best_row(cells, r, cols):
    vals = [(c, _value(cells[r][c])) for c in cols if _value(cells[r][c]) is not None]
    top = max(v for _, v in vals)
    return {(r, c) for c, v in vals if abs(v - top) < 1e-12}


def _rule_beats_context(cells, cols):
    """A starred model correlation (p < 0.01) above the context rule of the same column."""
    rule = next(row for row in cells if _is_rule(row))
    picks = set()
    for r, row in enumerate(cells):
        if _is_rule(row):
            continue
        for c in cols:
            v, ref = _value(row[c]), _value(rule[c])
            if v is not None and ref is not None and STAR in row[c] and v > ref:
                picks.add((r, c))
    return picks, cols


def rule_timing(cells):
    """Table 18: raw-context columns only (the context rule reads the raw contexts)."""
    return _rule_beats_context(cells, [1, 4])


def rule_shift(cells):
    """Table 19: every column has its own context rule."""
    return _rule_beats_context(cells, [1, 2, 3, 4, 5])


def rule_coverage(cells):
    """Table 6: the coverage closest to the nominal 0.80 in each coverage column; widths are not ranked."""
    rows = range(len(cells)); cols = [1, 2, 5, 6]
    return set().union(*(_best(cells, rows, c, lambda v: -abs(v - 0.80)) for c in cols)), cols


def rule_remedy_decomp(cells):
    """Table 25: the largest of the three terms (removed energy, future alignment, cross term) in magnitude."""
    picks = set()
    for r, row in enumerate(cells):
        vals = [(c, abs(_value(row[c]))) for c in (2, 3, 4)]
        top = max(v for _, v in vals)
        picks |= {(r, c) for c, v in vals if abs(v - top) < 1e-12}
    return picks, [2, 3, 4]


RULES = {"falling": rule_falling, "direction_accuracy": rule_direction_accuracy, "finetune": rule_finetune,
         "timing": rule_timing, "shift": rule_shift, "coverage": rule_coverage, "remedy_decomp": rule_remedy_decomp}


def apply(name, lines):
    """Return the table's lines with its bold rule applied (existing bold in the ruled columns is reset first)."""
    lines = list(lines)
    rows = _rows(lines)
    cells = [c for _, c in rows]
    picks, cols = RULES[name]([list(c) for c in cells])
    _mark(cells, picks, cols)
    for (i, _), c in zip(rows, cells):
        lines[i] = " & ".join(c) + r" \\"
    return lines


if __name__ == "__main__":
    for name in RULES:
        p = os.path.join("tables", name + ".tex")
        old = open(p).read()
        new = "\n".join(apply(name, old.rstrip("\n").split("\n"))) + "\n"
        if new != old:
            open(p, "w").write(new)
        print(f"{name}: {'updated' if new != old else 'unchanged'}")
    sys.exit(0)
