"""Merge the paired differences from the fitted drift (experiments/results/tier2_reanalysis.json, "smallcap_paired") into
the appendix small-cap table (tables/smallcap.tex) as a column next to the raw skill, so that one table carries every
small-cap number the paper uses and the separate main-text excerpt (tables/smallcap_main.tex) can be retired. The other
cells of tables/smallcap.tex are left as they are. The new column is checked cell by cell against the excerpt's
"Raw - drift" column, and the columns the two tables share (raw, mirror and sign-randomised skill) against each other.
Run once from finance/paper_revision (refuses to run on an already merged table):
  ~/miniconda3/envs/nature-figure/bin/python experiments/smallcap_table.py"""
import json, os, re, sys

T, X = os.path.join("tables", "smallcap.tex"), os.path.join("tables", "smallcap_main.tex")
B = json.load(open(os.path.join("experiments", "results", "tier2_reanalysis.json")))["smallcap_paired"]
LABEL = {"Chronos-T5": "chronos", "Chronos-Bolt": "chronosbolt", "Chronos-2": "chronos2", "TiRex": "tirex", "Moirai-1.1": "moirai",
         "Moirai-2.0": "moirai2", "TimesFM-2.0": "timesfm", "TimesFM-2.5": "timesfm25", "Time-MoE": "timemoe", "Sundial": "sundial", "FinCast": "fincast"}


def num(v, d=3):
    return f"{v:+.{d}f}".replace("-", "\\textminus{}")


def pm(o):
    return num(o["diff"]) + f"\\,\\textpm\\,{o['se']:.3f}"


new_cell = {"Persistence": "NE", "Drift (fit)": "0", "AR(1) (fit)": pm(B["ar1_minus_drift"]), "AR(5) (fit)": pm(B["ar5_minus_drift"])}
new_cell.update({lab: pm(B["models"][key]["minus_drift"]) for lab, key in LABEL.items()})

lines = open(T).read().splitlines()
assert lines[0] == r"\begin{tabular}{@{}lcccccc@{}}", "tables/smallcap.tex is not the unmerged seven-column table"
cells = lambda l: [c.strip() for c in l.rstrip("\\ ").split("&")]
plain = lambda c: re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
exc = {cells(l)[0]: cells(l) for l in open(X).read().splitlines() if "&" in l and not l.lstrip().startswith(("\\textbf{Forecast}",))}
bad, out = [], []
for l in lines:
    if l == r"\begin{tabular}{@{}lcccccc@{}}":
        out.append(r"\begin{tabular}{@{}lccccccc@{}}"); continue
    if l.startswith(r" & \multicolumn{5}{c}{\textbf{Smallest decile"):
        out.append(l.replace(r"\multicolumn{5}", r"\multicolumn{6}")); continue
    if l == r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}":
        out.append(r"\cmidrule(lr){2-7}\cmidrule(lr){8-8}"); continue
    if l.startswith(r" & \multicolumn{2}{c}{\textbf{Raw}}"):
        out.append(l.replace(r"\multicolumn{2}{c}{\textbf{Raw}}", r"\multicolumn{3}{c}{\textbf{Raw}}")); continue
    if l == r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}":
        out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}"); continue
    if l.startswith(r"\textbf{Forecast} & \textbf{Skill} & \textbf{AR(1) share}"):
        out.append(l.replace(r"\textbf{Skill} & \textbf{AR(1) share}", r"\textbf{Skill} & \textbf{$-$\,drift} & \textbf{AR(1) share}")); continue
    c = cells(l) if "&" in l else None
    if c and c[0] in new_cell:
        e = exc[c[0]]                                          # excerpt row: Forecast, Raw, Raw - drift, Mirror, Sign-randomised
        if e[2] != new_cell[c[0]]:
            bad.append(f"{c[0]}: new column {new_cell[c[0]]} vs excerpt {e[2]}")
        for name, a, b in (("raw", c[1], e[1]), ("mirror", c[3], e[3]), ("sign-randomised", c[5], e[4])):
            if plain(a) != plain(b) and not (plain(a) in ("0", "NE") and plain(b) in ("0", "NE")):
                bad.append(f"{c[0]} {name}: full table {a} vs excerpt {b}")
        out.append(" & ".join(c[:2] + [new_cell[c[0]]] + c[2:]) + r" \\")
        continue
    out.append(l)
if bad:
    print("MISMATCH:\n  " + "\n  ".join(bad)); sys.exit(1)
open(T, "w").write("\n".join(out) + "\n")
print(f"merged {len(new_cell)} rows; the new column equals the excerpt's, and the shared columns agree")
