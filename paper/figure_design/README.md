# Figure scripts

Run from `paper/`; every script reads result files only and writes `figures/` (PDF for the paper, SVG, PNG preview and
a panel-alignment record when the alignment audit is importable).

| Figure | Script |
|---|---|
| Figure 1 (`fig_intro`) | `fig_intro_20260924/plot_fig_intro.py` |
| Figure 2 (`fig_framework`) | `fig_framework_20260926/plot_fig_framework.py` (a schematic of the analysis; it reads no results) |
| Figure 3 (`fig_probe`) | `fig_probe_20260924/plot_fig_probe.py` |
| Figures 4 and 6 (`fig_direction`, `fig_real`) and the appendix figures `fig_declaration`, `fig_level`, `fig_scale` | `fig_paper_20260924/make_paper_figures_rev.py`: `code/make_paper_figures.py` with the panel labels under the panels; it checks that its drawn values equal `figures/figures_facts.json` |
| Figure 5 (`fig_corpus`) | `fig_corpus_20260925/plot_fig_corpus.py` (reads the corpus-experiment runs; checks its seed means against `arch_summary.json` and `arch_scale_summary.json`) |
| Figure 7 (`fig_timing`) | `fig_timing_20260924/plot_fig_timing.py` |
| Appendix figure `fig_shiftcheck` | `fig_shiftcheck_20260924/plot_fig_shiftcheck.py` |
| Panel labels "(a) name" under each panel | `fig_common/subcaptions.py`, imported by the scripts above |

`fig_summary_20260923/` drew an earlier version of Figure 1 and is kept for reference.
