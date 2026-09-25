# Experiments of Appendices E.3, E.4, F.5, F.6 and G.8

Scripts are run from `paper/` (the parent of this directory), with the release's `code/` on the path for the
model drivers; they read `../results/`, `../data/real/` and write `tables/` and `experiments/results/`.
Predictions were written down before the runs in `PREREGISTRATION.md` (fine-tuning, falling markets, larger
corpus models), `PREREGISTRATION_shift_anchors.md` (calendar-shift check on the paper's anchors) and
`PREREGISTRATION_corpus_match.md` (search of public training data, directional accuracy); their outcome
sections were added afterwards, and the text above each outcome section is as written.

| Paper item | Script(s) | Results |
|---|---|---|
| Uncertainty of upward fractions, paired small-cap differences, sampling-noise share (Tables mc-share, smallcap) | `tier2_reanalysis.py` | `results/tier2_reanalysis.json` |
| Mirror augmentation during fine-tuning (Table finetune) | `ft_bolt.py`, `eval_bolt.py`, `finetune_summary.py` | `results/finetune/` |
| Falling-market test (Table falling) | `falling_windows.py`, `code/real_probe.py`, `falling_summary.py` | `results/falling/`, `results/falling_summary.json` |
| Market timing (Table timing, Figure timing) | `timing_check.py`, `../figure_design/fig_timing_*/plot_fig_timing.py` | `results/timing_check.json` |
| Calendar-shift check (Table shift) | `shift_windows.py`, `shift_windows_anchor.py`, `server/shard_probe.py`, `server/run_shift_anchor.sh`, `shift_check.py`, `shift_bias_sim.py` | `results/shift/`, `results/shift_anchor/`, `results/shift_size/`, `results/shift_check.json`, `results/shift_bias_sim.json` |
| Corpus trend statistics (Table corpus-trend) | `server/dl_corpora.py`, `corpus_trend.py`, `corpus_summary.py` | `results/corpus/` |
| Larger controlled-corpus models (Table arch-scale) | `code/arch_train.py --d 256 --layers 6`, `arch_scale_summary.py` | `results/arch_scale/` |
| Search of the public training data for the US market's daily path (Appendix F.5) | `corpus_match_summary.py --targets`, `corpus_match.py` (server), `server/corpus_match_extract.py`, `corpus_match_summary.py` | `results/corpus_match/corpus_match.json`, `corpus_match_summary.json` |
| Directional accuracy (Table direction-accuracy) | `direction_accuracy.py` | `results/direction_accuracy.json` |
| Paired differences from the fitted drift in Table smallcap (merged once into the restyled table) | `smallcap_table.py` | `../tables/smallcap.tex` |
| Numbers quoted in the text of these appendices | `check_text_numbers.py` | |

Most of these runs used a CPU server. `server/code_cpu_patch/` holds the model drivers as run there: the only
change from `code/` is that the device and backend strings (`"cuda"`, `"gpu"`) are read from the environment
variables `TSFM_DEVICE` and `TSFM_BACKEND`, with the original values as defaults. Chronos-T5's falling-market
passes were split across processes (`server/shard_probe.py`, `server/shard_mirror.py`), which changes only its
sampling seeds. The calendar-shift runs on the paper's anchors used a GPU server and the unmodified drivers
(`server/run_shift_anchor.sh`); their unshifted forecasts reproduce the published ones (`shift_check.json`,
"reproduction"). The windows files are derived from the Kenneth R. French data library and the ECB reference
rates and are not shipped: `falling_windows.py`, `shift_windows.py` and `shift_windows_anchor.py` rebuild them
from the downloaded files (`../data/real/`), the last after `code/real_data.py` and `code/smallcap_data.py` have
rebuilt the published windows, which it checks bit for bit. `corpus_match_summary.py --targets` rebuilds the Ken French
return targets (`kf_targets.npz`, not shipped) and `server/corpus_match_extract.py` re-reads the matched corpus series
(`corpus_match_series.npz`, not shipped) from the public corpora downloaded by `server/dl_chronos.py` and `dl_corpora.py`.
