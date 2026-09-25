# Reproduction guide

How every table and figure is rebuilt from the shipped results, and how the model runs are repeated. Paths are relative to the repository root.

The paper feeds frozen time series foundation models sequences on which the optimal forecast is
known to be the last observed value (a ladder of martingale nulls, then real financial increments with
their signs randomised) and measures what the models forecast anyway. Every number in the paper is
generated from a file in `results/` by a script in `code/`; nothing is typed by hand.

Eleven models from seven families: Chronos-T5 (`amazon/chronos-t5-{tiny,mini,small,base,large}`),
Chronos-Bolt (`amazon/chronos-bolt-small`), Chronos-2 (`amazon/chronos-2`), Moirai-1.1
(`Salesforce/moirai-1.1-R-small`) and Moirai-2.0 (`Salesforce/moirai-2.0-R-small`), TimesFM-2.0
(`google/timesfm-2.0-500m-pytorch`) and TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`), Time-MoE
(`Maple728/TimeMoE-200M`), TiRex (`NX-AI/TiRex`, xLSTM), Sundial (`thuml/sundial-base-128m`) and FinCast (`Vincent05R/FinCast`, `v1.pth`, with the
authors' code from `github.com/vincent05r/FinCast-fts`). All are used as released, with no fine-tuning.

## Layout

```
code/            every script (Python 3.10/3.11, numpy); see "Code map" below
scripts/         environment, install and run scripts, one per model family, plus the CPU-side analysis
data/real/       SOURCES.md, download_real_data.sh, CHECKSUMS.sha256 (the source files themselves are not redistributed)
data/intraday/   SOURCES.md, download_intraday.sh, CHECKSUMS.md5 (the ITCH day is not redistributed either)
data/etth1/      SOURCES.md, download_etth1.sh, CHECKSUMS.sha256 (ETTh1, for the Time-MoE harness validation only)
results/         the result files the paper is built from (model outputs, summaries, metadata)
paper/tables/    the LaTeX tables the paper \inputs, generated from results/ (+ facts.json: every number used in the text)
paper/figures/   the seven figures (PDF/SVG) + figures_facts.json, generated from results/
```

## Reproducing the paper from the shipped results (CPU, minutes)

```bash
bash scripts/run_analysis.sh          # PY=/path/to/python bash scripts/run_analysis.sh to choose the interpreter
```

runs the generator tests (`test_nulls.py`), the numerical verification of the analytical claims
(`verify_theory.py`, Appendix "numerical verification"), downloads the two public data sources and
rebuilds the real-data windows (checked against `results/real_windows_fingerprint.json`; see "Data"),
then regenerates every table and figure from `results/`. Needs numpy, pandas, matplotlib
(`scripts/install_chronos.sh` provides them; any Python 3.11 with those packages works).

| Paper item | Generated file | Script | Reads |
|---|---|---|---|
| (anchor + correction, the main-text table) | `paper/tables/main.tex` | `remedy_summary.py` (+ `real_summary.py` for the martingale column) | `remedy_<model>.npz`, `real_<model>{,_mirror,_mirror_sur}.npz`, `real_windows.npz` |
| (earlier draft, kept for the record) | `paper/tables/real.tex`, `remedy.tex` | `real_summary.py`, `remedy_summary.py` | the two earlier main-text tables, still generated for the record |
| (the eleven models: objective, native output, point forecast read) | written in the appendix source from the drivers' docstrings (`code/pilot_a_*.py`, `model_adapters.py`); not generated | -- | -- |
| (null ladder, the first five models) | `paper/tables/ladder.tex` | `make_paper_tables.py` | `pilot_a.jsonl` |
| (null ladder, the six later models) | `paper/tables/ladder_cx.tex` | `make_paper_tables.py` | `pilot_a.jsonl` |
| (direction) | `paper/tables/direction.tex` | `make_paper_tables.py` | `diag_direction_*.json`, `diag_direction_control.json` |
| (Chronos sizes) | `paper/tables/scale.tex` | `make_paper_tables.py` | `scale_chronos.jsonl` |
| (context-length sweep) | `paper/tables/context.tex` | `make_paper_tables.py` | `diag_context_<model>.json` |
| (anchor, per split) | `paper/tables/real_detail.tex` | `real_summary.py` | as Table 1 |
| (correction, ladder / real) | `paper/tables/remedy_detail.tex`, `remedy_detail_real.tex` | `remedy_summary.py` | as Table 1 |
| (Proposition (v) on the raw windows: gain / drift / cross, prior and break-even per year) | `paper/tables/remedy_decomp.tex` | `remedy_summary.py` | as Table 1 |
| (the estimated-drift gate against raw and mirror, real windows) | `paper/tables/remedy_gate.tex` | `remedy_summary.py` | as Table 1 |
| (the intraday anchor) | `paper/tables/intraday.tex` | `intraday_summary.py` | `intraday_<model>*.npz`, `intraday_windows.npz`, `intraday_windows_meta.json` |
| (the ETTh1 positive control) | `paper/tables/etth1.tex` | `etth1_summary.py` | `etth1_<model>.npz`, `etth1_windows.npz` (rebuilt by `etth1_probe.py` from `data/etth1/ETTh1.csv`) |
| (the small-cap anchor) | `paper/tables/smallcap.tex` | `smallcap_summary.py` | `smallcap_<model>*.npz`, `smallcap_windows.npz` (rebuilt by `smallcap_data.py` from the Ken French size-decile file) |
| (level / shape decoupling: the same z-scored paths at five raw levels) | `paper/tables/level_all.tex` | `make_paper_tables.py` | `diag_level_<model>.json` |
| (the Time-MoE decoding loop against its published ETTh1 numbers) | `paper/tables/timemoe_check.tex` | `make_paper_tables.py` | `validate_timemoe_etth1*.json` |
| (controlled architecture comparison on a synthetic corpus) | `paper/tables/arch.tex` | `arch_summary.py` | `arch_<arch>_<corpus>_s<seed>.json` |
| (calibration under the null: 80 percent interval coverage and width) | `paper/tables/coverage.tex` | `make_paper_tables.py` | `pilot_a.jsonl` (`coverage_80` fields), `diag_coverage_chronos.json` |
| (teaser) | `paper/figures/fig_teaser.pdf` | `make_paper_figures.py` | `departures_*_N1_H128.npz`, `departures_ctrl_pos_H128.npz` |
| (direction) | `fig_direction.pdf` | `make_paper_figures.py` | the same departure files + `diag_direction_*.json` |
| (declared frequency / patch size) | `fig_declaration.pdf` | `make_paper_figures.py` | `pilot_a.jsonl`, `diag_freq_direction_fincast.json`, `departures_fincast_freq*_H128.npz` |
| (Chronos sizes) | `fig_scale.pdf` | `make_paper_figures.py` | `scale_chronos.jsonl` |
| (price level) | `fig_level.pdf` | `make_paper_figures.py` | `diag_direction_level.jsonl` |
| (real-data anchor) | `fig_real.pdf` | `make_paper_figures.py` | `real_windows.npz`, `real_<model>.npz` |
| (the correction) | `fig_remedy.pdf` | `make_paper_figures.py` | `remedy_summary.json` |
| numbers quoted in the text | `paper/tables/facts.json`, `paper/figures/figures_facts.json` | the two scripts above | |

`make_paper_figures.py` also runs a render-time panel-alignment audit when the module
`audit_panel_alignment` is on the `PYTHONPATH`; without it the figures are drawn and the audit is
reported as not run.

## Data

The real-data anchor uses two public sources: the European Central Bank's euro foreign exchange
reference rates and the Kenneth R. French data library (49 daily industry portfolios and the daily
Fama/French factors). They are **not redistributed** here: ECB data may be reproduced with the source
acknowledged, and the Ken French library is provided for research use with acknowledgement, so
`data/real/download_real_data.sh` fetches them from the providers (about 6 MB).
`data/real/SOURCES.md` records the URLs, coverage, terms and exactly what is used.

Both providers extend their files over time and windows are counted backwards from the last
observation, so `code/real_data.py` cuts a later download at the last dates of the paper's snapshot
of 2026-09-16 (`--end-fx 2026-09-15`, `--end-eq 2026-07-31`, the defaults) and

```bash
python code/real_data.py --data data/real --out results/real_windows.npz --k 0 --n-sur 4 \
       --expect results/real_windows_fingerprint.json
```

reports `exact` when the 1280 rebuilt windows (380 FX, 900 equity; context 512 + horizon 128 trading
days; four sign-randomised surrogate copies per window, seeds `5000 + window + 100000 * copy`) are
byte-identical to the paper's, `float-noise` when only last bits differ, and `MISMATCH` when the
provider has revised history. `results/real_windows_meta.json` (asset, dates) is shipped;
`results/real_windows.npz` itself is derived from the sources and is rebuilt by the command above.

The intraday anchor uses one day of Nasdaq TotalView-ITCH sample data (2019-12-30, 3.5 GB), Nasdaq's
public developer sample at emi.nasdaq.com, likewise not redistributed: `data/intraday/download_intraday.sh`
fetches it (its md5 is in `data/intraday/CHECKSUMS.md5`), `code/itch_trades.py --symbols ALL` extracts
every execution (about six minutes, 2 GB of memory), and

```bash
python code/intraday_data.py --trades data/intraday/itch_trades_2019-12-30.npz \
       --out results/intraday_windows.npz --expect results/intraday_windows_fingerprint.json
```

rebuilds the 792 windows (40 symbols, 512 + 16 consecutive transactions, four sign copies each) and
checks them against the paper's fingerprint. `data/intraday/SOURCES.md` records what is used.

## Reproducing the model runs (GPU)

Six conda environments, one per dependency set (`scripts/install_*.sh`, idempotent; each script prints
the versions it installed and downloads the checkpoints it needs):

| Script | env | Models | Tested with |
|---|---|---|---|
| `install_chronos.sh` | `tsfmfin` | Chronos-T5 x5, Chronos-Bolt, Chronos-2 | torch 2.5.1+cu121, chronos-forecasting 2.3.2, transformers 5.17.0, numpy 2.4.6, Python 3.11 |
| `install_fincast.sh` | `fincast_v1` | FinCast, Time-MoE, Sundial | torch 2.5.0+cu124, transformers 4.53, Python 3.11.11; clones FinCast-fts and fetches the 3.97 GB `v1.pth` |
| `install_moirai.sh` | `moirai` | Moirai-1.1-R-small | torch 2.4.1+cu121, gluonts 0.14.4, uni2ts, Python 3.10 |
| `install_timesfm.sh` | `timesfm` | TimesFM-2.0-500m | timesfm 1.3.0 (the last version with the frequency indicator), torch 2.5.1+cu121, Python 3.11 |
| `install_tirex.sh` | `tirex` | TiRex | tirex-ts without its CUDA extra (PyTorch sLSTM fallback), torch 2.5.1+cu121, Python 3.11 |
| `install_tsfm25.sh` | `tsfm25` | TimesFM-2.5 | timesfm 3.0.2, torch 2.5.1+cu121, Python 3.11 |
| (`install_moirai.sh`) | `moirai` | Moirai-2.0 too | uni2ts 2.0.0 ships the moirai2 module |

`scripts/env_*.sh` set `PYTHONPATH`, the result locations and the GPU (the card with the most free
memory, by UUID); `CONDA_ENVS` (default `$HOME/anaconda3/envs`) is where `install_*.sh` create the
environments and where `env_*.sh` look for them, so point it at a disk with room (5-7 GB each), and uncomment
`HF_ENDPOINT` if the Hugging Face hub is reachable only through a mirror. The runs were made on a
shared machine with 48 GB cards; Chronos-T5 at S=100 sample paths and batch 64 needs about 38 GB.

```bash
bash scripts/run_chronos.sh    # ladder, controls, spectra, declared frequency, random init, diagnostics, five sizes
bash scripts/run_fincast.sh    # FinCast ladder, declared frequency, direction
bash scripts/run_moirai.sh     # Moirai ladder, patch-size sweep, direction
bash scripts/run_timesfm.sh    # TimesFM ladder, declared frequency, direction
bash scripts/run_timemoe.sh    # Time-MoE ladder, direction
bash scripts/run_remedy.sh     # correction ladder + real-data anchor (raw, copies, mirrors) for the five models above; the longest
                               #   (Chronos-T5 alone forecasts 12 800 real windows at about 1.2 s each with S = 100)
bash scripts/run_chronosx.sh   # Chronos-Bolt and Chronos-2: everything above (minutes: both are point forecasters)
bash scripts/run_tirex.sh      # TiRex: smoke test, then everything above including its context sweep (about half an hour)
bash scripts/run_moirai2.sh    # Moirai-2.0: the same, in the moirai env (minutes)
bash scripts/run_tsfm25.sh     # TimesFM-2.5: the same plus the direction stage with its flip-invariance averaging off (about 20 minutes)
bash scripts/run_sundial.sh    # Sundial: the same, in the fincast_v1 env (S = 100 sample paths; about 10 minutes)
bash scripts/run_context.sh    # context-length sweep, all eleven models (minutes)
bash scripts/run_intraday.sh   # intraday anchor, all eleven models (about 40 minutes; Chronos-T5 is most of it); needs results/intraday_windows.npz
bash scripts/run_level_all.sh  # level / shape decoupling, all eleven models (minutes)
bash scripts/run_etth1.sh              # ETTh1 positive control (minutes per model)
bash scripts/run_smallcap.sh           # small-cap anchor (raw + copies + mirrors; needs results/smallcap_windows.npz from code/smallcap_data.py)
bash scripts/run_arch.sh               # controlled architecture comparison, 12 small training runs (~8 min each on one card)
bash scripts/run_validate_timemoe.sh   # Time-MoE decoding loop against the model's own ETTh1 benchmark (hours to a day; one horizon per process and --official-only shorten it); needs data/etth1/ETTh1.csv
```

`run_remedy.sh`, `run_chronosx.sh`, `run_tirex.sh`, `run_moirai2.sh`, `run_tsfm25.sh` and `run_sundial.sh` need `results/real_windows.npz` (see "Data"). Every driver
appends to `results/pilot_a.jsonl` or writes its own file and skips configurations already present, so
an interrupted run is resumed by starting the same script again. Delete (or move) the shipped
`results/` first to reproduce from nothing; with it in place the scripts find every configuration
done and only the analysis runs.

Settings (all in the scripts; the paper's Section 6 and the appendix state them): 512 contexts x 3
seeds per rung and model for the ladder at the main horizon H = 16, 128 contexts at H = 128 / 256 for
the spectral and direction analyses (direction contexts: seed 4000, shared by every model), 256
contexts for the artefact controls, 256 antithetic contexts x K = 4 sign copies per rung for the
correction ladder (data seed 6000, sign seeds 7000 + rung index), 1280 real windows x K = 4
surrogate copies for the anchor, S = 100 sample paths for the sampled models (Chronos-T5, Moirai),
2000 cluster-bootstrap replicates for the real-data standard errors.

## Result files shipped

| File(s) | Written by | Content |
|---|---|---|
| `pilot_a.jsonl` | `pilot_a*.py` | one row per (model, stage, rung, seed, horizon, ...): the ladder, artefact controls, declared-frequency test, random-init reference, spectral stage |
| `scale_chronos.jsonl` | `diag_scale_chronos.py` | the five Chronos-T5 sizes on N1 and N4 |
| `spectral_deep_H{128,256}.json`, `departures_{N1,N2,N4}_H*.npz` | `spectral_deep.py` | drift / seasonal-band decomposition of the departure; per-series departures |
| `diag_direction_control.json`, `departures_ctrl_*_H128.npz` | `diag_direction_control.py` | sign / level / mirror controls of the direction (Chronos-T5-small) |
| `diag_direction_{fincast,moirai,timesfm,timemoe,bolt,chronos2,tirex,moirai2,timesfm25,timesfm25-noflip,sundial}.json`, `departures_<model>_*_H128.npz` | the drivers / `diag_direction_*.py` | direction at H = 128 on the shared contexts |
| `diag_direction_level.jsonl`, `departures_level_*_H128.npz` | `diag_direction_level.py` | the price-level sweep |
| `diag_context_<model>.json` | `diag_context.py` | the context-length sweep: nested suffixes of 128 to 2048 points of the same 128 random walks |
| `diag_freq_direction_fincast.json`, `departures_fincast_freq*_H128.npz` | `diag_freq_direction_fincast.py` | FinCast's direction under its three frequency labels |
| `diag_dispersion.json`, `diag_n2_interval.json` | `diag_dispersion.py`, `diag_n2_interval.py` | is the mean driven by a few series; do N2 intervals track conditional volatility |
| `diag_level_<model>.json` | `diag_level_all.py` | the same increments at five raw levels: level / shape decoupling |
| `validate_timemoe_etth1_h<H>.json` | `validate_timemoe.py` (one horizon per process) | the Time-MoE decoding loop on ETTh1 in the official protocol, against the numbers the Time-MoE paper reports |
| `diag_coverage_chronos.json` | `diag_coverage_chronos.py` | Chronos-T5's 80 percent interval coverage and width on the ladder (the other drivers record theirs in `pilot_a.jsonl`); Table 7 |
| `real_<model>.npz` (+`.json`) | `real_probe.py` | point forecasts on the 1280 raw windows (`yhat_raw`) and their K = 4 surrogates (`yhat_sur`, shape (n, K, H)); `mcvar_*` = Monte Carlo variance for the sampled models |
| `real_<model>_mirror.npz`, `real_<model>_mirror_sur.npz` | `real_probe.py --mirror`, `--mirror-sur` | forecasts on the multiplicative mirror of every raw window / of every surrogate copy |
| `remedy_<model>.npz` (+`.json`) | `remedy_ladder.py` | per rung: contexts, futures, oracle, forecasts on the contexts and on their K = 4 sign copies |
| `real_summary.json`, `remedy_summary.json` | `real_summary.py`, `remedy_summary.py` | every real-data / correction metric with cluster-bootstrap standard errors (return units) |
| `remedy_summary_k16.json` | `remedy_summary.py` on a K = 16 run | the K = 16 extension for Chronos-T5 and Moirai (`real_probe.py --sur-from 4` on a `--n-sur 16` windows file; not used in the paper, forecasts not shipped for size) |
| `real_windows_meta.json`, `real_windows_fingerprint.json` | `real_data.py` | window metadata (asset, dates) and the fingerprint of the paper's windows |
| `intraday_<model>{,_mirror,_mirror_sur}.npz` (+`.json`), `intraday_summary.json`, `intraday_windows_meta.json`, `intraday_windows_fingerprint.json` | `real_probe.py --prefix intraday`, `intraday_summary.py`, `intraday_data.py` | the intraday anchor's forecasts and metrics; the windows themselves are rebuilt from Nasdaq's ITCH day (see `data/intraday/`) |

## Code map

| File | Role |
|---|---|
| `nulls.py`, `test_nulls.py` | the four null generators N1-N4, N4's closed-form optimal forecast, departure metrics; 30 checks |
| `probe_metrics.py` | `make_data / summarise / spectra`: the measurement code every driver shares |
| `verify_theory.py` | numerical verification of the paper's analytical claims (risk decomposition, gate, the symmetry-averaged correction, the mirror group on N4, the assumption-free identity of part (v)); 23 checks |
| `pilot_a.py` | Chronos-T5 driver (`--stage ladder/floor/spectral/freq/all`, resumable) |
| `arch_corpus.py` / `arch_train.py` / `arch_summary.py` | the controlled architecture comparison: a synthetic 'general' corpus (growth or symmetric), a masked encoder and a decoder-only model of identical size trained on it, probed like the released models (`scripts/run_arch.sh`) |
| `etth1_probe.py` / `etth1_summary.py` | the ETTh1 positive control: every model on 924 ETTh1 windows and their mirrors, skill against persistence and the seasonal naive (`scripts/run_etth1.sh`) |
| `smallcap_data.py` / `smallcap_summary.py` | the small-cap anchor: windows and sign-randomised copies from the Ken French size deciles (same layout as `real_data.py`), forecast by `real_probe.py --prefix smallcap` (`scripts/run_smallcap.sh`), scored at h=16 against fitted drift / AR(1) / AR(5) benchmarks |
| `n4_bayes.py` | N4's exact conditional mean (two-state filter on the trade directions, `nulls.oracle_forecast_exact`) against the optimal linear MA(1) forecast on the ladder's own draws; every share of the N4 optimum in the paper is taken against the exact one (`results/n4_optima.json`) |
| `validate_timemoe.py` | reproduces the Time-MoE paper's ETTh1 zero-shot row through this paper's decoding loop (Appendix, the Time-MoE harness) |
| `pilot_a_fincast.py`, `pilot_a_moirai.py`, `pilot_a_timesfm.py`, `pilot_a_timemoe.py`, `pilot_a_chronosx.py`, `pilot_a_tirex.py`, `pilot_a_moirai2.py`, `pilot_a_timesfm25.py`, `pilot_a_sundial.py` | the other drivers, same JSONL schema; `pilot_a_chronosx.py --model bolt/chronos2`; `pilot_a_timesfm25.py --flip off` switches the model's built-in flip-invariance averaging off; `pilot_a_sundial.py` calls the model's forward directly (its packaged `generate()` needs transformers 4.40 internals) |
| `spectral_deep.py`, `test_decompose.py` | drift / seasonal-band separation of the departure, with its tests |
| `diag_*.py` | the diagnostics listed under "Result files shipped" (`diag_context.py` is the context-length sweep) |
| `smoke*.py` | load / shape / finiteness / prefix-equivalence checks run before each model's first experiment |
| `model_adapters.py` | one `get_forecaster(model, H)` interface over the drivers, used by the two scripts below |
| `real_data.py`, `real_probe.py`, `real_summary.py` | the real-data anchor: windows, forecasts, metrics |
| `itch_trades.py`, `intraday_data.py`, `intraday_summary.py` | the intraday anchor: transaction prices from a Nasdaq ITCH day, its windows (with K sign copies and the trade directions), and its metrics; `real_probe.py --prefix intraday` makes the forecasts |
| `remedy_ladder.py`, `remedy_summary.py` | the symmetry-averaged correction on the ladder and on the anchor |
| `merge_real_k.py` | joins a K = 4 file with a `--sur-from 4` file into a K = 16 file |
| `make_paper_tables.py`, `make_paper_figures.py` | the tables and figures |
| `make_results_section.py`, `render_*.py`, `report_pilot_a.py`, `diag_direction_chronos.py`, `diag_trend_chronos.py` | human-readable reports from the same files |
| `paths.py` | repository-relative default locations (`results/`, `weights/`); every script also takes an environment-variable override |
