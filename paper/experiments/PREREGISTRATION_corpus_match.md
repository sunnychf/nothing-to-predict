# Remembered series in public corpora, and directional accuracy (written 2026-09-24 18:10 CST, before the full runs)

## What was seen before writing this

A pilot on 2026-09-24 (M4 daily of the Chronos datasets only, against the Ken French market's daily log returns
1985-2019, best lag, overlap at least 250 days) found 31 of 3731 usable series with correlation above 0.9, on spans
between 1997 and 2017. Nothing else below has been computed: not the full scan, not the industry matches, not the
coverage of any window set, not the timing split, not any directional accuracy.

## A. Series in the public pretraining corpora that match the US market (experiments/corpus_match.py)

Corpora: the downloaded copies used by the corpus audit (Chronos datasets, LOTSA, Time-300B), every series of every
subset except synthetic ones, values all positive and at least 251 observations. Targets: daily log returns of the Ken
French market (Mkt-RF + RF) 1963-07 to 2026-07 and, for the series whose best market correlation exceeds 0.6, of the
49 industry portfolios. Statistic: the largest correlation of daily log returns over all alignments with an overlap
of at least 250 days. A series matches a target when that correlation is at least 0.9 (chance maxima are near 0.3 at
this overlap, 0.45 over millions of series). The union of matched spans, in market trading days, is the covered set.

Coverage (design, computed before any model output is split by it): an end date of a window set is covered when at
least half of its 128 future trading days lie in the covered set of the market.

Predictions:
C1 The covered set includes part of 1996-2014 and none of the daily anchor's futures (2017-09 to 2026-07).
C2 For each of Chronos-Bolt, TiRex, Moirai-2.0 and TimesFM-2.0, the timing correlation of timing_check.py (date means
   of forecast and realised 128-day returns, raw contexts) is higher on the covered end dates of the 1996-2014 windows
   than on the uncovered ones. p-value for each model: permutation of the covered labels across end dates (20,000).
   If fewer than 8 end dates fall in either group, C2 is not tested and only the coverage is reported.
Controls, reported without a prediction: Chronos-T5, Chronos-2, Moirai-1.1, TimesFM-2.5, Time-MoE, Sundial, FinCast.
Decision rule for the paper: if C1 holds, the text names the corpus series that contain the market's path; if C2 holds
for a model, the text links its timing to them; if C2 fails, the text says the timing is not confined to covered dates.

## B. Directional accuracy on the daily anchor (experiments/direction_accuracy.py; forecasts already published)

Directional accuracy: the share of windows in which the forecast and the realised h-step move from the last value
have the same sign (ties count one half), at h = 16 and h = 128, on the anchor's raw equity and exchange-rate windows,
on the sign-randomised copies, and for the mirror-corrected forecast (the sign of the odd part). Baselines: always up,
and the sign of the context's mean return. Standard errors: cluster bootstrap over (family, end date), 2,000 draws.

D1 On raw equities at h = 128, no model's directional accuracy exceeds the always-up rule's by more than two
   standard errors.
D2 On the sign-randomised copies, every model's directional accuracy is within two standard errors of one half.
D3 On raw equities at h = 128, the mirror correction lowers the directional accuracy of the five models whose N1
   forecasts lean upward (Chronos-T5, Chronos-Bolt, Chronos-2, TiRex, Moirai-1.1).
Reported without a prediction: the same measures on the 1996-2014 windows.

# Outcomes (added 2026-09-24 evening after the runs; the text above is unchanged, and its sha256 as written was 0ab2bdf8…)

## A (corpus_match.py on the CPU server, corpus_match_summary.py)
287,251 series in 213 subsets scanned (Chronos datasets 129,053; LOTSA 29,872; Time-300B 128,326). 79 series match the
market at r >= 0.9 (0.90 to 0.98), all in the three copies of M4 daily (33 in the Chronos datasets, 23 each in LOTSA and
Time-300B), 69 distinct by value; every match survives dropping the five largest market days (r >= 0.85) and has under
20 percent zero returns. The only other series above 0.8 are eight wind-farm series whose correlation comes from
1987-10-19 alone. Matched spans run from 1984 to 2017-10-19 and cover 82 percent of 1996-2014 trading days (73 percent in
the LOTSA and Time-300B copies) and 8 percent of 2017-2026.
C1 not met as written: it holds for 1996-2014, but the future of the anchor's earliest equity end date (2017-11-30) is
77 percent covered; the other 17 equity end dates are not covered at all.
C2 not tested: 31 of the 37 end dates of 1996-2014 are covered and only 6 are not, below the 8 required. Only the coverage
is reported in the paper. (Descriptive, not used: timing on covered versus uncovered dates for the four models 0.75 to
0.91 against 0.29 to 0.48; Chronos-T5, Moirai-1.1 and Sundial also differ in that direction.)
Training data (checked 2026-09-24): Chronos's in-domain datasets, whose training portions trained Chronos, include
m4_daily (chronos-forecasting scripts/evaluation/configs/in-domain.yaml); TimesFM-2.0 contains the pretraining set of
TimesFM 1.0, which lists M4 daily (model card; Das et al. Table 1); LOTSA and Time-300B contain it; TiRex's card lists
the Chronos datasets; Moirai-2.0's card lists the training portions of the GIFT-Eval datasets, which include M4;
Chronos-Bolt's card lists no datasets.

## B (direction_accuracy.py)
D1 met: on raw equities at h = 128 always up scores 0.691 +/- 0.055 and every model less (0.399 to 0.664).
D2 not met: on the copies Chronos-T5 (h = 16: 0.517 +/- 0.006), TiRex (h = 128: 0.476 +/- 0.007) and Moirai-1.1 (h = 128:
0.483 +/- 0.007) are more than two standard errors from one half. One half was the wrong reference: compounding makes
rises less likely than falls on the copies (always up scores 0.492 at h = 16 and 0.467 at h = 128), so a forecast that
mostly says up scores below one half; Chronos-T5's 0.517 at h = 16 is not explained by this.
D3 met: the mirror correction lowers directional accuracy on raw equities at h = 128 for all five upward-leaning models.
