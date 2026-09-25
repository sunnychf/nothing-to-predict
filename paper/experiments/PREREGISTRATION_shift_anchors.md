# Calendar-shift check on the paper's own anchors (written 2026-09-24 11:31 CST, before any shifted forecast existed)

## Why

The falling-market test (PREREGISTRATION.md, Experiment 1) and its post hoc checks found that Chronos-Bolt, TiRex,
Moirai-2.0 and TimesFM-2.0 follow the calendar on 1996-2014 equity windows: moving the context end by 20 or 40
trading days moves their forecasts with the realised future (within-date correlation 0.52 to 0.87, permutation
p < 0.001; controls Moirai-1.1 +0.06 and Chronos-2 -0.29; experiments/shift_check.json). The paper's daily anchor
(2017-2026) was checked only by the timing check over 18 end dates, which has little power, and the size-decile
anchor (1963-2026) not at all. Both anchors support claims in the paper: the anchor carries the raw-equity and
mirror results, the size deciles carry the statement that the models use real short-horizon structure.

## Design (identical to experiments/shift_windows.py and shift_check.py)

Windows: experiments/shift_windows_anchor.py. Every raw window of the daily anchor (1280: 900 equity windows of
50 series at 18 end dates, 380 exchange-rate windows of 20 series at 19 end dates) and of the size-decile anchor
(240: 120 equal-weighted smallest-decile windows, 120 value-weighted largest-decile windows) is re-cut with its
context end moved by k in (-40, -20, +20, +40) trading days, keeping the 512-day context and the 128-day future;
windows that would leave the sample are skipped. The k = 0 windows equal the published ones bit for bit and are
forecast again as a consistency check against the published forecasts.

Models: Chronos-Bolt, TiRex, Moirai-2.0, TimesFM-2.0 (the four that followed the calendar on 1996-2014) and the two
controls of that check, Moirai-1.1 and Chronos-2. Raw contexts only, h = 128 point forecasts, the paper's adapters
(code/model_adapters.py, unchanged) on the GPU server GPUs, as the published anchor forecasts were made.

Statistic, per model and window set: cells are (end date, shift). A cell averages the windows of one family at
that date and shift (equities: 50 windows; exchange rates: 20), or is a single window for each decile series.
Within-date correlation of the cell means of the h = 128 forecast return and the realised return, after removing
each end date's mean over its shifts; permutation p-value shuffling the shifts within each date (20,000
permutations, per-model seed); ratio of the within-date standard deviations (forecast / realised). Also the
timing correlation across end dates at each shift, as in tables/shift.tex.

## Predictions and decision rules (fixed now)

A model "follows the calendar" on a window set when its within-date correlation exceeds 0.3 with permutation
p < 0.01 (on 1996-2014 the four models were at 0.52 to 0.87 with p < 0.001; the controls at +0.06 and -0.29).

S1 Daily anchor, equities (2017-2026). None of the four models follows the calendar. This is the paper's
   current reading of the anchor. A model that fails S1 is reported as possibly remembering the anchor, and the
   text on its raw anchor skill and on the mirror's cost there is qualified accordingly.
S2 Daily anchor, exchange rates (2017-2026). None of the four models follows the calendar. Same consequence.
S3 Size deciles (1963-2026). No prediction. Decision rule: a model that follows the calendar on a decile series is
   named in the small-cap text as possibly remembering that series, and the text no longer reads its skill there
   as the use of short-horizon structure alone; if none does, the text says the check found no sign of it.
Controls: Moirai-1.1 and Chronos-2 are expected not to follow the calendar on any set.
Descriptive only (no decision attached): the within-date correlation split at end dates before 2024 and from 2024.

# Outcomes (added 2026-09-24 after the runs; the text above is unchanged, and its sha256 as written was 6d0d986c…)

Runs: GPU server, 11:31-11:39 CST, experiments/server/run_shift_anchor.sh; the k = 0 forecasts reproduce the published ones
(max |difference| in the h = 128 return 0 to 7e-3 for the deterministic models; Moirai-1.1, sampled, correlation 0.89
and 0.94).

Statistic as written (returns from each shifted context's end, within-date permutation):
S1 met: none of the four models follows the calendar on the anchor's equities (-0.12 to +0.06).
S2 not met: TiRex +0.53 (p 0.009) and TimesFM-2.0 +0.54 (p < 0.001) on the exchange rates (Moirai-2.0 +0.40, p 0.02).
S3: no model flagged on either decile. Controls: neither follows the calendar anywhere.

After the runs a simulation (shift_bias_sim.py) showed the written statistic to be biased by construction: a later
shift's context contains part of the earlier shifts' futures, so trend-opposing forecasts score about +0.35 and
trend-following ones about -0.35 on a random walk. We therefore replaced it, after seeing the results above, by the
forecast change from each shifted context's end against the realised level at the target date (unbiased on a random
walk with constant drift; p-value by a random sign per date), with drift and momentum rules on the same windows as
references. With it: S1 met (largest +0.27, Chronos-Bolt, p 0.16), S2 met (largest +0.09), S3 no model flagged
(largest +0.18), controls never follow. The paper reports the replacement statistic and states the result of the
written one. The same replacement changes the 1996-2014 result (see PREREGISTRATION.md, correction section).
