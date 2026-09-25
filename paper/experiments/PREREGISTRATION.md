# Pre-registered predictions for the tier-3 experiments (written 2026-09-23, before any model was run)

## Experiment 1: falling markets (windows: experiments/results/falling_windows.npz)

1850 raw windows of 50 Ken French equity series, futures ending 1996-09-10 .. 2014-12-31 (37 end dates),
one sign-randomised copy each; every model forecasts each raw window, its copy, and the multiplicative
mirror of the raw window (the same three passes as the paper's daily anchor, code/real_probe.py).
A window is "down" if the market's compounded return over its future dates is negative (8 end dates,
400 windows) and "up" otherwise (29 end dates, 1450 windows); the two S&P 500 bear markets of the
period (2000-03-24 .. 2002-10-09, 2007-10-09 .. 2009-03-09) give a second, calendar-based split.

Groups, fixed from the paper's N1 results before this run:
  upward-biased: Chronos-T5, Chronos-Bolt, Chronos-2, TiRex, Moirai-1.1
  near balance:  Moirai-2.0, TimesFM-2.0, TimesFM-2.5, Time-MoE, Sundial (FinCast reported separately)

Predictions (skill at h = 128 in return units, pooled squared errors, as in the paper):
  P1  For each upward-biased model, raw skill is lower on down windows than on up windows.
  P2  For each upward-biased model, the mirror correction lowers skill on up windows and raises it on
      down windows (the alignment term of Proposition 2(v) changes sign with the market).
  P3  The mean up-minus-down gap in raw skill is larger for the upward-biased group than for the
      near-balance group.
  P4  On the sign-randomised copies every model loses to the last value in both regimes.
Analysis: skill and paired mirror changes with a cluster bootstrap over end dates; the decomposition
of Proposition 2(v) per regime. A prediction counts as met for a model when the point estimates go
the stated way; we report the standard errors and do not claim significance beyond them.
If P1-P2 fail for most upward-biased models, the paper's reading of the raw-equity losses as drift
alignment is weakened and will be revised.

## Experiment 2: training-time symmetrisation of a released model

Chronos-Bolt-small is fine-tuned from its released weights on the same data twice: plainly, and with
mirror augmentation (each training window replaced by its increment-negated mirror with probability
1/2). Predictions: the augmented model's upward fraction on N1 at h = 128 is closer to 0.5 than both
the released and the plainly fine-tuned model's; its ETTh1 skill at h = 16 is within 0.02 of the
plainly fine-tuned model's; its raw-equity skill at h = 128 on the paper's daily windows is lower than
the plainly fine-tuned model's, as the inference-time correction's was.

## Experiment 3: trend statistics of public pretraining corpora

For windows of 512 + 128 steps sampled from public pretraining corpora, the fraction whose future
128-step change (in units of the context's step standard deviation) is positive, and its mean.
No directional prediction is registered for the released-model split; the statistic is descriptive.

## Experiment 4: larger controlled corpus experiment

The masked-encoder and decoder-only designs of the paper, scaled up, with more seeds, trained on the
growth and symmetric corpora. Prediction: the growth corpus produces an upward N1 bias in both designs
at every size tested, as at 0.54M parameters.

---

# Outcomes (added 2026-09-24 after the runs; the text above is unchanged, sha256 of the registered version b092cb32…)

## Experiment 1 (falling_summary.py; h = 128, point estimates as registered)
P1 met for all five upward-biased models. P2 met for Chronos-T5 (mirror change -0.131 up, +0.064 down) and
Moirai-1.1 (-0.110 up, +0.208 down); not met for Chronos-Bolt, Chronos-2 and TiRex. P3 met (mean up-minus-down gap
+0.24 for the upward-biased group, -0.33 for the others). P4 met for all eleven models. By the registered rule the
drift reading is confirmed for Chronos-T5 and Moirai-1.1 and not for the other three.
Post hoc (not registered): Chronos-2's even part is downward on these windows (-1.2 and -3.8 %/yr), so part (v)
predicts the observed direction. Chronos-Bolt, TiRex, Moirai-2.0 and TimesFM-2.0 time the 1996-2014 market
(timing_check.py: r = 0.71 to 0.87 across the 37 end dates, from raw contexts; Bolt, TiRex and Moirai-2.0 not from
mirrored or sign-randomised contexts) and their forecasts follow the calendar when contexts are shifted by 20 or 40
trading days (shift_check.py: within-date r = 0.52 to 0.87; Moirai-1.1 +0.06, Chronos-2 -0.29), which indicates
that they saw these years in pretraining. On remembered windows the even part carries half the remembered move,
which is how P2 failed for Bolt and TiRex.

## Experiment 2 (finetune_summary.py; three seeds)
All three predictions met in every seed pair: N1 upward fraction at h = 128 0.79 released, 0.71 plain, 0.58 mirror;
ETTh1 skill at h = 16 0.697 plain vs 0.708 mirror; raw-equity skill at h = 128 0.126 plain vs 0.064 mirror.

## Experiment 3 (corpus_summary.py; descriptive)
At h = 128 the median subset's rising share is 0.50 to 0.51; pooled over series 0.52 (Time-300B) to 0.57
(Chronos datasets, LOTSA), almost all from the M4 subsets; without M4 0.49 to 0.50.

## Experiment 4 (arch_scale_summary.py; d = 256, six layers, 3.18M parameters, three seeds)
Prediction met in every seed: on the growth corpus the mean N1 departure at h = 128 is +1.39 (encoder) and +1.93
(decoder) sigma, per seed between +1.23 and +2.38, with 80 to 88 percent of forecasts upward. On the symmetric corpus
the encoder stays near balance (+0.16) and the decoder is seed-dependent (+0.19 to +1.30), as at 0.54M parameters.

## Correction to the post hoc calendar-shift statistic (added 2026-09-24 afternoon)
The within-date correlation of returns quoted above (0.52 to 0.87) is biased by construction: a later shift's
context contains part of the earlier shifts' futures, so on a random walk a forecast that opposes the last 20 days'
move scores about +0.35 and one that follows it about -0.35 (shift_bias_sim.py). shift_check.py now correlates the
forecast change from each shifted context's end with the realised level at the target date (unbiased on a random walk
with constant drift) and reports drift and momentum rules on the same windows. On 1996-2014: TimesFM-2.0 +0.58
(p < 0.001), Chronos-Bolt +0.41, Moirai-2.0 +0.33, TiRex +0.29 (p 0.04 to 0.11), Moirai-1.1 -0.05, Chronos-2 +0.11;
the best context rule reaches 0.29 in absolute value (250-day momentum, reversed). The timing check is unaffected
(context rules reach at most 0.26 there). The memorisation reading therefore rests on the timing check for all four
models and on the calendar-shift check clearly for TimesFM-2.0 only. The return version's numbers are kept in
shift_check.json ("returns_biased").
