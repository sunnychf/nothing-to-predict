"""Post hoc check (not registered) of the timing found in the falling-market test: calendar-shifted windows.
For every (series, end date) window of experiments/falling_windows.py, the context end is moved by k trading
days, k in (-40, -20, +20, +40), keeping the 512-day context and the 128-day future; windows that would leave
1994-01-01 .. 2014-12-31 are skipped. A forecast that recalls the series should follow the calendar, so its
error should not grow with |k|; a forecast that reacts to the context should not track the realised future any
better at k = 0 than at the other shifts. Only raw contexts are used.
Writes experiments/results/shift_windows.npz (ctx_raw, fut_raw, and ctx_sur = ctx_raw so that the model runner's
K = 1 file format holds) and shift_windows_meta.json.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/shift_windows.py
"""
import io, json, os
import numpy as np, pandas as pd

DATA = os.path.join("..", "data", "real"); OUT = os.path.join("experiments", "results", "shift_windows.npz")
START, END, CTX, H = "1994-01-01", "2014-12-31", 512, 128
L = CTX + H; SHIFTS = (-40, -20, 20, 40)


def read_ff_block(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "," in l and len(l.split(",")) > 3)
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
    return df


ind = read_ff_block(os.path.join(DATA, "49_Industry_Portfolios_Daily.csv")); ind = ind.loc[(ind.index >= START) & (ind.index <= END)]
fac = read_ff_block(os.path.join(DATA, "F-F_Research_Data_Factors_daily.csv")); fac = fac.loc[(fac.index >= START) & (fac.index <= END)]
series = {"eq:Mkt": 100.0 * (1.0 + (fac["Mkt-RF"] + fac["RF"]) / 100.0).cumprod()}
for c in ind.columns:
    if (ind[c] <= -99).any():
        continue
    series[f"eq:{c}"] = 100.0 * (1.0 + ind[c] / 100.0).cumprod()
base = json.load(open(os.path.join("experiments", "results", "falling_windows_meta.json")))
ctx, fut, meta = [], [], []
for b in base:
    s = series[b["asset"]]; y = s.to_numpy(dtype=np.float64); dates = s.index
    e0 = int(np.where(dates == pd.Timestamp(b["fut_end"]))[0][0])        # last future index of the base window
    for k in SHIFTS:
        e = e0 + k; st = e + 1 - L
        if st < 0 or e >= len(y):
            continue
        w = y[st:e + 1]
        ctx.append(w[:CTX]); fut.append(w[CTX:])
        meta.append({"base_i": b["i"], "asset": b["asset"], "base_fut_end": b["fut_end"], "shift": k,
                     "ctx_end": str(dates[st + CTX - 1].date()), "fut_end": str(dates[e].date())})
ctx, fut = np.array(ctx), np.array(fut)
np.savez_compressed(OUT, ctx_raw=ctx, fut_raw=fut, ctx_sur=ctx, fut_sur=fut)
json.dump(meta, open(OUT.replace(".npz", "_meta.json"), "w"), indent=1)
print(f"{len(meta)} shifted windows from {len(base)} base windows; per shift:", {k: sum(m['shift'] == k for m in meta) for k in SHIFTS})
