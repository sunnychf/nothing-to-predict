"""Windows for the falling-market test (tier 3, experiment 1): the same construction as
code/real_data.py (context 512 + future 128 trading days, stride 128 counted back from the last
observation, one multiplicative sign-randomised copy per window with the same seeding rule), but on
the Ken French equity series only and on an earlier period, observations 1994-01-01 .. 2014-12-31,
so every future ends before the paper's daily anchor begins and before 2015.

Each window also records the market's (Mkt-RF + RF) compounded return over the window's future
dates, mkt_fut_ret, and whether the future overlaps one of the two bear markets of the period as
dated by the S&P 500 peak and trough (2000-03-24 .. 2002-10-09 and 2007-10-09 .. 2009-03-09).
Prediction registered before the models are run (see experiments/PREREGISTRATION.md): for the
models whose null forecasts lean upward, raw skill is lower and the mirror correction more helpful
in windows whose market future falls than in those whose market future rises; the balanced
decoder-only models show a smaller gap.

Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/falling_windows.py
"""
import io, json, os
import numpy as np, pandas as pd

DATA = os.path.join("..", "data", "real"); OUT = os.path.join("experiments", "results", "falling_windows.npz")
START, END, CTX, H = "1994-01-01", "2014-12-31", 512, 128
L = CTX + H
BEARS = [("2000-03-24", "2002-10-09"), ("2007-10-09", "2009-03-09")]


def read_ff_block(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "," in l and len(l.split(",")) > 3)
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
    return df


ind = read_ff_block(os.path.join(DATA, "49_Industry_Portfolios_Daily.csv"))
ind = ind.loc[(ind.index >= START) & (ind.index <= END)]
fac = read_ff_block(os.path.join(DATA, "F-F_Research_Data_Factors_daily.csv"))
fac = fac.loc[(fac.index >= START) & (fac.index <= END)]
mkt_r = (fac["Mkt-RF"] + fac["RF"]) / 100.0
series = {"eq:Mkt": 100.0 * (1.0 + mkt_r).cumprod()}
for c in ind.columns:
    r = ind[c]
    if (r <= -99).any():
        continue
    series[f"eq:{c}"] = 100.0 * (1.0 + r / 100.0).cumprod()
mkt_level = series["eq:Mkt"]

ctx_raw, fut_raw, ctx_sur, fut_sur, sig_raw, sig_sur, rsig_raw, rsig_sur, meta = [], [], [], [], [], [], [], [], []
widx = 0
for name, s in series.items():
    y = s.to_numpy(dtype=np.float64); dates = s.index; n = len(y)
    for e in [n - 1 - k * H for k in range((n - L) // H + 1)]:
        st = e + 1 - L
        if st < 0:
            break
        w = y[st:e + 1]; r = w[1:] / w[:-1] - 1.0
        rng = np.random.default_rng(5000 + widx)
        eps = rng.choice(np.array([-1.0, 1.0]), size=r.shape)
        ws = np.empty_like(w); ws[0] = w[0]; ws[1:] = w[0] * np.cumprod(1.0 + eps * r)
        ctx_raw.append(w[:CTX]); fut_raw.append(w[CTX:]); sig_raw.append(np.std(np.diff(w[:CTX]))); rsig_raw.append(np.std(r[:CTX - 1]))
        ctx_sur.append(ws[:CTX]); fut_sur.append(ws[CTX:]); sig_sur.append(np.std(np.diff(ws[:CTX]))); rsig_sur.append(np.std(ws[1:CTX] / ws[:CTX - 1] - 1.0))
        d0, d1 = dates[st + CTX - 1], dates[e]
        m0, m1 = float(mkt_level.loc[d0]), float(mkt_level.loc[d1])
        bear = any(not (d1 < pd.Timestamp(a) or d0 > pd.Timestamp(b)) for a, b in BEARS)
        meta.append({"i": widx, "asset": name, "family": "eq", "ctx_start": str(dates[st].date()), "ctx_end": str(d0.date()),
                     "fut_end": str(d1.date()), "mkt_fut_ret": m1 / m0 - 1.0, "in_bear": bool(bear), "n_obs": int(L)})
        widx += 1

arrays = dict(ctx_raw=np.array(ctx_raw), fut_raw=np.array(fut_raw), ctx_sur=np.array(ctx_sur), fut_sur=np.array(fut_sur),
              sigma_raw=np.array(sig_raw), sigma_sur=np.array(sig_sur), rsigma_raw=np.array(rsig_raw), rsigma_sur=np.array(rsig_sur))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **arrays); json.dump(meta, open(OUT.replace(".npz", "_meta.json"), "w"), indent=1)
ends = sorted({m["fut_end"] for m in meta}); down = [d for d in ends if next(m for m in meta if m["fut_end"] == d)["mkt_fut_ret"] < 0]
print(f"{len(meta)} windows from {len(series)} series; {len(ends)} future-end dates, first {ends[0]}, last {ends[-1]}")
print(f"market future falls in {len(down)} of {len(ends)} end dates: {down}")
print(f"windows overlapping a bear market: {sum(m['in_bear'] for m in meta)}")
