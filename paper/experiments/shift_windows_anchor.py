"""Post hoc check, specified before the runs in experiments/PREREGISTRATION_shift_anchors.md: calendar-shifted windows
for the paper's daily anchor (2017-2026; ECB exchange rates and Ken French equities, code/real_data.py) and for the
size-decile anchor (1963-2026; code/smallcap_data.py). Same construction as experiments/shift_windows.py: the context end
of every raw window is moved by k in (-40, -20, +20, +40) trading days of its own series, keeping the 512-day context
and the 128-day future; windows that would leave the series' sample are skipped. The series are rebuilt exactly as the
two scripts build them (their default arguments, i.e. the paper's snapshot), and every k = 0 window is checked to equal
the published window bit for bit. k = 0 is also written, so that the run can be compared with the published forecasts.
Writes experiments/results/shift_anchor_windows.npz and shift_size_windows.npz (ctx_raw, fut_raw, and ctx_sur = ctx_raw
so that the K = 1 file format of experiments/server/shard_probe.py holds) with _meta.json.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/shift_windows_anchor.py
"""
import io, json, os
import numpy as np, pandas as pd

DATA, RES, OUT = os.path.join("..", "data", "real"), os.path.join("..", "results"), os.path.join("experiments", "results")
CTX, H = 512, 128
L = CTX + H; SHIFTS = (-40, -20, 0, 20, 40)
FX = ["USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK", "NOK", "NZD", "MXN",
      "ZAR", "BRL", "KRW", "PLN", "HUF", "CZK", "SGD", "INR", "ILS", "IDR"]


def read_ff_block(path):                      # as in code/real_data.py
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "," in l and len(l.split(",")) > 3)
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
    return df


def anchor_series(start="2015-01-01", end_fx="2026-09-15", end_eq="2026-07-31"):     # code/real_data.py defaults
    fx = pd.read_csv(os.path.join(DATA, "eurofxref-hist.csv"))
    fx = fx.loc[:, [c for c in fx.columns if not c.startswith("Unnamed")]]
    fx["Date"] = pd.to_datetime(fx["Date"]); fx = fx.sort_values("Date").set_index("Date")
    fx = fx.loc[(fx.index >= start) & (fx.index <= end_fx)]
    series = {f"fx:EUR{c}": pd.to_numeric(fx[c], errors="coerce").dropna() for c in FX}
    ind = read_ff_block(os.path.join(DATA, "49_Industry_Portfolios_Daily.csv")); ind = ind.loc[(ind.index >= start) & (ind.index <= end_eq)]
    fac = read_ff_block(os.path.join(DATA, "F-F_Research_Data_Factors_daily.csv")); fac = fac.loc[(fac.index >= start) & (fac.index <= end_eq)]
    series["eq:Mkt"] = 100.0 * (1.0 + (fac["Mkt-RF"] + fac["RF"]) / 100.0).cumprod()
    for c in ind.columns:
        if (ind[c] <= -99).any():
            continue
        series[f"eq:{c}"] = 100.0 * (1.0 + ind[c] / 100.0).cumprod()
    return series


def size_series(start="1963-07-01", end="2026-07-31", spec="ew:Lo 10,vw:Hi 10"):          # code/smallcap_data.py defaults
    lines = open(os.path.join(DATA, "Portfolios_Formed_on_ME_Daily.csv"), encoding="latin-1").read().splitlines()

    def block(title):
        i = next(k for k, l in enumerate(lines) if title in l); j = i + 1
        while not lines[j].startswith(","):
            j += 1
        end_ = j + 1
        while end_ < len(lines) and lines[end_].strip() and lines[end_].split(",")[0].strip().isdigit():
            end_ += 1
        df = pd.read_csv(io.StringIO("\n".join(lines[j:end_])), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
        return df.replace([-99.99, -999.0], np.nan)
    blocks = {"vw": block("Average Value Weighted Returns -- Daily"), "ew": block("Average Equal Weighted Returns -- Daily")}
    series = {}
    for s in spec.split(","):
        w, col = s.split(":"); r = blocks[w][col].dropna() / 100.0
        r = r.loc[(r.index >= start) & (r.index <= end)]
        series[f"{w}:{col.replace(' ', '')}"] = 100.0 * (1.0 + r).cumprod()
    return series


def build(tag, series, base_npz, base_meta):
    B = np.load(base_npz); bm = json.load(open(base_meta))
    ctx, fut, meta, n_exact = [], [], [], 0
    for b in bm:
        s = series[b["asset"]]; y = s.to_numpy(dtype=np.float64); dates = s.index
        e0 = int(np.where(dates == pd.Timestamp(b["fut_end"]))[0][0])
        w0 = y[e0 + 1 - L:e0 + 1]
        assert np.array_equal(w0[:CTX], B["ctx_raw"][b["i"]]) and np.array_equal(w0[CTX:], B["fut_raw"][b["i"]]), (tag, b)
        n_exact += 1
        for k in SHIFTS:
            e = e0 + k; st = e + 1 - L
            if st < 0 or e >= len(y):
                continue
            w = y[st:e + 1]
            ctx.append(w[:CTX]); fut.append(w[CTX:])
            meta.append({"base_i": b["i"], "asset": b["asset"], "family": b["family"], "base_fut_end": b["fut_end"], "shift": k,
                         "ctx_end": str(dates[st + CTX - 1].date()), "fut_end": str(dates[e].date())})
    ctx, fut = np.array(ctx), np.array(fut)
    path = os.path.join(OUT, f"shift_{tag}_windows.npz")
    np.savez_compressed(path, ctx_raw=ctx, fut_raw=fut, ctx_sur=ctx, fut_sur=fut)
    json.dump(meta, open(path.replace(".npz", "_meta.json"), "w"), indent=1)
    fams = sorted({m["family"] for m in meta})
    print(f"{tag}: {len(bm)} base windows, all {n_exact} rebuilt bit for bit; {len(meta)} windows written; per shift "
          + str({k: sum(m['shift'] == k for m in meta) for k in SHIFTS}) + "; per family " + str({f: sum(m['family'] == f for m in meta) for f in fams}))


build("anchor", anchor_series(), os.path.join(RES, "real_windows.npz"), os.path.join(RES, "real_windows_meta.json"))
build("size", size_series(), os.path.join(RES, "smallcap_windows.npz"), os.path.join(RES, "smallcap_windows_meta.json"))
