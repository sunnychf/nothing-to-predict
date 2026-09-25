"""Real-data anchor for the null probe: build forecasting windows from public daily
financial series, plus an exact-martingale surrogate of every window.

Sources (downloaded 2026-09-16, files kept under data/real/, provenance in
data/real/SOURCES.md):
  * ECB euro foreign exchange reference rates, eurofxref-hist.csv: daily EUR/XXX
    rates since 1999. Twenty floating currencies are used.
  * Kenneth R. French data library: 49 industry portfolios (daily, value-weighted
    returns) and the Fama/French daily factors (Mkt-RF + RF = market return).
    Portfolio levels are rebuilt as 100 * prod(1 + r).

Windows: context 512 trading days + horizon 128, non-overlapping futures per
series (stride 128), taken backwards from the last observation: --k 8 keeps the
latest eight (the first anchor), --k 0 keeps every window since --start (the
second, larger anchor), so the futures of the most recent windows postdate every
model's training cut-off either way.

Surrogates (the "real-increment null"): the 639 simple returns of a window are
multiplied by i.i.d. Rademacher signs (seed 5000 + window index for the first
draw, 5000 + window index + 100000 * draw for the others) and the level is
rebuilt multiplicatively; --n-sur K keeps K independent draws per window
(ctx_sur has shape (n, K, T)), which the sign-averaged correction needs. Because the signs are independent of everything,
E[y_{t+1} | F_t] = y_t exactly: the surrogate is a martingale in levels while
keeping the real |return| path (volatility clustering, heavy tails, calendar
gaps). On it persistence is optimal and every departure is imported; on the raw
window the same models are scored against persistence with the real drift left
in, which is the confounded quantity the paper says cannot be read on its own.

Snapshot: the paper's windows were built from the files downloaded on 2026-09-16 (ECB file
ending 2026-09-15, Ken French files ending 2026-07-31; checksums in data/real/CHECKSUMS.sha256).
Both providers extend their files, and windows are counted backwards from the last
observation, so --end-fx / --end-eq cut a later download at the snapshot's last dates and
reproduce the paper's windows exactly unless the provider has revised history. The script
writes <out>_fingerprint.json (sha256 of every array plus summary statistics); --expect
compares a rebuilt file against a published fingerprint and reports exact / float-noise /
mismatch.

Usage:  python code/real_data.py --data data/real --out results/real_windows.npz --k 0 --n-sur 4
        python code/real_data.py --data data/real --out results/real_windows.npz --k 0 --n-sur 4 --expect results/real_windows_fingerprint.json
        python code/real_data.py --data data/real --out results/real_windows_v1.npz --k 8 --n-sur 1     # the first anchor
"""
import argparse, hashlib, io, json, os, zipfile
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/real")
ap.add_argument("--out", default="results/real_windows.npz")
ap.add_argument("--k", type=int, default=8, help="latest non-overlapping windows per series; 0 = all since --start")
ap.add_argument("--n-sur", type=int, default=1, help="independent sign-randomised surrogates per window")
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--h", type=int, default=128)
ap.add_argument("--start", default="2015-01-01", help="ignore observations before this date")
ap.add_argument("--end-fx", default="2026-09-15", help="ignore ECB observations after this date (the paper's snapshot)")
ap.add_argument("--end-eq", default="2026-07-31", help="ignore Ken French observations after this date (the paper's snapshot)")
ap.add_argument("--expect", default=None, help="fingerprint JSON of a published windows file to compare against")
args = ap.parse_args()
CTX, H, L = args.ctx, args.h, args.ctx + args.h

FX = ["USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK", "NOK", "NZD", "MXN",
      "ZAR", "BRL", "KRW", "PLN", "HUF", "CZK", "SGD", "INR", "ILS", "IDR"]      # floating regimes

# ---------------------------------------------------------------- ECB reference rates
fx = pd.read_csv(os.path.join(args.data, "eurofxref-hist.csv"))
fx = fx.loc[:, [c for c in fx.columns if not c.startswith("Unnamed")]]
fx["Date"] = pd.to_datetime(fx["Date"]); fx = fx.sort_values("Date").set_index("Date")
fx = fx.loc[(fx.index >= args.start) & (fx.index <= args.end_fx)]
series = {}
for c in FX:
    s = pd.to_numeric(fx[c], errors="coerce").dropna()
    series[f"fx:EUR{c}"] = s

# ---------------------------------------------------------------- Ken French daily
def read_ff_block(path, header_marker=None):
    """Return the first daily block (value-weighted returns) of a Ken French CSV as a DataFrame."""
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "," in l and len(l.split(",")) > 3)
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    return df

ind = read_ff_block(os.path.join(args.data, "49_Industry_Portfolios_Daily.csv"))
ind = ind.loc[(ind.index >= args.start) & (ind.index <= args.end_eq)]
fac = read_ff_block(os.path.join(args.data, "F-F_Research_Data_Factors_daily.csv"))
fac = fac.loc[(fac.index >= args.start) & (fac.index <= args.end_eq)]
mkt = (fac["Mkt-RF"] + fac["RF"]) / 100.0
series["eq:Mkt"] = 100.0 * (1.0 + mkt).cumprod()
for c in ind.columns:
    r = ind[c]
    if (r <= -99).any():
        continue                                                   # missing data in the sample
    series[f"eq:{c}"] = 100.0 * (1.0 + r / 100.0).cumprod()

# ---------------------------------------------------------------- windows + surrogates
ctx_raw, fut_raw, ctx_sur, fut_sur, sig_raw, sig_sur, rsig_raw, rsig_sur, meta = [], [], [], [], [], [], [], [], []
widx = 0
for name, s in series.items():
    y = s.to_numpy(dtype=np.float64); dates = s.index
    n = len(y)
    n_win = args.k if args.k > 0 else (n - L) // H + 1
    ends = [n - 1 - k * H for k in range(n_win)]
    for e in ends:
        st = e + 1 - L
        if st < 0:
            break
        w = y[st:e + 1]
        r = w[1:] / w[:-1] - 1.0
        cs, fs, ss, rs = [], [], [], []
        for d in range(args.n_sur):
            rng = np.random.default_rng(5000 + widx + 100000 * d)
            eps = rng.choice(np.array([-1.0, 1.0]), size=r.shape)
            ws = np.empty_like(w); ws[0] = w[0]; ws[1:] = w[0] * np.cumprod(1.0 + eps * r)
            cs.append(ws[:CTX]); fs.append(ws[CTX:]); ss.append(np.std(np.diff(ws[:CTX])))
            rs.append(np.std(ws[1:CTX] / ws[:CTX - 1] - 1.0))
        ctx_raw.append(w[:CTX]); fut_raw.append(w[CTX:]); sig_raw.append(np.std(np.diff(w[:CTX])))
        rsig_raw.append(np.std(r[:CTX - 1]))
        ctx_sur.append(np.array(cs)); fut_sur.append(np.array(fs)); sig_sur.append(np.array(ss)); rsig_sur.append(np.array(rs))
        meta.append({"i": widx, "asset": name, "family": name.split(":")[0],
                     "ctx_start": str(dates[st].date()), "ctx_end": str(dates[st + CTX - 1].date()),
                     "fut_end": str(dates[e].date()), "n_obs": int(L)})
        widx += 1

os.makedirs(os.path.dirname(args.out), exist_ok=True)
ctx_sur, fut_sur, sig_sur, rsig_sur = np.array(ctx_sur, np.float64), np.array(fut_sur, np.float64), np.array(sig_sur), np.array(rsig_sur)
if args.n_sur == 1:                       # keep the first anchor's (n, T) layout
    ctx_sur, fut_sur, sig_sur, rsig_sur = ctx_sur[:, 0], fut_sur[:, 0], sig_sur[:, 0], rsig_sur[:, 0]
# sigma_* = std of the context's level increments (the first anchor's unit); rsigma_* = std of the
# context's simple returns, the unit in which the multiplicative surrogate is exact and the
# sign-averaged correction is applied
arrays = dict(ctx_raw=np.array(ctx_raw, np.float64), fut_raw=np.array(fut_raw, np.float64),
              ctx_sur=ctx_sur, fut_sur=fut_sur, sigma_raw=np.array(sig_raw), sigma_sur=sig_sur,
              rsigma_raw=np.array(rsig_raw), rsigma_sur=rsig_sur)
np.savez_compressed(args.out, **arrays)
json.dump(meta, open(args.out.replace(".npz", "_meta.json"), "w"), indent=1)

# fingerprint: exact hash of every array's bytes (C order, float64) and a few statistics that
# survive last-bit differences between CSV parsers or BLAS builds; --expect grades a rebuild
fp = {"n_windows": len(meta), "families": {f: sum(1 for m in meta if m["family"] == f) for f in ("fx", "eq")},
      "fut_end": {f: [min(m["fut_end"] for m in meta if m["family"] == f), max(m["fut_end"] for m in meta if m["family"] == f)] for f in ("fx", "eq")},
      "arrays": {k: {"shape": list(a.shape), "sha256": hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest(),
                     "mean": float(a.mean()), "std": float(a.std()), "first": float(a.ravel()[0]), "last": float(a.ravel()[-1])}
                 for k, a in arrays.items()}}
json.dump(fp, open(args.out.replace(".npz", "_fingerprint.json"), "w"), indent=1)
if args.expect:
    ref = json.load(open(args.expect)); bad, noise = [], []
    if ref["n_windows"] != fp["n_windows"] or ref["fut_end"] != fp["fut_end"]:
        verdict = "MISMATCH (different windows: check --start/--end-fx/--end-eq and the files' last dates)"
    else:
        for k, r in ref["arrays"].items():
            a = fp["arrays"][k]
            if r["sha256"] == a["sha256"]:
                continue
            close = all(abs(r[q] - a[q]) <= 1e-9 * max(1.0, abs(r[q])) for q in ("mean", "std", "first", "last"))
            (noise if close else bad).append(k)
        verdict = (f"MISMATCH in {bad} (the provider has revised the data)" if bad else
                   f"float-noise in {noise} (same windows, last-bit differences)" if noise else "exact")
    print(f"fingerprint vs {args.expect}: {verdict}")
