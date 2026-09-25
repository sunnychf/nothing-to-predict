"""A fourth anchor with known structure: daily size-decile portfolios from the Kenneth R. French data
library (Portfolios_Formed_on_ME_Daily.csv, downloaded 2026-09-19). The equal-weighted smallest
decile has a documented positive first-order autocorrelation of daily returns (nonsynchronous
trading of its micro-caps; Lo and MacKinlay 1988), about +0.15 since 1990 in this file, and the
value-weighted largest decile a negative one (about -0.10 since 2000): real series on which the
optimal linear forecast is not persistence, so that the probe's positive control can be repeated
on a market rather than on the synthetic N4.

Windows, surrogates and the output layout are exactly those of real_data.py (context 512, horizon
128, non-overlapping futures at stride 128 counted back from the last observation, K sign-
randomised surrogates per window with the same seed rule), so real_probe.py --prefix smallcap runs
every model on them unchanged. Levels are rebuilt as 100 * prod(1 + r) over the whole history.

    python code/smallcap_data.py --data data/real --out results/smallcap_windows.npz --n-sur 4
"""
import argparse, hashlib, io, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/real")
ap.add_argument("--out", default="results/smallcap_windows.npz")
ap.add_argument("--n-sur", type=int, default=4)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--h", type=int, default=128)
ap.add_argument("--start", default="1963-07-01", help="the CRSP daily era")
ap.add_argument("--end", default="2026-07-31", help="the snapshot's last date")
ap.add_argument("--series", default="ew:Lo 10,vw:Hi 10", help="weighting:column pairs from the file")
args = ap.parse_args()
CTX, H, L = args.ctx, args.h, args.ctx + args.h

lines = open(os.path.join(args.data, "Portfolios_Formed_on_ME_Daily.csv"), encoding="latin-1").read().splitlines()
def block(title):
    i = next(k for k, l in enumerate(lines) if title in l); j = i + 1
    while not lines[j].startswith(","): j += 1
    end = j + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit(): end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[j:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
    return df.replace([-99.99, -999.0], np.nan)
blocks = {"vw": block("Average Value Weighted Returns -- Daily"), "ew": block("Average Equal Weighted Returns -- Daily")}
series = {}
for spec in args.series.split(","):
    w, col = spec.split(":"); r = blocks[w][col].dropna() / 100.0
    r = r.loc[(r.index >= args.start) & (r.index <= args.end)]
    series[f"{w}:{col.replace(' ', '')}"] = 100.0 * (1.0 + r).cumprod()

ctx_raw, fut_raw, ctx_sur, fut_sur, sig_raw, sig_sur, rsig_raw, rsig_sur, meta = [], [], [], [], [], [], [], [], []
widx = 0
for name, s in series.items():
    y = s.to_numpy(dtype=np.float64); dates = s.index; n = len(y)
    n_win = (n - L) // H + 1; ends = [n - 1 - k * H for k in range(n_win)]
    for e in ends:
        st = e + 1 - L
        if st < 0: break
        w = y[st:e + 1]; r = w[1:] / w[:-1] - 1.0
        cs, fs, ss, rs = [], [], [], []
        for d in range(args.n_sur):
            rng = np.random.default_rng(5000 + widx + 100000 * d)
            eps = rng.choice(np.array([-1.0, 1.0]), size=r.shape)
            ws = np.empty_like(w); ws[0] = w[0]; ws[1:] = w[0] * np.cumprod(1.0 + eps * r)
            cs.append(ws[:CTX]); fs.append(ws[CTX:]); ss.append(np.std(np.diff(ws[:CTX]))); rs.append(np.std(ws[1:CTX] / ws[:CTX - 1] - 1.0))
        ctx_raw.append(w[:CTX]); fut_raw.append(w[CTX:]); sig_raw.append(np.std(np.diff(w[:CTX]))); rsig_raw.append(np.std(r[:CTX - 1]))
        ctx_sur.append(np.array(cs)); fut_sur.append(np.array(fs)); sig_sur.append(np.array(ss)); rsig_sur.append(np.array(rs))
        rho1 = float(pd.Series(r[:CTX - 1]).autocorr(1))
        meta.append({"i": widx, "asset": name, "family": name.split(":")[0], "ctx_start": str(dates[st].date()), "ctx_end": str(dates[st + CTX - 1].date()),
                     "fut_end": str(dates[e].date()), "n_obs": int(L), "rho1_ctx": rho1})
        widx += 1
os.makedirs(os.path.dirname(args.out), exist_ok=True)
arrays = dict(ctx_raw=np.array(ctx_raw), fut_raw=np.array(fut_raw), ctx_sur=np.array(ctx_sur), fut_sur=np.array(fut_sur),
              sigma_raw=np.array(sig_raw), sigma_sur=np.array(sig_sur), rsigma_raw=np.array(rsig_raw), rsigma_sur=np.array(rsig_sur))
np.savez_compressed(args.out, **arrays)
json.dump(meta, open(args.out.replace(".npz", "_meta.json"), "w"), indent=1)
fp = {k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest() for k, v in arrays.items()}
fp["n_windows"] = len(meta); fp["per_series"] = {nm: sum(1 for m in meta if m["asset"] == nm) for nm in series}
fp["rho1_ctx_mean"] = {nm: float(np.mean([m["rho1_ctx"] for m in meta if m["asset"] == nm])) for nm in series}
json.dump(fp, open(args.out.replace(".npz", "_fingerprint.json"), "w"), indent=1)
print(f"{len(meta)} windows: " + ", ".join(f"{nm} {fp['per_series'][nm]} (mean context rho1 {fp['rho1_ctx_mean'][nm]:+.3f}, {series[nm].index.min().date()} to {series[nm].index.max().date()})" for nm in series))
