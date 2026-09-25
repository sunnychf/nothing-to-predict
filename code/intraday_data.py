"""Build the intraday anchor: windows of consecutive transaction prices with a real bid-ask
bounce, the microstructure of rung N4 as it occurs, from the executions extracted by
itch_trades.py (Nasdaq TotalView-ITCH, one trading day).

    python intraday_data.py --trades data/intraday/itch_trades_2019-12-30.npz --out results/intraday_windows.npz

Per symbol, fills that share a timestamp are one transaction (a marketable order matched
against several resting orders prints one trade at the last fill's price; the resting side of
that fill gives the trade direction q = -resting side: a resting bid hit is a sale at the bid,
q = -1; a resting ask lifted is a purchase at the ask, q = +1). Windows are consecutive,
non-overlapping blocks of CTX + H transactions in event time within the regular session, for
every symbol with at least one block. The same K multiplicative sign-randomised copies as the
daily anchor (real_data.py) are drawn per window (seed 9000 + window index + 100000 * copy), so
real_probe.py runs on this file unchanged (--prefix intraday); q is saved for the side-aware
benchmark of intraday_summary.py, which the models never see.
"""
import argparse, hashlib, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trades", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--n-sur", type=int, default=4)
ap.add_argument("--min-price", type=float, default=1.0, help="drop symbols whose day-low is below this (sub-dollar tick regime)")
ap.add_argument("--top", type=int, default=40, help="keep the symbols with the most transactions in the session")
ap.add_argument("--max-windows", type=int, default=25, help="windows per symbol, evenly spaced over the session's blocks")
ap.add_argument("--expect", default=None, help="fingerprint JSON of the paper's windows to compare against")
ap.add_argument("--clock", type=float, default=0.0, help="sampling clock in seconds: the last transaction price of every clock interval with a trade (0 = every transaction)")
args = ap.parse_args()
CTX, H, L = args.ctx, args.horizon, args.ctx + args.horizon

z = np.load(args.trades, allow_pickle=False)
symbols = [str(s) for s in z["symbols"]]
ctx_raw, fut_raw, q_ctx, q_fut, ctx_sur, fut_sur, sig_raw, sig_sur, rsig_raw, rsig_sur, meta = [], [], [], [], [], [], [], [], [], [], []
widx = 0; per_sym = {}; series = {}
for sym in symbols:
    t, p, side = z[f"{sym}/t"], z[f"{sym}/p"].astype(np.float64) / 1e4, z[f"{sym}/side"].astype(np.int64)
    if len(t) < L or p.min() < args.min_price:
        continue
    keep = np.r_[t[1:] != t[:-1], True]                   # one transaction per timestamp: the last fill of each
    t, p, q = t[keep], p[keep], -side[keep]               # q = -resting side = trade direction
    if args.clock > 0:                                    # last transaction of every clock interval that has one
        b = t // int(args.clock * 1e9); keep = np.r_[b[1:] != b[:-1], True]
        t, p, q = t[keep], p[keep], q[keep]
    if len(t) < L: continue
    series[sym] = (t, p, q)
ranked = sorted(series, key=lambda s: -len(series[s][0]))[:args.top]
for sym in sorted(ranked):
    t, p, q = series[sym]
    n_blocks = len(p) // L
    blocks = sorted(set(np.linspace(0, n_blocks - 1, min(args.max_windows, n_blocks)).round().astype(int).tolist()))
    per_sym[sym] = {"transactions": int(len(p)), "blocks": int(n_blocks), "windows": int(len(blocks)), "price_low": float(p.min()), "price_high": float(p.max()),
                    "lag1_autocorr_returns": float(np.corrcoef(np.diff(np.log(p))[:-1], np.diff(np.log(p))[1:])[0, 1])}
    for k in blocks:
        w = p[k * L:(k + 1) * L]; qw = q[k * L:(k + 1) * L]
        r = w[1:] / w[:-1] - 1.0
        cs, fs, ss, rs = [], [], [], []
        for d in range(args.n_sur):
            rng = np.random.default_rng(9000 + widx + 100000 * d)
            eps = rng.choice(np.array([-1.0, 1.0]), size=r.shape)
            ws = np.empty_like(w); ws[0] = w[0]; ws[1:] = w[0] * np.cumprod(1.0 + eps * r)
            cs.append(ws[:CTX]); fs.append(ws[CTX:]); ss.append(np.std(np.diff(ws[:CTX]))); rs.append(np.std(ws[1:CTX] / ws[:CTX - 1] - 1.0))
        ctx_raw.append(w[:CTX]); fut_raw.append(w[CTX:]); q_ctx.append(qw[:CTX]); q_fut.append(qw[CTX:])
        sig_raw.append(np.std(np.diff(w[:CTX]))); rsig_raw.append(np.std(r[:CTX - 1]))
        ctx_sur.append(np.array(cs)); fut_sur.append(np.array(fs)); sig_sur.append(np.array(ss)); rsig_sur.append(np.array(rs))
        meta.append({"i": widx, "asset": sym, "family": "intraday", "t_start_ns": int(t[k * L]), "t_end_ns": int(t[(k + 1) * L - 1]), "n_obs": int(L)})
        widx += 1

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
arrays = dict(ctx_raw=np.array(ctx_raw, np.float64), fut_raw=np.array(fut_raw, np.float64),
              ctx_sur=np.array(ctx_sur, np.float64), fut_sur=np.array(fut_sur, np.float64),
              sigma_raw=np.array(sig_raw), sigma_sur=np.array(sig_sur), rsigma_raw=np.array(rsig_raw), rsigma_sur=np.array(rsig_sur),
              q_ctx=np.array(q_ctx, np.int8), q_fut=np.array(q_fut, np.int8))
np.savez_compressed(args.out, **arrays)
json.dump({"windows": meta, "symbols": per_sym, "ctx": CTX, "horizon": H, "n_sur": args.n_sur, "top": args.top, "max_windows": args.max_windows,
           "min_price": args.min_price, "clock_seconds": args.clock, "source": os.path.basename(args.trades)},
          open(args.out.replace(".npz", "_meta.json"), "w"), indent=1)
fp = {"n_windows": int(widx), "symbols": sorted(per_sym),
      "arrays": {k: {"shape": list(a.shape), "sha256": hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest(),
                     "mean": float(a.mean()), "std": float(a.std()), "first": float(a.ravel()[0]), "last": float(a.ravel()[-1])}
                 for k, a in arrays.items()}}
json.dump(fp, open(args.out.replace(".npz", "_fingerprint.json"), "w"), indent=1)
if args.expect:
    ref = json.load(open(args.expect)); bad, noise = [], []
    if ref["n_windows"] != fp["n_windows"] or ref["symbols"] != fp["symbols"]:
        verdict = "MISMATCH (different windows: check the ITCH file and the selection options)"
    else:
        for k, r in ref["arrays"].items():
            a = fp["arrays"][k]
            if r["sha256"] == a["sha256"]: continue
            close = all(abs(r[qq] - a[qq]) <= 1e-9 * max(1.0, abs(r[qq])) for qq in ("mean", "std", "first", "last"))
            (noise if close else bad).append(k)
        verdict = f"MISMATCH in {bad}" if bad else f"float-noise in {noise}" if noise else "exact"
    print(f"fingerprint vs {args.expect}: {verdict}")
print(f"{widx} windows from {len(per_sym)} symbols (" + ", ".join(f"{s}:{v['windows']}" for s, v in per_sym.items()) + f"); wrote {args.out}")
