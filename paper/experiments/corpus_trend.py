"""Experiment 3: trend statistics of public pretraining corpora (descriptive; no directional prediction registered).
For each subset, up to --max-series series and up to 4 random end points per series: with a context of up to 512
observations (at least 64) before the end point and a future of h steps, z = (x_{t+h} - x_t) / sd(diff(context)).
Reports, per subset, the fraction of windows with z > 0 and the mean and median of z, for h = 16 and 128; per corpus,
the median over subsets and the series-weighted pool.
Formats: chronos (autogluon/chronos_datasets parquet), lotsa (Salesforce/lotsa_data arrow), time300b (Maple728 .bin + meta.json).
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.expanduser("~/tsfm_rev/data"))
ap.add_argument("--max-series", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.expanduser("~/tsfm_rev/results/corpus_trend.json")); args = ap.parse_args()
HS = (16, 128); CTX, MINCTX = 512, 64
rng = np.random.default_rng(args.seed)


def stats(series):
    zs = {h: [] for h in HS}
    for s in series:
        s = np.asarray(s, dtype=np.float64); s = s[np.isfinite(s)]
        for h in HS:
            if len(s) < MINCTX + h: continue
            for _ in range(4):
                e = int(rng.integers(MINCTX, len(s) - h + 1)); c = s[max(0, e - CTX):e]
                sd = np.std(np.diff(c))
                if sd > 0: zs[h].append((s[e + h - 1] - s[e - 1]) / sd)
    def summ(v):
        a = np.array(v); q = np.quantile(a, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        up, dn = float(np.mean(a > 0)), float(np.mean(a < 0))
        return {"n": len(a), "frac_up": up, "frac_down": dn, "frac_zero": 1.0 - up - dn,
                "up_share_of_moves": (up / (up + dn)) if up + dn > 0 else None,
                "mean_z": float(np.mean(a)), "median_z": float(q[3]), "mean_z_clip10": float(np.mean(np.clip(a, -10, 10))),
                "quantiles_z": [float(x) for x in q]}
    return {str(h): (summ(v) if v else None) for h, v in zs.items()}


def pick(lst):
    if len(lst) > args.max_series:
        idx = rng.choice(len(lst), size=args.max_series, replace=False); return [lst[i] for i in idx]
    return lst


def chronos_subsets(root):
    import pyarrow.parquet as pq
    for d in sorted(os.listdir(root)):
        arrs = []
        for f in sorted(glob.glob(os.path.join(root, d, "*.parquet"))):
            t = pq.read_table(f)
            for c in t.schema.names:
                ty = str(t.schema.field(c).type)
                if c in ("id", "timestamp") or not ty.startswith("list<") or "timestamp" in ty: continue
                arrs += [np.array([np.nan if x is None else x for x in v], dtype=np.float64) for v in t.column(c).to_pylist()]
        if arrs: yield d, pick(arrs)


def lotsa_subsets(root):
    from datasets import load_from_disk
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p): continue
        try: ds = load_from_disk(p)
        except Exception as e: print("skip", d, e); continue
        idx = list(range(len(ds))); idx = pick(idx); arrs = []
        for i in idx:
            t = np.asarray(ds[int(i)]["target"], dtype=np.float64)
            arrs += [t] if t.ndim == 1 else [row for row in t]
        yield d, arrs


def t300b_subsets(root):
    for meta in sorted(glob.glob(os.path.join(root, "*", "*", "meta.json"))):
        d = os.path.relpath(os.path.dirname(meta), root); m = json.load(open(meta))
        files = sorted(m["files"]); sizes = [m["files"][f] for f in files]; starts = np.cumsum([0] + sizes)
        mm = {f: np.memmap(os.path.join(os.path.dirname(meta), f), dtype=m.get("dtype", "float32"), mode="r") for f in files}
        sc = pick(m["scales"]); arrs = []
        for s in sc:
            off, ln = int(s["offset"]), int(s["length"]); k = int(np.searchsorted(starts, off, side="right") - 1)
            arrs.append(np.asarray(mm[files[k]][off - starts[k]: off - starts[k] + ln], dtype=np.float64))
        yield d, arrs


out = {}
for corpus, gen in (("chronos", chronos_subsets(os.path.join(args.root, "chronos_datasets"))),
                    ("lotsa", lotsa_subsets(os.path.join(args.root, "lotsa"))),
                    ("time300b", t300b_subsets(os.path.join(args.root, "time300b")))):
    res = {}
    for name, arrs in gen:
        res[name] = {"series": len(arrs), "stats": stats(arrs)}
        s = res[name]["stats"]["128"]
        print(corpus, name, len(arrs), "h128 up", None if s is None else round(s["frac_up"], 3), flush=True)
    agg = {}
    for h in map(str, HS):
        ok = [(v["series"], v["stats"][h]) for v in res.values() if v["stats"][h]]
        if not ok: continue
        fu = np.array([s["frac_up"] for _, s in ok]); w = np.array([n for n, _ in ok], dtype=float)
        us = np.array([s["up_share_of_moves"] if s["up_share_of_moves"] is not None else np.nan for _, s in ok])
        mc = np.array([s["mean_z_clip10"] for _, s in ok]); md = np.array([s["median_z"] for _, s in ok]); mz = np.array([s["mean_z"] for _, s in ok])
        agg[h] = {"subsets": len(ok), "median_frac_up": float(np.median(fu)), "weighted_frac_up": float(np.sum(fu * w) / w.sum()),
                  "share_subsets_frac_up_gt_0.55": float(np.mean(fu > 0.55)), "share_subsets_frac_up_lt_0.45": float(np.mean(fu < 0.45)),
                  "median_up_share_of_moves": float(np.nanmedian(us)), "weighted_up_share_of_moves": float(np.nansum(us * w) / w[np.isfinite(us)].sum()),
                  "share_subsets_up_share_gt_0.55": float(np.mean(us[np.isfinite(us)] > 0.55)), "share_subsets_up_share_lt_0.45": float(np.mean(us[np.isfinite(us)] < 0.45)),
                  "median_mean_z_clip10": float(np.median(mc)), "weighted_mean_z_clip10": float(np.sum(mc * w) / w.sum()),
                  "median_median_z": float(np.median(md)), "median_mean_z": float(np.median(mz)), "weighted_mean_z": float(np.sum(mz * w) / w.sum())}
    out[corpus] = {"subsets": res, "aggregate": agg}
    print(corpus, "AGG", agg, flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
print("done")
