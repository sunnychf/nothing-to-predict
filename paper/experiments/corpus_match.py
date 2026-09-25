"""Search the public pretraining corpora of the corpus audit for series that reproduce the US equity market's daily path
(experiments/PREREGISTRATION_corpus_match.md, part A). Corpora: the downloaded Chronos datasets, LOTSA and Time-300B
(experiments/server/dl_chronos.py, dl_corpora.py), every series of every subset except the synthetic or mixed-up ones,
with at least 251 positive finite values. Targets: daily log returns of the Ken French market (Mkt-RF + RF) and of the 49
industry portfolios (kf_targets.npz, written by experiments/corpus_match_summary.py --targets). For each series, the
largest correlation between its log returns and the market's over all alignments with an overlap of at least 250 days
(computed for every lag at once with FFTs and cumulative sums); series whose best market correlation exceeds 0.6 are
also compared with every industry. Output: one JSON with the scan counts per subset and every series above 0.6.
Run on the server:  python corpus_match.py --root ~/tsfm_rev/data --targets kf_targets.npz --out corpus_match.json --workers 64"""
import argparse, glob, json, os, time
import numpy as np
from multiprocessing import Pool

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--targets", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--min-overlap", type=int, default=250)
ap.add_argument("--stage2", type=float, default=0.6); args = ap.parse_args()
SKIP = ("synth", "tsmixup", "mixup")                                  # synthetic or mixed-up corpora carry no real calendar


def load_targets():
    z = np.load(args.targets, allow_pickle=False)
    T = []
    for i, name in enumerate(z["names"]):
        r = z["R"][i]; d = z["dates"]; ok = np.concatenate([[False], np.isfinite(r), [False]]).astype(int)
        edges = np.flatnonzero(np.diff(ok)); runs = list(zip(edges[::2], edges[1::2]))
        a, b = max(runs, key=lambda q: q[1] - q[0])                          # longest run of valid returns of this target
        T.append((str(name), r[a:b].astype(np.float64), d[a:b]))
    return T


TARGETS = None
FFT_CACHE = {}


def best_lag(x, ti, W):
    """max over alignments of corr(x_i, y_{i+k}) with overlap >= W; returns r, lag k (index into y), overlap."""
    name, y, _ = TARGETS[ti]
    n, N = len(x), len(y)
    if n < W or N < W:
        return None
    L = 1 << int(np.ceil(np.log2(n + N)))
    key = (ti, L)
    if key not in FFT_CACHE:
        cy, cy2 = np.concatenate([[0.0], np.cumsum(y)]), np.concatenate([[0.0], np.cumsum(y * y)])
        FFT_CACHE[key] = (np.fft.rfft(y, L), cy, cy2)
    Y, cy, cy2 = FFT_CACHE[key]
    cc = np.fft.irfft(np.conj(np.fft.rfft(x, L)) * Y, L)                 # cc[k] = sum_i x_i y_{i+k}; negative k wrap around
    ks = np.arange(-(n - 1), N)
    sxy = np.concatenate([cc[L - (n - 1):], cc[:N]])
    a = np.maximum(0, -ks); b = np.minimum(n, N - ks); Lk = b - a
    ok = Lk >= W
    a, b, Lk, ks, sxy = a[ok], b[ok], Lk[ok], ks[ok], sxy[ok]
    cx, cx2 = np.concatenate([[0.0], np.cumsum(x)]), np.concatenate([[0.0], np.cumsum(x * x)])
    sx, sxx = cx[b] - cx[a], cx2[b] - cx2[a]; sy, syy = cy[b + ks] - cy[a + ks], cy2[b + ks] - cy2[a + ks]
    vx, vy = sxx - sx * sx / Lk, syy - sy * sy / Lk
    good = (vx > 1e-12 * Lk) & (vy > 0)
    if not good.any():
        return None
    r = np.full(len(ks), -1.0); r[good] = (sxy[good] - sx[good] * sy[good] / Lk[good]) / np.sqrt(vx[good] * vy[good])
    j = int(np.argmax(r))
    return float(r[j]), int(ks[j]), int(Lk[j]), int(a[j])


def chronos_series(root, d, part=(0, 1)):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(root, d, "*.parquet"))):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist() if "id" in t.schema.names else [None] * t.num_rows
        for c in t.schema.names:
            ty = str(t.schema.field(c).type)
            if c in ("id", "timestamp") or not ty.startswith("list<") or "timestamp" in ty:
                continue
            col = t.column(c)
            for i in range(part[0], t.num_rows, part[1]):
                v = col[i].as_py()
                yield f"{ids[i]}:{c}", np.array([np.nan if q is None else q for q in v], dtype=np.float64)


def lotsa_series(root, d, part=(0, 1)):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(root, d))
    for i in range(part[0], len(ds), part[1]):
        rec = ds[i]; t = np.asarray(rec["target"], dtype=np.float64); sid = rec.get("item_id", i)
        if t.ndim == 1:
            yield f"{sid}", t
        else:
            for j, row in enumerate(t):
                yield f"{sid}:{j}", row


def t300b_series(root, d, part=(0, 1)):
    meta = os.path.join(root, d, "meta.json"); m = json.load(open(meta))
    files = sorted(m["files"]); sizes = [m["files"][f] for f in files]; starts = np.cumsum([0] + sizes)
    mm = {f: np.memmap(os.path.join(root, d, f), dtype=m.get("dtype", "float32"), mode="r") for f in files}
    for i in range(part[0], len(m["scales"]), part[1]):
        s = m["scales"][i]
        off, ln = int(s["offset"]), int(s["length"]); k = int(np.searchsorted(starts, off, side="right") - 1)
        yield f"{i}", np.asarray(mm[files[k]][off - starts[k]: off - starts[k] + ln], dtype=np.float64)


def subsets():
    R = args.root; out = []
    c = os.path.join(R, "chronos_datasets")
    out += [("chronos", c, d) for d in sorted(os.listdir(c)) if os.path.isdir(os.path.join(c, d)) and not d.startswith(".")]
    l = os.path.join(R, "lotsa")
    out += [("lotsa", l, d) for d in sorted(os.listdir(l)) if os.path.isdir(os.path.join(l, d)) and not d.startswith(".")]
    t = os.path.join(R, "time300b")
    out += [("time300b", t, os.path.relpath(os.path.dirname(p), t)) for p in sorted(glob.glob(os.path.join(t, "*", "*", "meta.json")))]
    out = [s for s in out if not any(k in s[2].lower() for k in SKIP)]
    tasks = []                                                             # split large subsets into interleaved parts
    for corpus, root, d in out:
        n = size(corpus, root, d); k = max(1, int(np.ceil(n / 20000)))
        tasks += [(corpus, root, d, (j, k)) for j in range(k)]
    return tasks


def size(corpus, root, d):
    try:
        if corpus == "chronos":
            import pyarrow.parquet as pq
            return sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(os.path.join(root, d, "*.parquet")))
        if corpus == "lotsa":
            from datasets import load_from_disk
            return len(load_from_disk(os.path.join(root, d)))
        return len(json.load(open(os.path.join(root, d, "meta.json")))["scales"])
    except Exception:
        return 1


def init():
    global TARGETS
    TARGETS = load_targets()


def scan(task):
    corpus, root, d, part = task
    gen = {"chronos": chronos_series, "lotsa": lotsa_series, "time300b": t300b_series}[corpus]
    t0 = time.time(); n_seen = n_used = 0; hits = []
    try:
        for sid, v in gen(root, d, part):
            n_seen += 1
            if len(v) < args.min_overlap + 1 or not np.all(np.isfinite(v)) or np.any(v <= 0):
                continue
            x = np.diff(np.log(v))
            if np.std(x) == 0:
                continue
            n_used += 1
            m = best_lag(x, 0, args.min_overlap)
            if m is None or m[0] < args.stage2:
                continue
            rec = {"corpus": corpus, "subset": d, "series": sid, "length": int(len(v)), "market": list(m), "industries": {}}
            for ti in range(1, len(TARGETS)):
                q = best_lag(x, ti, args.min_overlap)
                if q is not None and q[0] >= args.stage2:
                    rec["industries"][TARGETS[ti][0]] = list(q)
            hits.append(rec)
    except Exception as e:
        return {"corpus": corpus, "subset": d, "part": list(part), "error": repr(e)[:300], "n_seen": n_seen, "n_used": n_used, "hits": hits}
    return {"corpus": corpus, "subset": d, "part": list(part), "n_seen": n_seen, "n_used": n_used, "seconds": round(time.time() - t0, 1), "hits": hits}


if __name__ == "__main__":
    tasks = subsets(); init()
    print(len(tasks), "tasks over", len({t[:3] for t in tasks}), "subsets;", len(TARGETS), "targets", flush=True)
    res = []
    with Pool(args.workers, initializer=init) as pool:
        for r in pool.imap_unordered(scan, tasks):
            res.append(r)
            print(f"[{time.strftime('%H:%M:%S')}] {r['corpus']}/{r['subset']} part {r['part'][0] + 1}/{r['part'][1]}: {r.get('n_used', 0)} of {r.get('n_seen', 0)} series scanned, "
                  f"{len(r['hits'])} above {args.stage2}" + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
    names = [t[0] for t in TARGETS]; dates = {t[0]: [str(t[2][0]), str(t[2][-1])] for t in TARGETS}
    json.dump({"min_overlap": args.min_overlap, "stage2": args.stage2, "targets": names, "target_spans": dates,
               "subsets": sorted(res, key=lambda r: (r["corpus"], r["subset"], r["part"][0]))}, open(args.out, "w"))
    print("DONE", sum(r.get("n_used", 0) for r in res), "series scanned,", sum(len(r["hits"]) for r in res), "above", args.stage2, flush=True)
