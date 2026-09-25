"""Part A of experiments/PREREGISTRATION_corpus_match.md: which trading days of the US market are contained in public
pretraining corpora, and does the models' market timing concentrate on them?
  --targets    write experiments/results/corpus_match/kf_targets.npz: daily log returns of the Ken French market
               (Mkt-RF + RF) and of the 49 industry portfolios, 1963-07-01 to 2026-07-31, for experiments/corpus_match.py
  (default)    read experiments/results/corpus_match/corpus_match.json (the server scan) and summarise: matched series
               (best correlation >= 0.9, overlap >= 250 days), the covered set of market trading days, the coverage of
               the 1996-2014 falling-market futures, of the daily anchor's and of the size deciles' futures, and the
               timing split of prediction C2; writes experiments/results/corpus_match/corpus_match_summary.json and
               tables/corpus_match.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/corpus_match_summary.py [--targets]"""
import argparse, io, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--targets", action="store_true"); ap.add_argument("--perm", type=int, default=20000)
args = ap.parse_args()
D = os.path.join("experiments", "results", "corpus_match"); os.makedirs(D, exist_ok=True)
DATA = os.path.join("..", "data", "real")
THR, W = 0.9, 250


def read_ff_block(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "," in l and len(l.split(",")) > 3)
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].split(",")[0].strip().isdigit():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d"); df.columns = [c.strip() for c in df.columns]
    return df


if args.targets:
    fac = read_ff_block(os.path.join(DATA, "F-F_Research_Data_Factors_daily.csv")).loc["1963-07-01":"2026-07-31"]
    ind = read_ff_block(os.path.join(DATA, "49_Industry_Portfolios_Daily.csv")).reindex(fac.index)
    R = [np.log1p((fac["Mkt-RF"] + fac["RF"]).to_numpy() / 100.0)]; names = ["Mkt"]
    for c in ind.columns:
        v = ind[c].to_numpy(dtype=np.float64, copy=True); v[v <= -99] = np.nan
        R.append(np.log1p(v / 100.0)); names.append(c)
    np.savez(os.path.join(D, "kf_targets.npz"), dates=np.array([str(d.date()) for d in fac.index]), names=np.array(names), R=np.array(R))
    print(f"{len(names)} targets, {len(fac)} trading days {fac.index[0].date()} .. {fac.index[-1].date()}")
    raise SystemExit(0)

# ------------------------------------------------------------------ summary of the server scan (default mode)
J = json.load(open(os.path.join(D, "corpus_match.json"))); K = np.load(os.path.join(D, "kf_targets.npz"))
SV = np.load(os.path.join(D, "corpus_match_series.npz")); SER = {k: SV[f"s{i}"] for i, k in enumerate(SV["keys"])}
dates, names, TR = K["dates"], list(K["names"]), K["R"]; mkt = TR[0]; didx = {d: i for i, d in enumerate(dates)}
CORP = {"chronos": "Chronos datasets", "lotsa": "LOTSA", "time300b": "Time-300B"}
hits = []
for s in J["subsets"]:
    for h in s["hits"]:
        if h["market"][0] < 0.8:
            continue
        r, k, Lk, a = h["market"]; x = np.diff(np.log(SER[f"{h['corpus']}|{h['subset']}|{h['series']}"]))
        xo, yo = x[a:a + Lk], mkt[a + k:a + k + Lk]
        assert abs(np.corrcoef(xo, yo)[0, 1] - r) < 1e-6, h["series"]
        keep = np.argsort(np.abs(yo))[:-5]                                   # without the five largest market moves
        ind = max(h["industries"].items(), key=lambda kv: kv[1][0]) if h["industries"] else (None, [None])
        hits.append({"corpus": h["corpus"], "subset": h["subset"], "series": h["series"], "length": h["length"], "r": r,
                     "r_without_5_extreme_days": float(np.corrcoef(xo[keep], yo[keep])[0, 1]), "zero_share": float(np.mean(xo == 0)),
                     "span": [str(dates[a + k]), str(dates[a + k + Lk - 1])], "i0": int(a + k), "n": int(Lk),
                     "best_industry": ind[0], "best_industry_r": ind[1][0]})
import hashlib
for h in hits:
    x = np.diff(np.log(SER[f"{h['corpus']}|{h['subset']}|{h['series']}"]))
    h["values_sha"] = hashlib.sha256(np.round(x, 10).tobytes()).hexdigest()[:16]
matched = [h for h in hits if h["r"] >= THR]
robust = [h for h in matched if h["r_without_5_extreme_days"] >= 0.85 and h["zero_share"] < 0.2]
cov = {}
for c in list(CORP) + ["all"]:
    m = np.zeros(len(dates), bool)
    for h in robust:
        if c in ("all", h["corpus"]):
            m[h["i0"]:h["i0"] + h["n"]] = True
    cov[c] = m
uniq = {h["values_sha"]: h for h in robust}


def share(mask, a, b):
    i, j = didx[a], didx[b]; return float(mask[i:j + 1].mean())


def fut_cov(meta_path, fam=None):
    meta = json.load(open(meta_path)); out = {}
    for m in meta:
        if fam and m.get("family") != fam:
            continue
        i = didx[m["ctx_end"]]; out[m["fut_end"]] = float(cov["all"][i + 1:i + 129].mean())
    return out


fall = fut_cov(os.path.join("experiments", "results", "falling_windows_meta.json"))
anch = fut_cov(os.path.join("..", "results", "real_windows_meta.json"), "eq")
size = fut_cov(os.path.join("..", "results", "smallcap_windows_meta.json"), "ew")
covered = sorted(d for d, v in fall.items() if v >= 0.5); uncovered = sorted(d for d, v in fall.items() if v < 0.5)
# C2: timing on covered and uncovered end dates (timing_check.py's statistic, raw contexts)
B = np.load(os.path.join("experiments", "results", "falling_windows.npz")); bm = json.load(open(os.path.join("experiments", "results", "falling_windows_meta.json")))
ends = np.array([m["fut_end"] for m in bm]); last = B["ctx_raw"][:, -1]; Y = B["fut_raw"][:, -1] / last - 1.0
Dd = sorted(set(ends)); ym = np.array([Y[ends == d].mean() for d in Dd]); cvl = np.array([d in covered for d in Dd])
MODELS = ["chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast"]
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
split = {}
testable = cvl.sum() >= 8 and (~cvl).sum() >= 8
for mi, m in enumerate(MODELS):
    p = os.path.join("experiments", "results", "falling", f"falling_{m}.npz")
    F = np.load(p)["yhat_raw"][:, -1] / last - 1.0; fm = np.array([F[ends == d].mean() for d in Dd])
    rc = float(np.corrcoef(fm[cvl], ym[cvl])[0, 1]); ru = float(np.corrcoef(fm[~cvl], ym[~cvl])[0, 1])
    o = {"label": LABEL[m], "covered": rc, "uncovered": ru, "difference": rc - ru}
    if testable:
        rng = np.random.default_rng([41, mi]); null = np.empty(args.perm)
        for b in range(args.perm):
            s = rng.permutation(cvl); null[b] = np.corrcoef(fm[s], ym[s])[0, 1] - np.corrcoef(fm[~s], ym[~s])[0, 1]
        o["perm_p"] = float(np.mean(null >= rc - ru - 1e-12))
    split[m] = o
FOUR = ("chronosbolt", "tirex", "moirai2", "timesfm")
res = {"threshold": THR, "min_overlap": W, "n_scanned": {c: sum(s["n_used"] for s in J["subsets"] if s["corpus"] == c) for c in CORP},
       "n_subsets": len({(s["corpus"], s["subset"]) for s in J["subsets"]}),
       "hits_r_ge_0.8": len(hits), "matched_r_ge_0.9": len(matched), "matched_robust": len(robust), "unique_robust_series": len(uniq),
       "matched_by_subset": {f"{h['corpus']}|{h['subset']}": sum(1 for q in robust if (q["corpus"], q["subset"]) == (h["corpus"], h["subset"])) for h in robust},
       "not_robust": [{k: h[k] for k in ("corpus", "subset", "series", "r", "r_without_5_extreme_days", "zero_share", "span")} for h in matched if h not in robust],
       "spans": sorted({(h["span"][0], h["span"][1], round(h["r"], 3), h["best_industry"]) for h in uniq.values()}),
       "covered_share": {c: {"1996-2014": share(cov[c], "1996-01-02", "2014-12-31"), "2017-2026": share(cov[c], "2017-01-03", "2026-07-31"),
                             "1963-1995": share(cov[c], "1963-07-01", "1995-12-29")} for c in cov},
       "last_covered_day": str(dates[np.flatnonzero(cov["all"])[-1]]) if cov["all"].any() else None,
       "first_covered_day": str(dates[np.flatnonzero(cov["all"])[0]]) if cov["all"].any() else None,
       "falling_future_coverage": fall, "falling_covered_dates": covered, "falling_uncovered_dates": uncovered,
       "anchor_eq_future_coverage_max": max(anch.values()), "size_future_coverage": {"n_dates": len(size), "n_covered": sum(v >= 0.5 for v in size.values())},
       "c1_part_of_1996_2014_and_none_of_anchor": bool(share(cov["all"], "1996-01-02", "2014-12-31") > 0 and max(anch.values()) == 0),
       "c2_testable": bool(testable), "timing_split": split,
       "c2_met": {m: bool(split[m]["difference"] > 0) for m in FOUR}}
for h in hits:
    h.pop("i0"); h.pop("n")
res["hits"] = hits
json.dump(res, open(os.path.join(D, "corpus_match_summary.json"), "w"), indent=1)
print(f"scanned {res['n_scanned']} series in {res['n_subsets']} subsets; r>=0.8: {len(hits)}; r>=0.9: {len(matched)}; robust: {len(robust)}; unique robust series: {len(uniq)}")
print("robust matches by subset:", res["matched_by_subset"]); print("not robust:", [(q['corpus'], q['subset'], round(q['r'], 3), round(q['r_without_5_extreme_days'], 3), round(q['zero_share'], 2)) for q in res["not_robust"]])
print("covered share:", {c: {k: round(v, 3) for k, v in d_.items()} for c, d_ in res["covered_share"].items()}, "first/last covered day:", res["first_covered_day"], res["last_covered_day"])
print(f"falling futures covered (>= half): {len(covered)} of {len(fall)}; anchor max coverage {res['anchor_eq_future_coverage_max']:.2f}; size deciles covered {res['size_future_coverage']}")
print("C1:", res["c1_part_of_1996_2014_and_none_of_anchor"], "| C2 testable:", testable)
for m, o in split.items():
    print(f"   {o['label']:12s} covered {o['covered']:+.2f} uncovered {o['uncovered']:+.2f} diff {o['difference']:+.2f}" + (f" p {o['perm_p']:.3f}" if "perm_p" in o else ""))
