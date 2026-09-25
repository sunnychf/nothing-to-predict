"""Re-read the corpus series that corpus_match.py found with a market correlation of at least 0.8 and save their values,
so that the matches can be checked locally (share of zero returns, correlation without the most extreme days)."""
import json, sys
import numpy as np
ROOT = sys.argv[1]                                                     # the corpus root, e.g. ~/tsfm_rev/data
sys.argv = ["x", "--root", ROOT, "--targets", "kf_targets.npz", "--out", "/dev/null"]
exec(open("corpus_match.py").read().split("if __name__")[0])
d = json.load(open("corpus_match.json")); want = {}
for s in d["subsets"]:
    for h in s["hits"]:
        if h["market"][0] >= 0.8:
            want.setdefault((h["corpus"], h["subset"]), set()).add(h["series"])
roots = {"chronos": "chronos_datasets", "lotsa": "lotsa", "time300b": "time300b"}
gen = {"chronos": chronos_series, "lotsa": lotsa_series, "time300b": t300b_series}
out = {}
for (corpus, subset), ids in want.items():
    for sid, v in gen[corpus](f"{ROOT}/{roots[corpus]}", subset):
        if sid in ids:
            out[f"{corpus}|{subset}|{sid}"] = v
np.savez_compressed("corpus_match_series.npz", keys=np.array(list(out)), **{f"s{i}": v for i, v in enumerate(out.values())})
print(len(out), "series saved of", sum(len(v) for v in want.values()))
