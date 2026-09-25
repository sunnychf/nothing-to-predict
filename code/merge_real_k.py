"""Merge a K=4 real_<model>.npz with the copies-4.. file of the K=16 run into one K=16 file.

    python code/merge_real_k.py results_v2/real_chronos.npz results_v2/real_chronos_from4.npz results_v2/real_chronos_k16.npz
The two runs used the same windows (the first four sign draws of the K=16 file are the K=4
file's draws), so the surrogate axes concatenate; raw forecasts come from the K=4 file.
"""
import sys, json
import numpy as np
a, b, out = sys.argv[1:4]
A, B = np.load(a), np.load(b)
ja, jb = json.load(open(a.replace(".npz", ".json"))), json.load(open(b.replace(".npz", ".json")))
assert ja["model"] == jb["model"] and ja["n"] == jb["n"] and ja["H"] == jb["H"] and jb["sur_from"] == A["yhat_sur"].shape[1], (ja, jb)
np.savez_compressed(out, yhat_raw=A["yhat_raw"], mcvar_raw=A["mcvar_raw"],
                    yhat_sur=np.concatenate([A["yhat_sur"], B["yhat_sur"]], axis=1), mcvar_sur=np.concatenate([A["mcvar_sur"], B["mcvar_sur"]], axis=1))
json.dump({**ja, "K": int(A["yhat_sur"].shape[1] + B["yhat_sur"].shape[1]), "merged_from": [a, b]}, open(out.replace(".npz", ".json"), "w"), indent=1)
print("wrote", out, "K =", A["yhat_sur"].shape[1] + B["yhat_sur"].shape[1])
