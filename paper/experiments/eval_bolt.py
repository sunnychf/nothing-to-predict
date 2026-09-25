"""Experiment 2 evaluation: one Chronos-Bolt-small checkpoint (released, or a fine-tuned state dict)
on the paper's probes, with the paper's forecasting helper (pilot_a_chronosx.forecast_cx, the median):
  N1 direction      128 series, seed 4000, H = 128 (the direction contexts of the paper)
  ladder skill      N1 and N4, 512 series, seed 0, H = 16; N4 share of the exact conditional mean at h = 1
  ETTh1             the 924 windows of etth1_windows.npz, skill at h = 1, 16 in context-sigma units
  daily windows     real_windows.npz: raw skill (all / eq / fx) at h = 16, 128 in return units, the
                    mirror-corrected raw skill, and the sign-randomised copies' skill and upward fraction
  python eval_bolt.py --tag released
  python eval_bolt.py --tag plain_s0 --ckpt runs/plain_s0/model.pt
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chronos import BaseChronosPipeline
from nulls import RUNGS, oracle_forecast_exact
from probe_metrics import make_data
import pilot_a_chronosx as px

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ckpt", default=None)
ap.add_argument("--threads", type=int, default=32); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--data", default=os.path.expanduser("~/tsfm_rev/data")); ap.add_argument("--out", default=os.path.expanduser("~/tsfm_rev/results"))
args = ap.parse_args(); torch.set_num_threads(args.threads)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
if args.ckpt:
    pipe.model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
pipe.model.eval()
F = lambda c, H: px.forecast_cx(pipe, c, H, args.batch)[0].astype(np.float64)
R = {"tag": args.tag, "ckpt": args.ckpt}

# N1 direction
y, info = RUNGS["N1"](128, 512, 128, np.random.default_rng(4000)); ctx = y[:, :512]; sig = info["sigma"]
dep = F(ctx, 128) - ctx[:, -1:]
R["N1_direction"] = {str(h): {"frac_up": float((dep[:, h - 1] > 0).mean()), "mean_dep_sigma": float(dep[:, h - 1].mean() / sig),
                              "rms_dep_sigma": float(np.sqrt((dep[:, h - 1] ** 2).mean()) / sig)} for h in (16, 128)}
log(f"N1 direction: {R['N1_direction']}")
# ladder skill
for rung in ("N1", "N4"):
    c, fu, inf = make_data(rung, 512, 512, 16, 0); yh = F(c, 16); last = c[:, -1:]
    ep = ((fu - last) ** 2).mean(0); em = ((fu - yh) ** 2).mean(0)
    r = {"skill_h1": float(1 - em[0] / ep[0]), "skill_h16": float(1 - em[15] / ep[15])}
    if rung == "N4":
        orc = oracle_forecast_exact("N4", c, inf, 16); eo = ((fu - orc) ** 2).mean(0)
        r["oracle_share_h1"] = float((ep[0] - em[0]) / (ep[0] - eo[0]))
    R[f"ladder_{rung}"] = r
log(f"ladder: {R['ladder_N1']} {R['ladder_N4']}")
# ETTh1
W = np.load(os.path.join(args.data, "etth1_windows.npz")); c, fu = W["ctx"], W["fut"]
s = np.std(np.diff(c, axis=1), axis=1)[:, None]; yh = F(c, 128); last = c[:, -1:]
ep = ((fu - last) ** 2 / s ** 2).mean(0); em = ((fu - yh) ** 2 / s ** 2).mean(0)
R["etth1"] = {str(h): float(1 - em[h - 1] / ep[h - 1]) for h in (1, 16, 64, 128)}
log(f"ETTh1: {R['etth1']}")
# daily windows
W = np.load(os.path.join(args.data, "real_windows.npz")); meta = json.load(open(os.path.join(args.data, "real_windows_meta.json")))
fam = np.array([m["family"] for m in meta]); cr, fr_ = W["ctx_raw"], W["fut_raw"]; lr = cr[:, -1:]; fr = fr_ / lr - 1.0
def mirror_of(x):
    r = x[:, 1:] / x[:, :-1] - 1.0; m = np.empty_like(x); m[:, 0] = x[:, 0]; m[:, 1:] = x[:, :1] * np.cumprod(1.0 - r, axis=1); return m
rel = F(cr, 128) / lr - 1.0
mr = mirror_of(cr); rel_m = F(mr, 128) / mr[:, -1:] - 1.0; rel_c = rel - 0.5 * (rel + rel_m)
def skill(rr, idx, j): return float(1 - ((fr[idx, j] - rr[idx, j]) ** 2).mean() / (fr[idx, j] ** 2).mean())
R["daily_raw"] = {}
for sp, idx in (("all", np.arange(len(cr))), ("eq", np.where(fam == "eq")[0]), ("fx", np.where(fam == "fx")[0])):
    R["daily_raw"][sp] = {str(h): {"skill": skill(rel, idx, h - 1), "skill_mirror": skill(rel_c, idx, h - 1),
                                   "frac_up": float((rel[idx, h - 1] > 0).mean()),
                                   "even_mean_annualised": float(100 * (252 / 128) * (0.5 * (rel + rel_m))[idx, h - 1].mean()) if h == 128 else None} for h in (16, 128)}
cs, fs = W["ctx_sur"], W["fut_sur"]; n, K, T = cs.shape; flat = cs.reshape(n * K, T); ls = flat[:, -1:]
rs = F(flat, 128) / ls - 1.0; frs = fs.reshape(n * K, -1) / ls - 1.0
R["daily_sign_randomised"] = {str(h): {"skill": float(1 - ((frs[:, h - 1] - rs[:, h - 1]) ** 2).mean() / (frs[:, h - 1] ** 2).mean()),
                                       "frac_up": float((rs[:, h - 1] > 0).mean())} for h in (16, 128)}
log(f"daily raw eq h128: {R['daily_raw']['eq']['128']} | null: {R['daily_sign_randomised']['128']}")
os.makedirs(args.out, exist_ok=True)
json.dump(R, open(os.path.join(args.out, f"eval_bolt_{args.tag}.json"), "w"), indent=1)
log("done")
