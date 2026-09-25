"""Is the upward drift a directional prior, or an artefact of positive inputs?

All conditions are built from ONE set of Gaussian increments z with sigma
passed explicitly (=1), so they differ only by offset and sign:
  pos     100 + cumsum(z)      the baseline every other run used
  neg     -pos                 exact mirror of pos, all values negative
  zero    cumsum(z)            centred on zero, crosses it freely
  mirror  -zero                exact mirror of zero

A model with no directional prior is sign-equivariant: dep(mirror) = -dep(zero)
and dep(neg) = -dep(pos), so the fractions-up sum to one and the signed means
cancel. A pure 'up' prior gives fraction-up > 1/2 in every condition.

History: the first version of this script derived sigma from level0 and got
sigma = -1 at level0 = -100 (flipping every reported sign) and sigma = 0 at
level0 = 0 (a constant series). METHOD_NOTES M7. Fixed by generating from
shared increments.
"""
import json, os, sys, numpy as np
from pilot_a import load_pipeline, forecast, log
from paths import RESULTS_DIR
OUT = os.environ.get("SPEC_OUT", RESULTS_DIR)
H, n, S, T = 128, 128, 100, 512
want = sys.argv[1:] or ["pos", "neg", "zero", "mirror"]
rng = np.random.default_rng(4000)
z = rng.standard_normal((n, T + H)) * 1.0            # sigma = 1, explicit
base = np.cumsum(z, axis=1)
series = {"pos": 100.0 + base, "neg": -(100.0 + base), "zero": base, "mirror": -base}
pipe = load_pipeline("amazon/chronos-t5-small", "pretrained", "cuda")
recs = {}
for tag in want:
    y = series[tag]; ctx = y[:, :T]
    s = forecast(pipe, ctx, H, S, batch=8, seed=0)
    dep = s.mean(1) - ctx[:, -1:]
    np.savez_compressed(f"{OUT}/departures_ctrl_{tag}_H{H}.npz", dep=dep.astype(np.float32),
                        last=ctx[:, -1].astype(np.float32), sigma=1.0)
    r = {"tag": tag, "H": H, "n": n, "sigma": 1.0,
         "mean_dep_sigma": {h + 1: float(dep[:, h].mean()) for h in (0, 15, 63, H - 1)},
         "median_dep_sigma": {h + 1: float(np.median(dep[:, h])) for h in (0, 15, 63, H - 1)},
         "rms_dep_sigma": {h + 1: float(np.sqrt((dep[:, h] ** 2).mean())) for h in (0, 15, 63, H - 1)},
         "frac_up": {h + 1: float((dep[:, h] > 0).mean()) for h in (0, 15, 63, H - 1)}}
    recs[tag] = r
    log(f"{tag:>6}: h=1 mean {r['mean_dep_sigma'][1]:+.3f} up {r['frac_up'][1]:.2f} | "
        f"h=16 mean {r['mean_dep_sigma'][16]:+.3f} up {r['frac_up'][16]:.2f} | "
        f"h=128 mean {r['mean_dep_sigma'][128]:+.3f} median {r['median_dep_sigma'][128]:+.3f} up {r['frac_up'][128]:.2f}")
# mirror-pair asymmetries from whatever npz files exist (this run or earlier)
for a, b in (("pos", "neg"), ("zero", "mirror")):
    try:
        da = np.load(f"{OUT}/departures_ctrl_{a}_H{H}.npz")["dep"]; db = np.load(f"{OUT}/departures_ctrl_{b}_H{H}.npz")["dep"]
    except FileNotFoundError:
        continue
    asym = da + db                                   # zero for a sign-equivariant model
    recs[f"{a}+{b}"] = {"pair": [a, b], "H": H,
        "mean_asym_sigma": {h + 1: float(asym[:, h].mean()) for h in (0, 15, 63, H - 1)},
        "rms_asym_over_rms_dep": {h + 1: float(np.sqrt((asym[:, h] ** 2).mean()) / np.sqrt((da[:, h] ** 2).mean())) for h in (0, 15, 63, H - 1)},
        "note": "asym = dep(a) + dep(b). Sign-equivariant model: 0. Its mean is twice the "
                "directional prior; rms ratio is the share of the departure that does not mirror."}
    log(f"{a}+{b}: asym mean h=128 {recs[f'{a}+{b}']['mean_asym_sigma'][128]:+.3f}σ  non-mirroring share {recs[f'{a}+{b}']['rms_asym_over_rms_dep'][128]:.2f}")
json.dump(recs, open(f"{OUT}/diag_direction_control.json", "w"), indent=1)
log("wrote diag_direction_control.json")
