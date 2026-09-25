"""Calendar-shift check: do a model's forecasts follow the calendar when the context end is moved by a few weeks?
Window sets (context 512, future 128; context end moved by k in -40, -20, 0, +20, +40 trading days of each series):
  falling_eq   1996-2014, 50 Ken French equity series, 37 end dates (experiments/falling_windows.py, shift_windows.py;
               k = 0 is the falling-market run, the shifted windows were run separately; post hoc, prompted by that test)
  anchor_eq    the paper's daily anchor, 50 equity series, 2017-2026, 18 end dates    } experiments/shift_windows_anchor.py;
  anchor_fx    the paper's daily anchor, 20 exchange rates, 2017-2026, 19 end dates   } predictions written before the runs
  size_small   equal-weighted smallest size decile, 1963-2026, 120 end dates         } in PREREGISTRATION_shift_anchors.md;
  size_large   value-weighted largest size decile, 1963-2026, 120 end dates          } all k from one run
Cells are (end date, shift); a cell averages the windows of the set at that date and shift (50, 50, 20, 1, 1).
Statistic (change against target): per cell, the mean h = 128 forecast change from the shifted context's end, and the
mean realised level at the shift's target date relative to the level at the unshifted context's end; the correlation of
their deviations from each date's mean over its shifts, pooled over dates. The target deviations use only moves at least
48 trading days after the latest shifted context end, so a forecast that reads only its context has no correlation with
them on a random walk with constant drift, while a forecast that recalls the series has (experiments/shift_bias_sim.py).
p-value: the realised deviations of each date are multiplied by an independent random sign (20,000 draws), which keeps
their within-date structure. Where the drift itself changes sign, readers of the context can also score (up to about
0.45 in the simulation), so the same statistic is given for rules that read only the context: the fitted drift and
momentum over 20, 60, 128 and 250 days, each followed or reversed; the table reports the largest absolute value.
A model follows the calendar on a set when the correlation exceeds 0.3 with p < 0.01.
Also recorded, not used: the version with returns from each shifted context's end (the one written down before the
anchor runs, and the one first used on 1996-2014), which is biased by construction because a later shift's context
contains part of the earlier shifts' futures (on a random walk trend-opposing forecasts score about +0.35 and trend-
following ones about -0.35), with its within-date permutation p; the version with forecast levels, which shares the
drift of context ends and targets; and the timing across end dates at each shift, which is not affected.
Writes experiments/results/shift_check.json and tables/shift.tex.
Run from finance/paper_revision:  ~/miniconda3/envs/nature-figure/bin/python experiments/shift_check.py"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=20000); args = ap.parse_args()
R = os.path.join("experiments", "results")
LABEL = {"chronosbolt": "Chronos-Bolt", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm": "TimesFM-2.0", "moirai": "Moirai-1.1", "chronos2": "Chronos-2"}
SETS = ("falling_eq", "anchor_eq", "anchor_fx", "size_small", "size_large")
MOM = (20, 60, 128, 250)
THRESH_R, THRESH_P = 0.3, 0.01


def records(sname, key):
    """Per window: end date, shift, level at the unshifted context end, last context value, realised and forecast target
    levels, and context-only forecasts (fitted drift, momentum). None if the model's forecasts are missing."""
    if sname == "falling_eq":
        B = np.load(os.path.join(R, "falling_windows.npz")); bm = json.load(open(os.path.join(R, "falling_windows_meta.json")))
        S = np.load(os.path.join(R, "shift_windows.npz")); sm = json.load(open(os.path.join(R, "shift_windows_meta.json")))
        pb, ps = os.path.join(R, "falling", f"falling_{key}.npz"), os.path.join(R, "shift", f"shift_{key}_raw_part0_{len(sm)}.npz")
        if not (os.path.exists(pb) and os.path.exists(ps)):
            return None
        c0 = B["ctx_raw"][:, -1]
        ctx = np.concatenate([B["ctx_raw"], S["ctx_raw"]]); fut = np.concatenate([B["fut_raw"], S["fut_raw"]])
        fc = np.concatenate([np.load(pb)["yhat_raw"][:, -1], np.load(ps)["yhat"][:, -1]])
        date = [m["fut_end"] for m in bm] + [m["base_fut_end"] for m in sm]; shift = [0] * len(bm) + [m["shift"] for m in sm]
        base = np.concatenate([c0, c0[[m["base_i"] for m in sm]]])
    else:
        tag, fam = {"anchor_eq": ("anchor", "eq"), "anchor_fx": ("anchor", "fx"), "size_small": ("size", "ew"), "size_large": ("size", "vw")}[sname]
        S = np.load(os.path.join(R, f"shift_{tag}_windows.npz")); sm = json.load(open(os.path.join(R, f"shift_{tag}_windows_meta.json")))
        ps = os.path.join(R, f"shift_{tag}", f"shift_{tag}_{key}_raw_part0_{len(sm)}.npz")
        if not os.path.exists(ps):
            return None
        sel = np.array([m["family"] == fam for m in sm])
        k0 = {m["base_i"]: S["ctx_raw"][i, -1] for i, m in enumerate(sm) if m["shift"] == 0}
        ctx, fut, fc = S["ctx_raw"][sel], S["fut_raw"][sel], np.load(ps)["yhat"][:, -1][sel]
        msel = [m for m, s in zip(sm, sel) if s]
        date = [m["base_fut_end"] for m in msel]; shift = [m["shift"] for m in msel]; base = np.array([k0[m["base_i"]] for m in msel])
    r = ctx[:, 1:] / ctx[:, :-1] - 1.0
    rec = {"date": np.array(date), "shift": np.array(shift), "base": base, "last": ctx[:, -1], "real": fut[:, -1], "fc": fc,
           "context_only": {"Fitted drift": (1.0 + r.mean(axis=1)) ** 128 - 1.0}}
    for m in MOM:
        rec["context_only"][f"Momentum {m}"] = ctx[:, -1] / ctx[:, -1 - m] - 1.0
    return rec


def cells(rec, F, Y):
    C = {}
    for d, k, f, y in zip(rec["date"], rec["shift"], F, Y):
        C.setdefault((d, int(k)), []).append((f, y))
    return {c: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for c, v in C.items()}


def change(rec, chg):      # forecast change from the shifted context's end vs realised target level (the statistic used)
    return cells(rec, chg, rec["real"] / rec["base"] - 1.0)


def within(C, rng, test):
    dates = sorted({d for d, _ in C}); shifts = sorted({k for _, k in C})
    groups = [[(d, k) for k in shifts if (d, k) in C] for d in dates]
    groups = [g for g in groups if len(g) >= 3]
    dF, dY, gid = [], [], []
    for i, g in enumerate(groups):
        F = np.array([C[c][0] for c in g]); Y = np.array([C[c][1] for c in g])
        dF += list(F - F.mean()); dY += list(Y - Y.mean()); gid += [i] * len(g)
    dF, dY, gid = np.array(dF), np.array(dY), np.array(gid)
    r = float(np.corrcoef(dF, dY)[0, 1]); norm = np.linalg.norm(dF) * np.linalg.norm(dY)
    o = {"corr": r, "n_date_groups": len(groups), "n_cells": int(len(dF))}
    if test == "signflip":           # an independent random sign per date on the realised deviations
        a = np.bincount(gid, weights=dF * dY); s = rng.choice([-1.0, 1.0], size=(args.draws, len(groups)))
        o["p"] = float(np.mean(s @ a / norm >= r - 1e-12))
    else:                            # shuffle the shifts within each date (the test written down for the returns version)
        null = np.zeros(args.draws); by = {}
        for i in range(len(groups)):
            by.setdefault(int(np.sum(gid == i)), []).append(np.where(gid == i)[0])
        for n, idx in by.items():
            I = np.array(idx); Fm, Ym = dF[I], dY[I]
            for b0 in range(0, args.draws, 1000):
                b1 = min(args.draws, b0 + 1000); P = np.argsort(rng.random((b1 - b0,) + Ym.shape), axis=2)
                null[b0:b1] += np.einsum("gs,bgs->b", Fm, np.take_along_axis(np.broadcast_to(Ym, P.shape), P, axis=2))
        o["p"] = float(np.mean(null / norm >= r - 1e-12))
    return o


def timing(C):
    dates = sorted({d for d, _ in C}); out = {}
    for k in sorted({k for _, k in C}):
        ds = [d for d in dates if (d, k) in C]
        out[str(k)] = {"n_dates": len(ds), "corr": float(np.corrcoef([C[(d, k)][0] for d in ds], [C[(d, k)][1] for d in ds])[0, 1])}
    return out


out = {"draws": args.draws, "rule": {"corr_above": THRESH_R, "p_below": THRESH_P}, "sets": {}}
for si, sname in enumerate(SETS):
    S = out["sets"][sname] = {"models": {}, "context_only": {}}
    for mi, key in enumerate(LABEL):
        rec = records(sname, key)
        if rec is None:
            print("missing", sname, key); continue
        chg = within(change(rec, rec["fc"] / rec["last"] - 1.0), np.random.default_rng([17, si, mi]), "signflip")
        ret = within(cells(rec, rec["fc"] / rec["last"] - 1.0, rec["real"] / rec["last"] - 1.0), np.random.default_rng([7, si, mi]), "perm")
        lev = within(cells(rec, rec["fc"] / rec["base"] - 1.0, rec["real"] / rec["base"] - 1.0), np.random.default_rng([19, si, mi]), "signflip")
        S["models"][key] = {"label": LABEL[key], "change_vs_target": chg, "returns_biased": ret, "levels_drift_biased": lev,
                            "timing_by_shift": timing(cells(rec, rec["fc"] / rec["last"] - 1.0, rec["real"] / rec["last"] - 1.0)),
                            "follows_calendar": bool(chg["corr"] > THRESH_R and chg["p"] < THRESH_P),
                            "n_windows": int(len(rec["fc"])), "dates": [str(min(rec["date"])), str(max(rec["date"]))]}
        if not S["context_only"]:
            for ci, (name, F) in enumerate(rec["context_only"].items()):
                S["context_only"][name] = within(change(rec, F), np.random.default_rng([29, si, ci]), "signflip")
            k = max(S["context_only"], key=lambda k: abs(S["context_only"][k]["corr"]))
            S["context_rule_max_abs"] = {"rule": k, "sign": "followed" if S["context_only"][k]["corr"] > 0 else "reversed", "abs_corr": abs(S["context_only"][k]["corr"])}
        print(f"{sname:10s} {LABEL[key]:12s} change vs target r {chg['corr']:+.2f} (p {chg['p']:.4f}) | returns r {ret['corr']:+.2f} (perm p {ret['p']:.4f})"
              f" | levels r {lev['corr']:+.2f} | follows {S['models'][key]['follows_calendar']}")
    print(f"{sname:10s} context only: " + ", ".join(f"{k} {v['corr']:+.2f}" for k, v in S["context_only"].items()) + f" | largest |r| {S['context_rule_max_abs']}")
# the unshifted (k = 0) forecasts of the anchor runs against the published ones (results/real_<model>.npz, smallcap_<model>.npz)
out["reproduction"] = {}
for tag, pub in (("anchor", "real"), ("size", "smallcap")):
    S = np.load(os.path.join(R, f"shift_{tag}_windows.npz")); sm = json.load(open(os.path.join(R, f"shift_{tag}_windows_meta.json")))
    k0 = np.array([m["shift"] == 0 for m in sm]); bi = np.array([m["base_i"] for m in sm])[k0]
    B0 = np.load(os.path.join("..", "results", f"{pub}_windows.npz"))["ctx_raw"][bi][:, -1]
    for key in LABEL:
        new_ = np.load(os.path.join(R, f"shift_{tag}", f"shift_{tag}_{key}_raw_part0_{len(sm)}.npz"))["yhat"][k0][:, -1] / B0 - 1.0
        old_ = np.load(os.path.join("..", "results", f"{pub}_{key}.npz"))["yhat_raw"][bi][:, -1] / B0 - 1.0
        out["reproduction"][f"{tag}|{key}"] = {"n": int(k0.sum()), "max_abs_diff_return": float(np.max(np.abs(new_ - old_))), "corr": float(np.corrcoef(new_, old_)[0, 1])}
print("reproduction of the published k = 0 forecasts:", {k: (round(v["max_abs_diff_return"], 6), round(v["corr"], 5)) for k, v in out["reproduction"].items()})
FOUR = ("chronosbolt", "tirex", "moirai2", "timesfm")
out["summary"] = {s: [k for k, v in out["sets"][s]["models"].items() if v["follows_calendar"]] for s in SETS}
ret_flag = {s: [k for k, v in out["sets"][s]["models"].items() if v["returns_biased"]["corr"] > THRESH_R and v["returns_biased"]["p"] < THRESH_P] for s in SETS}
out["written_predictions"] = {
    "S1_anchor_eq": {"statistic_used": not any(k in out["summary"]["anchor_eq"] for k in FOUR), "returns_as_written": not any(k in ret_flag["anchor_eq"] for k in FOUR)},
    "S2_anchor_fx": {"statistic_used": not any(k in out["summary"]["anchor_fx"] for k in FOUR), "returns_as_written": not any(k in ret_flag["anchor_fx"] for k in FOUR),
                     "flagged_by_returns": [k for k in ret_flag["anchor_fx"] if k in FOUR]},
    "S3_size_flagged": {"statistic_used": {s: out["summary"][s] for s in ("size_small", "size_large")}, "returns_as_written": {s: ret_flag[s] for s in ("size_small", "size_large")}},
    "controls_following": {s: [k for k in ("moirai", "chronos2") if k in out["summary"][s]] for s in SETS}}
json.dump(out, open(os.path.join(R, "shift_check.json"), "w"), indent=1)
print("follows the calendar:", out["summary"]); print("written predictions:", out["written_predictions"])


def num(v, d=2):
    return "0.00" if round(v, d) == 0 else f"{v:+.{d}f}".replace("-", "\\textminus{}")


def cell(o):
    return num(o["corr"]) + ("\\textsuperscript{*}" if o["p"] < THRESH_P else "")


L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & \textbf{1996--2014} & \multicolumn{2}{c}{\textbf{Daily anchor, 2017--2026}} & \multicolumn{2}{c}{\textbf{Size deciles, 1963--2026}} \\",
     r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"\textbf{Forecast} & \textbf{Equities} & \textbf{Equities} & \textbf{Exchange rates} & \textbf{Smallest} & \textbf{Largest} \\", r"\midrule"]
for key in LABEL:
    L.append(" & ".join([LABEL[key]] + [cell(out["sets"][s]["models"][key]["change_vs_target"]) if key in out["sets"][s]["models"] else "--" for s in SETS]) + r" \\")
    if key == "timesfm":
        L.append(r"\addlinespace[3pt]")
L.append(r"\addlinespace[3pt]")
L.append(" & ".join([r"Context rule, largest $|r|$"] + [f"{out['sets'][s]['context_rule_max_abs']['abs_corr']:.2f}" for s in SETS]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join("tables", "shift.tex"), "w").write("\n".join(L) + "\n")
