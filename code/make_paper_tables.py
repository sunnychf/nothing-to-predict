"""Emit the paper's result tables as LaTeX bodies from results/ (numpy only for N4's exact optimum).

Each file under paper/tables/ is a language-neutral tabular body: both
main.tex and main_zh.tex \\input the same file under their own caption, so the
two versions cannot disagree on a number, and no number is typed by hand.
"""
import json, os, sys, math
from collections import defaultdict
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/tables"
os.makedirs(OUT, exist_ok=True)

def mean(xs): xs = list(xs); return sum(xs) / len(xs)
def rows(path):
    out = []
    for l in open(path):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        if not r.get("error"): out.append(r)
    return out
def agg(rs, field, h):
    return mean(r[field][h] for r in rs if field in r)

recs = rows(os.path.join(RES, "pilot_a.jsonl"))
lad = defaultdict(lambda: defaultdict(list))
for r in recs:
    if r["stage"] == "ladder" and r.get("init") == "pretrained":
        lad[r["model"]][r["rung"]].append(r)
RUNGS = [("N1", "N1 random walk"), ("N2", "N2 GARCH"), ("N3_nu3", r"N3 $\nu{=}3$"),
         ("N3_nu5", r"N3 $\nu{=}5$"), ("N4", "N4 microstructure")]
HS = [0, 15]      # h=1 and h=16; the prose cites only these, and four model groups do not fit wider

# ------------------------------------------------------------ ladder table (N models)
LABELS = {"amazon/chronos-t5-small": "Chronos-small", "fincast_v1": "FinCast", "timemoe-200m": "Time-MoE-200M",
          "moirai-1.1-R-small": "Moirai-small", "timesfm-2.0-500m": "TimesFM-2.0",
          "chronos-bolt-small": "Chronos-Bolt-small", "chronos-2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "timesfm25-noflip": "TimesFM-2.5 (flip off)", "sundial": "Sundial"}   # sizes are in the Setup text
MODELS = [m for m in ("amazon/chronos-t5-small", "moirai-1.1-R-small", "timesfm-2.0-500m", "timemoe-200m", "fincast_v1") if m in lad]
def dep_of(rs, h):
    return agg(rs, "departure_in_sigma_mc_corrected", h) if "departure_in_sigma_mc_corrected" in rs[0] else agg(rs, "departure_in_sigma", h)
nh, nm = len(HS), len(MODELS)
L = []
L.append(r"\begin{tabular}{l|" + "|".join("c" * nh for _ in MODELS) + "}")
L.append(r"\toprule")
L.append(" & " + " & ".join(rf"\multicolumn{{{nh}}}{{c{'|' if i < nm-1 else ''}}}{{{LABELS.get(m, m)}}}" for i, m in enumerate(MODELS)) + r" \\")
L.append("Rung & " + " & ".join(" & ".join(f"$h{{=}}{h+1}$" for h in HS) for _ in MODELS) + r" \\")
L.append(r"\midrule")
L.append(rf"\multicolumn{{{1+nh*nm}}}{{l}}{{\textit{{Departure from persistence, $\sqrt{{\Ex[(\hat y-y_t)^2]}}/\sigma$ (sampled models: Monte Carlo component removed)}}}}\\")
for key, label in RUNGS:
    L.append(f"{label} & " + " & ".join(" & ".join(f"{dep_of(lad[m].get(key, []), h):.2f}" for h in HS) for m in MODELS) + r" \\")
L.append(r"\midrule")
L.append(rf"\multicolumn{{{1+nh*nm}}}{{l}}{{\textit{{Skill against persistence, $1-\Risk(\hat y)/\Risk(y_t)$; negative is worse than doing nothing}}}}\\")
for key, label in RUNGS:
    L.append(f"{label} & " + " & ".join(" & ".join(f"${agg(lad[m].get(key, []), 'skill_vs_persistence', h):+.3f}$" for h in HS) for m in MODELS) + r" \\")
L.append(r"\bottomrule"); L.append(r"\end{tabular}")
open(os.path.join(OUT, "ladder.tex"), "w").write("\n".join(L) + "\n")
# the 2025 Chronos family (Bolt, Chronos-2), same layout, its own table so that neither is too wide
MODELS_CX = [m for m in ("chronos-bolt-small", "chronos-2", "tirex", "moirai2", "timesfm25", "sundial") if m in lad]
if MODELS_CX:
    nmx = len(MODELS_CX); C = [r"\begin{tabular}{l|" + "|".join("c" * nh for _ in MODELS_CX) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{{nh}}}{{c{'|' if i < nmx-1 else ''}}}{{{LABELS.get(m, m)}}}" for i, m in enumerate(MODELS_CX)) + r" \\",
         "Rung & " + " & ".join(" & ".join(f"$h{{=}}{h+1}$" for h in HS) for _ in MODELS_CX) + r" \\", r"\midrule",
         rf"\multicolumn{{{1+nh*nmx}}}{{l}}{{\textit{{Departure from persistence, $\sqrt{{\Ex[(\hat y-y_t)^2]}}/\sigma$ (point forecast: the median; Sundial: the sample mean, Monte Carlo component removed)}}}}\\"]
    for key, label in RUNGS:
        C.append(f"{label} & " + " & ".join(" & ".join(f"{dep_of(lad[m].get(key, []), h):.2f}" for h in HS) for m in MODELS_CX) + r" \\")
    C += [r"\midrule", rf"\multicolumn{{{1+nh*nmx}}}{{l}}{{\textit{{Skill against persistence, $1-\Risk(\hat y)/\Risk(y_t)$}}}}\\"]
    for key, label in RUNGS:
        C.append(f"{label} & " + " & ".join(" & ".join(f"${agg(lad[m].get(key, []), 'skill_vs_persistence', h):+.3f}$" for h in HS) for m in MODELS_CX) + r" \\")
    C += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "ladder_cx.tex"), "w").write("\n".join(C) + "\n")
n4c = lad["amazon/chronos-t5-small"]["N4"]; n4f = lad["fincast_v1"]["N4"]
# N4's optimum. The drivers stored the optimal *linear* forecast's skill (MA(1) filter, oracle_skill_vs_persistence);
# the exact conditional mean (two-state filter on the trade directions, nulls.oracle_forecast_exact) is
# recomputed here on the same draws, and every share below is taken against it; the linear one is kept for the record.
from nulls import n4_optima_skill
_opt = {r["seed"]: n4_optima_skill(r["n"], 512, r["H"], 1000 + r["seed"]) for r in n4c}
for r in n4c:                                                    # the recomputation must reproduce the stored linear skill
    assert abs(_opt[r["seed"]]["linear"][0] - r["oracle_skill_vs_persistence"][0]) < 1e-9, (r["seed"], _opt[r["seed"]]["linear"][0], r["oracle_skill_vs_persistence"][0])
facts = {
  "codec_grid_N1_sigma": math.sqrt(agg(lad["amazon/chronos-t5-small"]["N1"], "codec_floor", 0)) / lad["amazon/chronos-t5-small"]["N1"][0]["sigma"],
  "n4_oracle_skill_h1": mean(_opt[sd]["exact"][0] for sd in _opt),          # the exact optimum: the reference for every share
  "n4_linear_skill_h1": agg(n4c, "oracle_skill_vs_persistence", 0),          # the optimal linear forecast, for the record
  "n4_oracle_skill_h16": mean(_opt[sd]["exact"][15] for sd in _opt),
  "n4_linear_skill_h16": mean(_opt[sd]["linear"][15] for sd in _opt),
  "n4_chronos_skill_h1": agg(n4c, "skill_vs_persistence", 0),
  "n4_fincast_skill_h1": agg(n4f, "skill_vs_persistence", 0),
  "n_series": lad["amazon/chronos-t5-small"]["N1"][0]["n"],
  "seeds": len(lad["amazon/chronos-t5-small"]["N1"]),
  "S": lad["amazon/chronos-t5-small"]["N1"][0]["num_samples"],
  "models_in_ladder": MODELS,
}
facts["n4_exact_over_linear_h1"] = facts["n4_oracle_skill_h1"] / facts["n4_linear_skill_h1"]
for m in MODELS + [m for m in ("chronos-bolt-small", "chronos-2", "tirex", "moirai2", "timesfm25", "sundial") if m in lad]:
    if m not in ("amazon/chronos-t5-small", "fincast_v1") and "N4" in lad[m]:
        facts[f"n4_{m}_pct"] = 100 * agg(lad[m]["N4"], "skill_vs_persistence", 0) / facts["n4_oracle_skill_h1"]
        facts[f"n4_{m}_pct_linear"] = 100 * agg(lad[m]["N4"], "skill_vs_persistence", 0) / facts["n4_linear_skill_h1"]
facts["n4_chronos_pct"] = 100 * facts["n4_chronos_skill_h1"] / facts["n4_oracle_skill_h1"]

# ------------------------------------------------------------ main-text ladder table (M31, 09-23): all eleven models in
# one table, rows grouped by the direction split of Section 6.3; same records and aggregation as ladder.tex/ladder_cx.tex.
GROUPS_MAIN = [("Not decoder-only", ["amazon/chronos-t5-small", "chronos-bolt-small", "chronos-2", "tirex", "moirai-1.1-R-small"]),
               ("Decoder-only, general corpus", ["moirai2", "timesfm-2.0-500m", "timesfm25", "timemoe-200m", "sundial"]),
               ("Decoder-only, financial corpus", ["fincast_v1"])]
MAIN_NAMES = {"amazon/chronos-t5-small": "Chronos-T5", "chronos-bolt-small": "Chronos-Bolt", "chronos-2": "Chronos-2", "tirex": "TiRex",
              "moirai-1.1-R-small": "Moirai-1.1", "moirai2": "Moirai-2.0", "timesfm-2.0-500m": "TimesFM-2.0", "timesfm25": "TimesFM-2.5",
              "timemoe-200m": "Time-MoE", "sundial": "Sundial", "fincast_v1": "FinCast"}
def _sk(m, key, h): return agg(lad[m].get(key, []), "skill_vs_persistence", h)
M = [r"\begin{tabular}{l|cc|cccc|c}", r"\toprule",
     r" & \multicolumn{2}{c|}{N1 departure$/\sigma$} & \multicolumn{4}{c|}{Skill at $h{=}16$} & N4 share \\",
     r"Model & $h{=}1$ & $h{=}16$ & N1 & N2 & N3 $\nu{=}3$ & N3 $\nu{=}5$ & $h{=}1$ \\"]
facts["ladder_main"] = {}
for gname, ms in GROUPS_MAIN:
    ms = [m for m in ms if m in lad]
    M += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\textit{{{gname}}}}} \\"]
    for m in ms:
        d1, d16 = dep_of(lad[m]["N1"], 0), dep_of(lad[m]["N1"], 15)
        sk = [_sk(m, k, 15) for k in ("N1", "N2", "N3_nu3", "N3_nu5")]
        share = 100 * _sk(m, "N4", 0) / facts["n4_oracle_skill_h1"]
        facts["ladder_main"][m] = {"dep_h1": d1, "dep_h16": d16, "skill_h16": sk, "n4_share_h1_pct": share}
        M.append(f"{MAIN_NAMES[m]} & {d1:.2f} & {d16:.2f} & " + " & ".join(f"${v:+.3f}$" for v in sk) + f" & {share:.0f}\\% \\\\")
M += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "ladder_main.tex"), "w").write("\n".join(M) + "\n")
facts["n4_fincast_pct"] = 100 * facts["n4_fincast_skill_h1"] / facts["n4_oracle_skill_h1"]
facts["n4_chronos_pct_linear"] = 100 * facts["n4_chronos_skill_h1"] / facts["n4_linear_skill_h1"]
facts["n4_fincast_pct_linear"] = 100 * facts["n4_fincast_skill_h1"] / facts["n4_linear_skill_h1"]

# ------------------------------------------------------------ controls (numbers only)
mc = sorted([r for r in recs if r["stage"] == "floor_mc"], key=lambda r: r["num_samples"])
S = [r["num_samples"] for r in mc]; D = [r["imported"][0] for r in mc]; sig = mc[0]["sigma"]
# least squares D = a + b/S
x = [1 / s for s in S]; xm = mean(x); dm = mean(D)
b = sum((xi - xm) * (di - dm) for xi, di in zip(x, D)) / sum((xi - xm) ** 2 for xi in x); a = dm - b * xm
pred = [a + b * xi for xi in x]
r2 = 1 - sum((di - pi) ** 2 for di, pi in zip(D, pred)) / sum((di - dm) ** 2 for di in D)
facts["mc_dinf_sigma"] = math.sqrt(max(a, 0)) / sig; facts["mc_r2"] = r2
facts["mc_S100_sigma"] = math.sqrt(D[S.index(100)]) / sig
sc = sorted([r for r in recs if r["stage"] == "floor_scale"], key=lambda r: r["sigma_rel"])
facts["scale"] = [(r["sigma_rel"], r["departure_in_sigma_mc_corrected"][0] / (math.sqrt(r["codec_floor"][0]) / r["sigma"]),
                   r["departure_in_sigma_mc_corrected"][-1] / (math.sqrt(r["codec_floor"][0]) / r["sigma"])) for r in sc]
lv = [r for r in recs if r["stage"] == "floor_level"]
facts["level_vals"] = sorted({round(r["departure_in_sigma_mc_corrected"][0], 4) for r in lv})
disp = json.load(open(os.path.join(RES, "diag_dispersion.json")))
facts["top1pct_share"] = mean(d["share_top1pct"] for d in disp)
rnd = [r for r in recs if r["stage"] == "ladder" and r.get("init") == "random" and r["rung"] == "N1"]
facts["random_init_dep_h1"] = agg(rnd, "departure_in_sigma", 0)
n2 = json.load(open(os.path.join(RES, "diag_n2_interval.json")))
facts["n2_corr_h1"] = [r["corr_width_true_condvol"][0] for r in n2]
facts["n2_cov_h1"] = mean(r["coverage_80"][0] for r in n2)
facts["n2_frac_not_horizon"] = mean(r["frac_width_var_not_explained_by_horizon"] for r in n2)
sp = {H: json.load(open(os.path.join(RES, f"spectral_deep_H{H}.json"))) for H in (128, 256)}
facts["drift_share_H256"] = {d["rung"]: d["share_drift_periodGtH4"] for d in sp[256]}
facts["band24_H128"] = {d["rung"]: d["band24"]["excess_over_continuum"] for d in sp[128]}
facts["band24_H256"] = {d["rung"]: d["band24"]["excess_over_continuum"] for d in sp[256]}
facts["max_band_H256"] = max(d[f"band{p}"]["excess_over_continuum"] for d in sp[256] for p in (7, 12, 24))

# ------------------------------------------------------------ direction table
ctrl = json.load(open(os.path.join(RES, "diag_direction_control.json")))
fdir = json.load(open(os.path.join(RES, "diag_direction_fincast.json")))
ffreq = json.load(open(os.path.join(RES, "diag_freq_direction_fincast.json")))
D = []
D.append(r"\begin{tabular}{ll|rr|rr}")
D.append(r"\toprule")
D.append(r"Model & Condition & \multicolumn{2}{c|}{$h{=}16$} & \multicolumn{2}{c}{$h{=}128$} \\")
D.append(r" & & mean $/\sigma$ & frac.\ up & mean $/\sigma$ & frac.\ up \\")
D.append(r"\midrule")
for tag, lab in (("pos", r"N1, level $+100$"), ("neg", r"mirror of the above"), ("zero", "N1, centred on zero"), ("mirror", r"mirror of the above")):
    c = ctrl[tag]
    D.append(f"Chronos-small & {lab} & ${c['mean']['16']:+.2f}$ & {c['frac_up']['16']:.2f} & ${c['mean']['128']:+.2f}$ & {c['frac_up']['128']:.2f} \\\\")
D.append(r"\midrule")
_mp = os.path.join(RES, "diag_direction_moirai.json")
mdir = json.load(open(_mp)) if os.path.exists(_mp) else None
if mdir:
    mn1 = next(r for r in mdir if r["rung"] == "N1")
    D.append(f"Moirai-small & N1, patch auto (default) & ${mn1['mean_dep_sigma']['16']:+.2f}$ & {mn1['frac_up']['16']:.2f} & ${mn1['mean_dep_sigma']['128']:+.2f}$ & {mn1['frac_up']['128']:.2f} \\\\")
    for r in sorted((x for x in recs if x["stage"] == "freq" and x.get("model") == "moirai-1.1-R-small"), key=lambda x: x["freq"]):
        band = {8: "yearly/quarterly", 16: "weekly/daily", 32: "daily/hourly", 64: "hourly/minute", 128: "minute/second"}[r["freq"]]
        D.append(f"Moirai-small & N1, patch {r['freq']} ({band}) & ${r['departure_mean_signed_sigma'][15]:+.2f}$ & {r['frac_departure_up'][15]:.2f} & ${r['departure_mean_signed_sigma'][-1]:+.2f}$ & {r['frac_departure_up'][-1]:.2f} \\\\")
    D.append(r"\midrule")
_tp = os.path.join(RES, "diag_direction_timesfm.json")
if os.path.exists(_tp):
    tn1 = next(r for r in json.load(open(_tp)) if r["rung"] == "N1")
    D.append(f"TimesFM-2.0 & N1, default (freq 0) & ${tn1['mean_dep_sigma']['16']:+.2f}$ & {tn1['frac_up']['16']:.2f} & ${tn1['mean_dep_sigma']['128']:+.2f}$ & {tn1['frac_up']['128']:.2f} \\\\")
    for r in sorted((x for x in recs if x["stage"] == "freq" and x.get("model") == "timesfm-2.0-500m"), key=lambda x: x["freq"]):
        lab = {0: "declared $\\le$ daily", 1: "declared weekly/monthly", 2: "declared quarterly/yearly"}[r["freq"]]
        D.append(f"TimesFM-2.0 & N1, {lab} & ${r['departure_mean_signed_sigma'][15]:+.2f}$ & {r['frac_departure_up'][15]:.2f} & ${r['departure_mean_signed_sigma'][-1]:+.2f}$ & {r['frac_departure_up'][-1]:.2f} \\\\")
    D.append(r"\midrule")
_mo = os.path.join(RES, "diag_direction_timemoe.json")
if os.path.exists(_mo):
    mon1 = next(r for r in json.load(open(_mo)) if r["rung"] == "N1")
    D.append(f"Time-MoE-200M & N1, default (no declaration) & ${mon1['mean_dep_sigma']['16']:+.2f}$ & {mon1['frac_up']['16']:.2f} & ${mon1['mean_dep_sigma']['128']:+.2f}$ & {mon1['frac_up']['128']:.2f} \\\\")
    D.append(r"\midrule")
    facts["timemoe_direction_N1"] = mon1
for _short, _lab in (("bolt", "Chronos-Bolt-small"), ("chronos2", "Chronos-2"), ("tirex", "TiRex"), ("moirai2", "Moirai-2.0"), ("timesfm25", "TimesFM-2.5"), ("timesfm25-noflip", "TimesFM-2.5 (flip off)"), ("sundial", "Sundial")):
    _cp = os.path.join(RES, f"diag_direction_{_short}.json")
    if os.path.exists(_cp):
        _c1 = next(r for r in json.load(open(_cp)) if r["rung"] == "N1")
        D.append(f"{_lab} & N1, default (no declaration) & ${_c1['mean_dep_sigma']['16']:+.2f}$ & {_c1['frac_up']['16']:.2f} & ${_c1['mean_dep_sigma']['128']:+.2f}$ & {_c1['frac_up']['128']:.2f} \\\\")
        D.append(r"\midrule"); facts[f"{_short}_direction_N1"] = _c1
fn1 = next(r for r in fdir if r["rung"] == "N1" and r["H"] == 128)
D.append(f"FinCast & N1, default & ${fn1['mean_dep_sigma']['16']:+.2f}$ & {fn1['frac_up']['16']:.2f} & ${fn1['mean_dep_sigma']['128']:+.2f}$ & {fn1['frac_up']['128']:.2f} \\\\")
for r in ffreq:
    lab = {0: "declared $\\le$ daily", 1: "declared weekly/monthly", 2: "declared yearly"}[r["freq"]]
    D.append(f"FinCast & N1, {lab} & ${r['mean_dep_sigma']['16']:+.2f}$ & {r['frac_up']['16']:.2f} & ${r['mean_dep_sigma']['128']:+.2f}$ & {r['frac_up']['128']:.2f} \\\\")
D.append(r"\bottomrule")
D.append(r"\end{tabular}")
open(os.path.join(OUT, "direction.tex"), "w").write("\n".join(D) + "\n")
facts["asym_pos_neg_128"] = ctrl["pos+neg"]["asym_mean"]["128"]; facts["asym_zero_mirror_128"] = ctrl["zero+mirror"]["asym_mean"]["128"]
facts["fincast_freq_rms_diff"] = next(r for r in recs if r["stage"] == "freq_delta")

# ------------------------------------------------------------ scale table
scl = rows(os.path.join(RES, "scale_chronos.jsonl"))
by = {(r["size"], r["rung"]): r for r in scl}
T = []
T.append(r"\begin{tabular}{l|cc|c|cc|c}")
T.append(r"\toprule")
T.append(r"Chronos-T5 & \multicolumn{2}{c|}{N1 skill} & N1 mean $/\sigma$ & \multicolumn{2}{c|}{N1 frac.\ up} & N4 skill \\")
T.append(r"size & $h{=}1$ & $h{=}64$ & $h{=}64$ & $h{=}16$ & $h{=}64$ & \% of optimum \\")
T.append(r"\midrule")
_sc_opt = n4_optima_skill(128, 512, 64, 4000)                    # the sweep's own N4 draw (diag_scale_chronos.py: rng(4000), n=128, H=64)
assert abs(_sc_opt["linear"][0] - by[("46M", "N4")]["oracle_skill"][0]) < 1e-9
facts["scale_n4_oracle_skill_h1"] = _sc_opt["exact"][0]; facts["scale_n4_linear_skill_h1"] = _sc_opt["linear"][0]
for sz in ("8M", "20M", "46M", "200M", "710M"):
    a = by[(sz, "N1")]; b = by[(sz, "N4")]
    T.append(f"{sz} & ${a['skill'][0]:+.3f}$ & ${a['skill'][-1]:+.3f}$ & ${a['mean_dep_sigma'][-1]:+.2f}$ & {a['frac_up'][15]:.2f} & {a['frac_up'][-1]:.2f} & {100*b['skill'][0]/_sc_opt['exact'][0]:.0f} \\\\")
T.append(r"\bottomrule")
T.append(r"\end{tabular}")
open(os.path.join(OUT, "scale.tex"), "w").write("\n".join(T) + "\n")
facts["scale_rows"] = {sz: {"skill_h1": by[(sz,"N1")]["skill"][0], "frac_up_64": by[(sz,"N1")]["frac_up"][-1],
                            "mean_64": by[(sz,"N1")]["mean_dep_sigma"][-1], "n4_pct": 100*by[(sz,"N4")]["skill"][0]/_sc_opt["exact"][0], "n4_pct_linear": 100*by[(sz,"N4")]["skill"][0]/by[(sz,"N4")]["oracle_skill"][0],
                            "corr_slope_64": by[(sz,"N1")]["corr_slope64"][-1]} for sz in ("8M","20M","46M","200M","710M")}
# ---- context-length sweep (diag_context.py): rows = (model, metric), columns = context length
CTX_LABEL = {"chronos": "Chronos-small", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-small", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5",
             "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "sundial": "Sundial", "fincast": "FinCast"}
ctx_files = [(m, os.path.join(RES, f"diag_context_{m}.json")) for m in CTX_LABEL]
ctx = {m: json.load(open(p)) for m, p in ctx_files if os.path.exists(p)}
if ctx:
    lens = sorted({r["T"] for d in ctx.values() for r in d["records"]})
    C = [r"\begin{tabular}{ll|" + "r" * len(lens) + "}", r"\toprule",
         r"Model & context $T$ & " + " & ".join(f"${t}$" for t in lens) + r" \\", r"\midrule"]
    facts["context"] = {}
    for m, d in ctx.items():
        by = {r["T"]: r for r in d["records"]}
        cell = lambda t, key, h, fmt: (fmt(by[t][key][h]) if t in by else "--")
        C.append(f"{CTX_LABEL[m]} & mean dep. $h{{=}}128$ / $\\sigma$ & " + " & ".join((f"${by[t]['mean_dep']['128']:+.2f}\\pm{by[t]['mean_dep_se']['128']:.2f}$" if t in by else "--") for t in lens) + r" \\")
        C.append(f" & fraction up, $h{{=}}128$ & " + " & ".join(cell(t, "frac_up", "128", lambda v: f"${v:.2f}$") for t in lens) + r" \\")
        C.append(f" & skill $h{{=}}16$ & " + " & ".join(cell(t, "skill", "16", lambda v: f"${v:+.3f}$") for t in lens) + r" \\")
        C.append(r"\midrule")
        facts["context"][m] = {str(t): {"dep128": by[t]["mean_dep"]["128"], "dep128_se": by[t]["mean_dep_se"]["128"], "up128": by[t]["frac_up"]["128"],
                                        "skill16": by[t]["skill"]["16"], "skill128": by[t]["skill"]["128"], "dep16": by[t]["mean_dep"]["16"]} for t in by}
    C[-1] = r"\bottomrule"; C.append(r"\end{tabular}")
    open(os.path.join(OUT, "context.tex"), "w").write("\n".join(C) + "\n")
# ---- level / shape decoupling (diag_level_all.py): the same increments at five raw levels, per model
LV_ORDER = [("chronos", "Chronos-small"), ("chronosbolt", "Chronos-Bolt"), ("chronos2", "Chronos-2"), ("tirex", "TiRex"), ("moirai", "Moirai-small"),
            ("moirai2", "Moirai-2.0"), ("timesfm", "TimesFM-2.0"), ("timesfm25", "TimesFM-2.5"), ("timemoe", "Time-MoE-200M"), ("sundial", "Sundial"), ("fincast", "FinCast")]
lv = {m: json.load(open(os.path.join(RES, f"diag_level_{m}.json"))) for m, _ in LV_ORDER if os.path.exists(os.path.join(RES, f"diag_level_{m}.json"))}
if lv:
    def sci(x):
        mant, ex = f"{x:.1e}".split("e"); return f"${mant}\\times10^{{{int(ex)}}}$" if x > 0 else "$0$"
    levels = next(iter(lv.values()))["levels"]
    V = [r"\begin{tabular}{l|" + "r" * len(levels) + "|" + "r" * len(levels) + "|r}", r"\toprule",
         r"Model & \multicolumn{" + str(len(levels)) + r"}{c|}{mean departure at $h{=}128$ / $\sigma$, by level} & \multicolumn{" + str(len(levels)) + r"}{c|}{fraction up at $h{=}128$} & max $|\Delta|$ \\",
         " & " + " & ".join(f"${int(l):+d}$" for l in levels) + " & " + " & ".join(f"${int(l):+d}$" for l in levels) + r" & vs $+100$ \\", r"\midrule"]
    facts["level_all"] = {}
    for m, lab in LV_ORDER:
        if m not in lv: continue
        by = {r["level"]: r for r in lv[m]["records"]}
        V.append(f"{lab} & " + " & ".join(f"${by[l]['mean_dep']['128']:+.2f}$" for l in levels) + " & " + " & ".join(f"{by[l]['frac_up']['128']:.2f}" for l in levels)
                 + " & " + sci(max(r['max_abs_diff_from_level100'] for r in lv[m]['records'])) + r" \\")
        facts["level_all"][m] = {str(int(l)): {"dep128": by[l]["mean_dep"]["128"], "up128": by[l]["frac_up"]["128"], "dep16": by[l]["mean_dep"]["16"]} for l in levels}
        facts["level_all"][m]["max_abs_diff_from_level100"] = max(r["max_abs_diff_from_level100"] for r in lv[m]["records"])
    V += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "level_all.tex"), "w").write("\n".join(V) + "\n")
# ---- the Time-MoE decoding loop on the model's own ETTh1 benchmark (validate_timemoe.py)
import glob as _glob
_vts = sorted(_glob.glob(os.path.join(RES, "validate_timemoe_etth1*.json")))
if _vts:
    vt = {"horizons": {}, "average": {}}
    for _p in _vts:                                            # one file per horizon (parallel runs) or one file with all
        _d = json.load(open(_p)); vt["horizons"].update(_d["horizons"]); vt["paper_timemoe_large"] = _d["paper_timemoe_large"]
    hs = sorted(vt["horizons"], key=int)
    from statistics import mean as _mean
    _pw = [h for h in hs if "per_window_normalisation" in vt["horizons"][h]]        # decoding (b) may be absent (--official-only)
    vt["average"] = {k: {"mse": _mean(vt["horizons"][h][k]["mse"] for h in hs), "mae": _mean(vt["horizons"][h][k]["mae"] for h in hs)} for k in ("official_protocol", "paper")}
    vt["average"]["per_window_normalisation"] = {"mse": _mean(vt["horizons"][h]["per_window_normalisation"]["mse"] for h in _pw), "mae": _mean(vt["horizons"][h]["per_window_normalisation"]["mae"] for h in _pw)} if len(_pw) == len(hs) else None
    vt["average"]["paper"] = {"mse": vt["paper_timemoe_large"]["avg"][0], "mae": vt["paper_timemoe_large"]["avg"][1]} if len(hs) == 4 else vt["average"]["paper"]
    full = len(hs) == 4
    W = [r"\begin{tabular}{l|" + ("cc|" * (len(hs) - 1) + ("cc|cc" if full else "cc")) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c{'|' if (full or j < len(hs) - 1) else ''}}}{{$h{{=}}{h}$ (${vt['horizons'][h]['context']}$)}}" for j, h in enumerate(hs)) + (r" & \multicolumn{2}{c}{average}" if full else "") + r" \\",
         "Decoding & " + " & ".join("MSE & MAE" for _ in hs) + (r" & MSE & MAE" if full else "") + r" \\", r"\midrule"]
    rows = [("paper (Time-MoE$_{\\text{large}}$)", [vt["horizons"][h]["paper"] for h in hs] + ([vt["average"]["paper"]] if full else [])),
            ("our loop, official protocol", [vt["horizons"][h]["official_protocol"] for h in hs] + ([vt["average"]["official_protocol"]] if full else [])),
            ("our loop, per-window normalisation", [vt["horizons"][h].get("per_window_normalisation") for h in hs] + ([vt["average"]["per_window_normalisation"]] if full else []))]
    for lab, cells in rows:
        W.append(lab + " & " + " & ".join(f"{c['mse']:.3f} & {c['mae']:.3f}" if c else "-- & --" for c in cells) + r" \\")
    W += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "timemoe_check.tex"), "w").write("\n".join(W) + "\n")
    facts["timemoe_check"] = {"horizons": {h: {"n_windows": vt["horizons"][h]["n_windows"], "paper": vt["horizons"][h]["paper"], "official": vt["horizons"][h]["official_protocol"],
                                              "per_window": vt["horizons"][h].get("per_window_normalisation")} for h in hs}, "average": vt["average"]}
# ---- calibration under the null: 80 percent interval coverage and width on N1 and N2, h=1 and h=16.
# The drivers record `coverage_80` / `interval80_width_over_sigma` in their ladder rows (0.1-0.9 quantile
# band: sample quantiles for the sampled models, the model's own quantiles for the quantile models,
# FinCast's and TimesFM-2.0's quantile heads); Chronos-T5's come from diag_coverage_chronos.py, the same
# contexts and seeds; Time-MoE has no interval. The martingale's own 80 percent band is 2 * 1.2816 * sqrt(h).
COV_ORDER = [("amazon/chronos-t5-small", "Chronos-small"), ("chronos-bolt-small", "Chronos-Bolt-small"), ("chronos-2", "Chronos-2"),
             ("tirex", "TiRex"), ("moirai-1.1-R-small", "Moirai-small"), ("moirai2", "Moirai-2.0"), ("timesfm-2.0-500m", "TimesFM-2.0"),
             ("timesfm25", "TimesFM-2.5"), ("sundial", "Sundial"), ("fincast_v1", "FinCast")]
cov_rows = {m: {r: list(rs) for r, rs in lad[m].items() if rs and "coverage_80" in rs[0]} for m, _ in COV_ORDER if m in lad}
_cc = os.path.join(RES, "diag_coverage_chronos.json")
if os.path.exists(_cc):
    _rows = json.load(open(_cc)); cov_rows["amazon/chronos-t5-small"] = {}
    for r in _rows:
        cov_rows["amazon/chronos-t5-small"].setdefault(r["rung"], []).append(r)
cov_rows = {m: v for m, v in cov_rows.items() if v}
if cov_rows:
    from statistics import mean as _mean
    NORM = 2 * 1.2815515655446004                                        # 80 percent band of a unit-variance Gaussian
    V = [r"\begin{tabular}{l|cc|cc|cc|cc}", r"\toprule",
         r" & \multicolumn{4}{c|}{N1 random walk} & \multicolumn{4}{c}{N2 GARCH} \\",
         r" & \multicolumn{2}{c|}{coverage} & \multicolumn{2}{c|}{width$/(2.56\,\sigma\sqrt{h})$} & \multicolumn{2}{c|}{coverage} & \multicolumn{2}{c}{width$/(2.56\,\sigma\sqrt{h})$} \\",
         r"Model & $h{=}1$ & $h{=}16$ & $h{=}1$ & $h{=}16$ & $h{=}1$ & $h{=}16$ & $h{=}1$ & $h{=}16$ \\", r"\midrule"]
    facts["coverage"] = {}
    for m, lab in COV_ORDER:
        if m not in cov_rows: continue
        cells = []; facts["coverage"][m] = {}
        for rung in ("N1", "N2"):
            rs = cov_rows[m].get(rung, [])
            if not rs: cells += ["--"] * 4; continue
            c1, c16 = _mean(r["coverage_80"][0] for r in rs), _mean(r["coverage_80"][15] for r in rs)
            w1, w16 = _mean(r["interval80_width_over_sigma"][0] for r in rs) / NORM, _mean(r["interval80_width_over_sigma"][15] for r in rs) / (NORM * 4)
            cells += [f"{c1:.2f}", f"{c16:.2f}", f"{w1:.2f}", f"{w16:.2f}"]
            facts["coverage"][m][rung] = {"cov_h1": c1, "cov_h16": c16, "width_ratio_h1": w1, "width_ratio_h16": w16, "seeds": len(rs)}
        V.append(f"{lab} & " + " & ".join(cells) + r" \\")
    V += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "coverage.tex"), "w").write("\n".join(V) + "\n")
# ---- the sample cross term of eq:decomp as a share of the irreducible term (the prose quotes the maxima):
# seeds are pooled by averaging cross and irreducible separately, then dividing; h = 1 and 16 (the table's horizons).
_cross = {}
for m in list(MODELS) + list(MODELS_CX):
    per = {}
    for key, _lab in RUNGS:
        rs = lad[m].get(key, [])
        if not rs or "cross" not in rs[0]: continue
        per[key] = {str(h + 1): abs(agg(rs, "cross", h)) / agg(rs, "irreducible", h) for h in HS}
    if per:
        null_vals = [(v, key, h) for key, d in per.items() if key != "N4" for h, v in d.items()]
        _cross[LABELS.get(m, m)] = {"per_rung": per, "n1n3_max": max(null_vals), "n4_h1": per["N4"]["1"] if "N4" in per else None}
facts["cross_share"] = _cross
facts["cross_share_n1n3_max_over_models"] = max((v["n1n3_max"][0], k) for k, v in _cross.items())
facts["cross_share_n1n3_max_excluding_max_model"] = sorted(((v["n1n3_max"][0], k) for k, v in _cross.items()), reverse=True)[1]
facts["cross_share_n4_h1_range"] = (min(v["n4_h1"] for v in _cross.values() if v["n4_h1"] is not None), max(v["n4_h1"] for v in _cross.values() if v["n4_h1"] is not None))
print("  cross share of irreducible, N1-N3 max:", {k: f"{v['n1n3_max'][0]*100:.2f}% ({v['n1n3_max'][1]}, h={v['n1n3_max'][2]})" for k, v in _cross.items()})
print("  cross share on N4 at h=1:", {k: f"{v['n4_h1']*100:.1f}%" for k, v in _cross.items()})
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=1, default=float)
print("wrote", sorted(os.listdir(OUT)))
for k in ("codec_grid_N1_sigma","mc_dinf_sigma","mc_r2","n4_chronos_pct","n4_fincast_pct","top1pct_share","random_init_dep_h1","max_band_H256","asym_pos_neg_128","asym_zero_mirror_128"):
    print(f"  {k}: {facts[k]:.4g}")
print("  scale (sr, h1 ratio, h16 ratio):", [(s, round(a,2), round(b,1)) for s,a,b in facts["scale"]])
print("  n2 corr h1:", [round(v,3) for v in facts["n2_corr_h1"]], " level vals:", facts["level_vals"])
