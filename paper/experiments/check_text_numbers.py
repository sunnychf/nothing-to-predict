"""Check that the numbers the revision's new text quotes equal the values in the result files.
Each entry: (source file, the exact LaTeX snippet that must appear, the value computed from the results, format).
Prints every mismatch and exits 1 if any. Run from finance/paper_revision after all summaries have been rerun:
  ~/miniconda3/envs/nature-figure/bin/python experiments/check_text_numbers.py"""
import json, os, re, sys

R = os.path.join("experiments", "results")
TXT = {f: open(os.path.join("sec", f)).read() for f in os.listdir("sec") if f.endswith(".tex")}
ALL = "\n".join(TXT.values()) + open(os.path.join("sec", "0_abstract.tex")).read()
fs = json.load(open(os.path.join(R, "falling_summary.json")))["models"]
tc = json.load(open(os.path.join(R, "timing_check.json")))
SC = json.load(open(os.path.join(R, "shift_check.json")))
SIM = json.load(open(os.path.join(R, "shift_bias_sim.json")))
ft = json.load(open(os.path.join(R, "finetune", "finetune_summary.json")))
cs = json.load(open(os.path.join(R, "corpus", "corpus_summary.json")))
rs = json.load(open(os.path.join("..", "results", "remedy_summary.json")))["models"]
bad, n = [], 0


def fmt(v, d, sign=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s


def need(snippet, where="any"):
    """The snippet must occur in the text; runs of whitespace (line breaks) are compared as one space."""
    global n
    n += 1
    hay = ALL if where == "any" else TXT[where]
    if re.sub(r"\s+", " ", snippet) not in re.sub(r"\s+", " ", hay):
        bad.append(snippet)


def claim(ok, what):
    """A qualitative statement in the text that must hold in the result files."""
    global n
    n += 1
    if not ok:
        bad.append("CLAIM: " + what)


def tex(x):   # LaTeX minus as used in the text
    return x.replace("+", "+").replace("-", "-")


# falling-market test
m = fs["moirai"]
need(f"${fmt(m['up']['128']['dskill'], 3)}\\pm{m['up']['128']['dskill_se']:.3f}$")
need(f"${fmt(m['down']['128']['dskill'], 3)}\\pm{m['down']['128']['dskill_se']:.3f}$")
need(f"${fmt(m['up']['128']['alignment'], 2)}$ and ${fmt(m['down']['128']['alignment'], 2)}$")
need(f"${fmt(m['up']['128']['even_mean_annualised_pct'], 1)}$ and ${fmt(m['down']['128']['even_mean_annualised_pct'], 1)}$ for Moirai-1.1")
t5 = fs["chronos"]
need(f"${fmt(t5['up']['128']['dskill'], 3)}\\pm{t5['up']['128']['dskill_se']:.3f}$")
need(f"${fmt(t5['down']['128']['dskill'], 3)}\\pm{t5['down']['128']['dskill_se']:.3f}$")
need(f"(${fmt(t5['up']['128']['alignment'], 2)}$ and ${fmt(t5['down']['128']['alignment'], 2)}$ for Chronos-T5")
need(f"${fmt(t5['up']['128']['even_mean_annualised_pct'], 1)}$ and\n${fmt(t5['down']['128']['even_mean_annualised_pct'], 1)}$ percent a year for Chronos-T5")
gap = json.load(open(os.path.join(R, "falling_summary.json")))["preregistered"]["mean_gap"]
need(f"${fmt(gap['upward'], 2)}$ for the upward-biased group and ${fmt(gap['balanced'], 2)}$ for the others")
c2 = fs["chronos2"]["up"]["128"]["even_mean_annualised_pct"], fs["chronos2"]["down"]["128"]["even_mean_annualised_pct"]
need(f"(${fmt(c2[0], 1)}$ and ${fmt(c2[1], 1)}$ percent a year)")
b, t = fs["chronosbolt"], fs["tirex"]
need(f"(${fmt(b['up']['128']['even_mean_annualised_pct'], 1)}$ against ${fmt(b['down']['128']['even_mean_annualised_pct'], 1)}$ and ${fmt(t['up']['128']['even_mean_annualised_pct'], 1)}$ against ${fmt(t['down']['128']['even_mean_annualised_pct'], 1)}$ percent a year)")
bears = [fs[k]["bear"]["128"]["skill"] for k in ("chronosbolt", "tirex", "moirai2", "timesfm")]
need(f"(${fmt(min(bears), 2)}$ to\n${fmt(max(bears), 2)}$)")
# timing
F = tc["falling"]["models"]
four = [F[k]["raw"] for k in ("chronosbolt", "tirex", "moirai2", "timesfm")]
need(f"${min(four):.2f}$ to ${max(four):.2f}$")
sl = [F[k]["raw_slope"] for k in ("chronosbolt", "tirex", "moirai2", "timesfm")]
need(f"slopes of ${min(sl):.2f}$ to ${max(sl):.2f}$")
w8 = [F[k]["raw_without_2008_11"] for k in ("chronosbolt", "tirex", "moirai2", "timesfm")]
need(f"${min(w8):.2f}$ to ${max(w8):.2f}$ without the crash")
mir = [F[k]["mirror"] for k in ("chronosbolt", "tirex", "moirai2")]
need(f"${fmt(min(mir), 2)}$ to ${fmt(max(mir), 2)}$")
rnd = [F[k]["sign_randomised"] for k in ("chronosbolt", "tirex", "moirai2")]
need(f"${min(rnd):.2f}$ to ${max(rnd):.2f}$")
need(f"(${fmt(F['timesfm']['mirror'], 2)}$)")
w = {d["asset"]: d for d in F["chronosbolt"]["2008_11_worst5"]}
need(f"falls of ${-100 * w['eq:Mines']['forecast']:.0f}$ and ${-100 * w['eq:Steel']['forecast']:.0f}$")
need(f"which fell ${-100 * w['eq:Mines']['realised']:.0f}$ and ${-100 * w['eq:Steel']['realised']:.0f}$")
A = tc["anchor_eq"]["models"]
need(f"at most ${max(v['raw'] for v in A.values()):.2f}$")
need(f"slopes of at most ${max(v['raw_slope'] for v in A.values()):.2f}$")
need(f"(up to ${max(max(v['mirror'], v['sign_randomised']) for v in A.values()):.2f}$)")
# timing: context-only rules
need(f"against at most ${tc['falling']['context_rule_max_abs']['abs_corr']:.2f}$ for drift or momentum rules")
need(f"followed or reversed, reach at most ${tc['falling']['context_rule_max_abs']['abs_corr']:.2f}$")
need(f"below what $60$-day momentum reaches there (${tc['anchor_eq']['context_rules']['Momentum 60']:.2f}$)")
claim(tc["anchor_eq"]["context_rule_max_abs"]["rule"] == "Momentum 60", "60-day momentum is the best timing rule on the anchor")
claim(min(F[k]["raw"] for k in ("chronosbolt", "tirex", "moirai2", "timesfm")) > 2 * tc["falling"]["context_rule_max_abs"]["abs_corr"], "four models time the market far better than trend rules")
# calendar-shift check (change against target)
c = lambda s_, k: SC["sets"][s_]["models"][k]["change_vs_target"]
claim(c("falling_eq", "timesfm")["p"] < 1e-3, "TimesFM-2.0 p < 10^-3 on 1996-2014")
need(f"TimesFM-2.0 scores ${fmt(c('falling_eq', 'timesfm')['corr'], 2)}$ ($p<10^{{-3}}$)")
ps = [c("falling_eq", k)["p"] for k in ("chronosbolt", "moirai2", "tirex")]
need(f"Chronos-Bolt ${fmt(c('falling_eq', 'chronosbolt')['corr'], 2)}$, Moirai-2.0 ${fmt(c('falling_eq', 'moirai2')['corr'], 2)}$ and TiRex ${fmt(c('falling_eq', 'tirex')['corr'], 2)}$ ($p$ from ${min(ps):.2f}$ to ${max(ps):.2f}$)")
need(f"Moirai-1.1 and Chronos-2 ${fmt(c('falling_eq', 'moirai')['corr'], 2)}$ and ${fmt(c('falling_eq', 'chronos2')['corr'], 2)}$, and the best context rule ${SC['sets']['falling_eq']['context_rule_max_abs']['abs_corr']:.2f}$ in absolute value")
claim(SC["sets"]["falling_eq"]["context_rule_max_abs"]["sign"] == "reversed", "the best 1996-2014 rule opposes the long-run trend")
claim(SC["summary"]["falling_eq"] == ["timesfm"], "only TimesFM-2.0 follows the calendar on 1996-2014")
claim(all(SC["summary"][s_] == [] for s_ in ("anchor_eq", "anchor_fx", "size_small", "size_large")), "no model follows the calendar on the anchors")
mx = lambda s_: max(v["change_vs_target"]["corr"] for v in SC["sets"][s_]["models"].values())
need(f"the largest correlations are ${fmt(mx('anchor_eq'), 2)}$ on the daily anchor's equities, ${fmt(mx('anchor_fx'), 2)}$ on its exchange rates and ${fmt(max(mx('size_small'), mx('size_large')), 2)}$ on the deciles")
need(f"while momentum over $60$ days reaches ${SC['sets']['anchor_eq']['context_only']['Momentum 60']['corr']:.2f}$ on the anchor's equities")
rb = lambda k: SC["sets"]["anchor_fx"]["models"][k]["returns_biased"]["corr"]
need(f"(${fmt(rb('tirex'), 2)}$ and ${fmt(rb('timesfm'), 2)}$), which the unbiased version does not reproduce (${fmt(c('anchor_fx', 'tirex')['corr'], 2)}$ and ${fmt(c('anchor_fx', 'timesfm')['corr'], 2)}$)")
claim(SC["written_predictions"]["S2_anchor_fx"]["flagged_by_returns"] == ["tirex", "timesfm"], "the return version flags TiRex and TimesFM-2.0 on exchange rates")
nA = sum(1 for _ in json.load(open(os.path.join(R, "shift_anchor_windows_meta.json")))); nS = sum(1 for _ in json.load(open(os.path.join(R, "shift_size_windows_meta.json"))))
need(f"${nA}$ daily-anchor and ${nS}$ size-decile windows per model")
rep = SC["reproduction"]
claim(all(v["max_abs_diff_return"] < 0.01 for k, v in rep.items() if not k.endswith("|moirai")) and all(v["corr"] > 0.8 for k, v in rep.items() if k.endswith("|moirai")),
      "unshifted forecasts reproduce the published ones (up to sampling for Moirai-1.1)")
# simulation of the statistic
rd = [v["change"][0] for k, v in SIM.items() if k.split("|")[0] in ("none", "constant") and k.split("|")[1] in ("trend_following", "trend_opposing", "fitted_drift")]
need(f"average between ${fmt(min(rd), 2)}$ and ${fmt(max(rd), 2)}$ whether they follow or oppose recent")
mm = [v["change"][0] for k, v in SIM.items() if k.split("|")[1] == "remembering"]
need(f"recalls the path averages ${min(mm):.2f}$ to ${max(mm):.2f}$")
need(f"then scores ${fmt(SIM['none|trend_opposing|m=20']['returns'][0], 2)}$ and one that follows it ${fmt(SIM['none|trend_following|m=20']['returns'][0], 2)}$")
sw = [abs(v["change"][0]) for k, v in SIM.items() if k.split("|")[0] == "switching" and k.split("|")[1] in ("trend_following", "trend_opposing", "fitted_drift")]
need(f"(up to ${max(sw):.2f}$ in absolute value in the")
# small-cap decile (main text now carries the numbers of the retired main-text table)
SP = json.load(open(os.path.join(R, "tier2_reanalysis.json")))["smallcap_paired"]; SM = SP["models"]
top = max(SM, key=lambda k: SM[k]["skill"])
claim(top == "moirai2", "Moirai-2.0 has the highest raw small-cap skill")
SCS = json.load(open(os.path.join("..", "results", "smallcap_summary.json")))["models"]["moirai2"]["raw|lo10"]["16"]
claim(abs(SCS["skill"] - SM[top]["skill"]) < 1e-9, "the reanalysis and the small-cap summary agree on Moirai-2.0's skill")
need(f"${fmt(SCS['skill'], 3)}\\pm{SCS['skill_se']:.3f}$ for Moirai-2.0", "6_experiments.tex")
need(f"(the largest differences are ${fmt(SM['moirai2']['minus_drift']['diff'], 3)}\\pm{SM['moirai2']['minus_drift']['se']:.3f}$ for Moirai-2.0 and ${fmt(SM['chronos2']['minus_drift']['diff'], 3)}\\pm{SM['chronos2']['minus_drift']['se']:.3f}$ for Chronos-2)", "6_experiments.tex")
srt = sorted(SM, key=lambda k: -SM[k]["minus_drift"]["diff"])
claim(srt[:2] == ["moirai2", "chronos2"] and not SP["count_above_drift_gt_2se"], "Moirai-2.0 and Chronos-2 have the largest differences, none above drift by 2 SE")
claim(sorted(SP["count_below_drift_gt_2se"]) == ["fincast", "sundial", "timemoe"], "Time-MoE, Sundial and FinCast are behind the drift by more than 2 SE")
# Figure 2 (worked example) and the appendix note on how its examples were chosen
FP = json.load(open(os.path.join("figure_design", "fig_probe_20260924", "fig_probe_facts.json")))
ab, cf = FP["a_b"], FP["c"]
need(f"Chronos-T5 forecasts a rise of ${ab['departure_h128']:.1f}\\sigma$ after the history and of ${ab['departure_mirror_h128']:.1f}\\sigma$ after its mirror")
need(f"the forecasts rise ${cf['raw_departure_h128_sigma_r']:.1f}$ and ${cf['copy_departure_h128_sigma_r']:.1f}$ return standard deviations, and the market rose ${cf['raw_realised_h128_sigma_r']:.1f}$")
need(f"closest to the median over all ${cf['n_equity_copies']}$ equity copies")
need(f"middle half of ${ab['n_contexts']}$ contexts")
need(f"{cf['ctx_end']}")
claim(ab["departure_h128"] > 0 and ab["departure_mirror_h128"] > 0 and ab["even_h128"] > 0, "the example forecast rises after the history and after its mirror")
# Figure 1 panel a (one random walk) and the arrows of panel c
FI = json.load(open(os.path.join("figure_design", "fig_intro_20260924", "fig_intro_facts.json")))
need(f"Forecasts of the eleven models ${FI['a_horizon']}$ steps ahead on one zero-drift random walk (the\nlast ${FI['a_tail_steps']}$ of ${FI['a_context_length']}$ steps shown)", "1_intro.tex")
claim(len(FI["a_departure_h128_sigma"]) == 11, "panel a shows eleven models")
claim(sorted(FI["c_clipped_below_axis"]) == ["Sundial", "Time-MoE"], "the arrows of panel c are Time-MoE and Sundial")
# Figure 18 (calendar-shift design) caption
FS = json.load(open(os.path.join("figure_design", "fig_shiftcheck_20260924", "fig_shiftcheck_facts.json")))
need(f"lies at least ${FS['gap_days']}$ days after the latest")
readers = [v for k, v in FS["rows"].items() if not k.startswith("recalls")]
claim(max(abs(v["returns"][0]) for v in readers) <= 0.51 and max(abs(v["returns"][0]) for v in readers) >= 0.45, "returns version reaches about +/-0.5 for context-only forecasters")
claim(max(abs(v["change"][0]) for v in readers) < 0.02, "statistic used is near zero for context-only forecasters")
need(f"over ${FS['rows']['fitted drift']['reps']}$ replications, ${FS['rows']['recalls the path']['reps']}$ for the two recalling forecasts")
# Section 3: N4 range restated from the appendix
need("the models capture $26$ to $65$ percent of its skill at $h{=}1$")
need("FinCast $26$ percent; the remaining six models capture $58$--$65$ percent")
# post-2025 anchor (main text and appendix)
p = rs["chronos"]["real|post2025"]
need(f"${fmt(p['raw']['128']['skill'], 3)}\\pm{p['raw']['128']['skill_se']:.3f}$")
need(f"${fmt(p['mirror']['128']['skill'], 3)}\\pm{p['mirror']['128']['skill_se']:.3f}$")
# fine-tuning
g = lambda arm, key: ft[arm][key]["mean"]
need(f"to ${g('mirror', 'N1 upward fraction, $h{=}128$'):.2f}$ against ${g('plain', 'N1 upward fraction, $h{=}128$'):.2f}$ (${g('released', 'N1 upward fraction, $h{=}128$'):.2f}$ as released)")
need(f"from ${fmt(g('plain', 'Raw equities, skill, $h{=}128$'), 3)}$ to ${fmt(g('mirror', 'Raw equities, skill, $h{=}128$'), 3)}$")
# corpus audit
rp = sorted(cs[c]["rising_pooled"] for c in cs)
need(f"${100 * rp[0]:.0f}$ to ${100 * rp[-1]:.0f}$ percent of $128$-step moves are rises")
# larger corpus models
ar = json.load(open(os.path.join(R, "arch_scale", "arch_scale_summary.json")))
eu, du, es, ds = ar["encoder|up"], ar["decoder|up"], ar["encoder|sym"], ar["decoder|sym"]
need(f"${fmt(eu['dir']['128']['mean_dep'][0], 2)}\\pm{eu['dir']['128']['mean_dep'][1]:.2f}\\sigma$ with ${100 * eu['dir']['128']['frac_up'][0]:.0f}$ percent")
need(f"${fmt(du['dir']['128']['mean_dep'][0], 2)}\\pm{du['dir']['128']['mean_dep'][1]:.2f}\\sigma$ with ${100 * du['dir']['128']['frac_up'][0]:.0f}$ percent")
g8 = [x[1] for x in eu["per_seed_dir128"] + du["per_seed_dir128"]]
need(f"between ${fmt(min(g8), 2)}$ and ${fmt(max(g8), 2)}\\sigma$")
need(f"(${fmt(es['dir']['128']['mean_dep'][0], 2)}\\pm{es['dir']['128']['mean_dep'][1]:.2f}\\sigma$, ${100 * es['dir']['128']['frac_up'][0]:.0f}$ percent")
d8 = [x[1] for x in ds["per_seed_dir128"]]
need(f"from ${fmt(min(d8), 2)}$ to ${fmt(max(d8), 2)}\\sigma$ (${fmt(ds['dir']['128']['mean_dep'][0], 2)}\\pm{ds['dir']['128']['mean_dep'][1]:.2f}$ on")
need(f"${eu['params'] / 1e6:.2f}$M parameters")
# remembered series in public training data (corpus_match_summary.py) and directional accuracy (direction_accuracy.py)
CM = json.load(open(os.path.join(R, "corpus_match", "corpus_match_summary.json")))
need(f"${sum(CM['n_scanned'].values()):,}$".replace(",", "{,}") + f" series of the\n${CM['n_subsets']}$ non-synthetic subsets")
rr = [h["r"] for h in CM["hits"] if h["r"] >= 0.9]
need(f"${CM['matched_r_ge_0.9']}$ series\nexceed $0.9$ (${min(rr):.2f}$ to ${max(rr):.2f}$)")
need(f"and ${CM['unique_robust_series']}$ of them\ndistinct")
claim(CM["matched_robust"] == CM["matched_r_ge_0.9"] and not CM["not_robust"], "every match stays above 0.85 without the five largest market days")
claim(set(CM["matched_by_subset"]) == {"chronos|m4_daily", "lotsa|m4_daily", "time300b|other/m4_daily"}, "all matches are in the copies of M4 daily")
claim(CM["first_covered_day"].startswith("1984") and CM["last_covered_day"].startswith("2017-10"), "spans run from 1984 to October 2017")
cv = round(100 * CM["covered_share"]["all"]["1996-2014"])
need(f"cover ${cv}$ percent of the trading days of 1996--2014")
need(f"on ${cv}$ percent\nof the 1996--2014 trading days and on none after October 2017", "6_experiments.tex")
claim(min(CM["falling_future_coverage"][d] for d in ("2008-05-22", "2008-11-21")) == 1.0, "the crash of 2008 is covered")
need(f"they include ${round(100 * CM['anchor_eq_future_coverage_max'])}$ percent of the future of its earliest equity end date and nothing of the\nother $17$")
an = json.load(open(os.path.join("..", "results", "real_windows_meta.json")))
claim(len({m["fut_end"] for m in an if m["family"] == "eq"}) == 18, "18 equity end dates on the anchor")
need(f"only\n${len(CM['falling_uncovered_dates'])}$ of the $37$ are uncovered, fewer than the $8$ we required")
claim(not CM["c2_testable"], "the timing split was not testable")
DA = json.load(open(os.path.join(R, "direction_accuracy.json"))); Ae = DA["sets"]["anchor"]["eq"]; Aa = DA["sets"]["anchor"]["all"]; Fe = DA["sets"]["falling"]["eq"]
au = Ae["baselines"]["128"]["always_up"]
need(f"${100 * Ae['baselines']['128']['rising_share']:.0f}$ percent\nrose over $128$ days, so always forecasting a rise scores ${100 * au['da']:.0f}\\pm{100 * au['se']:.0f}$ percent")
das = [v["128"]["raw"]["da"] for v in Ae["models"].values()]
need(f"(${100 * min(das):.0f}$ to ${100 * max(das):.0f}$ percent; Table~\\ref{{tab:direction-accuracy}})")
claim(max(das) < au["da"] and DA["predictions"]["D1_no_model_beats_always_up_by_2se"], "always up beats every model on raw equities")
need(f"(correlation ${DA['anchor_eq_128_corr_upshare_da']:.2f}$)")
claim(DA["predictions"]["D3_all"], "the mirror correction lowers all five upward models' directional accuracy")
t = Ae["models"]["tirex"]["128"]
need(f"TiRex's from ${100 * t['raw']['da']:.0f}$ to ${100 * t['mirror']['da']:.0f}$ percent")
need(f"(${100 * Aa['baselines']['128']['always_up_randomised']['da']:.0f}$ percent at $h{{=}}128$), and the models score ${100 * min(v['128']['randomised']['da'] for v in Aa['models'].values()):.0f}$ to ${100 * max(v['128']['randomised']['da'] for v in Aa['models'].values()):.0f}$ percent")
claim(sorted(k for k, v in DA["predictions"]["D2_randomised_within_2se_of_half"].items() if not v) == ["chronos", "moirai", "tirex"], "D2 fails for Chronos-T5, TiRex and Moirai-1.1")
need(f"Chronos-T5's ${100 * Aa['models']['chronos']['16']['randomised']['da']:.0f}$ percent at $h{{=}}16$")
claim(round(100 * Fe["models"]["timesfm"]["128"]["raw"]["da"]) == round(100 * Fe["models"]["chronosbolt"]["128"]["raw"]["da"]) == 72, "TimesFM-2.0 and Chronos-Bolt score 72 on 1996-2014")
need(f"${round(100 * Fe['models']['timesfm']['128']['raw']['da'])}$ percent against ${100 * Fe['baselines']['128']['always_up']['da']:.0f}$ for always forecasting a rise")
# final read-through fixes (2026-09-25)
TF = tc["falling"]["models"]; four = ("chronosbolt", "tirex", "moirai2", "timesfm")
need(f"(${TF['chronos']['raw']:.2f}$ and ${TF['moirai']['raw']:.2f}$; Table~\\ref{{tab:timing}})")
need(f"(${TF['moirai']['raw']:.2f}$ and ${TF['timemoe']['raw']:.2f}$ against ${min(TF[k]['raw'] for k in four):.2f}$ to ${max(TF[k]['raw'] for k in four):.2f}$; Table~\\ref{{tab:timing}})")
claim(TF["chronos"]["raw"] < tc["falling"]["context_rule_max_abs"]["abs_corr"] < min(TF["moirai"]["raw"], TF["timemoe"]["raw"]),
      "Chronos-T5 does not beat the best context rule; Moirai-1.1 and Time-MoE do, weakly")
need(f"with slopes of ${min(TF[k]['raw_slope'] for k in four):.2f}$ to ${max(TF[k]['raw_slope'] for k in four):.2f}$, and still ${min(TF[k]['raw_without_2008_11'] for k in four):.2f}$ to ${max(TF[k]['raw_without_2008_11'] for k in four):.2f}$")
need(f"${SP['ar1_minus_drift']['diff']:.3f}\\pm{SP['ar1_minus_drift']['se']:.3f}$ higher in the paired comparison")
eq16 = sum(m["real|eq"]["decomp"]["16"]["dskill"] > 0 for m in rs.values()); fx16 = sum(m["real|fx"]["decomp"]["16"]["dskill"] > 0 for m in rs.values())
claim(eq16 == 8 and fx16 == 7, "at h=16 the mirror correction raises equity skill for eight models and FX skill for seven")
need("raises raw-window skill on equities for eight models and on exchange rates for seven")
eq128 = [m["real|eq"]["decomp"]["128"] for m in rs.values()]
claim(sum(d["dskill"] < 0 for d in eq128) == 10 and sum(-d["dskill"] > 2 * d["dskill_se"] for d in eq128) == 4,
      "raw equities at h=128: ten of eleven lower, four beyond two paired SEs (Table remedy-decomp)")
FFR = json.load(open(os.path.join("figures", "figures_facts.json")))
sr = [FFR["real"][k]["sur"] for k in ("Chronos-small", "Chronos-Bolt-small", "Moirai-small", "TiRex")]
need("are " + ", ".join(f"${v['mean_h128']:+.1f}$" for v in sr[:3]) + f" and ${sr[3]['mean_h128']:+.1f}\\sigma_r$, with ${100 * min(v['frac_up_h128'] for v in sr):.0f}$ to ${100 * max(v['frac_up_h128'] for v in sr):.0f}$ percent of forecasts upward")
need(f"keeps ETTh1 skill at ${g('mirror', 'ETTh1 skill, $h{=}16$'):.2f}$ against ${g('plain', 'ETTh1 skill, $h{=}16$'):.2f}$")
need(f"the inference-time correction leaves ${fmt(g('plain', 'Raw equities, mirror-corrected'), 3)}$")
LA = json.load(open(os.path.join("tables", "facts.json")))["level_all"]["chronos"]["100"]["dep128"]
need(f"(for example ${LA:+.2f}$ against ${FFR['level']['+100']['mean_h128']:+.2f}$ at level $+100$)")
TZ = FFR["teaser"]
claim(TZ["series_index"] == 115, "Figure 1a is direction context 115")
need(f"context ${TZ['series_index']}$ of the $128$")
# addendum (2026-09-25): the four weaker timers of F.5
weak = ("sundial", "timemoe", "moirai", "fincast")
need("Sundial, Time-MoE, Moirai-1.1 and FinCast also exceed the best rule, but only at " + ", ".join(f"${TF[k]['raw']:.2f}$" for k in weak[:3]) + f" and ${TF['fincast']['raw']:.2f}$")
claim(all(tc["falling"]["context_rule_max_abs"]["abs_corr"] < TF[k]["raw"] < min(TF[j]["raw"] for j in four) for k in weak)
      and sorted(k for k in TF if k not in four and TF[k]["raw"] > tc["falling"]["context_rule_max_abs"]["abs_corr"]) == sorted(weak),
      "exactly Sundial, Time-MoE, Moirai-1.1 and FinCast lie between the best context rule and the four timers")
# Figure 4 (controlled corpus experiment) and its main-text numbers
FC = json.load(open(os.path.join("figure_design", "fig_corpus_20260925", "fig_corpus_facts.json")))["means"]
claim(len(FC) == 8, "Figure 4 has two designs x two corpora x two sizes")
need(f"(${100 * FC['0.54M|encoder|up']['frac_up_h128']:.0f}$ and ${100 * FC['0.54M|decoder|up']['frac_up_h128']:.0f}$ percent of\nN1 contexts)")
need(f"falls from ${fmt(FC['0.54M|encoder|up']['mean_dep_h128'], 2)}$ to ${fmt(FC['0.54M|encoder|sym']['mean_dep_h128'], 2)}\\sigma$ for the encoder")
need(f"from ${fmt(FC['0.54M|decoder|up']['mean_dep_h128'], 2)}$ to ${fmt(FC['0.54M|decoder|sym']['mean_dep_h128'], 2)}\\sigma$ for the decoder")
print(f"{n - len(bad)} of {n} quoted numbers found in the text")
for s in bad:
    print("  NOT FOUND:", s.replace("\n", "\\n"))
sys.exit(1 if bad else 0)
