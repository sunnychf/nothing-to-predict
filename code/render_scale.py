"""Render the Chronos scale sweep (results/scale_chronos.jsonl) as the §4.11 table."""
import json, sys
J = sys.argv[1] if len(sys.argv) > 1 else "results/scale_chronos.jsonl"
rows = [json.loads(l) for l in open(J) if l.strip()]
order = ["8M", "20M", "46M", "200M", "710M"]
by = {(r["size"], r["rung"]): r for r in rows}
L = []
P = L.append
P("### 4.11 模型尺寸：先验随规模变大是缩小还是放大\n")
P("五个 Chronos-T5 尺寸，同一批 N1 / N4 上下文（seed 4000），H=64，n=128，S=100。")
P("`dep-MC` 为 MC 修正偏离，`grid` 为码本分辨率（五个尺寸共用同一 tokenizer，所以相同）。\n")
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nulls import n4_optima_skill                       # N4 的精确最优（条件均值）；存储的 oracle_skill 是线性 MA(1) 最优
_opt = n4_optima_skill(128, 512, 64, 4000); assert abs(_opt["linear"][0] - by[("46M", "N4")]["oracle_skill"][0]) < 1e-9
P("| 尺寸 | dep-MC h=1 | dep-MC h=64 | grid | skill h=1 | skill h=64 | 带符号均值 h=64 | 向上比例 h=64 | corr(斜率) h=64 | N4 抓到的最优结构（精确最优 / 线性最优） |")
P("|---|---|---|---|---|---|---|---|---|---|")
for sz in order:
    a = by.get((sz, "N1")); b = by.get((sz, "N4"))
    if not a:
        P(f"| {sz} | 未跑 | | | | | | | | |"); continue
    n4 = f"{100*b['skill'][0]/_opt['exact'][0]:.0f}% / {100*b['skill'][0]/b['oracle_skill'][0]:.0f}%" if b else "—"
    P(f"| {sz} | {a['dep_mc_sigma'][0]:.3f}σ | {a['dep_mc_sigma'][-1]:.2f}σ | {a['grid_sigma'][0]:.3f}σ | "
      f"{a['skill'][0]:+.3f} | {a['skill'][-1]:+.3f} | **{a['mean_dep_sigma'][-1]:+.2f}σ** | **{a['frac_up'][-1]:.2f}** | "
      f"{a['corr_slope64'][-1]:+.2f} | {n4} |")
P("")
have = [sz for sz in order if (sz, "N1") in by]
if len(have) >= 3:
    ups = [by[(sz, "N1")]["frac_up"][-1] for sz in have]
    means = [by[(sz, "N1")]["mean_dep_sigma"][-1] for sz in have]
    deps = [by[(sz, "N1")]["dep_mc_sigma"][-1] for sz in have]
    sk = [by[(sz, "N1")]["skill"][0] for sz in have]
    P(f"从 {have[0]} 到 {have[-1]}：h=64 向上比例 {ups[0]:.2f} → {ups[-1]:.2f}，带符号均值 {means[0]:+.2f}σ → {means[-1]:+.2f}σ，")
    P(f"MC 修正偏离 {deps[0]:.2f}σ → {deps[-1]:.2f}σ，h=1 的 skill {sk[0]:+.3f} → {sk[-1]:+.3f}。")
    mono_up = all(b >= a - 0.02 for a, b in zip(ups, ups[1:]))
    mono_dn = all(b <= a + 0.02 for a, b in zip(ups, ups[1:]))
    if mono_up: P("**上偏随规模单调增强。** 更大的模型带的先验更强，不是更弱。")
    elif mono_dn: P("**上偏随规模单调减弱。** 规模在往正确方向走，但最大的尺寸仍未到零。")
    else: P("上偏与规模的关系**不单调**，见表。")
print("\n".join(L))
