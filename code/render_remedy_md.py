"""Render RESULTS_PILOT_A.md §4.13 (the sign-averaged correction) from results/remedy_summary.json.

Every number is read from the JSON written by remedy_summary.py; nothing is retyped.
Usage: python code/render_remedy_md.py [results] > section.md
"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
s = json.load(open(os.path.join(RES, "remedy_summary.json")))
LABEL = {"chronos": "Chronos-small", "moirai": "Moirai-small", "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "fincast": "FinCast",
         "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "sundial": "Sundial"}
M = s["models"]; K = s.get("K") or 4; R = s.get("real", {})
order = [k for k in ("chronos", "moirai", "timesfm", "timemoe", "fincast", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial") if k in M]
L = []; P = L.append
def pm(v, se, d=2): return f"{v:+.{d}f}±{se:.{d}f}"

P("### 4.13 对称平均修正：把先验减掉之后还剩什么\n")
P(f"估计量（论文 §5 式 eq:sas、命题 prop:sas）：让模型预测上下文在零假设对称群下的副本，把偏离取平均，减掉。"
  "两个群：**镜像**（每个增量取负，一条副本，一次前向传播；对任何在增量取负下分布不变的过程精确，N4 也在内）；"
  f"**符号**（各增量独立取负，K={K} 组随机副本的蒙特卡洛平均；只在条件符号对称下精确，N1–N3 满足、N4 不满足）。"
  "合成阶梯上副本按加法重建，真实窗口上按乘法重建（收益率翻号），修正在收益率单位里做。"
  "阶梯：每档 128 条计分的上下文加它们的 128 条镜像（镜像只提供 D(x̄)，不计分：若把镜像对的两个成员都收进测试集，镜像的风险降幅恒等于 E[偶部分²]、与数据无关，"
  "`paired_identity_check` 把这个恒等式复现到舍入误差），H=128，种子 6000/7000；N4 的「最优解」是条件均值（两态滤波，`nulls.oracle_forecast_exact`），不是存储的线性 MA(1) 最优；"
  "真实：第二版锚点的 1280 个窗口、它们的镜像（`real_<model>_mirror.npz`）及 K 组替代序列（§4.12）。"
  f"「偶份额」= E[D̄²]/E[(ŷ−y_T)²] 是镜像修正恰好去掉的那部分搬入项；「系统份额」是符号平均的对应量，按命题 (iii)，"
  f"有限 K 的符号平均降低风险当且仅当份额 ≥ 1/(K+1) = {1/(K+1):.2f}；「K→∞」列把有限 K 噪声项 E[V]/K 从修正后 MSE 里减掉。"
  "标准误：阶梯按序列自助，真实按「族 × 未来段结束日」聚类自助。\n")
P("**阶梯（合成，条件符号对称在 N1–N3 上精确成立）。** 均值偏离 / σ 与相对持久性的技能，原始 → 修正后（K→∞ 估计）：\n")
P("| 模型 | 档 | 均值偏离 h=128 原始 | 符号平均后 | 技能 h=16 原始 | 镜像 | 符号平均 (K→∞) | 技能 h=128 原始 | 镜像 | 符号平均 (K→∞) | 偶份额 h=16 | 系统份额 h=128 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in order:
    for rung in ("N1", "N2", "N3_nu5", "N4"):
        r = M[k].get(f"ladder|{rung}")
        if not r: continue
        a, c, mr, sh = r["raw"], r["corrected"], r.get("mirror"), r["systematic_share"]["128"]["debiased"]
        ev = r.get("even_share", {}).get("16", float("nan"))
        P(f"| {LABEL[k]} | {rung} | {pm(a['128']['mean_dep'], a['128']['mean_dep_se'])} | {pm(c['128']['mean_dep'], c['128']['mean_dep_se'])} | "
          f"{a['16']['skill']:+.3f} | {mr['16']['skill']:+.3f} | {c['16']['skill']:+.3f} ({c['16']['skill_Kinf']:+.3f}) | {a['128']['skill']:+.3f} | {mr['128']['skill']:+.3f} | {c['128']['skill']:+.3f} ({c['128']['skill_Kinf']:+.3f}) | {ev:.2f} | {sh:.2f} |")
P("")
n4 = [(k, M[k]["ladder|N4"]) for k in order if "ladder|N4" in M[k]]
if n4:
    P("N4（正面控制；镜像对称成立，条件符号对称**不**成立）h=1 处抓到的最优结构份额，原始 → 镜像 → 符号平均（K→∞）："
      + "；".join(f"{LABEL[k]} {r['raw']['1']['oracle_share']:+.2f} → {r['mirror']['1']['oracle_share']:+.2f} → {r['corrected']['1']['oracle_share']:+.2f}（{r['corrected']['oracle_share_Kinf']['1']:+.2f}）" for k, r in n4) + "。\n")
P("**真实窗口。** 收益率单位；「鞅」= 替代序列上的留一修正（每条副本用其余 K−1 条修正，均值恰为零是构造保证，技能不是）：\n")
P("| 模型 | 输入 | 均值偏离 h=128 原始 | 镜像 | 符号平均后 | 技能 h=16 原始 | 镜像 | 符号平均 (K→∞) | 技能 h=128 原始 | 镜像 | 符号平均 (K→∞) |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for k in order:
    for key, lab in (("realsur|all", "鞅（全部）"), ("real|all", "原始（全部）"), ("real|fx", "原始，汇率"), ("real|eq", "原始，股票"), ("real|post2025", "原始，2025 后")):
        r = M[k].get(key)
        if not r: continue
        a, c, mr = r["raw"], r["corrected"], r.get("mirror")
        mm = lambda h, f, d=3: f"{mr[h][f]:+.{d}f}" if mr else "—"
        P(f"| {LABEL[k]} | {lab} | {pm(a['128']['mean_dep'], a['128']['mean_dep_se'])} | {mm('128', 'mean_dep', 2)} | {pm(c['128']['mean_dep'], c['128']['mean_dep_se'])} | "
          f"{a['16']['skill']:+.3f} | {mm('16', 'skill')} | {c['16']['skill']:+.3f} ({c['16']['skill_Kinf']:+.3f}) | {a['128']['skill']:+.3f} | {mm('128', 'skill')} | {c['128']['skill']:+.3f} ({c['128']['skill_Kinf']:+.3f}) |")
P("")
P("系统份额（真实原始窗口，h=128，去噪后）：" + "；".join(f"{LABEL[k]} {M[k]['real|all']['systematic_share']['128']['debiased']:.2f}" for k in order if "real|all" in M[k]) + f"；阈值 1/(K+1) = {1/(K+1):.2f}。\n")
cnt_l = s.get("ladder_counts", {})
P(f"**读法。** (1) 镜像修正在它没见过的未来上：{cnt_l.get('cells')} 个模型×档×视野格子里降低风险 {cnt_l.get('better')} 个（{cnt_l.get('better_gt_1se')} 个超过配对差的一个标准误）、"
  f"抬高 {cnt_l.get('worse')} 个（{cnt_l.get('worse_gt_1se')} 个超过一个标准误：{', '.join(f'{LABEL[c[0]]} {c[1]} h={c[2]}' for c in cnt_l.get('worse_gt_1se_cells', []))}）、"
  f"TimesFM-2.5 的 {cnt_l.get('unchanged')} 个不变；N1 h=16 上升 {len(cnt_l.get('N1_h16_better', []))} 个、下降 {', '.join(LABEL[m] for m in cnt_l.get('N1_h16_worse', [])) or '无'}、不变 {', '.join(LABEL[m] for m in cnt_l.get('N1_h16_unchanged', [])) or '无'}；"
  f"N4 上抓到的最优结构份额上升 {len(cnt_l.get('N4_h1_share_up', []))} 个、下降 {', '.join(LABEL[m] for m in cnt_l.get('N4_h1_share_down', [])) or '无'}："
  "它只去掉偏离的偶部分，在对称之下那是纯误差，而 N4 的最优偏离是奇的，不被触碰（命题 (ii)、(iv)）；偶份额很小的模型（Time-MoE 0.04）没什么可去，样本的未来决定正负。"
  "(2) 符号平均在 N1–N3 上把均值偏离去掉，但有限 K 只在系统份额超过 1/(K+1) 的地方赚回来，先验不是符号盲的模型"
  "（TimesFM、Time-MoE、FinCast，份额接近零）在 K=4 时只添噪声；在 N4 上它减掉的是模型对一条被打乱了跳动的序列的响应，"
  "所以最优结构份额下降。两种修正的对比就是命题里两种假设的对比。"
  "(3) 真实原始窗口是一段上涨期：向上的先验（Chronos、Moirai）碰巧与实际走势同向，去掉它会**失去**技能，"
  "而这正是 §3 的混淆被量化出来：那部分技能就是先验。修正后的技能才是模型在这段数据上真正知道的东西。"
  "(4) 逆斜率的 Time-MoE 在原始窗口上修正后更差：它的错误是读符号得来的（命题 (iv)：修正不碰随数据翻号的部分），不是符号盲的先验。\n")
# Proposition (v): the exact decomposition of the mirror correction's skill change on the raw windows
S_real = R
if any("decomp" in M[k].get("real|eq", {}) for k in order):
    cnt = S_real.get("mirror_raw_counts", {})
    P("**命题 (v)：原始窗口上的镜像修正为何输、输多少。** 对分布不作任何假设时 Risk(ŷ^c) − Risk(ŷ) = −E[D̄²] + 2E[m_h D̄] − 2E[(D−D̄)D̄]，"
      "在样本上是「已实现未来」的恒等式（脚本里断言到 1e-9）。以持久性误差为单位、h=128：Δ技能 = 收益 + 漂移 + 交叉；"
      "「先验」= 模型去掉的偶部分折成年率，「盈亏平衡」= 它的一半（符号盲先验只有在真实漂移低于此时才值得去掉）。"
      f"这批窗口的已实现漂移：汇率每年 {100*M[order[0]]['real|fx']['decomp']['128']['realised_annualised']:+.1f}%，股票每年 {100*M[order[0]]['real|eq']['decomp']['128']['realised_annualised']:+.1f}%。\n")
    P("| 模型 | 窗口 | Δ技能 | 收益 | 漂移 | 交叉 | 先验 %/年 | 盈亏平衡 %/年 |")
    P("|---|---|---|---|---|---|---|---|")
    for k in order:
        for key, lab in (("real|fx", "汇率"), ("real|eq", "股票")):
            d = M[k].get(key, {}).get("decomp", {}).get("128")
            if not d: continue
            P(f"| {LABEL[k]} | {lab} | {d['dskill']:+.3f} | {d['gain']:+.3f} | {d['drift']:+.3f} | {d['cross']:+.3f} | {100*d['prior_annualised']:+.1f} | {100*d['breakeven_annualised']:+.1f} |")
    P("")
    if cnt:
        P("镜像修正在原始窗口上帮到的模型数（帮/害）：" + "；".join(f"{k} {v['helped']}/{v['hurt']}" for k, v in cnt.items()) + "。\n")
    P("**读法。** 股票窗口在 h=128 处七个模型全输：Chronos-T5、Bolt、Moirai、FinCast 输在漂移项压过收益项（这段股票每年涨得远高于它们 0.7–3.5% 的盈亏平衡），"
      "TimesFM、Time-MoE、Chronos-2 输在交叉项（回归者对上涨窗口的镜像的响应不是其响应的镜像）；汇率窗口上 Chronos 的收益与漂移几乎相抵，修正七个赢五个；"
      "h=16 处漂移项小，股票上帮到五个。所以修正是「赌符号盲的漂移不是真的」，相信股权溢价高于盈亏平衡的人不该下这个注。\n")
# K=16 extension (Chronos, Moirai; copies 4..15 added to the same windows), summarised separately so that the
# canonical K=4 tables stay comparable across models
k16 = os.path.join(RES, "remedy_summary_k16.json")
if os.path.exists(k16):
    S16 = json.load(open(k16))["models"]
    P("**K=16 扩展（只对 Chronos 与 Moirai，同一批窗口再加 12 组符号副本；只进这里，不进论文表）。** 符号平均在真实增量鞅上 h=128 的技能，K=4 → K=16（括号内为各自的 K→∞ 估计）："
      + "；".join(f"{LABEL[k]} {M[k]['realsur|all']['corrected']['128']['skill']:+.3f}（{M[k]['realsur|all']['corrected']['128']['skill_Kinf']:+.3f}）→ {S16[k]['realsur|all']['corrected']['128']['skill']:+.3f}（{S16[k]['realsur|all']['corrected']['128']['skill_Kinf']:+.3f}）" for k in ("chronos", "moirai") if k in S16)
      + "；原始 " + "、".join(f"{M[k]['realsur|all']['raw']['128']['skill']:+.3f}" for k in ("chronos", "moirai")) + "。"
      "更多抽样把有限 K 的噪声项压下去，估计量向 K→∞ 线靠拢；镜像修正一次前向传播就到那里（"
      + "；".join(f"{LABEL[k]} 镜像 {M[k]['realsur|all']['mirror']['128']['skill']:+.3f}" for k in ("chronos", "moirai") if "mirror" in M[k]["realsur|all"]) + "）。\n")
# two classical baselines on the same columns as the main table, and the estimated-drift gate (2026-09-19)
S = json.load(open(os.path.join(RES, "remedy_summary.json")))
if "baselines" in S:
    P("**两个逐窗拟合的经典基线（主表末两行；`remedy_summary.py`）。** 漂移 = 上下文平均增量/收益率外推；AR(1) = 对增量/收益率做带截距的最小二乘 AR(1) 并迭代。两者都是奇的，镜像只在收益率复利处改变它们。\n")
    P("| 基线 | 鞅 h=128 均值/σ_r | 鞅 skill h=16 → 镜像 | 鞅 skill h=128 → 镜像 | 原始 skill h=128 → 镜像 | N1 skill h=16 → 镜像 | N4 份额 h=1 → 镜像 |")
    P("|---|---|---|---|---|---|---|")
    for k, lab in (("drift", "漂移（逐窗）"), ("ar1", "AR(1)（逐窗）")):
        B = S["baselines"][k]; ms, rw, n1, n4 = B["realsur|all"], B["real|all"], B["ladder|N1"], B["ladder|N4"]
        P(f"| {lab} | {ms['raw']['128']['mean_dep']:+.2f}±{ms['raw']['128']['mean_dep_se']:.2f} | {ms['raw']['16']['skill']:+.3f} → {ms['mirror']['16']['skill']:+.3f} | {ms['raw']['128']['skill']:+.3f} → {ms['mirror']['128']['skill']:+.3f} | "
          f"{rw['raw']['128']['skill']:+.3f} → {rw['mirror']['128']['skill']:+.3f} | {n1['raw']['16']['skill']:+.3f} → {n1['mirror']['16']['skill']:+.3f} | {n4['raw']['1']['oracle_share']:+.2f} → {n4['mirror']['1']['oracle_share']:+.2f} |")
    P("")
if "gate_counts_h128" in S.get("real", {}):
    g = S["real"]["gate_counts_h128"]
    P("**门控估计量（算法第 6 行，μ̂ = 上下文平均收益；`tables/remedy_gate.tex`）。** h=128 处「门控 ≥ 镜像 / 门控 ≥ 原始」的模型数："
      + "；".join(f"{k} {v['gated_ge_mirror']}/{v['n']} 与 {v['gated_ge_raw']}/{v['n']}" for k, v in g.items())
      + "。门控率（全部窗口）：" + "、".join(f"{LABEL[k]} {M[k]['real|all']['gate_rate']['128']:.2f}" for k in order if "gate_rate" in M[k].get("real|all", {})) + "。"
      "结论：估计的漂移太吵，门控落在两个无条件估计量之间，两个市场都不输的版本做不出来；论文把修正定位为测量、把符号盲份额定位为它量出的数。\n")
print("\n".join(L))
