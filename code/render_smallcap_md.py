"""Render RESULTS_PILOT_A.md section 4.21 (the small-cap anchor) from results/smallcap_summary.json. Usage: python render_smallcap_md.py results"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
S = json.load(open(os.path.join(RES, "smallcap_summary.json"))); M = S["models"]; B = S["benchmarks"]; c = S["counts"]
LAB = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
       "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
order = [m for m in ("chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast") if m in M]
print("### 4.21 第四个锚点，带结构：Ken French 等权最小市值十分位（`smallcap_data.py` → `real_probe.py --prefix smallcap` → `smallcap_summary.py`）\n")
print(f"数据：`Portfolios_Formed_on_ME_Daily.csv`（2026-09-19 下载，来源与校验和在 `data/real/`），等权最小十分位（ew:Lo10，微型股非同步交易带来的正自相关，上下文上一阶自相关均值 {S['rho1_ctx']['lo10']['mean']:+.2f}）"
      f"与市值加权最大十分位（vw:Hi10，{S['rho1_ctx']['hi10']['mean']:+.2f}）对照；{S['period']['first_ctx_start']} 到 {S['period']['last_fut_end']}，每序列 {S['splits']['lo10']} 个窗口，K={S['K']} 副本；与日频锚点同构。"
      "指标按每条上下文收益率标准差归一化后汇总（否则 1987/2008/2020 决定一切；未归一化版本在 JSON 里）；基线只在上下文上拟合并迭代：漂移、AR(1)、AR(5)。**h=1 不读**（首步假象），表在 h=16。\n")
print("| 预测（h=16） | Lo10 原始 skill | 占 AR(1) 份额 | 镜像修正 | 偶份额 | 副本 skill | Hi10 原始 skill | h=1 偏离与 AR(1) 的相关 |")
print("|---|---|---|---|---|---|---|---|")
for k, lab in (("drift", "漂移（拟合）"), ("ar1", "AR(1)（拟合）"), ("ar5", "AR(5)（拟合）")):
    b = B[k]; print(f"| {lab} | {b['lo10']['16']['skill']:+.3f}±{b['lo10']['16']['skill_se']:.3f} | {b['lo10']['16'].get('share', 1.0):+.2f} | | | {B['ar1_on_copies']['lo10']['16']['skill']:+.3f} | {b['hi10']['16']['skill']:+.3f} | |")
for m in order:
    r = M[m]; a, mm, s_ = r["raw|lo10"]["16"], r.get("mirror|lo10", {}).get("16"), r["sur|lo10"]["16"]
    print(f"| {LAB[m]} | {a['skill']:+.3f}±{a['skill_se']:.3f} | {a['share']:+.2f} | {mm['skill'] if mm else float('nan'):+.3f} | {r.get('even_share|lo10', {}).get('16', float('nan')):.2f} | {s_['skill']:+.3f} | {r['raw|hi10']['16']['skill']:+.3f} | {r['raw|lo10']['1']['corr_with_ar1']:+.2f}±{r['raw|lo10']['1']['corr_with_ar1_se']:.2f} |")
r16 = {m: M[m]["raw|lo10"]["16"]["skill"] for m in M}; c16 = {m: M[m]["mirror|lo10"]["16"]["skill"] for m in M if "mirror|lo10" in M[m]}
beat = c["lo10_beat_persistence_h16"]; lose = [m for m in order if m not in beat]
print(f"\n**读法。** 这是逐笔锚点没做成的真实正面控制。Lo10 上拟合漂移 {B['drift']['lo10']['16']['skill']:+.2f}、AR(1) {B['ar1']['lo10']['16']['skill']:+.2f}、AR(5) {B['ar5']['lo10']['16']['skill']:+.2f}；副本上 AR(1) 回到 {B['ar1_on_copies']['lo10']['16']['skill']:+.2f}。"
      f"十一个模型里 {len(beat)} 个赢过持久性（{', '.join(LAB[m] for m in beat)}），{len(c['lo10_above_ar1_h16'])} 个高于 AR(1)（{', '.join(LAB[m] for m in c['lo10_above_ar1_h16'])}）；输的是 {', '.join(LAB[m] for m in lose)}。"
      f"副本上 {len(c['lo10_copies_beat_persistence_h16'])} 个赢过持久性，Hi10 上 {len(c['hi10_beat_persistence_h16'])} 个。h=1 处 {len(c['lo10_corr_positive_gt_1se'])} 个模型的单步偏离与 AR(1) 的相关为正（> 1 se），但没有一个变成技能。"
      f"镜像分解：修正后不低于原始的有 {', '.join(LAB[m] for m in c['lo10_mirror_ge_raw_h16'])}（读到的是奇结构）；塌掉的是 "
      + "、".join(f"{LAB[m]}（{r16[m]:+.2f} → {c16[m]:+.2f}，偶份额 {M[m]['even_share|lo10']['16']:.2f}）" for m in order if r16[m] > 0.1 and c16.get(m, 0) < 0.05)
      + "：在上涨序列上看似技能的是符号盲先验碰上同向漂移，即 §3 的混淆在真实大漂移序列上的样子；(v) 两种情形都读得通。")
