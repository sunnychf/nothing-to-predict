"""Render RESULTS_PILOT_A.md section 4.20 (the ETTh1 positive control) from results/etth1_summary.json. Usage: python render_etth1_md.py results"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
S = json.load(open(os.path.join(RES, "etth1_summary.json"))); M = S["models"]; B = S["benchmarks"]["seasonal_naive_24"]; c = S["counts"]
LAB = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
       "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
order = [m for m in ("chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast") if m in M]
print("### 4.20 金融之外的真实正面控制：ETTh1（`etth1_probe.py` → `etth1_summary.py`）\n")
print(f"目的：证明在鞅上全输的十一个模型不是迟钝，而是「没有东西可预测时才这样」。ETTh1 七列各 512 点上下文 + 128 点未来、步长 128（未来不重叠），共 {S['n_windows']} 个窗口；"
      "每个模型按阶梯的方式预测原窗口及其加法镜像；技能按上下文单步 σ 归一化后汇总，标准误按窗口自举；季节朴素预测 = 目标前一天（24 步前）的值。\n")
print("| 预测 | skill h=1 | h=16 | h=64 | h=128 | 镜像修正 h=16 | 偶份额 h=16 |")
print("|---|---|---|---|---|---|---|")
print(f"| 季节朴素（前一天） | {B['1']['skill']:+.3f} | {B['16']['skill']:+.3f} | {B['64']['skill']:+.3f} | {B['128']['skill']:+.3f} | | |")
for m in order:
    r = M[m]; print(f"| {LAB[m]} | " + " | ".join(f"{r['raw'][h]['skill']:+.3f}±{r['raw'][h]['skill_se']:.3f}" for h in ("1", "16", "64", "128")) + f" | {r['mirror']['16']['skill']:+.3f} | {r['even_share']['16']:.2f} |")
gen = [m for m in order if m != "fincast"]
fc_str = ", ".join(f"{M['fincast']['raw'][h]['skill']:+.2f}" for h in ("1", "16", "64", "128"))
print(f"\n**读法。** 通用语料模型在 h=16 处技能 {min(M[m]['raw']['16']['skill'] for m in gen):+.2f} 到 {max(M[m]['raw']['16']['skill'] for m in gen):+.2f}（季节朴素 {B['16']['skill']:+.2f}，{len(c['beat_seasonal_naive_h16'])} 个赢过它），"
      f"h=64/128 处全部 > 0；Moirai-1.1 的 h=1 为 {M['moirai']['raw']['1']['skill']:+.2f}（首步假象，日频锚点同样不读）。**FinCast 在每个视野上都输给持久性**（{fc_str}），"
      f"偶份额 {M['fincast']['even_share']['16']:.2f}，镜像修正后 {M['fincast']['mirror']['16']['skill']:+.2f}。通用模型的偶份额 ≤ {max(M[m]['even_share']['16'] for m in gen):.2f}，修正后 h=16 技能变化 ≤ {max(abs(M[m]['mirror']['16']['skill'] - M[m]['raw']['16']['skill']) for m in gen):.2f}：带真实结构的预测在镜像下是奇的，修正不碰它（命题 (iv)）。")
