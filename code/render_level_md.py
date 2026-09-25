"""Render RESULTS_PILOT_A.md section 4.17 (level / shape decoupling) from diag_level_<model>.json.
Usage: python render_level_md.py results"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
ORDER = [("chronos", "Chronos-small"), ("chronosbolt", "Chronos-Bolt"), ("chronos2", "Chronos-2"), ("tirex", "TiRex"), ("moirai", "Moirai-small"), ("moirai2", "Moirai-2.0"),
         ("timesfm", "TimesFM-2.0"), ("timesfm25", "TimesFM-2.5"), ("timemoe", "Time-MoE-200M"), ("sundial", "Sundial"), ("fincast", "FinCast")]
lv = {m: json.load(open(os.path.join(RES, f"diag_level_{m}.json"))) for m, _ in ORDER if os.path.exists(os.path.join(RES, f"diag_level_{m}.json"))}
levels = next(iter(lv.values()))["levels"]
print("### 4.17 水平与形状解耦：同一批增量放在五个原始水平上（`diag_level_all.py`）\n")
print("审稿意见（09-19）：水平扫描里 −1000 处漂移最强、+1000 处不漂，可能只是「归一化之后落在训练分布何处」的支撑效应，而不是先验。"
      "检验：同一批 128 条零漂移增量（种子 4000，σ=1，与 `diag_direction_level.py` 同一构造）放在 −100/−10/0/+10/+100 五个水平，z 分数轨迹逐点相同，只有位置不同；"
      "每个模型按各自默认设置预测（`model_adapters.py`）。\n")
print("| 模型 | " + " | ".join(f"均值偏离 h=128 @{int(l):+d}" for l in levels) + " | " + " | ".join(f"向上 @{int(l):+d}" for l in levels) + " | max\\|Δ\\| vs +100 | 均方根 Δ h=128 |")
print("|---|" + "---|" * (2 * len(levels) + 2))
mx = {m: max(r["max_abs_diff_from_level100"] for r in lv[m]["records"]) for m in lv}
rms = {m: max(r["rms_diff_from_level100_h128"] for r in lv[m]["records"]) for m in lv}
for m, lab in ORDER:
    if m not in lv: continue
    by = {r["level"]: r for r in lv[m]["records"]}
    print(f"| {lab} | " + " | ".join(f"{by[l]['mean_dep']['128']:+.2f}" for l in levels) + " | " + " | ".join(f"{by[l]['frac_up']['128']:.2f}" for l in levels)
          + f" | {mx[m]:.1e} | {rms[m]:.1e} |")
inv = [lab for m, lab in ORDER if m in lv and mx[m] < 0.5]
worst = sorted([m for m in lv if mx[m] < 0.5], key=lambda m: -mx[m])[:2]; LAB = dict(ORDER)
rest = max(mx[m] for m in lv if mx[m] < 0.5 and m not in worst)
print(f"\n**读法。** {len(inv)} 个模型（{', '.join(inv)}）在网络看到上下文之前先按自身均值与标准差做中心化与缩放，原始水平到不了它们：五个水平上的偏离一致到算术噪声（最差 "
      + "、".join(f"{LAB[m]} {mx[m]:.1f}σ（均方根 {rms[m]:.2f}σ）" for m in worst) + f"，其余 ≤ {rest:.2f}σ），"
      "所以它们的漂移（TiRex、Moirai-1.1、Bolt 向上；纯解码器没有）是轨迹形状的性质，不是缩放后位置的性质。"
      "Chronos-T5 是唯一例外，原因是码本（除以平均绝对值、不中心化、再量化）：幅度随水平变（−10/0/+10/+100/−100 处 "
      + "/".join(f"{lv['chronos']['records'][i]['mean_dep']['128']:+.1f}" for i in (1, 2, 3, 4, 0)) + "σ），符号不变（每个水平上 "
      + f"{min(r['frac_up']['128'] for r in lv['chronos']['records']):.2f}–{max(r['frac_up']['128'] for r in lv['chronos']['records']):.2f} 向上），包括缩放后输入关于零对称的水平 0。"
      "附录 app:level 原来那句「负水平被拉向语料所在处」只描述 Chronos-T5 的分词器，已改写；§6.4 加了一句。")
