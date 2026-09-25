"""Render RESULTS_PILOT_A.md section 4.16 (the intraday anchor) from results/intraday_summary.json
and results/intraday_windows_meta.json; the same numbers as paper/tables/intraday.tex.
Usage: python render_intraday_md.py results
"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
s = json.load(open(os.path.join(RES, "intraday_summary.json"))); M = s["models"]; c = s["counts"]
meta = json.load(open(os.path.join(RES, "intraday_windows_meta.json")))
LABEL = {"chronos": "Chronos-T5", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-1.1", "moirai2": "Moirai-2.0",
         "timesfm": "TimesFM-2.0", "timesfm25": "TimesFM-2.5", "timemoe": "Time-MoE", "sundial": "Sundial", "fincast": "FinCast"}
order = [m for m in ("chronos", "chronosbolt", "chronos2", "tirex", "moirai", "moirai2", "timesfm", "timesfm25", "timemoe", "sundial", "fincast") if m in M]
print("### 4.16 第三个锚点：逐笔成交价（`itch_trades.py` → `intraday_data.py` → `real_probe.py --prefix intraday` → `intraday_summary.py`）\n")
print(f"数据：纳斯达克公开的 TotalView-ITCH 样本一天（{meta['source']}，2019-12-30），抽出全部成交（E/C/P 消息，去掉集合竞价 Q），同一时间戳的多笔合并为一笔；"
      f"价格 ≥ ${meta['min_price']:.0f}、成交笔数最多的 {meta['top']} 个代码，每个均匀取 ≤{meta['max_windows']} 段 {meta['ctx']}+{meta['horizon']} 笔的连续窗口：共 {s['n_windows']} 个窗口、{s['n_symbols']} 个代码；"
      f"每窗 K={s['K']} 条乘法随机符号副本（种子 9000+窗号+100000·副本），指标全部按收益率单位。原意是把 N4（买卖价弹跳）搬到真实市场上做正面控制。\n")
print(f"**数据实际是什么。** 连续两笔同价的比例 {s['zero_return_share']:.2f}；交易方向翻转率 {s['direction_flip_rate']:.2f}（Roll 的独立方向应为 0.5）；"
      f"收益率一阶自相关均值 {s['rho1_mean']:+.3f}，低于 −0.1 的窗口占 {s['rho1_frac_below_-0.1']:.2f}；对每个上下文拟合的 MA(1) 在 h=1 处技能 "
      f"{s['benchmarks']['ma1_fit']['1']['skill']:+.3f}±{s['benchmarks']['ma1_fit']['1']['skill_se']:.3f}，在自相关 < −0.1 的窗口上 {s['ma1_fit_skill_h1_where_rho_below_-0.1']:+.3f}：\n"
      "在这个交易所、这个频率上弹跳不是拟合 MA(1) 能用的一步结构，所以这些窗口是第三个零假设锚点而不是正面控制（按此报告，不改口）。\n")
print("| 模型 | 原始 skill h=1 | 镜像 | 原始 skill h=16 | 镜像 | 副本 dep h=16 (σ_r) | 副本向上 | 副本 skill h=1 → 镜像 | 副本 skill h=16 → 镜像 | 偶份额 h=1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in order:
    a = M[m]; r, b, sr, sm = a["raw"], a.get("mirror"), a["sur"], a.get("sur_mirror")
    print(f"| {LABEL[m]} | {r['1']['skill']:+.3f}±{r['1']['skill_se']:.3f} | {b['1']['skill']:+.3f} | {r['16']['skill']:+.3f} | {b['16']['skill']:+.3f} | "
          f"{sr['16']['mean_dep']:+.2f}±{sr['16']['mean_dep_se']:.2f} | {sr['16']['frac_up']:.2f} | {sr['1']['skill']:+.3f} → {sm['1']['skill']:+.3f} | {sr['16']['skill']:+.3f} → {sm['16']['skill']:+.3f} | {a['even_share']['1']:.2f} |")
ups = {m: M[m]["sur"]["16"]["frac_up"] for m in order}
print(f"\n**读法。** 副本上日频锚点的符号在逐笔频率重现：向上比例 > 0.6 的是 {', '.join(LABEL[m] for m in c['copies_up_above_0.6'])}"
      f"（{', '.join(f'{ups[m]:.2f}' for m in c['copies_up_above_0.6'])}），< 0.5 的是 {', '.join(LABEL[m] for m in c['copies_up_below_0.5'])}。"
      f"原始窗口上 h=1 处高于持久性的有 {len(c['beat_persistence_raw_h1'])} 个（{', '.join(LABEL[m] for m in c['beat_persistence_raw_h1'])}），都不超过一个标准误；h=16 处全部低于持久性。"
      f"镜像修正：原始窗口 h=16 处帮到 {len(c['mirror_helps_raw_h16'])}/11、h=1 处 {len(c['mirror_helps_raw_h1'])}/11；副本 h=16 处 {len(c['mirror_helps_copies_h16'])}/11、h=1 处 {len(c['mirror_helps_copies_h1'])}/11。"
      f"Chronos-T5 只报告不解读：4096 格网格间距是上下文平均水平的固定比例，在这个尺度上比一步宽得多（中位 67 步），预测是码本的舍入；同族的 Bolt 与 Chronos-2 是连续输入，读数正常。")
