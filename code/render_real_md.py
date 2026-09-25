"""Render RESULTS_PILOT_A.md §4.12 (the real-data anchor) from results/real_summary.json.

Every number below is read from the JSON written by real_summary.py; nothing is retyped.
Usage: python code/render_real_md.py [results] > section.md
"""
import json, os, sys
import numpy as np
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
s = json.load(open(os.path.join(RES, "real_summary.json")))
meta = json.load(open(os.path.join(RES, "real_windows_meta.json")))
zw = np.load(os.path.join(RES, "real_windows.npz"))
famv = np.array([m["family"] for m in meta])
rs_fx, rs_eq = 100 * float(np.median(zw["rsigma_raw"][famv == "fx"])), 100 * float(np.median(zw["rsigma_raw"][famv == "eq"]))
K = s.get("K", 1)
LABEL = {"chronos": "Chronos-small", "moirai": "Moirai-small", "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "fincast": "FinCast",
         "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "sundial": "Sundial"}
M = s["models"]; R = s["realised"]
L = []; P = L.append
def pm(v, se, d=2): return f"{v:+.{d}f}±{se:.{d}f}"
fx_end = max(m["fut_end"] for m in meta if m["family"] == "fx"); eq_end = max(m["fut_end"] for m in meta if m["family"] == "eq")
first_end = min(m["fut_end"] for m in meta)

P("### 4.12 真实数据锚点：真实增量上的符号，与一段上涨期里的「技能」\n")
P(f"数据：ECB 欧元参考汇率（20 种浮动货币对 EUR）与 Ken French 日频 49 行业组合加市场组合（水平 100·∏(1+r)），"
  f"上下文 512 + 视野 128 个交易日，每条序列自 2015 年起取全部不重叠未来段：共 {s['n_windows']} 个窗口"
  f"（汇率 {s['families']['fx']}，股票 {s['families']['eq']}），未来段结束于 {first_end} 到 {max(fx_end, eq_end)}"
  f"（股票数据到 {eq_end}），其中 {s['n_post2025']} 个结束在 2025-01-01 之后。"
  f"替代序列（sur）：每个窗口的 639 个简单收益率乘以 i.i.d. ±1 符号后按乘法重建水平，每窗 {K} 组独立符号，"
  "是精确的鞅、保留真实的 |收益率| 路径。原始序列（raw）：真实窗口本身。"
  "**单位一律是收益率**：偏离 (ŷ_h/y_T−1)/σ_r（σ_r 为上下文简单收益率的标准差），误差 ((y−ŷ)/y_T)² 在窗口间汇总，"
  "这样窗口的权重与它水平的尺度无关（第一版按水平汇总，被 IDR/KRW/JPY 吞掉，见 METHOD_NOTES M12）。"
  f"标准误来自按「族 × 未来段结束日」聚类的自助法（2000 次，{s.get('n_clusters', '?')} 个聚类；一个窗口的替代序列与它同一聚类），因为 50 个股票组合共享每一个日历窗口。"
  "模型与 Pilot A 驱动完全相同（Chronos S=100；Moirai patch auto、S=100；TimesFM freq 0；FinCast freq 0；Time-MoE 贪心多视野解码）。\n")
P("| 模型 | sur 均值 h=128 | sur 向上 | sur skill h=16 | raw 均值 h=128 | raw 向上 | raw skill h=16 | raw skill h=128 |")
P("|---|---|---|---|---|---|---|---|")
for k in [k for k in ("chronos", "moirai", "timesfm", "timemoe", "fincast", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial") if k in M]:
    a, w = M[k]["sur|all"], M[k]["raw|all"]
    P(f"| {LABEL[k]} | {pm(a['128']['mean_dep'], a['128']['mean_dep_se'])}σ | {a['128']['frac_up']:.2f} | {pm(a['16']['skill'], a['16']['skill_se'], 3)} | "
      f"{pm(w['128']['mean_dep'], w['128']['mean_dep_se'])}σ | {w['128']['frac_up']:.2f} | {pm(w['16']['skill'], w['16']['skill_se'], 3)} | {pm(w['128']['skill'], w['128']['skill_se'], 3)} |")
P(f"| 数据本身 | {R['sur_h128']:+.2f}σ | {R['up_sur_h128']:.2f} | 0 | {R['raw_h128']:+.2f}σ | {R['up_raw_h128']:.2f} | 0 | 0 |\n")
c, mo, tf, tm, fc = (M[k] for k in ("chronos", "moirai", "timesfm", "timemoe", "fincast"))
# level-unit surrogate means (the first anchor's unit) for the slope-reverting models
cs = zw["ctx_sur"]; n_, K_, T_ = cs.shape; flat = cs.reshape(n_ * K_, T_); sig_lvl = np.std(np.diff(flat, axis=1), axis=1)
lvl = {k: float(np.mean((np.load(os.path.join(RES, f"real_{k}.npz"))["yhat_sur"].reshape(n_ * K_, -1)[:, -1] - flat[:, -1]) / sig_lvl)) for k in M}
order = [k for k in ("chronos", "moirai", "timesfm", "timemoe", "fincast", "chronosbolt", "chronos2", "tirex", "moirai2", "timesfm25", "sundial") if k in M]
neg16 = [k for k in order if M[k]["sur|all"]["16"]["skill"] < 0]; neg128 = [k for k in order if M[k]["sur|all"]["128"]["skill"] < 0]
P("**替代序列上（精确的鞅，真实的 |增量|）：合成阶梯上的符号原样重现。** "
  f"Chronos {c['sur|all']['128']['mean_dep']:+.2f}σ_r、Moirai {mo['sur|all']['128']['mean_dep']:+.2f}σ_r（h=128），"
  f"向上比例 {c['sur|all']['128']['frac_up']:.2f} / {mo['sur|all']['128']['frac_up']:.2f}；"
  f"FinCast {fc['sur|all']['128']['mean_dep']:+.2f}σ_r、向上 {fc['sur|all']['128']['frac_up']:.2f}。"
  f"TimesFM {tf['sur|all']['128']['mean_dep']:+.2f}±{tf['sur|all']['128']['mean_dep_se']:.2f}σ_r 与 "
  f"Time-MoE {tm['sur|all']['128']['mean_dep']:+.2f}±{tm['sur|all']['128']['mean_dep_se']:.2f}σ_r 在加法阶梯上没有方向，这里的正均值是单位效应，不是先验："
  f"两者都逆着近期斜率（与 64 步斜率相关 {tf['sur|all']['128']['corr_slope64']:+.2f} 与 {tm['sur|all']['128']['corr_slope64']:+.2f}），"
  "回归量在跌下去的副本上按 y_T 归一化后被放大、在涨上去的副本上被压缩，收益率单位里两者不抵消；"
  f"同一批预测按水平单位算是 {lvl['timesfm']:+.2f}σ 与 {lvl['timemoe']:+.2f}σ（M12）。"
  + (f"Sundial 同理：{M['sundial']['sur|all']['128']['mean_dep']:+.2f}±{M['sundial']['sur|all']['128']['mean_dep_se']:.2f}σ_r、斜率相关 "
     f"{M['sundial']['sur|all']['128']['corr_slope64']:+.2f}、按水平单位 {lvl['sundial']:+.2f}σ。" if "sundial" in M else "") +
  f"h=16 处 skill 为负的模型：{len(neg16)}/{len(order)}；h=128 处：{len(neg128)}/{len(order)}"
  "（" + "、".join(f"{LABEL[k]} {M[k]['sur|all']['16']['skill']:+.3f}/{M[k]['sur|all']['128']['skill']:+.3f}" for k in order) + "）。"
  "这一半回答审稿人「零真实金融数据」的问题：先验不是合成序列的性质，真实的波动率聚集、厚尾与日历都在，符号不变。\n")
pos128 = [k for k in order if M[k]["raw|all"]["128"]["skill"] > 0]
P("**原始序列上：这段时期涨了，混淆就照 §3 说的那样出现。** "
  f"数据本身 128 天平均 {R['raw_h128']:+.2f}σ_r，{R['up_raw_h128']:.0%} 的窗口收在上方。"
  f"Chronos 的预测同样上涨（{c['raw|all']['128']['mean_dep']:+.2f}σ_r），但其中 "
  f"{100*c['sur|all']['128']['mean_dep']/c['raw|all']['128']['mean_dep']:.0f}% 已经出现在符号随机化之后的替代序列上，是先验而不是信号；"
  f"它在股票上 h=128 的 skill 为 {pm(c['raw|eq']['128']['skill'], c['raw|eq']['128']['skill_se'], 2)}（涨了 {c['raw|eq']['128']['realised']:+.1f}σ_r），"
  f"2025 年后 {pm(c['raw|post2025']['128']['skill'], c['raw|post2025']['128']['skill_se'], 2)}，汇率上 {pm(c['raw|fx']['128']['skill'], c['raw|fx']['128']['skill_se'], 2)}（涨了 {c['raw|fx']['128']['realised']:+.1f}σ_r）："
  "只在涨势与先验同向的地方像有技能。"
  f"Time-MoE 逆着近期斜率预测（raw h=128 {tm['raw|all']['128']['mean_dep']:+.1f}σ_r，与 64 步斜率相关 {tm['raw|all']['128']['corr_slope64']:+.2f}），"
  f"在上涨期里相对持久性损失 {-100*tm['raw|all']['128']['skill']:.0f}%；2025 年后的 {s['n_post2025']} 个窗口更差（{tm['raw|post2025']['128']['mean_dep']:+.1f}σ_r，skill {tm['raw|post2025']['128']['skill']:+.2f}）。"
  f"汇总全部窗口，h=128 处 skill 为正的模型：{'、'.join(LABEL[k] + ' ' + format(M[k]['raw|all']['128']['skill'], '+.3f') + '±' + format(M[k]['raw|all']['128']['skill_se'], '.3f') for k in pos128) if pos128 else '没有'}"
  "（全部：" + "、".join(f"{M[k]['raw|all']['128']['skill']:+.3f}" for k in order) + "）；"
  "替代序列那一列说明其中多少是先验碰上了一段恰好同向的时期，§4.13 的修正把它减掉之后再读。\n")
P(f"**族别与 2025 年后拆分**见 `paper/tables/real_detail.tex`。汇率一族日收益率 σ_r 的中位数只有 {rs_fx:.2f}%，"
  f"Chronos 的码本（均值缩放后每格约 0.733% 的水平）在这个尺度上约 {0.733 / rs_fx:.1f}σ_r 一格，所以汇率的 h=1 不读（同 §4.2 的尺度告诫）；股票为 {rs_eq:.2f}%（{0.733 / rs_eq:.1f}σ_r 一格）。\n")
n_cl = len({(m["family"], m["fut_end"]) for m in meta}); n_cl_post = len({(m["family"], m["fut_end"]) for m in meta if m["fut_end"] >= "2025-01-01"})
P(f"**两条告诫。** 聚类自助法只有 {n_cl} 个日历聚类（2025 年后的拆分 {n_cl_post} 个），标准误相应地宽；"
  "公开的日频汇率与股票序列可能出现在预训练语料里（替代序列按构造是新序列；原始序列以「未来段结束于 2025-01-01 之后」单列）。"
  "来源、许可与预处理见 `data/real/SOURCES.md`；脚本 `code/real_data.py`（窗口）、`code/real_probe.py`（每模型一跑）、`code/real_summary.py`（指标与表）。\n")
print("\n".join(L))
