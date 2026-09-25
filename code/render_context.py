"""Render RESULTS_PILOT_A.md section 4.14 (the context-length sweep) from results/diag_context_*.json.
Usage: python code/render_context.py [results] > section.md   (numbers are never typed)"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
LABEL = {"chronos": "Chronos-small", "chronosbolt": "Chronos-Bolt", "chronos2": "Chronos-2", "tirex": "TiRex", "moirai": "Moirai-small", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "sundial": "Sundial",
         "timesfm": "TimesFM-2.0", "timemoe": "Time-MoE-200M", "fincast": "FinCast"}
D = {m: json.load(open(os.path.join(RES, f"diag_context_{m}.json"))) for m in LABEL if os.path.exists(os.path.join(RES, f"diag_context_{m}.json"))}
L = []; P = L.append
P("### 4.14 上下文长度扫描：先验随证据消退吗（`diag_context.py`）\n")
lens = sorted({r["T"] for d in D.values() for r in d["records"]})
n = next(iter(D.values()))["records"][0]["n"]
P(f"设计：一批 {n} 条零漂移随机游走（N1，种子 4000，水平 100、步长 1%）一次抽到 {max(lens)}+128 长；每个上下文长度 T 取每条序列**最后 T 点**做上下文，"
  "所以各长度下的未来完全相同、只有看到的历史在变（嵌套后缀）。Chronos-T5 最多读 512 点、FinCast 的接口在加载时固定上下文，二者只扫到 512。"
  "指标同方向实验：h=128 处均值带符号偏离（σ 单位，跨序列标准误）、预测高于末值的比例、h=16 处相对持久性的技能（n=128 时技能的标准误约 0.05，只看趋势）。\n")
P("| 模型 | 指标 | " + " | ".join(f"T={t}" for t in lens) + " |")
P("|---|---|" + "---|" * len(lens))
for m, d in D.items():
    by = {r["T"]: r for r in d["records"]}
    c = lambda t, f: (f(by[t]) if t in by else "—")
    P(f"| {LABEL[m]} | 偏离 h=128 /σ | " + " | ".join(c(t, lambda r: f"{r['mean_dep']['128']:+.2f}±{r['mean_dep_se']['128']:.2f}") for t in lens) + " |")
    P(f"| | 向上比例 h=128 | " + " | ".join(c(t, lambda r: f"{r['frac_up']['128']:.2f}") for t in lens) + " |")
    P(f"| | 偏离 h=16 /σ | " + " | ".join(c(t, lambda r: f"{r['mean_dep']['16']:+.2f}") for t in lens) + " |")
    P(f"| | 技能 h=16 | " + " | ".join(c(t, lambda r: f"{r['skill']['16']:+.3f}") for t in lens) + " |")
P("")
def dep(m, t): return {r["T"]: r for r in D[m]["records"]}[t]["mean_dep"]["128"]
def up(m, t): return {r["T"]: r for r in D[m]["records"]}[t]["frac_up"]["128"]
def sk(m, t): return {r["T"]: r for r in D[m]["records"]}[t]["skill"]["16"]
P("**读法。** 向上的漂移随上下文变长而消退但不消失："
  f"Chronos-T5 从 128 点的 {dep('chronos',128):+.2f}σ 到 512 点（它的上限）的 {dep('chronos',512):+.2f}σ（向上比例 {up('chronos',512):.2f}）；"
  f"Moirai 从 {dep('moirai',128):+.2f} 到 2048 点的 {dep('moirai',2048):+.2f}σ（{up('moirai',2048):.2f}）；"
  f"Chronos-Bolt 在 512/1024 点 {dep('chronosbolt',512):+.2f}/{dep('chronosbolt',1024):+.2f}，到 2048 点降到 {dep('chronosbolt',2048):+.2f}σ（{up('chronosbolt',2048):.2f}）；"
  f"Chronos-2 从 512 点的 {dep('chronos2',512):+.2f} 到 2048 点的 {dep('chronos2',2048):+.2f}σ（{up('chronos2',2048):.2f}）。"
  f"FinCast 向下的先验在每个长度上都在（向上比例 {up('fincast',128):.2f}/{up('fincast',256):.2f}/{up('fincast',512):.2f}）。"
  f"没有方向的两个模型彼此不同：TimesFM 随上下文变长向持久性靠拢（h=16 技能 {sk('timesfm',128):+.2f} → {sk('timesfm',2048):+.2f}），"
  f"Time-MoE 的误差随上下文增大（{sk('timemoe',128):+.2f} → {sk('timemoe',2048):+.2f}，h=128 处偏离的跨序列标准误从 "
  f"{ {r['T']: r for r in D['timemoe']['records']}[128]['mean_dep_se']['128']:.2f} 涨到 { {r['T']: r for r in D['timemoe']['records']}[2048]['mean_dep_se']['128']:.2f}σ：贪心多视野解码在长序列上越跑越散）。"
  "结论：更长的上下文是最便宜、也是不完整的缓解；这正是 §5 转而去掉先验的理由。"
  f"一致性：Chronos-T5 的扫描只到 512，它的序列就是 §4.10 方向实验的那 128 条（同一生成调用、同一种子），T=512 行（{dep('chronos',512):+.4f}σ，向上 {up('chronos',512):.4f}）"
  f"与 `diag_direction_control.json` 的 pos 记录（{json.load(open(os.path.join(RES, 'diag_direction_control.json')))['pos']['mean']['128']:+.4f}σ，{json.load(open(os.path.join(RES, 'diag_direction_control.json')))['pos']['frac_up']['128']:.4f}）逐位相同；其余模型的序列是 2048+128 长的另一批。\n")
print("\n".join(L))
