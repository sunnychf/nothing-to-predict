"""Render RESULTS_PILOT_A.md section 4.15 (calibration under the null) from the ladder rows'
`coverage_80` / `interval80_width_over_sigma` fields and diag_coverage_chronos.json; the same
numbers as paper/tables/coverage.tex (make_paper_tables.py). Usage: python render_coverage.py results
"""
import json, os, sys
from collections import defaultdict
from statistics import mean
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
ORDER = [("amazon/chronos-t5-small", "Chronos-small"), ("chronos-bolt-small", "Chronos-Bolt"), ("chronos-2", "Chronos-2"),
         ("tirex", "TiRex"), ("moirai-1.1-R-small", "Moirai-small"), ("moirai2", "Moirai-2.0"), ("timesfm-2.0-500m", "TimesFM-2.0"),
         ("timesfm25", "TimesFM-2.5"), ("sundial", "Sundial"), ("fincast_v1", "FinCast"), ("timemoe-200m", "Time-MoE-200M")]
rows = defaultdict(lambda: defaultdict(list))
for l in open(os.path.join(RES, "pilot_a.jsonl")):
    if not l.strip(): continue
    r = json.loads(l)
    if r.get("stage") == "ladder" and r.get("init") == "pretrained" and "coverage_80" in r:
        rows[r["model"]][r["rung"]].append(r)
p = os.path.join(RES, "diag_coverage_chronos.json")
if os.path.exists(p):
    for r in json.load(open(p)):
        rows["amazon/chronos-t5-small"][r["rung"]].append(r)
NORM = 2 * 1.2815515655446004
print("### 4.15 零假设下的区间校准：80% 区间的覆盖率与宽度（`coverage_80`，`diag_coverage_chronos.py`）\n")
print("每个有区间的模型在阶梯上 0.1–0.9 分位数带（采样模型取样本分位数、分位数模型取自身分位数）的实际覆盖率，"
      "以及带宽相对鞅自身 80% 带 2.56σ√h 的比值；h=1 与 h=16，三个种子平均。Time-MoE 只输出点预测，没有区间。\n")
print("| 模型 | 档 | 覆盖 h=1 | 覆盖 h=16 | 宽/2.56σ√h h=1 | h=16 |")
print("|---|---|---|---|---|---|")
for m, lab in ORDER:
    if m not in rows:
        print(f"| {lab} | — | 无区间 | | | |"); continue
    for rung in ("N1", "N2", "N3_nu3", "N3_nu5", "N4"):
        rs = rows[m].get(rung, [])
        if not rs: continue
        c1, c16 = mean(r["coverage_80"][0] for r in rs), mean(r["coverage_80"][15] for r in rs)
        w1, w16 = mean(r["interval80_width_over_sigma"][0] for r in rs) / NORM, mean(r["interval80_width_over_sigma"][15] for r in rs) / (NORM * 4)
        print(f"| {lab} | {rung} | {c1:.2f} | {c16:.2f} | {w1:.2f} | {w16:.2f} |")
n1 = {m: rows[m]["N1"] for m, _ in ORDER if m in rows and rows[m].get("N1")}
cov1 = {m: mean(r["coverage_80"][0] for r in rs) for m, rs in n1.items()}; cov16 = {m: mean(r["coverage_80"][15] for r in rs) for m, rs in n1.items()}
others = [m for m in n1 if m != "sundial"]
print(f"\n**读法。** N1 上，除 Sundial 外的 {len(others)} 个有区间的模型基本校准：h=1 覆盖 {min(cov1[m] for m in others):.2f}–{max(cov1[m] for m in others):.2f}，"
      f"h=16 覆盖 {min(cov16[m] for m in others):.2f}–{max(cov16[m] for m in others):.2f}（名义 0.80）——均值偏了四分之一步，区间照样罩住未来，"
      f"这就是 §3.1 说的「区间与点预测是两个互不牵连的输出」的直接证据。"
      + (f"例外是 Sundial：生成式样本在 h=1 处只罩住 {cov1['sundial']:.2f}、h=16 处 {cov16['sundial']:.2f}，带宽只有鞅的 "
         f"{mean(r['interval80_width_over_sigma'][0] for r in n1['sundial']) / NORM:.2f} 与 {mean(r['interval80_width_over_sigma'][15] for r in n1['sundial']) / (NORM * 4):.2f} 倍：它的流匹配头相信下一步比随机游走可预测得多，"
         "同一个语料信念，从位置换到了散布上表达。" if "sundial" in n1 else ""))
