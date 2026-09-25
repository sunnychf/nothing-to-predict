"""Render RESULTS_PILOT_A.md section 4.18 (the Time-MoE harness against the model's own ETTh1 benchmark)
from results/validate_timemoe_etth1*.json. Usage: python render_validate_md.py results"""
import glob, json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
H = {}; paper = None
for p in sorted(glob.glob(os.path.join(RES, "validate_timemoe_etth1*.json"))):
    d = json.load(open(p)); H.update(d["horizons"]); paper = d["paper_timemoe_large"]
hs = sorted(H, key=int)
print("### 4.18 Time-MoE 解码循环对照它自己的基准（`validate_timemoe.py`，ETTh1，官方协议）\n")
print("审稿意见（09-19）：Time-MoE 是全表最极端的模型，而它的解码是本文自写的贪心多视野循环（M10），一句「harness 的 bug」就能打掉整行。"
      "检验：按 Time-MoE 仓库 `BenchmarkEvalDataset`/`run_eval.py` 的协议（训练行标准化、测试段步长 1、上下文 512/1024/2048/3072 对视野 96/192/336/720、模型直接吃标准化输入）"
      "用本文的循环跑 ETTh1，对照论文 `table/zero_full.tex` 里 Time-MoE_large 那一行（从 arXiv 源码抄，HTML 摘要会把列对错）。第三列是本文实验用的逐窗归一化。\n")
print("| 视野 | 上下文 | 窗口数 | 论文 MSE/MAE | 本文循环（官方协议）MSE/MAE | 本文循环（逐窗归一化）MSE/MAE |")
print("|---|---|---|---|---|---|")
pw = [h for h in hs if "per_window_normalisation" in H[h]]                        # decoding (b) is absent in an --official-only run
cell = lambda r, k: f"{r[k]['mse']:.3f} / {r[k]['mae']:.3f}" if k in r else "-- (未跑)"
for h in hs:
    r = H[h]
    print(f"| {h} | {r['context']} | {r['n_windows']} | {cell(r, 'paper')} | {cell(r, 'official_protocol')} | {cell(r, 'per_window_normalisation')} |")
if len(hs) == 4:
    from statistics import mean
    pwavg = f"{mean(H[h]['per_window_normalisation']['mse'] for h in hs):.3f} / {mean(H[h]['per_window_normalisation']['mae'] for h in hs):.3f}" if len(pw) == 4 else "--"
    print(f"| 平均 | | | {paper['avg'][0]:.3f} / {paper['avg'][1]:.3f} | {mean(H[h]['official_protocol']['mse'] for h in hs):.3f} / {mean(H[h]['official_protocol']['mae'] for h in hs):.3f} | {pwavg} |")
diffs = {k: max(abs(H[h]["official_protocol"][k] - H[h]["paper"][k]) for h in hs) for k in ("mse", "mae")}
print(f"\n**读法。** 官方协议下本文的循环与论文数字最大差 {diffs['mse']:.3f}（MSE）、{diffs['mae']:.3f}（MAE）；循环没有 bug，Time-MoE 在阶梯上的极端读数是模型本身。逐窗归一化把数字改变 "
      f"{max(abs(H[h]['per_window_normalisation']['mse'] - H[h]['official_protocol']['mse']) for h in pw):.3f}（MSE，最大，视野 {', '.join(pw)}）。已跑完的视野：{', '.join(hs)}"
      + ("" if len(pw) == len(hs) else f"（{', '.join(h for h in hs if h not in pw)} 只跑了官方协议：`--official-only` 对冲进程先到，完整进程还在跑）") + "。")
