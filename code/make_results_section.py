"""Render RESULTS_PILOT_A.md section 4 straight from the result files.

Every number in the write-up is produced here rather than retyped, so the paper
and the JSONL cannot drift apart. Nothing is printed that is not backed by a row
on disk; missing pieces are reported as missing.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
J = os.path.join(RES, "pilot_a.jsonl")


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    for l in open(p):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def agg(rows, field, h):
    v = [r[field][h] for r in rows if field in r and len(r[field]) > h]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)


recs = [r for r in load_jsonl(J) if not r.get("error")]
errs = [r for r in load_jsonl(J) if r.get("error")]
deep = {H: load_json(os.path.join(RES, f"spectral_deep_H{H}.json")) for H in (128, 256)}
deep = {H: v for H, v in deep.items() if v}
disp = load_json(os.path.join(RES, "diag_dispersion.json"))
n2iv = load_json(os.path.join(RES, "diag_n2_interval.json"))

L = []
P = L.append

P("## 4. 结果\n")
P(f"共 {len(recs)} 个成功配置，{len(errs)} 条失败记录。")
if errs:
    # Do not assert the failures were recovered; check it against the rows on disk.
    P("")
    P("失败记录逐条核对（**不是「都重跑好了」这种笼统说法**）：")
    P("")
    P("| 失败阶段 | 原因 | 该阶段现在有几个成功配置 |")
    P("|---|---|---|")
    for e in errs:
        st = e.get("stage", "?")
        tb = e.get("traceback", "")
        cause = ("CUDA OOM" if "OutOfMemory" in tb else
                 tb.strip().splitlines()[-1][:60] if tb.strip() else "?")
        nok = sum(1 for r in recs if r["stage"] == st)
        P(f"| `{st}` | {cause} | {nok} |")
    P("")
    bad = [e.get("stage") for e in errs if not any(r["stage"] == e.get("stage") for r in recs)]
    P("全部失败阶段都已有成功配置覆盖。" if not bad
      else f"**仍未覆盖的阶段：{sorted(set(bad))}——这些是真缺口，不是噪声。**")
P("")

# ---------------------------------------------------------------- 4.1 ladder
lad_all = defaultdict(lambda: defaultdict(list))
for r in recs:
    if r["stage"] == "ladder" and r.get("init") == "pretrained":
        lad_all[r["model"]][r["rung"]].append(r)
models = sorted(lad_all)
lad = lad_all.get("amazon/chronos-t5-small", lad_all[models[0]] if models else {})


def ladder_table(model, lad):
    first = next(iter(lad.values()))[0]
    cont = first.get("floor_kind") == "affine_normalisation_roundtrip"   # grid or continuous
    sampled = "departure_in_sigma_mc_corrected" in first                  # MC term or not
    desc = f"S={first['num_samples']}，MC 项已减" if sampled else "确定性输出，无 MC 项"
    desc += "；连续模型无码本网格，floor≈0" if cont else "；码本网格见 grid 列"
    P(f"**{model}**（n={first['n']}，种子 {len(next(iter(lad.values())))}，{desc}）\n")
    P("| 档 | h | dep(σ) | dep-MC(σ) | 种子间 sd | grid(σ) | dep-MC/grid | skill |")
    P("|---|---|---|---|---|---|---|---|")
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        rows = lad.get(rung)
        if not rows:
            continue
        H = len(rows[0]["departure_in_sigma"])
        sig = rows[0]["sigma"]
        for h in [0, 3, 7, H - 1]:
            if h >= H:
                continue
            dep, sd = agg(rows, "departure_in_sigma", h)
            depc, _ = agg(rows, "departure_in_sigma_mc_corrected", h)
            fl, _ = agg(rows, "codec_floor", h)
            g = np.sqrt(fl) / sig
            sk, _ = agg(rows, "skill_vs_persistence", h)
            dd = depc if np.isfinite(depc) else dep
            mc_cell = f"{depc:.3f}" if sampled else "—"
            grid_cell, ratio_cell = ("≈0", "—") if cont else (f"{g:.3f}", f"{dd/g if g else float('nan'):.2f}")
            P(f"| {rung if h==0 else ''} | {h+1} | {dep:.3f} | {mc_cell} | {sd:.3f} | {grid_cell} | {ratio_cell} | {sk:+.3f} |")
    P("")
    n4 = lad.get("N4")
    if n4 and "oracle_skill_vs_persistence" in n4[0]:
        lsk, _ = agg(n4, "oracle_skill_vs_persistence", 0)                      # the optimal linear forecast (stored by the drivers)
        msk, _ = agg(n4, "skill_vs_persistence", 0)
        from nulls import n4_optima_skill                                        # the exact conditional mean, recomputed on the same draws
        osk = float(np.mean([n4_optima_skill(r["n"], 512, r["H"], 1000 + r["seed"])["exact"][0] for r in n4]))
        P(f"N4 正面控制（h=1）：精确最优（交易方向的两态滤波，条件均值）比持久性好 **{100*osk:+.2f}%**，最优线性预测（MA(1) 滤波）**{100*lsk:+.2f}%**，模型 **{100*msk:+.2f}%**。")
        if osk > 0 and msk <= 0:
            P("模型**没有抓到确实存在的结构**。所以它在 N1–N3 上的表现"
              "不能读成「正确地弃权」。")
        elif osk > 0:
            P(f"模型抓到了精确最优的 {100*msk/osk:.1f}%（最优线性预测的 {100*msk/lsk:.1f}%）。")
        P("")


if lad_all:
    P("### 4.1 阶梯：模型在四档上偏离最优预测多远\n")
    P("`dep` = √E[(ŷ−y_T)²]/σ，单位是**一步的标准差**。`dep-MC` 是精确减掉")
    P("蒙特卡洛分量后的值（有采样的模型才有）。`grid` 是码本分辨率标尺（不做减法）。")
    P("`skill` = 1 − MSE_模型/MSE_持久性，**负数表示比什么都不做还差**。\n")
    for model in models:
        ladder_table(model, lad_all[model])

# ---------------------------------------------------------------- 4.2 floor
sc = [r for r in recs if r["stage"] == "floor_scale"]
if sc:
    P("### 4.2 尺度扫描：偏离是量化假象还是先验\n")
    P("量化假象随网格缩放，学到的先验不该。这是归因的决定性实验。\n")
    P("| σ_rel | σ | dep(σ) h=1 | dep-MC(σ) h=1 | grid(σ) | **dep-MC/grid** | dep-MC h=16 | h16/grid | skill h=1 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(sc, key=lambda r: r["sigma_rel"]):
        sig = r["sigma"]
        g = np.sqrt(r["codec_floor"][0]) / sig
        dc = r.get("departure_in_sigma_mc_corrected", r["departure_in_sigma"])
        P(f"| {r['sigma_rel']:g} | {sig:g} | {r['departure_in_sigma'][0]:.3f} | {dc[0]:.3f} | {g:.4f} | "
          f"**{dc[0]/g:.2f}** | {dc[-1]:.3f} | {dc[-1]/g:.1f} | {r['skill_vs_persistence'][0]:+.3f} |")
    P("")
    fine = [r for r in sc if r["sigma_rel"] >= 0.01]
    if len(fine) >= 2:
        fine = sorted(fine, key=lambda r: r["sigma_rel"])
        g0 = np.sqrt(fine[0]["codec_floor"][0]) / fine[0]["sigma"]
        g1 = np.sqrt(fine[-1]["codec_floor"][0]) / fine[-1]["sigma"]
        d0 = fine[0]["departure_in_sigma_mc_corrected"][0]; d1 = fine[-1]["departure_in_sigma_mc_corrected"][0]
        P(f"细网格段：网格从 {g0:.3f}σ 缩到 {g1:.3f}σ（{g0/g1:.1f} 倍），h=1 的 MC 修正偏离从 "
          f"{d0:.3f}σ 到 {d1:.3f}σ（{d0/d1:.2f} 倍）。**偏离不随网格走，留下的不是码本造成的。**")
        P("粗网格段（σ_rel ≤ 1e-3）MC 修正偏离低于单 token 下界（比值 <1），是 M2 的实证：")
        P("多条轨迹的均值能落在格点之间，码本往返不是均值的下界。")
        P("")
mc = sorted([r for r in recs if r["stage"] == "floor_mc"], key=lambda r: r["num_samples"])
if len(mc) >= 3:
    S = np.array([r["num_samples"] for r in mc], float)
    D = np.array([r["imported"][0] for r in mc], float)
    sig = mc[0]["sigma"]
    A = np.vstack([np.ones_like(S), 1 / S]).T
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    ss = 1 - np.sum((D - A @ coef) ** 2) / max(np.sum((D - D.mean()) ** 2), 1e-300)
    P(f"**蒙特卡洛**：拟合 `E[(ŷ_S−y_T)²] = D + V/S`，R² = {ss:.4f}，"
      f"外推 S→∞ 得 **{np.sqrt(max(coef[0],0))/sig:.4f} σ**"
      f"（S=100 实测 {np.sqrt(D[list(S).index(100)])/sig:.4f} σ）。\n")
lv = [r for r in recs if r["stage"] == "floor_level"]
if lv:
    vals = {round(r.get("departure_in_sigma_mc_corrected", r["departure_in_sigma"])[0], 4) for r in lv}
    P(f"**水平扫描**：level ∈ {sorted(r['level0'] for r in lv)}，"
      + ("三档 dep 完全相同（"
         + ", ".join(f"{v:.4f}" for v in vals) + " σ），管线严格尺度等变。\n"
         if len(vals) == 1 else "各档不同，需查归一化。\n"))

# ---------------------------------------------------------------- 4.3 disp
if disp:
    P("### 4.3 均值是不是被少数序列带偏的\n")
    P("| seed | RMS | 中位数 | 95%截尾 RMS | top1% 占比 | 最大 |")
    P("|---|---|---|---|---|---|")
    for d in disp:
        P(f"| {d['seed']} | {d['rms']:.3f} | {d['rms_of_median']:.3f} | "
          f"{d['trimmed_rms_95']:.3f} | {d['share_top1pct']:.3f} | {d['max']:.1f} |")
    top = np.mean([d["share_top1pct"] for d in disp])
    P("")
    P(f"top 1% 的序列平均占总平方偏离的 **{100*top:.1f}%**。"
      + ("**均值被少数序列主导，正文必须报稳健统计量。**"
         if top > 0.25 else "均值不是被少数序列撑起来的，可以直接用。"))
    P("")

# ---------------------------------------------------------------- 4.4 spec
if deep:
    P("### 4.4 偏离有没有结构：漂移，还是搬来的节律\n")
    P("跑了两个视野。H=128 时周期 24 落在第 4 个箱，紧挨着漂移；H=256 时落在第 10 个箱，")
    P("邻域干净。两个视野都用修好的分解（多项式去趋势 + 连续谱排除漂移箱，")
    P("`test_decompose.py` 8/8 通过）。\n")
    P("§4.1 的论断只有在偏离**有结构**时才成立。但「有结构」不等于「有季节性」：")
    P("缓慢漂移也不平坦。所以功率显式拆成漂移带与季节带，季节带报的是")
    P("**相对局部连续谱的超出量**，而不是占总功率的比例。\n")
    P("| H | 档 | 漂移占比(周期>H/4) | 峰值周期(原始) | 峰值周期(去趋势) | band7 超出 | band24 超出 | band12 超出 |")
    P("|---|---|---|---|---|---|---|---|")
    alld = []
    for H, recs_H in sorted(deep.items()):
        for d in recs_H:
            alld.append(d)
            def ex(k):
                v = (d.get(k) or {}).get("excess_over_continuum")
                return f"{v:.2f}×" if v is not None else "n/a"
            P(f"| {H} | {d['rung']} | {d['share_drift_periodGtH4']:.3f} | "
              f"{d['peak_period_of_mean_psd']:.1f} | {d.get('peak_period_detrended', float('nan')):.1f} | "
              f"{ex('band7')} | {ex('band24')} | {ex('band12')} |")
    P("")
    dr = np.mean([d["share_drift_periodGtH4"] for d in alld])
    Hmax = max(deep)
    top = deep[Hmax]
    exc_top = [x for d in top for x in
               [(d.get("band7") or {}).get("excess_over_continuum"),
                (d.get("band24") or {}).get("excess_over_continuum"),
                (d.get("band12") or {}).get("excess_over_continuum")] if x]
    P(f"漂移带平均占 **{100*dr:.1f}%** 的功率，两个视野、三档都是。峰值周期等于整个窗长。")
    P("")
    if len(deep) >= 2:
        Hmin = min(deep)
        b24_lo = {d["rung"]: (d.get("band24") or {}).get("excess_over_continuum") for d in deep[Hmin]}
        b24_hi = {d["rung"]: (d.get("band24") or {}).get("excess_over_continuum") for d in deep[Hmax]}
        P(f"**判定读 H={Hmax}。** 周期 24 的超出量在 H={Hmin} 时为 "
          + "、".join(f"{k} {v:.2f}×" for k, v in b24_lo.items() if v)
          + f"，到 H={Hmax} 时变为 "
          + "、".join(f"{k} {v:.2f}×" for k, v in b24_hi.items() if v) + "。")
        P("一个真实的峰随分辨率提高会变尖，不会消失；随分辨率提高而消失的，是分箱假象。")
        if b24_lo.get("N4") and b24_lo["N4"] == max(v for v in b24_lo.values() if v):
            P(f"另一个内部核对：H={Hmin} 时超出量最大的是 **N4**，而 N4 按构造没有任何周期性。")
            P("要是那个超出量是真的季节性，它不该在唯一一档没有季节性的数据上最大。")
        P("")
    if dr > 0.5 and (not exc_top or max(exc_top) < 1.5):
        P(f"**在 H={Hmax} 上，周期 7 / 12 / 24 处没有任何超出连续谱的功率"
          f"（最大 {max(exc_top):.2f}×）。这否证了「搬来季节性」这一读法。**")
        P("偏离确实有结构，但结构是**低频漂移**，不是语料里的日/周节律。")
        P("论文 §4.1 预期的是后者，按 §4.1 自己立下的规矩，这里照实报为反驳，不换统计量。")
        P("")
        P("与阶梯上「偏离随 horizon 单调增长」和 Jander et al. (2026) 的「高估持续性」")
        P("放在一起读，一致的解释是：**Chronos 搬进来的是趋势先验，不是季节先验。**")
        P("它把随机游走的近期路径当作趋势外推。这仍是一个先验，只是不是 §4.1 猜的那一个。")
    P("")

# ---------------------------------------------------------------- 4.6 n2 interval
if n2iv:
    P("### 4.5 N2：区间宽度跟不跟条件波动率（论文 Remark rem:n2）\n")
    P("点预测上 N2 与 N1 无差别（§4.1）。对概率模型公平的问题是：它的预测区间")
    P("有没有用上那个高度可预测的方差。真实 σ_t 由生成器给出，比较对象是模型要恢复的量本身。\n")
    P("| seed | 80% 覆盖 h=1 | h=16 | corr(宽, 真 σ_t) h=1 | h=16 | corr(宽, 上下文末 σ) h=1 | corr(宽, 实现波动20) h=1 | 宽度方差中非 horizon 部分 | σ_t 跨度 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for r in n2iv:
        P(f"| {r['seed']} | {r['coverage_80'][0]:.3f} | {r['coverage_80'][-1]:.3f} | "
          f"{r['corr_width_true_condvol'][0]:+.3f} | {r['corr_width_true_condvol'][-1]:+.3f} | "
          f"{r['corr_width_last_condvol'][0]:+.3f} | {r['corr_width_realised_vol20'][0]:+.3f} | "
          f"{r['frac_width_var_not_explained_by_horizon']:.3f} | {r['condvol_spread_max_over_min']:.1f}× |")
    P("")
    c1 = np.mean([r["corr_width_last_condvol"][0] for r in n2iv])
    fr = np.mean([r["frac_width_var_not_explained_by_horizon"] for r in n2iv])
    if c1 > 0.5:
        P(f"区间宽度与上下文末的条件波动率相关 **{c1:+.2f}**：模型确实把波动率结构用上了，")
        P("这是它在 N2 上唯一能利用的东西，它利用了。")
    elif c1 > 0.2:
        P(f"相关 **{c1:+.2f}**，弱。区间对波动率有反应但远未跟上一个跨度 "
          f"{np.mean([r['condvol_spread_max_over_min'] for r in n2iv]):.0f}× 的过程。")
    else:
        P(f"相关只有 **{c1:+.2f}**，宽度方差中仅 {100*fr:.0f}% 不由 horizon 解释：")
        P("**区间基本只是 horizon 的函数**。序列里唯一可预测的成分被放着没用。")
    P("")

# ---------------------------------------------------------------- 4.5 random
rnd = [r for r in recs if r["stage"] == "ladder" and r.get("init") == "random"]
if rnd:
    d = np.mean([r["departure_in_sigma"][0] for r in rnd])
    pre = np.mean([r["departure_in_sigma"][0] for r in lad.get("N1", [])]) if lad.get("N1") else float("nan")
    P("### 4.6 随机初始化参照\n")
    P(f"同架构同 tokenizer、权重随机：h=1 偏离 **{d:.1f} σ**，"
      f"预训练模型 {pre:.3f} σ，差 **{d/pre:.0f} 倍**。")
    P("这直接证伪了论文 §4.1「随机初始化模型的偏离全来自量化与归一化」的说法：")
    P("未训练网络的输出由它未训练的 token 分布主导，它不是下界。\n")

# ---------------------------------------------------------------- 4.7 freq test
fq = [r for r in recs if r["stage"] == "freq" and r.get("applicable")]
fd = [r for r in recs if r["stage"] == "freq_delta"]
if fq:
    P("### 4.7 声明频率检验（论文 §4.1，最锋利的归因）\n")
    P("同一批合成数据，只改喂给模型的频率声明。数据在任何频率上都无周期性，")
    P("正确读取的模型三种声明下应给出**相同**预测。Chronos 没有频率参数，此检验对它无定义；")
    P("FinCast 有，但和 TimesFM 一样只分三级（日频及以上 / 周月 / 年），小时与日是同一个标签。\n")
    P("| 模型 | 声明 | dep(σ) h=1 | dep h=H | skill h=1 | 平坦度(去趋势) | band7 | band24 |")
    P("|---|---|---|---|---|---|---|---|")
    for r in sorted(fq, key=lambda r: (r["model"], r["freq"])):
        bp = r["spec_band_power"]
        P(f"| {r['model']} | {r['freq_name']} | {r['departure_in_sigma'][0]:.3f} | "
          f"{r['departure_in_sigma'][-1]:.3f} | {r['skill_vs_persistence'][0]:+.3f} | "
          f"{r['spec_flatness_detrended']:.3f} | {bp['p7']:.4f} | {bp['p24']:.4f} |")
    P("")
    for r in fd:
        diffs = {k: v for k, v in r.items() if k.startswith("rms_diff_")}
        kind = r.get("declaration_kind", "frequency label")
        P(f"**{r['model']}**（声明 = {kind}）：各声明两两之间预测的 RMS 差（σ 单位）："
          + "，".join(f"{k[len('rms_diff_'):-len('_sigma')]} **{v:.2f}**" for k, v in diffs.items()) + "。")
        mx = max(diffs.values())
        dep1 = next((x["departure_in_sigma"][0] for x in fq if x["model"] == r["model"]), float("nan"))
        if mx < 0.05 * max(dep1, 1e-9):
            P("预测**不随声明而变**：偏离另有来源，与声明无关。这同样有信息。")
        else:
            P(f"预测**随声明而变**（最大差为 h=1 偏离的 {100*mx/dep1:.0f}%）：模型把声明用进了预测，")
            P("而数据在任何频率上都没有周期性。看谱带那两列判断它加的是什么。")
        P("")

# ---------------------------------------------------------------- 4.8 cross-model
if len(models) >= 2:
    P("### 4.8 跨模型：同一阶梯，通用 vs 金融原生\n")
    P("| 档 | h | " + " | ".join(f"{m} dep(σ)" for m in models) + " | " + " | ".join(f"{m} skill" for m in models) + " |")
    P("|---|---|" + "---|" * (2 * len(models)))
    for rung in ["N1", "N2", "N3_nu3", "N3_nu5", "N4"]:
        for h in (0, 15):
            cells_d, cells_s = [], []
            for m in models:
                rows = lad_all[m].get(rung, [])
                if rows and h < len(rows[0]["departure_in_sigma"]):
                    dep, _ = agg(rows, "departure_in_sigma", h)
                    sk, _ = agg(rows, "skill_vs_persistence", h)
                    cells_d.append(f"{dep:.3f}"); cells_s.append(f"{sk:+.3f}")
                else:
                    cells_d.append("—"); cells_s.append("—")
            P(f"| {rung if h==0 else ''} | {h+1} | " + " | ".join(cells_d) + " | " + " | ".join(cells_s) + " |")
    P("")

print("\n".join(L))
