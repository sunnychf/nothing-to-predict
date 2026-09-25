"""Render RESULTS_PILOT_A.md section 4.19 (the controlled architecture comparison) from results/arch_summary.json.
Usage: python render_arch_md.py results"""
import json, os, sys
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
S = json.load(open(os.path.join(RES, "arch_summary.json")))
ARCH = {"encoder": "掩码编码器", "decoder": "纯解码器"}; CORP = {"up": "增长语料", "sym": "对称语料"}
any_ = next(iter(S.values()))
print("### 4.19 受控的架构对照：同一合成语料上从头训练的掩码编码器与纯解码器（`arch_corpus.py` → `arch_train.py` → `arch_summary.py`）\n")
print(f"第二轮审稿之后（M24 评分建议 2）：十一个发布模型里语料、架构、损失、归一化一起变，架构划分只能是观察。这里把数据固定：同一条合成「通用」语料流"
      f"（季节 AR 0.45、带趋势 0.25、带漂移随机游走 0.20、N4 式弹跳 0.10；「增长语料」的趋势与漂移均值为每步 +0.02，「对称语料」为零），"
      f"同样的块（4 层，d=128，4 头，补丁 16，{any_['params']/1e6:.2f}M 参数）、同样的优化器与步数（{any_['steps']} 步 × 256），只换架构："
      "掩码编码器（上下文 32 个补丁 + 8 个 [MASK]，双向注意力，一次解出未来 8 个补丁，损失只在未来上）对纯解码器（因果注意力，每个位置预测下一个补丁，推理时自回归 8 步）。"
      "三个训练种子。探针与发布模型完全相同：方向探针（N1，种子 4000，128 条，H=128）、阶梯 N1/N4（种子 1000–1002，512 条，H=16）、语料留出集 MSE（归一化单位）。\n")
print("| 语料 | 架构 | 留出 MSE（持久性） | N1 h=128 均值偏离/σ（种子间 sd） | 向上 | 偶份额 | 斜率相关 | N1 skill h=16 | N4 份额（精确最优） |")
print("|---|---|---|---|---|---|---|---|---|")
for corpus in ("up", "sym"):
    for arch in ("encoder", "decoder"):
        k = f"{arch}|{corpus}"
        if k not in S: continue
        r = S[k]; d = r["dir"]["128"]
        print(f"| {CORP[corpus]} | {ARCH[arch]} | {r['heldout_mse'][0]:.3f} ({r['heldout_persistence_mse'][0]:.3f}) | {d['mean_dep'][0]:+.2f} ({d['mean_dep'][1]:.2f}) | {d['frac_up'][0]:.2f} | {d['even_share'][0]:.2f} | {d['corr_slope64'][0]:+.2f} | {r['ladder']['N1']['skill_16'][0]:+.3f} | {r['ladder']['N4']['share_exact_1'][0]:.2f} |")
print("\n逐种子（h=128 均值偏离，向上比例）：" + "；".join(f"{CORP[k.split('|')[1]]}/{ARCH[k.split('|')[0]]} " + ", ".join(f"s{s}: {m:+.2f}/{u:.2f}" for s, m, u in r["per_seed_dir128"]) for k, r in S.items()) + "。")
eu, du, es, ds = S.get("encoder|up"), S.get("decoder|up"), S.get("encoder|sym"), S.get("decoder|sym")
if all((eu, du, es, ds)):
    print(f"\n**读法。** 增长语料让两种架构都在随机游走上向上漂（h=128：编码器 {eu['dir']['128']['mean_dep'][0]:+.2f}、解码器 {du['dir']['128']['mean_dep'][0]:+.2f}，向上 {eu['dir']['128']['frac_up'][0]:.2f} 与 {du['dir']['128']['frac_up'][0]:.2f}，偶份额 {eu['dir']['128']['even_share'][0]:.2f} 与 {du['dir']['128']['even_share'][0]:.2f}）；"
          f"对称语料下掩码编码器不漂（{es['dir']['128']['mean_dep'][0]:+.2f}，种子间 sd {es['dir']['128']['mean_dep'][1]:.2f}），解码器均值 {ds['dir']['128']['mean_dep'][0]:+.2f}（sd {ds['dir']['128']['mean_dep'][1]:.2f}）："
          + f"{sum(1 for _, m, _ in ds['per_seed_dir128'] if m > 2 * ds['dir_mean_dep_se_within']['128'])} 个种子在两倍标准误之外向上打破了对称、"
          + f"{sum(1 for _, m, _ in ds['per_seed_dir128'] if m < -2 * ds['dir_mean_dep_se_within']['128'])} 个向下，三个种子分不清这是随机的对称破缺还是系统的。"
          "在这个受控设定里，无结构输入上的方向由语料的增长常态决定，两种架构同等地传递它，架构本身不决定（这里反而是解码器漂得更多）：十一个发布模型上的「非解码器向上、纯解码器不漂」划分，不能由架构单独复现，"
          "只能来自与架构一起变化的其他东西（语料构成、分词、损失、归一化）。这是对 §6.4「与架构一致、不是检验」措辞的一次直接支持，也是对「架构导致」读法的否定。"
          "两种架构对语料的拟合相当（留出 MSE 相差不到 5%），N4 份额都在 0.3–0.5，说明两者都学到了语料里的弹跳结构，差异不是欠拟合。")
