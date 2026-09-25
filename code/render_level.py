"""Render the direction-vs-level sweep (results/diag_direction_level.jsonl), numpy-free."""
import json, sys
J = sys.argv[1] if len(sys.argv) > 1 else "results/diag_direction_level.jsonl"
rows = sorted((json.loads(l) for l in open(J) if l.strip()), key=lambda r: r["level"])
P = print
P("| 水平 | mean\\|ctx\\| | 网格(σ) | h=1 均值 | h=1 向上 | h=16 均值 | h=128 均值 | h=128 中位数 | h=128 向上 |")
P("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    m, md, u = r["mean_dep_sigma"], r["median_dep_sigma"], r["frac_up"]
    P(f"| {r['level']:+g} | {r['mean_abs_context']:.1f} | {r['grid_step_sigma']:.3f} | {m['1']:+.3f}σ | {u['1']:.2f} | "
      f"{m['16']:+.2f}σ | **{m['128']:+.2f}σ** | {md['128']:+.2f}σ | **{u['128']:.2f}** |")
ups = [r["frac_up"]["128"] for r in rows]
P("")
P(f"{len(rows)} 个水平全部向上（h=128 向上比例 {min(ups):.2f}–{max(ups):.2f}）。" if min(ups) > 0.5
  else f"有水平不向上：{[r['level'] for r in rows if r['frac_up']['128'] <= 0.5]}。")
