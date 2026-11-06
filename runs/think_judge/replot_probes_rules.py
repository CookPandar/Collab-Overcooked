"""Regenerate eval/probes/probe_l2v3_rules_{direct,think,delta}_{chef,assistant}.png/.txt (run from runs/think_judge)."""
import json, subprocess, sys
from pathlib import Path
def l2(arm):
    p = Path(f"eval/all_levels/think_{arm}_u50_l2.jsonl")
    return f"，L2 {json.loads(p.read_text().splitlines()[-1])['success_rate']:.2f}" if p.exists() and p.stat().st_size else ""
SPECS = [("base", "基座 Qwen3-1.7B", True), ("sft_l12", "SFT think L1+2（起点）", True),
         ("newsum_u50", "LLM 评委 sum u50（参考）", True)]
for arm in ["none", "mate", "mate_v2", "plan", "full", "ground_split", "full_v2_split"]:
    SPECS.append((f"r{arm}_u50", f"r{arm} u50（规则 {arm}{l2('l2r_' + arm)}）", False))
P = "eval/probes"
for role, R in (("chef", "Chef"), ("assistant", "Assistant")):
    for mode, desc in (("direct", "直接作答（每类 150 题）"), ("think", "先思考再作答（每类 100 题）"),
                       ("delta", "先思考再作答 − 直接作答（同一批题，每格约 100 题，差值 95% CI 约 ±13）")):
        sub = "" if mode == "delta" else "；热图：相对 SFT 起点的变化（百分点），括号内为准确率；每格 95% CI 约 ±10–13"
        title = f"{R}：L2 探针题，规则奖励消融，{desc}{sub}；选项轮换"
        out = f"{P}/probe_l2v3_rules_{mode}_{role}"
        args = [sys.executable, "plot_probes_rules.py", mode, role, f"{P}/scores_l2v3_direct.jsonl",
                f"{P}/scores_l2v3_think.jsonl", out + ".png", title, "sft_l12"] + \
               [f"{n}={l}" + ("=ref" if r else "") for n, l, r in SPECS]
        txt = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        Path(out + ".txt").write_text(txt)
        print(out, txt.splitlines()[0] if txt.startswith("missing") else "")
