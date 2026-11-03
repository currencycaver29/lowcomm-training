"""Fills main_template.tex from results/ -> main.tex"""
import json, os, glob, sys, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
sys.path.insert(0, os.path.join(H, "..")); os.chdir(os.path.join(H, "..")); from analyze import load_sim
sim, _ = load_sim()
agg = sim.groupby(["name", "method", "H", "H_g", "bits", "wan_syncs"]).final_val.agg(["mean", "min", "max"]).reset_index().set_index("name")
T = pd.read_csv(f"{R}/wallclock.csv")
rt = pd.read_csv(f"{R}/comm/rounds.csv").groupby(["profile", "scheme", "bits"]).round_s.median()
val = json.load(open(f"{R}/validation.json")); best = json.load(open(f"{R}/sim/best_hparams.json"))
author = json.load(open(f"{H}/author.json")) if os.path.exists(f"{H}/author.json") else {}
L = lambda n: agg.loc[n, "mean"]
tmin = lambda label, p: T[(T.label == label) & (T.profile == p)].total_s.iloc[0] / 60
HI8, DDP, DI8 = "Hierarchical H=10/50, INT8", "DDP (sync every step)", "DiLoCo H=50, INT8"
comp = [abs(L(f"{b}_b{q}") - L(f"{b}_b32")) for b in ("diloco_H50", "diloco_H100", "hier_Hr10_Hg50", "hier_Hr10_Hg100") for q in (8, 4)]
t_c = float(np.mean([json.load(open(f))["t_compute_per_step"] for f in glob.glob(f"{R}/dist/*/summary.json")]))
def under(tag, bits):
    v = val[tag]; pred = 300 * v["t_compute_per_step"] + 6 * rt[("constrained", "flat", bits)]
    return (v["t_total"] - pred) / v["t_total"] * 100
b2b = max(abs(rt[(p, "flat_b2b", 32)] / rt[(p, "flat", 32)] - 1) for p in ["regional", "intercontinental", "constrained", "poor"]) * 100
sci = lambda x: (lambda m, e: f"${float(m):g}\\times10^{{{int(e)}}}$")(*f"{x:.1e}".split("e"))
num = {
    "AUTHOR": author.get("name", "Author Name"),
    "REPO_URL": f"at \\url{{{author['repo']}}}" if author.get("repo") else "in the Zenodo record of this preprint",
    "n_params": f"{int(sim.iloc[0:1].shape[0] and 351744):,}", "kb32": f"{1406976 / 1000:.0f}", "kb8": f"{357240 / 1000:.0f}", "kb4": f"{181368 / 1000:.0f}",
    "ddp_lr": sci(best["ddp"]["lr"]), "dil_lr": sci(best["diloco"]["lr"]), "dil_olr": f"{best['diloco']['outer_lr']}",
    "tc_ms": f"{t_c * 1000:.0f}", "compute_min": f"{2000 * t_c / 60:.1f}",
    "ddp": f"{L('ddp'):.3f}", "dil10": f"{L('diloco_H10_b32'):.3f}", "dil50": f"{L('diloco_H50_b32'):.3f}", "dil100": f"{L('diloco_H100_b32'):.3f}",
    "hier50": f"{L('hier_Hr10_Hg50_b32'):.3f}", "hier100": f"{L('hier_Hr10_Hg100_b32'):.3f}", "max_comp_delta": f"{max(comp):.3f}",
    "ddp_constrained_min": f"{tmin(DDP, 'constrained'):.0f}", "hier8_constrained_min": f"{tmin(HI8, 'constrained'):.1f}",
    "dil8_constrained_min": f"{tmin(DI8, 'constrained'):.1f}", "ddp_poor_h": f"{tmin(DDP, 'poor') / 60:.0f}", "hier8_poor_min": f"{tmin(HI8, 'poor'):.1f}",
    "sp_regional": f"{tmin(DDP, 'regional') / tmin(HI8, 'regional'):.1f}", "sp_inter": f"{tmin(DDP, 'intercontinental') / tmin(HI8, 'intercontinental'):.1f}",
    "sp_constrained": f"{tmin(DDP, 'constrained') / tmin(HI8, 'constrained'):.0f}", "sp_poor": f"{tmin(DDP, 'poor') / tmin(HI8, 'poor'):.0f}",
    "ic_ratio": f"{rt[('intercontinental', 'flat', 32)] / rt[('intercontinental', 'flat', 8)]:.1f}",
    "ic_hier_red": f"{(1 - rt[('intercontinental', 'hier', 32)] / rt[('intercontinental', 'flat', 32)]) * 100:.0f}",
    "b2b_diff": f"{np.ceil(b2b):.0f}",
    "dist32": f"{val['constrained_b32']['dist_final_val']:.3f}", "dist8": f"{val['constrained_b8']['dist_final_val']:.3f}",
    "sim32": f"{val['constrained_b32']['sim_final_val']:.3f}", "sim8": f"{val['constrained_b8']['sim_final_val']:.3f}",
    "val_match": f"{max(abs(v['dist_final_val'] - v['sim_final_val']) for v in val.values()):.3f}",
    "dist_r32": f"{np.median(val['constrained_b32']['dist_round_times']):.2f}", "dist_r8": f"{np.median(val['constrained_b8']['dist_round_times']):.2f}",
    "bench_r32": f"{rt[('constrained', 'flat', 32)]:.2f}", "bench_r8": f"{rt[('constrained', 'flat', 8)]:.2f}",
    "under32": f"{under('constrained_b32', 32):.0f}", "under8": f"{under('constrained_b8', 8):.0f}",
}
order = [("ddp", "DDP (every step)"), ("diloco_H10_b32", "DiLoCo $H$=10")] + \
        [(f"diloco_H50_b{b}", f"DiLoCo $H$=50{'' if b == 32 else ', INT' + str(b)}") for b in (32, 8, 4)] + \
        [(f"hier_Hr10_Hg50_b{b}", f"Hierarchical 10/50{'' if b == 32 else ', INT' + str(b)}") for b in (32, 8, 4)] + \
        [(f"diloco_H100_b{b}", f"DiLoCo $H$=100{'' if b == 32 else ', INT' + str(b)}") for b in (32, 8, 4)] + \
        [(f"hier_Hr10_Hg100_b{b}", f"Hierarchical 10/100{'' if b == 32 else ', INT' + str(b)}") for b in (32, 8, 4)]
rows = []
for i, (n, lab) in enumerate(order):
    a = agg.loc[n]
    rows.append(f"{lab} & {int(a.wan_syncs):,} & {a['mean']:.3f} & [{a['min']:.3f}, {a['max']:.3f}] \\\\")
    if i in (1, 4, 7, 10): rows.append("\\addlinespace[2pt]")
num["TABLE_QUALITY"] = "\n".join(rows)
nr = [T[(T.profile == p) & (T.label == DDP)].comm_s.iloc[0] / T[(T.profile == p) & (T.label == HI8)].comm_s.iloc[0]
      for p in ["regional", "intercontinental", "constrained", "poor"]]
num["net_min"], num["net_max"] = f"{min(nr):.0f}", f"{max(nr):.0f}"
PL = {"regional": "Regional", "intercontinental": "Intercont.", "constrained": "Constrained", "poor": "Poor"}
num["TABLE_ROUNDS"] = "\n".join(
    f"{PL[p]} & {rt[(p, 'flat_b2b', 32)]:.2f} & " + " & ".join(f"{rt[(p, s, b)]:.2f}" for s in ("flat", "hier") for b in (32, 8, 4)) + " \\\\"
    for p in PL)
t = open(f"{H}/main_template.tex").read()
for k, v in num.items(): t = t.replace(f"<<{k}>>", str(v))
assert "<<" not in t, [l for l in t.splitlines() if "<<" in l][:3]
open(f"{H}/main.tex", "w").write(t)
for k, v in num.items():
    if not k.startswith("TABLE"): print(f"{k:22s} {v}")
