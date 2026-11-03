"""IEEE-sized vector figures for the low-communication training paper (needs analyze.py outputs)."""
import os, sys, json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, ".."); O = os.path.join(H, "figs")
sys.path.insert(0, ROOT); os.chdir(ROOT); from analyze import load_sim, PROFILES
INK2 = "#52514e"; GRID = "#e6e5e0"
FAM = {"ddp": "#8a8984", "diloco": "#2a78d6", "hier": "#eb6834"}; MARK = {32: "o", 8: "s", 4: "^"}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
sim, curves = load_sim()
T = pd.read_csv(os.path.join(ROOT, "results", "wallclock.csv"))

# Fig 1: learning curves (fp32), single column
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for name, lab, col, ls in [("ddp", "DDP (every step)", FAM["ddp"], "-"), ("diloco_H10_b32", "DiLoCo $H$=10", "#86b6ef", "-"),
                           ("diloco_H50_b32", "DiLoCo $H$=50", FAM["diloco"], "-"), ("diloco_H100_b32", "DiLoCo $H$=100", "#1c5cab", "--"),
                           ("hier_Hr10_Hg50_b32", "Hier. 10/50", FAM["hier"], "-"), ("hier_Hr10_Hg100_b32", "Hier. 10/100", "#b04a1c", "--")]:
    c = curves[curves.name == name].groupby("step").val.mean()
    ax.plot(c.index, c.values, color=col, lw=1.2, ls=ls, label=lab)
ax.set_xlabel("steps per worker"); ax.set_ylabel("validation loss"); ax.set_ylim(1.55, 2.5)
ax.grid(color=GRID, lw=0.5); ax.legend(frameon=False, fontsize=5.5, ncols=2, loc="upper right")
fig.savefig(f"{O}/fig_curves.pdf"); plt.close(fig)

# Fig 2: time vs quality, 4 panels, double column
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), sharey=True)
TITLES = {"regional": "Regional\n100 Mbit/s, 20 ms RTT", "intercontinental": "Intercontinental\n50 Mbit/s, 150 ms RTT",
          "constrained": "Constrained\n10 Mbit/s, 200 ms RTT", "poor": "Poor\n2 Mbit/s, 300 ms RTT"}
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
for ax, p in zip(axes, PROFILES):
    q = T[T.profile == p]
    for r in q.to_dict("records"):
        ax.scatter(r["total_s"] / 60, r["mean"], color=FAM[r["method"]], marker=MARK[int(r["bits"])], s=16, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xscale("log"); ax.grid(color=GRID, lw=0.5); ax.set_title(TITLES[p], fontsize=6.5, loc="left")
    span = q.total_s.max() / q.total_s.min()
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if span > 30 else (1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=5.5)
    ax.set_xlabel("est. training time (min)")
axes[0].set_ylabel("final validation loss")
h = [Line2D([], [], color=FAM[k], marker="o", ls="", ms=4, label=l) for k, l in [("ddp", "DDP"), ("diloco", "DiLoCo"), ("hier", "Hierarchical")]]
h += [Line2D([], [], color=INK2, marker=MARK[b], ls="", ms=4, label=("FP32" if b == 32 else f"INT{b}")) for b in (32, 8, 4)]
fig.legend(handles=h, frameon=False, ncols=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.08))
fig.savefig(f"{O}/fig_time_quality.pdf"); plt.close(fig)

# Fig 3: compression effect, single column
agg = sim.groupby(["method", "H", "H_g", "bits"]).final_val.agg(["mean", "min", "max"]).reset_index()
fig, ax = plt.subplots(figsize=(3.5, 2.1))
groups = [("diloco", 50, None, "DiLoCo 50"), ("diloco", 100, None, "DiLoCo 100"), ("hier", None, 50, "Hier. 10/50"), ("hier", None, 100, "Hier. 10/100")]
for gi, (m, Hh, Hg, lab) in enumerate(groups):
    for bi, b in enumerate([32, 8, 4]):
        q = agg[(agg.method == m) & (agg.bits == b) & ((agg.H == Hh) if Hh else (agg.H_g == Hg))]
        if not len(q): continue
        q = q.iloc[0]; x = gi + (bi - 1) * 0.24
        ax.bar(x, q["mean"] - 1.5, 0.22, bottom=1.5, color=["#1c5cab", "#3987e5", "#9ec5f4"][bi], label=("FP32" if b == 32 else f"INT{b}") if gi == 0 else None)
        ax.errorbar(x, q["mean"], yerr=[[q["mean"] - q["min"]], [q["max"] - q["mean"]]], fmt="none", ecolor=INK2, capsize=1.5, lw=0.6)
d = agg[agg.method == "ddp"]["mean"].iloc[0]
ax.axhline(d, color=FAM["ddp"], ls="--", lw=0.8); ax.text(3.5, d - 0.012, "DDP", ha="right", va="top", fontsize=6, color=INK2)
ax.set_xticks(range(4), [g[3] for g in groups]); ax.set_ylim(1.5, None); ax.set_ylabel("final validation loss")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6, ncols=3, loc="upper left")
fig.savefig(f"{O}/fig_compression.pdf"); plt.close(fig)
print(sorted(os.listdir(O)))
